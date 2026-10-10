# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite Hex8 lowering, reusing the explicit Procedure and history owner."""

from dataclasses import fields as dataclass_fields

import numpy as np

from ._transient_problems import ExplicitDynamicsStep
from .provenance import collective_call, collective_canonical_record


class FiniteUniformHexStep(ExplicitDynamicsStep):
    @property
    def material_residual(self):
        """The constitutive owner, distinct from a composed contact residual."""
        return getattr(self, "_material_residual", self.residual)

    def operator_lifecycle_summary(self):
        return {
            **super().operator_lifecycle_summary(),
            "stability_scope": (
                "caller_path_ceiling_with_signed_endpoint_screen"
                if self.material_residual.material_envelope is None
                else self.material_residual.summary()["stability_scope"]
            ),
            "material_tangent_policy": "accepted_trial_recomputed_each_increment",
        }

    def solve_result(self, *, field_variables=None, fields=(), **options):
        comm = self.material_residual.comm
        modes = comm.allgather((field_variables is None, bool(fields)))
        if any(mode != modes[0] for mode in modes):
            raise ValueError("Finite Hex8 output mode differs across ranks.")
        if field_variables is None:
            return super().solve_result(fields=fields, **options)
        if fields:
            raise ValueError(
                "Choose explicit live fields or field_variables, not both."
            )
        from .results._finite_hex import FiniteHexCellFields

        generated = FiniteHexCellFields(self.material_residual, field_variables)
        return super().solve_result(fields=(self.state.u.value, generated), **options)

    def summary(self):
        return {
            **super().summary(),
            "element_policy": self.element_policy.summary(),
            "execution_scope": "experimental_single_material_finite_reference_hex8",
            "contact_scope": (
                "none" if self.residual is self.material_residual else
                "single_frictionless_translating_plane_reference_surface_penalty"
            ),
            "bond_contact_scope": (
                "disjoint_trace_nodes_isotropic_elastic_reference_bond"
                if self.residual is not self.material_residual
                and self.material_residual.cohesive is not None else "none"
            ),
            "energy_balance_scope": "accepted_work_stored_artificial_and_material_dissipation",
            "stability_scope": (
                "caller_path_ceiling_and_signed_endpoint_screen_not_nonlinear_guarantee"
                if self.material_residual.material_envelope is None
                else self.material_residual.summary()["stability_scope"]
            ),
            "material_stability_envelope": self.material_residual.summary()["material_stability_envelope"],
            "material": self.material_residual.material.summary(),
        }


def lower(model, request):
    from .state import field_transaction

    # Construction can apply prescribed values before a later admission check
    # fails. Keep the user's field untouched on failure, just as an increment
    # protects accepted State. Register only after the transaction succeeds.
    with field_transaction(displacement=request.target):
        step = _prepare(model, request)
    return model.add_step(step)


def _prepare(model, request):
    from . import boundary_models, constraints, fracture, input_effects, problems, state, time
    from .constitutive.material_driver import MaterialQuadratureResponse
    from .constitutive.user_material import MaterialTangentConvention
    from .elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
    from .mechanics._finite_hex_energy import FiniteHexEnergyMonitor
    from .mechanics._finite_hex_explicit import FiniteHexExplicitResidual

    options = dict(request.options)
    policy = options.pop("element_policy")
    domain = request.target.value.function_space.mesh
    contact_pairs = tuple(model.boundary_models)

    def admit_assets():
        if model.study.dimension != 3 or model.study.physics != "solid_mechanics":
            raise ValueError("Finite Hex8 requires a 3D solid dynamics Study.")
        for key in ("K", "F", "material", "solver_options"):
            if options.pop(key, None) is not None:
                raise ValueError(
                    f"Finite Hex8 does not accept {key}; use registered model assets."
                )
        options.pop("output", None)
        options.pop("history", None)
        if model.eigenstrains:
            raise NotImplementedError(
                "Finite Hex8 eigenstrains are not admitted."
            )
        if contact_pairs:
            if len(contact_pairs) != 1:
                raise NotImplementedError("Finite Hex8 contact is not admitted beyond one pair.")
            pair = contact_pairs[0]
            if (not isinstance(pair, boundary_models.RigidContactPair)
                    or not isinstance(pair.rigid_body.surface, boundary_models.RigidPlaneSurface)
                    or pair.friction is not None):
                raise NotImplementedError("Finite Hex8 contact is not admitted beyond a frictionless rigid plane.")
            schedule = pair.rigid_body.motion_schedule
            if schedule is not None and np.any(np.asarray(schedule.motion.rotation) != 0):
                raise NotImplementedError("Finite Hex8 rotating contact is not admitted yet.")
        selected = options.pop("constraints", None)
        assets = tuple(model.constraints if selected is None else selected)
        if any(
            not isinstance(
                item,
                (constraints.DirichletConstraint, constraints.TimeDependentDirichlet),
            )
            for item in constraints.constraint_assets(assets)
        ):
            raise NotImplementedError(
                "Finite Hex8 currently requires strong Dirichlet constraints."
            )
        return assets

    assets = collective_call(
        admit_assets, comm=domain.comm, label="Finite Hex8 model admission"
    )
    contact_contract = collective_call(
        lambda: [pair.scientific_identity for pair in contact_pairs],
        comm=domain.comm, label="Finite Hex8 contact description",
    )
    collective_canonical_record(
        contact_contract, comm=domain.comm, label="Finite Hex8 contact contract",
    )

    def select_material():
        if len(model.materials) != 1 or model.materials[0].region is not None:
            raise NotImplementedError(
                "Finite Hex8 currently requires one full-domain material; regional history dispatch is not admitted."
            )
        material = model.materials[0].item
        if not callable(getattr(material, "initial_array_response", None)):
            raise TypeError(
                "Finite Hex8 material must declare initial_array_response without a fictitious material increment."
            )
        if (
            material.tangent_convention
            != MaterialTangentConvention.first_piola_deformation_gradient()
        ):
            raise ValueError(
                "Finite Hex8 requires a work-conjugate dP/dF material tangent."
            )
        density = getattr(material, "density", None)
        if density is None or not np.isfinite(density) or density <= 0:
            raise ValueError("Finite Hex8 material requires positive finite density.")
        return material

    material = collective_call(
        select_material, comm=domain.comm, label="Finite Hex8 material admission"
    )
    # Validate field-creation semantics before ranks allocate differing spaces.
    # A provider's friendly summary is not a substitute for the actual schema.
    field_contract = collective_call(
        lambda: {
            "state_schema": material.state_schema.summary(),
            "tangent": material.tangent_convention.summary(),
            "density": float(material.density),
            "energy_components": list(
                getattr(material, "stored_energy_component_names", ())
            ),
        },
        comm=domain.comm,
        label="Finite Hex8 material field description",
    )
    collective_canonical_record(
        field_contract,
        comm=domain.comm,
        label="Finite Hex8 material field contract",
    )
    # Preserve the selected formulation during readiness validation; otherwise
    # a generic material is incorrectly re-tested against the default solver.
    collective_call(
        lambda: model.check(target=request.target, step_options=request.options),
        comm=domain.comm,
        label="Finite Hex8 model validation",
    )
    history = state.second_order_state(request.target)
    response = MaterialQuadratureResponse.create(
        domain,
        material.state_schema,
        degree=1,
        stored_energy_component_names=getattr(
            material, "stored_energy_component_names", ()
        ),
    )
    internal = FiniteUniformHexResidual(
        history.u,
        response,
        density=material.density,
        hourglass_modulus=policy.hourglass_modulus,
        hourglass_scale=policy.hourglass_scale,
        chunk_size=policy.chunk_size,
    )

    def prepare_stability():
        from .constitutive.stability import FirstPiolaTangentEnvelope

        ceiling = options.pop("omega_squared_bound", None)
        if ceiling is not None:
            return float(ceiling), None
        provider = getattr(material, "explicit_stability_envelope", None)
        if not callable(provider):
            raise ValueError(
                "Finite Hex8 requires omega_squared_bound or a material-owned explicit_stability_envelope."
            )
        envelope = provider()
        if not isinstance(envelope, FirstPiolaTangentEnvelope):
            raise TypeError("Expected a FirstPiolaTangentEnvelope from the material provider.")
        upper, _ = internal.cells.tangent_envelope_bounds(
            positive_modulus=envelope.positive_modulus,
            negative_modulus=envelope.negative_modulus,
        )
        return upper, envelope

    ceiling, envelope = collective_call(
        prepare_stability, comm=domain.comm, label="Finite Hex8 material stability declaration"
    )
    collective_canonical_record(
        None if envelope is None else envelope.summary(),
        comm=domain.comm,
        label="Finite Hex8 material stability domain",
    )
    if envelope is not None:
        from mpi4py import MPI

        ceiling = domain.comm.allreduce(ceiling, op=MPI.MAX)

    def prepare_external():
        from ufl.algorithms import extract_coefficients

        external = model.external_force(request.target) if model.loads else None
        if external is None:
            return None
        expression = getattr(external, "expression", external)
        if any(
            coefficient is history.u.value
            for coefficient in extract_coefficients(expression)
        ):
            raise NotImplementedError(
                "Finite Hex8 natural loads must not depend on displacement; follower loads need a separate tangent bound."
            )
        return external

    external = collective_call(
        prepare_external, comm=domain.comm, label="Finite Hex8 natural loads"
    )
    prescribed = tuple(constraints.dirichlet_constraints(assets))

    def prepare_inputs():
        callbacks = [model._time_update_callback(include_constraints=False)]
        callbacks.extend(
            input_effects.from_asset(item)
            for item in prescribed
            if callable(getattr(item, "update", None))
        )
        update = input_effects.compose(*callbacks)
        if time.input_summary(update)["changes_operator"]:
            raise NotImplementedError(
                "Finite Hex8 does not admit externally changing mass/material operators."
            )
        return update

    update = collective_call(
        prepare_inputs, comm=domain.comm, label="Finite Hex8 input admission"
    )
    if update is not None:
        collective_call(
            lambda: update(0.0), comm=domain.comm, label="Finite Hex8 initial inputs"
        )
    constraints.apply_dirichlet_bcs(history.u, [item.bc for item in prescribed])
    residual = FiniteHexExplicitResidual(
        internal,
        material,
        omega_squared_bound=ceiling,
        material_envelope=envelope,
        maximum_negative_growth_per_increment=options.pop(
            "maximum_negative_growth_per_increment", None
        ),
        cohesive=options.pop("cohesive_force", None),
        external_force=external,
    )
    material_residual = residual
    stability = material_residual.stability
    if contact_pairs:
        pair = contact_pairs[0]
        if material_residual.cohesive is not None:
            from dolfinx import fem

            region = pair.slave_boundary
            contact_nodes = fem.locate_dofs_topological(
                history.u.value.function_space, domain.topology.dim - 1,
                region.facet_tags.find(region.tag),
            )
            bonded_nodes = np.union1d(
                material_residual.cohesive.negative_dofs,
                material_residual.cohesive.positive_dofs,
            )
            if domain.comm.allreduce(bool(np.intersect1d(contact_nodes, bonded_nodes[bonded_nodes >= 0]).size)):
                raise NotImplementedError(
                    "Finite Hex8 combined bonding/contact requires disjoint trace nodes; "
                    "overlap or post-failure contact switching is not admitted."
                )
        adapter = boundary_models.dolfinx_boundary_region_contact_trace(
            pair.slave_boundary, history.u.value.function_space,
        )
        estimate = boundary_models.estimate_dolfinx_contact_stability(
            adapter=adapter, lumped_mass=internal.mass_diagonal,
            normal_penalty=pair.law.penalty, safety_factor=material_residual.safety,
        )
        stability = time.combine_explicit_stability(
            (*stability.contributions, time.ExplicitStabilityContribution.from_time_increment(
                f"contact:{pair.name}", estimate.unsafed_limit,
                method="contact_trace_mass_spectral_bound",
            )), safety_factor=material_residual.safety,
        )
        residual = boundary_models.dolfinx_explicit_contact_residual(
            material_residual, adapter=adapter, displacement=history.u.value,
            contact_pair=pair, maximum_stable_time_increment=stability.selected,
            contact_stability_estimate=estimate, name=pair.name,
        )
    dt_option = options.pop("dt")

    def controls():
        steps = options.get("steps")
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 1
        ):
            raise ValueError("Finite Hex8 steps must be a positive integer.")
        dt = stability.selected if dt_option == "auto" else float(dt_option)
        material_residual.validate_time_increment(dt)
        residual.validate_time_increment(dt)
        return dt, int(steps)

    dt, steps = collective_call(
        controls, comm=domain.comm, label="Finite Hex8 time controls"
    )
    if any(value != (dt, steps) for value in domain.comm.allgather((dt, steps))):
        raise ValueError("Finite Hex8 time controls differ across ranks.")
    from .time.explicit import _owned_dirichlet_dofs

    material_residual.bind_prescribed_layout(
        _owned_dirichlet_dofs([item.bc for item in prescribed])
    )
    initial = collective_call(
        lambda: material.initial_array_response(
            len(response.state.reference_field.values)
        ),
        comm=domain.comm,
        label="Finite Hex8 declared virgin response",
    )
    material_residual.enable_energy(initial)
    if contact_pairs:
        residual.initialize_accepted_state(time=0.0)
    vector = residual.assemble_accepted_vector()
    try:
        from .kernel.dofs import assign_owned

        assign_owned(history.a, -vector.array * internal.inv_mass)
    finally:
        vector.destroy()
    from .time.explicit import _assign_prescribed_component, _prescribed_kinematics

    motion = collective_call(
        lambda: _prescribed_kinematics(prescribed, time=0.0, dt=dt),
        comm=domain.comm,
        label="Finite Hex8 initial kinematics",
    )
    _assign_prescribed_component(history.a, motion, component=1)
    _assign_prescribed_component(history.v, motion, component=2)
    options["checkpoint_policy"] = options.pop("checkpoint", None)
    options["name"] = options.get("name") or "finite_uniform_strain_hex8_explicit"
    base = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        prescribed=prescribed,
        constraints=assets,
        update_load=update,
        dt=dt,
        study=model.study,
        history_monitor=fracture.DynamicEnergyLedger(
            energy=FiniteHexEnergyMonitor(material_residual),
            state=history,
            mass=internal.mass_diagonal,
            residual=residual,
            natural_force=external,
            prescribed=prescribed,
        ),
        stability=stability if contact_pairs else {
            "dt_limit": stability.selected,
            "method": material_residual.summary()["stability_scope"],
            "scope": "experimental_not_general_nonlinear_stability",
        },
        **options,
    )
    step = FiniteUniformHexStep(
        **{
            item.name: getattr(base, item.name)
            for item in dataclass_fields(base)
            if item.init
        }
    )
    step.element_policy = policy
    step._material_residual = material_residual
    return step
