# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite Hex8 lowering, reusing the explicit Procedure and history owner."""

from dataclasses import fields as dataclass_fields

import numpy as np

from ._transient_problems import ExplicitDynamicsStep
from .provenance import collective_call


class FiniteUniformHexStep(ExplicitDynamicsStep):
    def solve_result(self, *, field_variables=None, fields=(), **options):
        comm = self.residual.comm
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

        generated = FiniteHexCellFields(self.residual, field_variables)
        return super().solve_result(fields=(self.state.u.value, generated), **options)

    def summary(self):
        return {
            **super().summary(),
            "element_policy": self.element_policy.summary(),
            "execution_scope": "experimental_single_material_finite_reference_hex8",
            "energy_balance_scope": "accepted_work_stored_artificial_and_material_dissipation",
            "stability_scope": "caller_path_ceiling_and_signed_endpoint_screen_not_nonlinear_guarantee",
            "material": self.residual.material.summary(),
        }


def lower(model, request):
    from . import constraints, fracture, input_effects, problems, state, time
    from .constitutive.material_driver import MaterialQuadratureResponse
    from .constitutive.user_material import MaterialTangentConvention
    from .elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
    from .mechanics._finite_hex_energy import FiniteHexEnergyMonitor
    from .mechanics._finite_hex_explicit import FiniteHexExplicitResidual

    options = dict(request.options)
    policy = options.pop("element_policy")
    domain = request.target.value.function_space.mesh

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
        if model.boundary_models or model.eigenstrains:
            raise NotImplementedError(
                "Finite Hex8 boundary models/contact and eigenstrains are not admitted."
            )
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
    if update is not None:
        collective_call(
            lambda: update(0.0), comm=domain.comm, label="Finite Hex8 initial inputs"
        )
    constraints.apply_dirichlet_bcs(history.u, [item.bc for item in prescribed])
    residual = FiniteHexExplicitResidual(
        internal,
        material,
        omega_squared_bound=options.pop("omega_squared_bound"),
        maximum_negative_growth_per_increment=options.pop(
            "maximum_negative_growth_per_increment", None
        ),
        cohesive=options.pop("cohesive_force", None),
        external_force=external,
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
        dt = residual.stability.selected if dt_option == "auto" else float(dt_option)
        residual.validate_time_increment(dt)
        return dt, int(steps)

    dt, steps = collective_call(
        controls, comm=domain.comm, label="Finite Hex8 time controls"
    )
    if any(value != (dt, steps) for value in domain.comm.allgather((dt, steps))):
        raise ValueError("Finite Hex8 time controls differ across ranks.")
    from .time.explicit import _owned_dirichlet_dofs

    residual.bind_prescribed_layout(
        _owned_dirichlet_dofs([item.bc for item in prescribed])
    )
    initial = collective_call(
        lambda: material.initial_array_response(
            len(response.state.reference_field.values)
        ),
        comm=domain.comm,
        label="Finite Hex8 declared virgin response",
    )
    residual.enable_energy(initial)
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
            energy=FiniteHexEnergyMonitor(residual),
            state=history,
            mass=internal.mass_diagonal,
            residual=residual,
            natural_force=external,
            prescribed=prescribed,
        ),
        stability={
            "dt_limit": residual.stability.selected,
            "method": "declared_path_ceiling_with_signed_endpoint_screen",
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
    return model.add_step(step)
