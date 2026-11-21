# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Hex8 lowering to the existing explicit Procedure and result owner."""

from dataclasses import fields as dataclass_fields, replace

import numpy as np

from ._transient_problems import ExplicitDynamicsStep
from .provenance import collective_call


from ._linear_explicit_body import _material_partition, _Residual, _Energy


class UniformHexStep(ExplicitDynamicsStep):
    def solve_result(self, *, field_variables=None, fields=(), **options):
        # All ranks must enter the same live-field collectives, including when
        # one caller accidentally mixes explicit fields and named selection.
        comm = self.residual.internal.comm
        modes = comm.allgather((field_variables is None, bool(fields)))
        if any(mode != modes[0] for mode in modes):
            raise ValueError("Uniform Hex8 output mode differs across MPI ranks.")
        if field_variables is None:
            return super().solve_result(fields=fields, **options)
        if fields:
            raise ValueError(
                "Choose explicit live fields or field_variables, not both."
            )
        from .results._uniform_hex import UniformHexCellFields

        generated = UniformHexCellFields(self.residual.internal, field_variables)
        return super().solve_result(fields=(self.state.u.value, generated), **options)

    def checkpoint_capabilities(self):
        return replace(
            super().checkpoint_capabilities(),
            rank_count_portability="unsupported",
            evidence=("fixed-operator same-partition restart",),
            limitations=("no cross-partition restore",),
        )

    def save_checkpoint(self, path, *, portable=False):
        if portable:
            raise NotImplementedError("Uniform Hex8 portable restart is not verified.")
        return super().save_checkpoint(path, portable=False)

    def summary(self):
        return {
            **super().summary(),
            "element_policy": self.element_policy.summary(),
            "energy_balance_scope": "accepted_path_work_with_explicit_artificial_energy",
            "execution_scope": "small_strain_elastic_owned_cell_assembly",
            "geometry_admission": "bounded_bernstein_with_floating_point_margin",
            "interface": None
            if self.residual.cohesive is None
            else self.residual.cohesive.summary(),
        }


def lower(model, request):
    from . import constraints, fracture, input_effects, problems, state, time
    from .elements import UniformStrainHex8
    from .elements._uniform_hex_dolfinx import UniformHexResidual

    options = dict(request.options)
    policy = options.pop("element_policy")
    if not isinstance(policy, UniformStrainHex8):
        raise TypeError("Expected UniformStrainHex8 element policy.")
    if model.study.dimension != 3 or model.study.physics != "solid_mechanics":
        raise ValueError("Uniform Hex8 requires a 3D solid dynamics Study.")
    for key in ("K", "F", "material", "solver_options"):
        if options.pop(key, None) is not None:
            raise ValueError(
                f"Uniform Hex8 does not accept {key}; use registered model inputs."
            )
    options.pop("output", None)
    options.pop("history", None)
    steps = options.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("Uniform Hex8 steps must be a positive integer.")
    if model.boundary_models or model.eigenstrains:
        raise NotImplementedError(
            "Uniform Hex8 boundary models and eigenstrains are not yet supported."
        )
    selected_constraints = options.pop("constraints", None)
    assets = tuple(
        model.constraints if selected_constraints is None else selected_constraints
    )
    if any(
        not isinstance(
            item, (constraints.DirichletConstraint, constraints.TimeDependentDirichlet)
        )
        for item in constraints.constraint_assets(assets)
    ):
        raise NotImplementedError(
            "Uniform Hex8 currently accepts strong Dirichlet constraints only."
        )
    model.check(target=request.target)
    history = state.second_order_state(request.target)
    domain = history.u.value.function_space.mesh
    matrices, density = collective_call(
        lambda: _material_partition(model, domain),
        comm=domain.comm,
        label="Uniform Hex8 material partition",
    )
    internal = UniformHexResidual(
        history.u,
        matrices,
        density=density,
        hourglass_modulus=policy.hourglass_modulus,
        hourglass_scale=policy.hourglass_scale,
        chunk_size=policy.chunk_size,
    )
    external = model.external_force(request.target) if model.loads else None
    cohesive = options.pop("cohesive_force", None)
    if cohesive is not None:
        if history.u.value.function_space.mesh.comm.size > 1:
            raise NotImplementedError("Distributed nonmatching linear Hex8 history is not admitted yet.")
        from ._nonmatching_force import NonmatchingCohesiveForce
        from ._elastic_cohesive import ElasticCohesiveLaw

        if not isinstance(cohesive, NonmatchingCohesiveForce) or not isinstance(
            cohesive.assembler.law, ElasticCohesiveLaw
        ):
            raise NotImplementedError(
                "Uniform Hex8 currently composes only fixed nonmatching elastic interfaces."
            )
        if cohesive.displacement is not history.u.value:
            raise ValueError("Interface force must use the same displacement field.")
        if cohesive.assembler.pairing.method != "coplanar-affine-q1-common-refinement":
            raise ValueError("Uniform Hex8 needs original Q1 common-refinement traces.")
    residual = _Residual(internal, external, cohesive)
    dt = options.pop("dt")
    interface_bound = (
        0
        if cohesive is None
        else cohesive.elastic_stability_bound(internal.mass_diagonal)
    )
    bulk_dt = internal.stable_dt(safety=0.8)
    bulk_bound = (1.6 / bulk_dt) ** 2
    stable = 0.8 * 2 / np.sqrt(bulk_bound + interface_bound)

    def select_dt():
        selected = stable if dt == "auto" else float(dt)
        if not np.isfinite(selected) or selected <= 0 or selected > stable:
            raise ValueError(
                f"Hex8 dt must be positive and at most the conservative bound {stable:g}."
            )
        return selected

    dt = collective_call(select_dt, comm=domain.comm, label="Hex8 time increment")
    controls = domain.comm.allgather((dt, int(steps)))
    if any(value != controls[0] for value in controls):
        raise ValueError("Hex8 time increment/step count differs across ranks.")
    # Record the selected time-input contract for restart, not excluded model
    # constraints. The explicit integrator also consumes these same assets.
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
            "Uniform Hex8 requires fixed material/operator inputs."
        )
    if update is not None:
        collective_call(
            lambda: update(0.0), comm=domain.comm, label="Initial Hex8 inputs"
        )
    bcs = [item.bc for item in prescribed]
    residual.bind_initial_inputs(bcs)
    accepted = history.snapshot()
    initial = None
    try:
        constraints.apply_dirichlet_bcs(history.u, bcs)
        initial = residual.assemble_vector()
        from .kernel.dofs import assign_owned

        assign_owned(history.a, -initial.array * internal.inv_mass)
        from .time.explicit import _assign_prescribed_component, _prescribed_kinematics

        kinematics = collective_call(
            lambda: _prescribed_kinematics(prescribed, time=0.0, dt=dt),
            comm=domain.comm,
            label="Initial Hex8 prescribed kinematics",
        )
        _assign_prescribed_component(history.a, kinematics, component=1)
        _assign_prescribed_component(history.v, kinematics, component=2)
        residual.commit()
    except Exception:
        residual.rollback()
        history.restore(accepted)
        raise
    finally:
        if initial is not None:
            initial.destroy()
    options["checkpoint_policy"] = options.pop("checkpoint", None)
    options["name"] = options.get("name") or "uniform_strain_hex8_explicit"
    base = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        study=model.study,
        dt=dt,
        prescribed=prescribed,
        constraints=assets,
        update_load=update,
        history_monitor=fracture.DynamicEnergyLedger(
            energy=_Energy(internal, cohesive),
            state=history,
            mass=internal.mass_diagonal,
            residual=residual,
            natural_force=external,
            prescribed=prescribed,
        ),
        stability={
            "dt_limit": stable,
            "method": "positive_cell_mass_rayleigh_bound",
            "scope": "fixed_elastic_bulk_hourglass_and_optional_interface",
            "bulk_omega_squared_bound": bulk_bound,
            "interface_omega_squared_bound": interface_bound,
        },
        **options,
    )
    step = UniformHexStep(
        **{
            item.name: getattr(base, item.name)
            for item in dataclass_fields(base)
            if item.init
        }
    )
    step.element_policy = policy
    return model.add_step(step)
