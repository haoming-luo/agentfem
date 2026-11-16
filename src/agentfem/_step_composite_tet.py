# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Composite Tet10 lowering into the existing explicit state/result lifecycle."""

from dataclasses import fields as dataclass_fields, replace
import numpy as np
from ._transient_problems import ExplicitDynamicsStep
from ._linear_explicit_body import _material_partition, _Residual, _Energy
from .provenance import collective_call


class CompositeTetStep(ExplicitDynamicsStep):
    def solve_result(self, *, field_variables=None, fields=(), **options):
        modes = self.residual.internal.comm.allgather(
            (field_variables is None, bool(fields))
        )
        if any(mode != modes[0] for mode in modes):
            raise ValueError("Composite output mode differs across MPI ranks.")
        if field_variables is not None:
            if fields:
                raise ValueError("Choose fields or field_variables, not both.")
            from .results._composite_tet import CompositeTetCellFields

            generated = CompositeTetCellFields(self.residual.internal, field_variables)
            fields = (
                self.state.u.value,
                self.state.v.value,
                self.state.a.value,
                generated,
            )
        elif not fields:
            fields = (self.state.u.value, self.state.v.value, self.state.a.value)
        return super().solve_result(fields=fields, **options)

    def checkpoint_capabilities(self):
        return replace(
            super().checkpoint_capabilities(),
            rank_count_portability="unsupported",
            evidence=("fixed-operator same-partition restart",),
            limitations=("no cross-partition restore",),
        )

    def save_checkpoint(self, path, *, portable=False):
        if portable:
            raise NotImplementedError(
                "Composite Tet10 supports same-partition restart only."
            )
        return super().save_checkpoint(path, portable=False)

    def summary(self):
        return {
            **super().summary(),
            "element_policy": self.element_policy.summary(),
            "execution_scope": "affine_small_strain_elastic_owned_cell_assembly",
            "stabilization_energy_channel": "hourglass_energy",
            "field_interpolation": "piecewise_linear_12_subtets",
        }


def lower(model, request):
    from . import (
        constraints,
        loads,
        operators,
        state,
        time,
        problems,
        fracture,
        input_effects,
    )
    from .elements._composite_tet_dolfinx import CompositeTetResidual

    options = dict(request.options)
    policy = options.pop("element_policy", None) or getattr(
        request.target.function_space, "_agentfem_element_policy", None
    )
    existing = getattr(request.target.function_space, "_agentfem_element_policy", None)
    if existing is not None and existing != policy:
        raise ValueError("Step and field composite policies must agree.")
    for key in ("K", "F", "material", "solver_options"):
        if options.pop(key, None) is not None:
            raise ValueError(f"Composite Tet10 uses registered inputs, not {key}.")
    options.pop("output", None)
    options.pop("history", None)
    if model.study.dimension != 3 or model.study.physics != "solid_mechanics":
        raise ValueError("Composite Tet10 requires 3D solid dynamics.")
    steps = options.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if model.boundary_models or model.eigenstrains:
        raise NotImplementedError(
            "Composite Tet10 boundary models/eigenstrains are not supported."
        )
    selected = options.pop("constraints", None)
    assets = tuple(model.constraints if selected is None else selected)
    if any(
        not isinstance(
            item, (constraints.DirichletConstraint, constraints.TimeDependentDirichlet)
        )
        for item in constraints.constraint_assets(assets)
    ):
        raise NotImplementedError(
            "Composite Tet10 supports strong Dirichlet constraints only."
        )
    model.check(target=request.target)
    history = state.second_order_state(request.target)
    domain = history.u.function_space.mesh
    matrices, density = collective_call(
        lambda: _material_partition(model, domain, label="Composite Tet10"),
        comm=domain.comm,
        label="Composite material partition",
    )
    internal = CompositeTetResidual(history.u, matrices, density, policy)
    for name in ("u", "v", "a", "u_next", "v_next", "a_next", "v_mid"):
        getattr(history, name).function_space._agentfem_composite_tet10 = True
    from .elements._composite_tet_loads import CompositeDistributedLoad

    for key, method in (
        ("u", "primary_finite_element_solution"),
        ("v", "accepted_velocity"),
        ("a", "accepted_acceleration"),
    ):
        getattr(history, key).value._agentfem_processing = {
            "method": method,
            "formulation": "composite_tet10",
            "interpolation": "piecewise_linear_12_subtets",
            "coefficient_storage": "P2_nodal_layout_only",
            "abaqus_c3d10m_equivalent": False,
        }
    parts = [
        item.bind(request.target)
        if isinstance(item, loads.NodalForce)
        else CompositeDistributedLoad(internal, item)
        for item in model.loads
    ]
    external = operators.combine(*parts, name="F") if parts else None
    residual = _Residual(internal, external)
    stable = internal.stable_dt()
    requested_dt = options.pop("dt")

    def select_dt():
        dt = stable if requested_dt == "auto" else float(requested_dt)
        if not np.isfinite(dt) or dt <= 0 or dt > stable:
            raise ValueError(f"Composite Tet10 dt must be positive and <= {stable:g}.")
        return dt

    dt = collective_call(select_dt, comm=domain.comm, label="Composite time increment")
    controls = domain.comm.allgather((dt, int(steps), policy.summary()))
    if any(item != controls[0] for item in controls):
        raise ValueError("Composite Tet10 controls differ across MPI ranks.")
    prescribed = tuple(constraints.dirichlet_constraints(assets))
    callbacks = [model._time_update_callback(include_constraints=False)]
    callbacks += [
        input_effects.from_asset(item)
        for item in prescribed
        if callable(getattr(item, "update", None))
    ]
    update = input_effects.compose(*callbacks)
    if time.input_summary(update)["changes_operator"]:
        raise NotImplementedError(
            "Composite Tet10 requires fixed material/operator inputs."
        )
    if update is not None:
        collective_call(
            lambda: update(0.0), comm=domain.comm, label="Composite initial inputs"
        )
    bcs = [item.bc for item in prescribed]
    residual.bind_initial_inputs(bcs)
    accepted = history.snapshot()
    initial = None
    try:
        constraints.apply_dirichlet_bcs(history.u, bcs)
        initial = residual.assemble_vector()
        from .kernel.dofs import assign_owned
        from .time.explicit import _assign_prescribed_component, _prescribed_kinematics

        assign_owned(history.a, -initial.array * internal.inv_mass)
        kinematics = collective_call(
            lambda: _prescribed_kinematics(prescribed, time=0.0, dt=dt),
            comm=domain.comm,
            label="Composite initial constraints",
        )
        _assign_prescribed_component(history.a, kinematics, component=1)
        _assign_prescribed_component(history.v, kinematics, component=2)
    except Exception:
        history.restore(accepted)
        raise
    finally:
        if initial is not None:
            initial.destroy()
    options["checkpoint_policy"] = options.pop("checkpoint", None)
    options["name"] = options.get("name") or "composite_tet10_explicit"
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
            energy=_Energy(internal),
            state=history,
            mass=internal.mass_diagonal,
            residual=residual,
            natural_force=external,
            prescribed=prescribed,
        ),
        stability={
            "dt_limit": stable,
            "method": "positive_cell_mass_rayleigh_bound",
            "scope": "fixed_composite_elasticity_and_projection_stabilization",
        },
        **options,
    )
    step = CompositeTetStep(
        **{f.name: getattr(base, f.name) for f in dataclass_fields(base) if f.init}
    )
    step.element_policy = policy
    return model.add_step(step)
