# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Time-domain dynamics and modal Step builders."""

from __future__ import annotations

import numpy as np

from . import constraints as constraint_api
from . import loads as load_api
from . import state as state_api
from .operators.core import LumpedMassOperator


def explicit_dynamics(
    model,
    *,
    target,
    dt: float,
    steps: int,
    residual=None,
    state=None,
    mass=None,
    cohesive_force=None,
    prescribed=(),
    constraints=None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint=None,
    name: str = "explicit_dynamics",
):
    """Build small-strain or expert-residual central-difference dynamics."""

    from . import problems
    from . import time as time_api
    from .constitutive import hyperelasticity

    if (
        residual is None
        and len(model.materials) == 1
        and hyperelasticity.is_finite_strain_hyperelastic(model.materials[0].item)
    ):
        if prescribed:
            raise ValueError(
                "Use registered constraints for automatic finite-strain "
                "Explicit, or pass an expert residual explicitly."
            )
        return finite_strain_explicit_dynamics(
            model,
            target=target,
            dt=dt,
            steps=steps,
            material=model.materials[0].item,
            state=state,
            mass=mass,
            cohesive_force=cohesive_force,
            constraints=constraints,
            update_load=update_load,
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint=checkpoint,
            name=name,
        )
    selected_constraints = model.constraints if constraints is None else constraints
    model.check(
        target=target,
        step_options={
            "mass": mass,
            "residual": residual,
            "method": "central_difference",
            "dt": dt,
            "steps": steps,
            "constraints": selected_constraints,
        },
    )
    selected_state = (
        state if state is not None else state_api.second_order_state(target)
    )
    selected_mass = mass if mass is not None else model.lumped_mass(target)
    integrator = time_api.explicit.central_difference(
        state=selected_state,
        mass=selected_mass,
    )
    energy_stiffness = None
    if residual is None:
        energy_stiffness = model.stiffness(target)
        residual = model.force_balance(
            internal=model.internal_force(selected_state.u),
            external=(model.external_force(target) if model.loads else None),
        )
    selected_prescribed = tuple(_as_tuple(prescribed)) + tuple(
        constraint_api.dirichlet_constraints(selected_constraints)
    )
    step = problems.explicit_dynamics(
        state=selected_state,
        integrator=integrator,
        residual=residual,
        stiffness=energy_stiffness,
        study=model.study,
        prescribed=selected_prescribed,
        constraints=selected_constraints,
        update_load=model._time_update_callback(
            update_load,
            include_constraints=False,
        ),
        dt=dt,
        steps=steps,
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint,
        name=name,
    )
    return model.add_step(step)


def finite_strain_explicit_dynamics(
    model,
    *,
    target,
    dt: float | str | None = "auto",
    steps: int,
    material=None,
    state=None,
    mass=None,
    cohesive_force=None,
    constraints=None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    history_every: int = 1,
    progress=True,
    status_file=None,
    checkpoint=None,
    stability_safety: float = 0.8,
    mass_damping: float = 0.0,
    name: str = "finite_strain_explicit_dynamics",
):
    """Build current-state Total-Lagrangian central-difference dynamics."""

    import ufl

    from . import fracture, problems
    from . import time as time_api
    from .constitutive import hyperelasticity

    selected_constraints = model.constraints if constraints is None else constraints
    model.check(
        target=target,
        step_options={
            "material": material,
            "method": "central_difference",
            "dt": dt,
            "steps": steps,
            "constraints": selected_constraints,
        },
    )
    if hasattr(model.study, "require"):
        model.study.require(analysis="second_order_dynamics", physics="solid_mechanics")
    record = (
        _single_material(model, "model.step with finite-strain Explicit")
        if material is None
        else model._material_record(material)
    )
    properties = record.item
    if not hyperelasticity.is_finite_strain_hyperelastic(properties):
        raise TypeError(
            "model.step with finite-strain Explicit requires a supported "
            "hyperelastic material."
        )
    if not hyperelasticity.supports_hyperelastic_study(
        properties,
        dimension=getattr(model.study, "dimension", 0),
        assumption=getattr(model.study, "assumption", None),
    ):
        raise ValueError(
            "The Study dimension/assumption has no formulation for the "
            f"selected hyperelastic material {properties.name!r}."
        )
    if properties.density is None:
        raise ValueError("Finite-strain Explicit requires material density.")
    selected_measure = record.region.measure if record.region is not None else ufl.dx
    selected_state = (
        state if state is not None else state_api.second_order_state(target)
    )
    if cohesive_force is not None:
        cohesive_force = cohesive_force.for_displacement(selected_state.u)
    selected_mass = (
        mass
        if mass is not None
        else LumpedMassOperator.assemble(
            _space(target),
            density=properties.density,
            measure=selected_measure,
        )
    )
    if hyperelasticity.is_plane_stress_hyperelastic(properties):
        reference_gradient = np.eye(2)
        membrane_modes = (
            fracture.incremental_wave_speeds(
                reference_gradient,
                direction,
                properties,
                direction_configuration="reference",
            )
            for direction in ((1.0, 0.0), (0.0, 1.0))
        )
        body_screening_speed = max(
            float(mode.reference_speeds[-1]) for mode in membrane_modes
        )
    else:
        body_screening_speed = fracture.isotropic_reference_wave_speeds(
            properties
        ).pressure
    interface_stability = (
        {} if cohesive_force is None else cohesive_force.stability_inputs(selected_mass)
    )
    stability = fracture.estimate_stable_time_increment(
        characteristic_length=fracture.minimum_cell_nodal_spacing(_domain(model.mesh)),
        dilatational_speed=body_screening_speed,
        safety_factor=stability_safety,
        **interface_stability,
    )
    if dt is None or str(dt).strip().lower() == "auto":
        selected_dt = stability.selected
    else:
        selected_dt = float(dt)
        if selected_dt <= 0.0:
            raise ValueError("Finite-strain Explicit requires dt > 0.")
        if selected_dt > stability.selected:
            raise ValueError(
                "The requested dt exceeds the current body/interface screening limit "
                f"({selected_dt:.6g} > {stability.selected:.6g}; "
                f"controller={stability.controller})."
            )
    internal = fracture.finite_strain_internal_force(
        selected_state.u,
        target.test,
        properties,
        measure=selected_measure,
    )
    external = model.external_force(target) if model.loads else None
    residual = model.force_balance(internal=internal, external=external)
    if cohesive_force is not None:
        residual = fracture.FiniteStrainCohesiveResidual(residual, cohesive_force)
    damping_residual = None
    if float(mass_damping) != 0.0:
        damping_residual = fracture.MassProportionalDampingResidual(
            residual,
            mass=selected_mass,
            velocity=selected_state.v_mid,
            coefficient=mass_damping,
            dt=selected_dt,
        )
        residual = damping_residual
    selected_prescribed = constraint_api.dirichlet_constraints(selected_constraints)
    base_energy = (
        fracture.FiniteStrainEnergyMonitor(
            mass=selected_mass,
            material=properties,
            measure=selected_measure,
        )
        if cohesive_force is None
        else fracture.FiniteStrainCohesiveEnergyMonitor(
            bulk=fracture.FiniteStrainEnergyMonitor(
                mass=selected_mass,
                material=properties,
                measure=selected_measure,
            ),
            cohesive=cohesive_force,
        )
    )
    if damping_residual is not None:
        base_energy = fracture.DampingEnergyMonitor(
            energy=base_energy,
            damping=damping_residual,
        )
    integrator = time_api.explicit.central_difference(
        state=selected_state,
        mass=selected_mass,
    )
    step = problems.explicit_dynamics(
        state=selected_state,
        integrator=integrator,
        residual=residual,
        stiffness=None,
        study=model.study,
        prescribed=selected_prescribed,
        constraints=selected_constraints,
        update_load=model._time_update_callback(
            update_load,
            include_constraints=False,
        ),
        dt=selected_dt,
        steps=steps,
        save_every=save_every,
        print_every=print_every,
        history_every=history_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint,
        history_monitor=fracture.DynamicEnergyLedger(
            energy=base_energy,
            state=selected_state,
            mass=selected_mass,
            residual=residual,
            natural_force=external,
            prescribed=selected_prescribed,
        ),
        stability=stability,
        name=name,
    )
    return model.add_step(step)


def modal(
    model,
    *,
    target,
    modes: int,
    M=None,
    K=None,
    constraints=None,
    target_frequency: float | None = None,
    tolerance: float = 1.0e-9,
    maximum_iterations: int = 1000,
    rigid_mode_tolerance: float = 1.0e-10,
    name: str = "modal_analysis",
):
    """Build a constrained linear structural modal analysis."""

    from . import problems

    selected_constraints = model.constraints if constraints is None else constraints
    model.check(
        target=target,
        step_options={
            "M": M,
            "K": K,
            "modes": modes,
            "constraints": selected_constraints,
        },
    )
    step = problems.modal_analysis(
        target=target,
        mass=model.mass(target) if M is None else M,
        stiffness=model.stiffness(target) if K is None else K,
        modes=modes,
        study=model.study,
        constraints=selected_constraints,
        target_frequency=target_frequency,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        rigid_mode_tolerance=rigid_mode_tolerance,
        name=name,
    )
    return model.add_step(step)


def implicit_dynamics(
    model,
    *,
    target,
    dt: float,
    steps: int,
    method: str = "newmark",
    spectral_radius: float = 0.8,
    M=None,
    C=None,
    K=None,
    F=None,
    state=None,
    constraints=None,
    solver_options=None,
    update_load=None,
    progress=True,
    status_file=None,
    checkpoint=None,
    save_every: int | None = None,
    print_every: int | None = None,
    name: str = "implicit_dynamics",
):
    """Build Newmark or generalized-alpha structural dynamics."""

    from . import problems
    from . import time as time_api

    selected_constraints = model.constraints if constraints is None else constraints
    model.check(
        target=target,
        step_options={
            "M": M,
            "K": K,
            "F": F,
            "method": method,
            "dt": dt,
            "steps": steps,
            "constraints": selected_constraints,
        },
    )
    selected_state = (
        state if state is not None else state_api.second_order_state(target)
    )
    selected_method = method.lower().replace("-", "_")
    if selected_method == "newmark":
        parameters = time_api.newmark()
    elif selected_method == "generalized_alpha":
        parameters = time_api.generalized_alpha(spectral_radius=spectral_radius)
    else:
        raise ValueError(
            "Implicit dynamics method must be 'newmark' or 'generalized_alpha'."
        )
    step = problems.implicit_dynamics(
        state=selected_state,
        mass=model.mass(target) if M is None else M,
        damping=C,
        stiffness=model.stiffness(target) if K is None else K,
        force=model.external_force(target) if F is None else F,
        dt=dt,
        steps=steps,
        parameters=parameters,
        study=model.study,
        constraints=selected_constraints,
        solver_options=solver_options,
        update_load=model._time_update_callback(update_load),
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint,
        save_every=save_every,
        print_every=print_every,
        name=name,
    )
    return model.add_step(step)


def _single_material(model, caller: str):
    if len(model.materials) != 1:
        raise ValueError(f"{caller} requires material=... or exactly one material.")
    return model.materials[0]


def _as_tuple(item) -> tuple:
    if item is None:
        return ()
    if isinstance(item, tuple):
        return item
    if isinstance(item, list):
        return tuple(item)
    return (item,)


def _domain(mesh):
    if mesh is None:
        return None
    return getattr(mesh, "domain", mesh)


def _space(target):
    if hasattr(target, "space"):
        return target.space
    if hasattr(target, "function_space"):
        return target.function_space
    if hasattr(target, "value") and hasattr(target.value, "function_space"):
        return target.value.function_space
    return target


__all__ = ()
