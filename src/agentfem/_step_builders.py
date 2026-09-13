"""Internal scientific builders used by public Step providers.

The stable user entry point is :meth:`agentfem.models.Model.step`.  Builders
live here so the Model remains a registry/facade rather than accumulating the
construction logic of every material and solution procedure.  Compatibility
methods on Model delegate to these functions throughout the 0.2.x series.
"""

from __future__ import annotations

import numpy as np

from . import constraints as constraint_api
from . import loads as load_api
from . import state as state_api
from .operators.core import LumpedMassOperator
from ._step_builders_thermal import heat_transfer, linear_static


def j2_plasticity(
    model,
    *,
    target,
    material=None,
    constraints=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    progress=True,
    status_file=None,
    amplitude=None,
    name: str = "j2_plasticity",
):
    """Build and register a global 3D small-strain J2 Step."""

    from . import mechanics
    from .constitutive.plasticity import (
        ChabocheCombinedHardening,
        J2LinearIsotropicHardening,
    )
    from .constitutive.quadrature import QuadratureMaterialMap

    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    material = _quadrature_material(
        model,
        target,
        material,
        material_type=(J2LinearIsotropicHardening, ChabocheCombinedHardening),
        label="model.step with J2 plasticity",
    )
    if not isinstance(
        material,
        (J2LinearIsotropicHardening, ChabocheCombinedHardening, QuadratureMaterialMap),
    ):
        raise TypeError("model.step requires a supported small-strain J2 material.")

    time_dependent_constraints = tuple(
        item
        for item in constraint_api.dirichlet_constraints(
            model.constraints if constraints is None else constraints
        )
        if isinstance(item, constraint_api.TimeDependentDirichlet)
    )
    if time_dependent_constraints:
        raise NotImplementedError(
            "The J2 step does not accept absolute time-dependent Dirichlet "
            "histories. Prescribe the end-of-step value and use the step "
            "amplitude as its dimensionless load path."
        )

    selected_loads, amplitude = _single_shared_amplitude_loads(
        model.loads,
        amplitude,
        label="J2",
        physical_time=False,
    )
    step = mechanics.j2_plasticity_step(
        displacement=target,
        material=material,
        external_force=(
            model.external_force(target, loads=selected_loads)
            if selected_loads
            else None
        ),
        constraints=model.constraints if constraints is None else constraints,
        study=model.study,
        incrementation=incrementation,
        solver_options=solver_options,
        quadrature_degree=quadrature_degree,
        progress=progress,
        status_file=status_file,
        amplitude=amplitude,
        name=name,
    )
    return model.add_step(step)


def finite_strain_j2(
    model,
    *,
    target,
    material=None,
    constraints=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    output=None,
    output_every: int | None = None,
    checkpoint=None,
    amplitude=None,
    progress=True,
    status_file=None,
    name: str = "finite_strain_j2",
):
    """Build stateful finite-strain J2 for supported global formulations."""

    from . import mechanics
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constitutive import QuadratureMaterialMap

    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    dimension = int(getattr(model.study, "dimension", -1))
    mixed_target = getattr(target, "kind", None) == "displacement_pressure"
    if dimension not in {2, 3} or (dimension == 2 and not mixed_target):
        raise NotImplementedError(
            "Finite-strain J2 currently requires a 3D Study or the explicit "
            "2D plane-strain mixed formulation."
        )
    if dimension == 2 and getattr(model.study, "assumption", None) != "plane_strain":
        raise NotImplementedError(
            "Two-dimensional mixed finite-strain J2 currently requires "
            "study assumption='plane_strain'."
        )
    if mixed_target:
        domain = target.space.mesh
        required_cell = "tetrahedron" if dimension == 3 else "quadrilateral"
        actual_cell = str(domain.topology.cell_name())
        if actual_cell != required_cell:
            raise ValueError(
                "Mixed finite-strain J2 currently has formulation evidence "
                f"only for {required_cell} cells in {dimension}D; received "
                f"{actual_cell}."
            )
        pressure_family = str(getattr(target, "pressure_family", "DG")).upper()
        interpolation = (
            int(getattr(target, "displacement_degree", -1)),
            pressure_family,
            int(getattr(target, "pressure_degree", -1)),
        )
        required = (2, "DG", 0) if dimension == 3 else (2, "DPC", 1)
        if interpolation != required:
            label = "P2/DG0" if dimension == 3 else "Q2/DPC1"
            raise ValueError(
                f"Mixed finite-strain J2 currently requires the {label} "
                "displacement-pressure unknown."
            )
        if hasattr(model.mesh, "require_formulation"):
            model.mesh.require_formulation(
                "hybrid",
                operation="model.step with mixed finite-strain J2",
            )
    properties = _quadrature_material(
        model,
        target,
        material,
        material_type=FiniteStrainJ2Logarithmic,
        label="model.step with finite-strain J2",
    )
    if not isinstance(
        properties,
        (FiniteStrainJ2Logarithmic, QuadratureMaterialMap),
    ):
        raise TypeError(
            "model.step finite-strain J2 requires one or more regional "
            "FiniteStrainJ2Logarithmic materials."
        )
    selected_constraints = _as_tuple(
        model.constraints if constraints is None else constraints
    )
    constraint_assets = constraint_api.constraint_assets(selected_constraints)
    affine = tuple(
        item
        for item in constraint_assets
        if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
    )
    if affine and (len(affine) != 1 or len(constraint_assets) != 1):
        raise NotImplementedError(
            "Finite-strain J2 cannot mix an affine/MPC constraint with ordinary "
            "strong boundary constraints in one Step."
        )
    if mixed_target and not affine:
        raise NotImplementedError(
            "The first mixed finite-strain J2 provider requires one exact "
            "AbaqusPeriodicConstraint. Ordinary strong-boundary mixed J2 "
            "needs a separate block-aware boundary lowering."
        )
    if output is not None and output_every is not None:
        raise ValueError("Pass output=... or output_every=..., not both.")
    selected_output_every = (
        getattr(output, "every", None)
        if output is not None
        else (None if output_every is None else int(output_every))
    )
    output_factors = (
        output.required_factors()
        if output is not None and hasattr(output, "required_factors")
        else ()
    )
    if affine:
        physical_loads = load_api.load_assets(
            model.loads,
            unwrap_amplitudes=True,
        )
        if physical_loads:
            raise NotImplementedError(
                "Affine finite-strain J2 currently accepts prescribed "
                "macroscopic deformation without body-force or natural-load "
                "power."
            )
        if amplitude is not None:
            raise NotImplementedError(
                "Affine finite-strain J2 reads its macroscopic deformation "
                "path from AbaqusPeriodicConstraint; do not also pass amplitude=."
            )
        if mixed_target:
            problem = mechanics.finite_strain_j2_mixed_affine_problem(
                target=target,
                material=properties,
                constraint=affine[0],
                incrementation=incrementation,
                solver_options=solver_options,
                quadrature_degree=quadrature_degree,
                output_every=selected_output_every,
                output_factors=output_factors,
                progress=progress,
                status_file=status_file,
                checkpoint_policy=checkpoint,
                name=name,
            )
        else:
            problem = mechanics.finite_strain_j2_affine_problem(
                displacement=target,
                material=properties,
                constraint=affine[0],
                incrementation=incrementation,
                solver_options=solver_options,
                quadrature_degree=quadrature_degree,
                output_every=selected_output_every,
                output_factors=output_factors,
                progress=progress,
                status_file=status_file,
                checkpoint_policy=checkpoint,
                name=name,
            )
        has_external_power = False
    else:
        concrete = constraint_assets
        if not concrete:
            raise ValueError(
                "Standard finite-strain J2 requires explicit strong boundary "
                "constraints that remove rigid-body motion."
            )
        unsupported = tuple(
            item
            for item in concrete
            if not isinstance(
                item,
                (
                    constraint_api.DirichletConstraint,
                    constraint_api.RemoteDisplacementConstraint,
                    constraint_api.TimeDependentDirichlet,
                ),
            )
        )
        if unsupported:
            names = tuple(type(item).__name__ for item in unsupported)
            raise NotImplementedError(
                "Standard finite-strain J2 currently requires ordinary strong "
                f"Dirichlet constraints; unsupported={names}."
            )
        time_dependent = tuple(
            item
            for item in concrete
            if isinstance(item, constraint_api.TimeDependentDirichlet)
        )
        if time_dependent:
            raise NotImplementedError(
                "Standard finite-strain J2 does not accept absolute "
                "TimeDependentDirichlet histories. Prescribe the end-of-step "
                "value and drive it with the shared step amplitude."
            )
        if model.boundary_models:
            names = tuple(
                getattr(item, "name", type(item).__name__)
                for item in model.boundary_models
            )
            raise NotImplementedError(
                "Standard finite-strain J2 does not yet consume weak boundary "
                f"models; unsupported={names}. Their residual and consistent "
                "tangent must be lowered together."
            )
        selected_loads, selected_amplitude = _single_shared_amplitude_loads(
            model.loads,
            amplitude,
            label="finite-strain J2",
            physical_time=False,
        )
        physical_loads = load_api.load_assets(
            selected_loads,
            unwrap_amplitudes=True,
        )
        follower = tuple(
            item
            for item in physical_loads
            if getattr(item, "configuration", "reference") != "reference"
        )
        if follower:
            raise NotImplementedError(
                "Standard finite-strain J2 currently supports dead loads in the "
                "reference configuration. Follower/current-configuration loads "
                "require their external-work tangent."
            )
        problem = mechanics.finite_strain_j2_standard_problem(
            displacement=target,
            material=properties,
            external_force=(
                model.external_force(target, loads=selected_loads)
                if selected_loads
                else None
            ),
            load_identity=tuple(_describe(item) for item in selected_loads),
            constraints=selected_constraints,
            incrementation=incrementation,
            solver_options=solver_options,
            quadrature_degree=quadrature_degree,
            amplitude=selected_amplitude,
            output_every=selected_output_every,
            output_factors=output_factors,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            name=name,
        )
        has_external_power = bool(selected_loads)
    if output is not None and hasattr(output, "bind"):
        output.bind(
            problem,
            properties,
            has_external_power=has_external_power,
        )
    return model.add_step(problem)


def creep(
    model,
    *,
    target,
    duration: float,
    material=None,
    constraints=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    creep_strain_error_tolerance: float | None = None,
    progress=True,
    status_file=None,
    amplitude=None,
    temperature=None,
    name: str = "implicit_creep",
):
    """Build and register a global 3D implicit power-law creep Step."""

    from . import mechanics
    from .constitutive.creep import IsotropicPowerLawCreepMaterial
    from .constitutive.quadrature import QuadratureMaterialMap

    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(
            analysis="nonlinear_transient",
            physics="solid_mechanics",
        )
    material = _quadrature_material(
        model,
        target,
        material,
        material_type=IsotropicPowerLawCreepMaterial,
        label="model.step with implicit creep",
    )
    if not isinstance(
        material,
        (IsotropicPowerLawCreepMaterial, QuadratureMaterialMap),
    ):
        raise TypeError("model.step requires IsotropicPowerLawCreepMaterial here.")

    selected_loads, amplitude = _single_shared_amplitude_loads(
        model.loads,
        amplitude,
        label="implicit creep",
        physical_time=True,
    )
    step = mechanics.implicit_creep_step(
        displacement=target,
        material=material,
        duration=duration,
        external_force=(
            model.external_force(target, loads=selected_loads)
            if selected_loads
            else None
        ),
        constraints=model.constraints if constraints is None else constraints,
        study=model.study,
        incrementation=incrementation,
        solver_options=solver_options,
        quadrature_degree=quadrature_degree,
        creep_strain_error_tolerance=creep_strain_error_tolerance,
        time_unit=getattr(model.unit_system, "time", None),
        progress=progress,
        status_file=status_file,
        amplitude=amplitude,
        temperature=temperature,
        name=name,
    )
    return model.add_step(step)


def viscoelastic(
    model,
    *,
    target,
    duration: float,
    steps: int | None = None,
    time_points=None,
    incrementation=None,
    time_error_tolerance: float | None = None,
    material=None,
    constraints=None,
    solver_options=None,
    quadrature_degree: int = 2,
    amplitude=None,
    temperature=None,
    progress=True,
    status_file=None,
    checkpoint=None,
    name: str = "viscoelastic",
):
    """Build and register a global 3D generalized-Maxwell Step."""

    from . import mechanics
    from .constitutive.quadrature import QuadratureMaterialMap
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(
            analysis="first_order_transient",
            physics="solid_mechanics",
        )
    properties = _quadrature_material(
        model,
        target,
        material,
        material_type=IsotropicGeneralizedMaxwell,
        label="model.step with generalized-Maxwell viscoelasticity",
    )
    if not isinstance(properties, (IsotropicGeneralizedMaxwell, QuadratureMaterialMap)):
        raise TypeError("model.step requires IsotropicGeneralizedMaxwell here.")
    selected_loads, selected_amplitude = _single_shared_amplitude_loads(
        model.loads,
        amplitude,
        label="viscoelastic",
        physical_time=True,
    )
    step = mechanics.quasistatic_viscoelastic_step(
        displacement=target,
        material=properties,
        duration=duration,
        steps=steps,
        time_points=time_points,
        incrementation=incrementation,
        time_error_tolerance=time_error_tolerance,
        external_force=(
            model.external_force(target, loads=selected_loads)
            if selected_loads
            else None
        ),
        constraints=model.constraints if constraints is None else constraints,
        study=model.study,
        solver_options=solver_options,
        quadrature_degree=quadrature_degree,
        amplitude=selected_amplitude,
        temperature=temperature,
        time_unit=getattr(model.unit_system, "time", None),
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint,
        name=name,
    )
    return model.add_step(step)


def direct_harmonic(
    model,
    *,
    target,
    K,
    F,
    M=None,
    C=None,
    K_loss=None,
    frequency: float | None = None,
    angular_frequency: float | None = None,
    frequencies=None,
    angular_frequencies=None,
    responses=(),
    execution_order: str = "forward",
    constraints=None,
    load_phase: float = 0.0,
    solver_options=None,
    status_file=None,
    name: str = "direct_harmonic",
):
    """Build and register a direct harmonic Step from explicit operators."""

    from . import mechanics, operators

    model.study.require(analysis="frequency_domain", physics="solid_mechanics")
    system = operators.direct_harmonic_system(
        K,
        F,
        M=M,
        C=C,
        K_loss=K_loss,
        name=f"{name}_system",
    )
    selected = tuple(
        value
        for value in (
            frequency,
            angular_frequency,
            frequencies,
            angular_frequencies,
        )
        if value is not None
    )
    if len(selected) != 1:
        raise ValueError(
            "Specify exactly one of frequency, angular_frequency, frequencies, "
            "or angular_frequencies."
        )
    sweep_frequencies = None
    point_frequency = frequency
    point_angular_frequency = angular_frequency
    if frequencies is not None:
        sweep_frequencies = tuple(float(value) for value in frequencies)
        if not sweep_frequencies:
            raise ValueError("frequencies must contain at least one point.")
        point_frequency = sweep_frequencies[0]
    elif angular_frequencies is not None:
        angular_axis = tuple(float(value) for value in angular_frequencies)
        if not angular_axis:
            raise ValueError("angular_frequencies must contain at least one point.")
        sweep_frequencies = tuple(value / (2.0 * np.pi) for value in angular_axis)
        point_angular_frequency = angular_axis[0]

    selected_constraints = model.constraints if constraints is None else constraints
    point_step = mechanics.direct_harmonic_step(
        displacement=target,
        system=system,
        frequency=point_frequency,
        angular_frequency=point_angular_frequency,
        constraints=selected_constraints,
        load_phase=load_phase,
        study=model.study,
        solver_options=solver_options,
        name=name,
    )
    if sweep_frequencies is None:
        return model.add_step(point_step)
    sweep = mechanics.harmonic_frequency_sweep_step(
        point_step,
        frequencies=sweep_frequencies,
        responses=responses,
        execution_order=execution_order,
        scientific_assets={
            "study": model.study,
            "materials": tuple(model.materials),
            "loads": tuple(model.loads),
            "constraints": tuple(selected_constraints),
            "boundary_models": tuple(model.boundary_models),
        },
        status_file=status_file,
        name=name,
    )
    return model.add_step(sweep)


def harmonic_viscoelastic(
    model,
    *,
    target,
    frequency: float | None = None,
    angular_frequency: float | None = None,
    material=None,
    constraints=None,
    density: float | None = None,
    load_phase: float = 0.0,
    temperature: float | None = None,
    solver_options=None,
    name: str = "harmonic_viscoelastic",
):
    """Build and register one direct generalized-Maxwell harmonic Step."""

    from . import mechanics
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    model.check(
        target=target,
        step_options={
            "material": material,
            "frequency": frequency,
            "angular_frequency": angular_frequency,
        },
    )
    model.study.require(analysis="frequency_domain", physics="solid_mechanics")
    properties = (
        model._material_record(material).item
        if material is not None
        else _single_material(model, "model.step harmonic response").item
    )
    if not isinstance(properties, IsotropicGeneralizedMaxwell):
        raise TypeError(
            "A harmonic generalized-Maxwell Step requires one "
            "IsotropicGeneralizedMaxwell material."
        )
    selected_loads = load_api.load_assets(model.loads)
    if any(isinstance(item, load_api.AmplitudeLoad) for item in selected_loads):
        raise ValueError(
            "Frequency-domain loads use load_phase on the Step; time-domain "
            "AmplitudeLoad assets are not valid in a harmonic Study."
        )
    step = mechanics.harmonic_viscoelastic_step(
        displacement=target,
        material=properties,
        frequency=frequency,
        angular_frequency=angular_frequency,
        external_force=(
            model.external_force(target, loads=selected_loads)
            if selected_loads
            else None
        ),
        constraints=model.constraints if constraints is None else constraints,
        density=density,
        load_phase=load_phase,
        temperature=temperature,
        study=model.study,
        solver_options=solver_options,
        name=name,
    )
    return model.add_step(step)


from ._step_builders_finite_strain import (
    fabric_membrane,
    hyperelastic,
    mixed_hyperelastic,
)


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


def _quadrature_material(model, target, material, *, material_type, label: str):
    from .constitutive.quadrature import QuadratureMaterialMap

    if material is not None:
        return model._material_record(material).item
    if not model.materials:
        raise ValueError(f"{label} requires registered material data.")
    selected = QuadratureMaterialMap.from_assignments(
        target.value.function_space.mesh,
        model.materials,
        material_type=material_type,
    )
    if len(model.materials) == 1 and model.materials[0].region is None:
        return model.materials[0].item
    return selected


def _single_shared_amplitude_loads(
    loads,
    amplitude,
    *,
    label: str,
    physical_time: bool,
):
    selected = load_api.load_assets(loads)
    amplitude_loads = tuple(
        item for item in selected if isinstance(item, load_api.AmplitudeLoad)
    )
    if not amplitude_loads:
        return selected, amplitude
    ordinary = tuple(
        item for item in selected if not isinstance(item, load_api.AmplitudeLoad)
    )
    histories = {id(item.amplitude): item.amplitude for item in amplitude_loads}
    if ordinary or len(histories) != 1:
        time_label = " physical-time" if physical_time else ""
        raise ValueError(
            f"A {label} step requires one shared{time_label} load path. "
            "Do not mix ordinary loads with amplitude-driven loads or use "
            "multiple amplitudes."
        )
    if amplitude is not None:
        raise ValueError(
            f"Pass a load amplitude or step amplitude to {label}, not both."
        )
    resolved = load_api.load_assets(tuple(item.load for item in amplitude_loads))
    if any(isinstance(item, load_api.AmplitudeLoad) for item in resolved):
        raise ValueError(
            f"A {label} step does not accept nested amplitude load wrappers. "
            "Declare one shared amplitude around the physical loads."
        )
    return resolved, next(iter(histories.values()))


def _describe(item):
    if item is None:
        return None
    if hasattr(item, "summary"):
        return item.summary()
    if hasattr(item, "as_dict"):
        return item.as_dict()
    return getattr(item, "name", repr(item))


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
