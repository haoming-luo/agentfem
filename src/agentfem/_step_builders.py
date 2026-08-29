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
from .materials.properties import constant_volumetric_heat_capacity


def linear_static(
    model,
    *,
    target,
    K=None,
    F=None,
    constraints=None,
    solver_options=None,
    name: str = "linear_static",
):
    """Build and register a linear static or steady conduction Step."""

    from . import operators, problems

    model.check(target=target, step_options={"K": K, "F": F})
    update_at_step_end = model._time_update_callback()
    if update_at_step_end is not None:
        update_at_step_end(1.0)

    if getattr(model.study, "is_heat_transfer", False):
        boundary_stiffness, boundary_source = _thermal_boundary_terms(model, target)
        if K is None:
            K = model.conduction(target)
        if boundary_stiffness:
            K = operators.combine(
                K,
                *boundary_stiffness,
                name="K_thermal",
                kind="conduction_and_exchange",
            )
        if F is None:
            sources = []
            if model.loads:
                sources.append(model.external_force(target))
            sources.extend(boundary_source)
            F = (
                operators.combine(*sources, name="Q", kind="thermal_source")
                if sources
                else operators.heat_source_vector(0.0, target)
            )
    else:
        K = K if K is not None else model.stiffness(target)
        foundation_terms = tuple(
            item.operator(target)
            for item in model.boundary_models
            if item.__class__.__name__ == "ElasticFoundation"
        )
        if foundation_terms:
            K = operators.combine(
                K,
                *foundation_terms,
                name="K_with_foundation",
                kind="solid_and_foundation_stiffness",
            )
        if F is None:
            if model.loads:
                F = model.external_force(target)
            else:
                value_shape = tuple(getattr(target.value, "ufl_shape", ()))
                zero = (
                    0.0
                    if not value_shape
                    else tuple(0.0 for _ in range(value_shape[0]))
                )
                F = model.external_force(
                    target,
                    load=load_api.body_force(
                        zero,
                        target=target,
                        name="zero_external_force",
                    ),
                )

    result_field_factory = None
    if (
        getattr(model.study, "is_solid_mechanics", False)
        and model.materials
        and all(_thermal_expansion_is_zero(record.item) for record in model.materials)
    ):
        assignments = tuple(model.materials)

        def result_field_factory(requested=None):
            from . import results

            isotropic = all(
                hasattr(record.item, "young") and hasattr(record.item, "poisson")
                for record in assignments
            )
            defaults = results.preselected_fields(
                physics="solid_mechanics",
                finite_strain=False,
            )[1:]
            variables = tuple(defaults if requested is None else requested)
            if (
                not isotropic
                and requested is None
                and getattr(model.study, "assumption", None) == "plane_strain"
            ):
                variables = tuple(item for item in variables if item != "MISES")
            return results.small_strain_partition_fields(
                target,
                assignments,
                study=model.study,
                variables=variables,
            )

    step = problems.linear_static(
        K,
        F,
        study=model.study,
        unknown=target,
        constraints=model.constraints if constraints is None else constraints,
        solver_options=solver_options,
        result_field_factory=result_field_factory,
        name=name,
    )
    return model.add_step(step)


def heat_transfer(
    model,
    *,
    target,
    dt: float,
    steps: int,
    material=None,
    C=None,
    K=None,
    Q=None,
    constraints=None,
    solver_options=None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint=None,
    name: str = "transient_heat",
):
    """Build and register one implicit-Euler heat-transfer Step."""

    import ufl
    from dolfinx import fem

    from . import operators, problems

    model.check(
        target=target,
        step_options={
            "material": material,
            "K": K,
            "F": Q,
            "dt": dt,
            "steps": steps,
        },
    )
    if hasattr(model.study, "require"):
        model.study.require(
            analysis="first_order_transient",
            physics="heat_transfer",
        )
    records = (
        (model._material_record(material),)
        if material is not None
        else tuple(model.materials)
    )
    if not records:
        raise ValueError("A transient heat Step requires at least one material.")
    previous = fem.Function(target.space, name="TemperaturePrevious")
    previous.x.array[:] = target.value.x.array
    previous.x.scatter_forward()
    if any(
        bool(getattr(record.item, "state_dependent_heat_transfer", False))
        for record in records
    ):
        if C is not None or K is not None:
            raise ValueError(
                "Temperature-dependent heat transfer builds one consistent "
                "enthalpy/conduction residual. Do not also pass C= or K=."
            )
        return model.add_step(
            _nonlinear_heat_transfer(
                model,
                target=target,
                previous=previous,
                records=records,
                dt=dt,
                steps=steps,
                Q=Q,
                constraints=constraints,
                solver_options=solver_options,
                update_load=update_load,
                save_every=save_every,
                print_every=print_every,
                progress=progress,
                status_file=status_file,
                checkpoint=checkpoint,
                name=name,
            )
        )
    capacity = model.heat_capacity(target, material) if C is None else C
    stiffness = model.conduction(target, material) if K is None else K
    boundary_stiffness, boundary_source = _thermal_boundary_terms(model, target)
    if boundary_stiffness:
        stiffness = operators.combine(
            stiffness,
            *boundary_stiffness,
            name="K_thermal",
            kind="conduction_and_exchange",
        )
    source = Q
    if source is None and model.loads:
        source = model.external_force(target)
    if boundary_source:
        source = operators.combine(
            *((() if source is None else (source,)) + tuple(boundary_source)),
            name="Q_thermal",
            kind="thermal_source",
        )
    history_parts = []
    for index, record in enumerate(records):
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material heat history requires a region for every material."
            )
        history_parts.append(
            operators.heat_capacity_vector(
                previous,
                target,
                constant_volumetric_heat_capacity(record.item),
                measure=record.region.measure if record.region is not None else ufl.dx,
            ).renamed(
                "Q_capacity_history"
                if len(records) == 1
                else f"Q_capacity_{getattr(record.region, 'name', index)}"
            )
        )
    history = (
        history_parts[0]
        if len(history_parts) == 1
        else operators.combine(
            *history_parts,
            name="Q_capacity_history",
            kind="partitioned_heat_capacity_history",
        )
    )
    return model.add_step(
        problems.first_order_transient_run(
            capacity=capacity,
            stiffness=stiffness,
            history=history,
            source=source,
            current=target.value,
            previous=previous,
            dt=dt,
            steps=steps,
            study=model.study,
            constraints=model.constraints if constraints is None else constraints,
            solver_options=solver_options,
            update_load=model._time_update_callback(update_load),
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            name=name,
        )
    )


def _nonlinear_heat_transfer(
    model,
    *,
    target,
    previous,
    records,
    dt,
    steps,
    Q,
    constraints,
    solver_options,
    update_load,
    save_every,
    print_every,
    progress,
    status_file,
    checkpoint,
    name,
):
    """Build conservative ``k(T), c_p(T)`` implicit heat transfer."""

    import ufl

    from . import operators, problems
    from .diagnostics import StateDependentThermalBalanceMonitor

    temperature = target.value
    test = ufl.TestFunction(target.space)
    direction = ufl.TrialFunction(target.space)
    residual = 0
    content_form = 0
    for record in records:
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material nonlinear heat transfer requires a region "
                "for every material."
            )
        selected = record.item
        measure = record.region.measure if record.region is not None else ufl.dx
        if not hasattr(selected, "conductivity") or not hasattr(
            selected, "specific_heat"
        ):
            raise ValueError(
                f"Material {_describe(selected)!r} does not define conductivity "
                "and specific heat."
            )
        conductivity = (
            selected.conductivity_at(temperature)
            if hasattr(selected, "conductivity_at")
            else selected.conductivity
        )
        if hasattr(selected, "volumetric_enthalpy"):
            current_enthalpy = selected.volumetric_enthalpy(temperature)
            previous_enthalpy = selected.volumetric_enthalpy(previous)
        else:
            capacity = selected.density * selected.specific_heat
            current_enthalpy = capacity * temperature
            previous_enthalpy = capacity * previous
        residual += (
            (current_enthalpy - previous_enthalpy) / float(dt) * test
            + conductivity * ufl.dot(ufl.grad(temperature), ufl.grad(test))
        ) * measure
        content_form += current_enthalpy * measure

    outward_forms = []
    boundary_sources = []
    unsupported = []
    for boundary in model.boundary_models:
        if hasattr(boundary, "residual"):
            residual += boundary.residual(temperature, test)
            if hasattr(boundary, "outward_heat_rate_form"):
                outward_forms.append(boundary.outward_heat_rate_form(temperature))
            if hasattr(boundary, "source"):
                boundary_sources.append(boundary.source(target))
        else:
            unsupported.append(getattr(boundary, "name", type(boundary).__name__))
    if unsupported:
        raise ValueError(
            "Nonlinear heat transfer cannot consume these boundary models: "
            f"{unsupported}."
        )

    source = Q
    if source is None and model.loads:
        source = model.external_force(target)
    if source is not None:
        residual -= getattr(source, "expression", source)
    monitor_source = source
    if boundary_sources:
        monitor_source = operators.combine(
            *((() if source is None else (source,)) + tuple(boundary_sources)),
            name="Q_thermal_ledger",
            kind="thermal_source",
        )
    jacobian = ufl.derivative(residual, temperature, direction)
    return problems.nonlinear_first_order_transient_run(
        residual=residual,
        jacobian=jacobian,
        current=temperature,
        previous=previous,
        dt=dt,
        steps=steps,
        study=model.study,
        constraints=model.constraints if constraints is None else constraints,
        solver_options=solver_options,
        update_load=model._time_update_callback(update_load),
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint,
        history_monitor=StateDependentThermalBalanceMonitor(
            content_form=content_form,
            source=monitor_source,
            dt=float(dt),
            outward_forms=tuple(outward_forms),
        ),
        name=name,
    )


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
    """Build stateful finite-strain J2 for affine or ordinary strong kinematics."""

    from . import mechanics
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constitutive import QuadratureMaterialMap

    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    if getattr(model.study, "dimension", None) != 3:
        raise NotImplementedError("Finite-strain J2 currently requires a 3D Study.")
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


def hyperelastic(
    model,
    *,
    target,
    material=None,
    constraints=None,
    solver_options=None,
    measure=None,
    name: str = "hyperelastic",
    petsc_options_prefix: str = "agentfem_hyperelastic_",
    incrementation=None,
    increments: int | None = None,
    load_factors=None,
    output=None,
    output_every: int | None = None,
    progress=True,
    status_file=None,
):
    """Build displacement-based finite-strain equilibrium."""

    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    from . import problems
    from .constitutive import hyperelasticity

    if hasattr(model.mesh, "require_formulation"):
        model.mesh.require_formulation(
            "displacement",
            operation="model.step with finite-strain hyperelasticity",
        )
    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    record = (
        _single_material(model, "model.step with finite-strain hyperelasticity")
        if material is None
        else model._material_record(material)
    )
    properties = record.item
    if not hyperelasticity.is_finite_strain_hyperelastic(properties):
        raise TypeError("model.step requires a supported hyperelastic material here.")
    if not hyperelasticity.supports_hyperelastic_study(
        properties,
        dimension=getattr(model.study, "dimension", 0),
        assumption=getattr(model.study, "assumption", None),
    ):
        raise ValueError(
            "The Study dimension/assumption has no formulation for the "
            f"selected hyperelastic material {properties.name!r}."
        )
    selected_measure = measure
    if selected_measure is None:
        selected_measure = record.region.measure if record.region is not None else ufl.dx
    internal_residual = hyperelasticity.internal_virtual_work(
        target.value,
        target.test,
        properties,
        measure=selected_measure,
    )
    selected_constraints = _as_tuple(
        model.constraints if constraints is None else constraints
    )
    affine_constraints = [
        item
        for item in selected_constraints
        if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
    ]
    if output is not None and output_every is not None:
        raise ValueError("Pass output=... or output_every=..., not both.")
    selected_output_every = (
        getattr(output, "every", None)
        if output is not None
        else (1 if output_every is None else int(output_every))
    )
    from . import steps as step_api

    selected_incrementation = step_api.normalize(
        incrementation,
        increments=increments,
        load_factors=load_factors,
    )
    if affine_constraints:
        if any(isinstance(item, load_api.AmplitudeLoad) for item in model.loads):
            raise NotImplementedError(
                "Amplitude-driven natural loads are not yet supported by the "
                "affine nonlinear path. Prescribed affine loading remains supported."
            )
        if len(affine_constraints) != 1 or len(selected_constraints) != 1:
            raise ValueError(
                "The affine hyperelastic path currently requires exactly one "
                "AbaqusPeriodicConstraint and no separate Dirichlet constraints."
            )
        residual = internal_residual
        if model.loads:
            residual -= model.external_force(target).expression
        jacobian = hyperelasticity.tangent(residual, target.value, target.trial)
        output_factors = (
            output.required_factors()
            if output is not None and hasattr(output, "required_factors")
            else ()
        )

        def finite_strain_acceptance():
            return _finite_strain_acceptance(
                model,
                target,
                properties,
                selected_measure,
                hyperelasticity,
            )

        problem = problems.affine_nonlinear(
            residual,
            target.value,
            jacobian=jacobian,
            constraint=affine_constraints[0],
            incrementation=selected_incrementation,
            solver_options=solver_options,
            output_every=selected_output_every,
            output_factors=output_factors,
            acceptance_check=finite_strain_acceptance,
            progress=progress,
            status_file=status_file,
            name=name,
        )
    else:
        load_factor = fem.Constant(_domain(model.mesh), PETSc.ScalarType(0.0))
        residual = internal_residual
        if model.loads:
            proportional_loads = tuple(
                item
                for item in model.loads
                if not isinstance(item, load_api.AmplitudeLoad)
            )
            amplitude_loads = tuple(
                item
                for item in model.loads
                if isinstance(item, load_api.AmplitudeLoad)
            )
            if proportional_loads:
                residual -= load_factor * model.external_force(
                    target,
                    loads=proportional_loads,
                ).expression
            if amplitude_loads:
                residual -= model.external_force(
                    target,
                    loads=amplitude_loads,
                ).expression
        jacobian = hyperelasticity.tangent(residual, target.value, target.trial)

        def finite_strain_acceptance():
            return _finite_strain_acceptance(
                model,
                target,
                properties,
                selected_measure,
                hyperelasticity,
            )

        problem = problems.incremental_nonlinear(
            residual,
            target.value,
            factor=load_factor,
            value_path=constraint_api.prescribed_value_path(selected_constraints),
            update_load=model._time_update_callback(include_constraints=False),
            acceptance_check=finite_strain_acceptance,
            jacobian=jacobian,
            incrementation=selected_incrementation,
            constraints=selected_constraints,
            solver_options=solver_options,
            output_every=selected_output_every,
            progress=progress,
            status_file=status_file,
            name=name,
            petsc_options_prefix=petsc_options_prefix,
        )
    if output is not None and hasattr(output, "bind"):
        output.bind(problem, properties, has_external_power=bool(model.loads))
    return model.add_step(problem)


def mixed_hyperelastic(
    model,
    *,
    target,
    material=None,
    constraints=None,
    solver_options=None,
    measure=None,
    name: str = "mixed_hyperelastic",
    petsc_options_prefix: str = "agentfem_mixed_hyperelastic_",
    incrementation=None,
    increments: int | None = None,
    load_factors=None,
    output=None,
    output_every: int | None = None,
    progress=True,
    status_file=None,
):
    """Build verified P2-displacement/DG0-pressure equilibrium."""

    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    from . import problems
    from .constitutive import hyperelasticity

    if getattr(target, "kind", None) != "displacement_pressure":
        raise TypeError(
            "model.step with mixed hyperelasticity requires "
            "fields.displacement_pressure(...)."
        )
    if int(getattr(target, "displacement_degree", -1)) != 2 or int(
        getattr(target, "pressure_degree", -1)
    ) != 0:
        raise ValueError("The verified constant-pressure hybrid route requires P2/DG0.")
    if hasattr(model.mesh, "require_formulation"):
        model.mesh.require_formulation(
            "hybrid",
            operation="model.step with mixed hyperelasticity",
        )
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    if getattr(model.study, "dimension", None) == 2 and getattr(
        model.study, "assumption", None
    ) != "plane_strain":
        raise NotImplementedError(
            "The mixed Neo-Hookean 2D route currently represents plane strain."
        )
    record = (
        _single_material(model, "model.step with mixed hyperelasticity")
        if material is None
        else model._material_record(material)
    )
    properties = record.item
    if not isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
        raise TypeError("model.step requires MixedNeoHookeanProperties here.")
    selected_measure = measure or (
        record.region.measure if record.region is not None else ufl.dx
    )
    w = target.value
    displacement, pressure = ufl.split(w)
    test = ufl.TestFunction(target.space)
    trial = ufl.TrialFunction(target.space)
    internal_energy = hyperelasticity.mixed_strain_energy_density(
        displacement,
        pressure,
        properties,
    ) * selected_measure
    residual = ufl.derivative(internal_energy, w, test)
    load_factor = fem.Constant(_domain(model.mesh), PETSc.ScalarType(0.0))
    if model.loads:
        residual -= load_factor * model.external_force(target.displacement).expression
    jacobian = ufl.derivative(residual, w, trial)
    selected_constraints = _as_tuple(
        model.constraints if constraints is None else constraints
    )
    affine_constraints = [
        item
        for item in selected_constraints
        if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
    ]
    if output is not None and output_every is not None:
        raise ValueError("Pass output=... or output_every=..., not both.")
    selected_output_every = (
        getattr(output, "every", None)
        if output is not None
        else (1 if output_every is None else int(output_every))
    )
    from . import steps as step_api

    selected_incrementation = step_api.normalize(
        incrementation,
        increments=increments,
        load_factors=load_factors,
    )

    def mixed_acceptance():
        from .results import finite_strain_diagnostics, integral

        displacement_field = target.collapsed_displacement()
        diagnostics = finite_strain_diagnostics(
            displacement_field,
            quadrature_degree=3,
        )
        minimum_j = float(diagnostics["minimum_quadrature_J"])
        return {
            "accepted": bool(minimum_j > 0.0),
            "minimum_quadrature_J": minimum_j,
            "maximum_quadrature_J": float(diagnostics["maximum_quadrature_J"]),
            "mixed_potential": float(
                integral(
                    hyperelasticity.mixed_strain_energy_density(
                        displacement,
                        pressure,
                        properties,
                    ),
                    measure=selected_measure,
                    comm=_domain(model.mesh).comm,
                )
            ),
            "message": (
                "deformation Jacobian became non-positive" if minimum_j <= 0.0 else ""
            ),
        }

    if affine_constraints:
        if len(affine_constraints) != 1 or len(selected_constraints) != 1:
            raise ValueError(
                "The mixed affine hyperelastic path requires exactly one "
                "AbaqusPeriodicConstraint and no separate Dirichlet constraints."
            )
        if model.loads:
            raise NotImplementedError(
                "Natural loads are not yet combined with the mixed affine "
                "periodic path; prescribe the macroscopic deformation gradient."
            )
        output_factors = (
            output.required_factors()
            if output is not None and hasattr(output, "required_factors")
            else ()
        )
        problem = problems.affine_nonlinear(
            residual,
            w,
            jacobian=jacobian,
            constraint=affine_constraints[0],
            incrementation=selected_incrementation,
            solver_options=solver_options,
            output_every=selected_output_every,
            output_factors=output_factors,
            acceptance_check=mixed_acceptance,
            progress=progress,
            status_file=status_file,
            name=name,
        )
    else:
        problem = problems.incremental_nonlinear(
            residual,
            w,
            factor=load_factor,
            value_path=constraint_api.prescribed_value_path(selected_constraints),
            update_load=model._time_update_callback(include_constraints=False),
            acceptance_check=mixed_acceptance,
            jacobian=jacobian,
            incrementation=selected_incrementation,
            constraints=selected_constraints,
            solver_options=solver_options,
            output_every=selected_output_every,
            progress=progress,
            status_file=status_file,
            name=name,
            petsc_options_prefix=petsc_options_prefix,
        )
    problem.primary_fields = {"U": target.displacement, "PRESSURE": target.pressure}
    problem.result_field_factory = lambda: (
        target.collapsed_displacement(name="U"),
        target.collapsed_pressure(name="PRESSURE"),
    )
    problem.snapshot_field_factory = lambda: (
        target.collapsed_displacement(name="U"),
        {"PRESSURE": target.collapsed_pressure(name="PRESSURE")},
    )
    if output is not None and hasattr(output, "bind"):
        output.bind(problem, properties, has_external_power=bool(model.loads))
    return model.add_step(problem)


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
    selected_state = state if state is not None else problems.second_order_state(target)
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
    selected_state = state if state is not None else problems.second_order_state(target)
    if cohesive_force is not None:
        cohesive_force = cohesive_force.for_displacement(selected_state.u)
    selected_mass = (
        mass
        if mass is not None
        else problems.LumpedMassOperator.assemble(
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
        characteristic_length=fracture.minimum_cell_nodal_spacing(
            _domain(model.mesh)
        ),
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
    selected_state = state if state is not None else problems.second_order_state(target)
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


def _finite_strain_acceptance(
    model,
    target,
    properties,
    measure,
    hyperelasticity,
):
    from .results import finite_strain_diagnostics, integral

    diagnostics = finite_strain_diagnostics(target, quadrature_degree=2)
    minimum_j = float(diagnostics["minimum_quadrature_J"])
    return {
        "accepted": bool(minimum_j > 0.0),
        "minimum_quadrature_J": minimum_j,
        "maximum_quadrature_J": float(diagnostics["maximum_quadrature_J"]),
        "recoverable_strain_energy": float(
            integral(
                hyperelasticity.strain_energy_density(target.value, properties),
                measure=measure,
                comm=_domain(model.mesh).comm,
            )
        ),
        "message": (
            "deformation Jacobian became non-positive" if minimum_j <= 0.0 else ""
        ),
    }


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


def _thermal_boundary_terms(model, target):
    stiffness = []
    source = []
    unsupported = []
    for item in model.boundary_models:
        if hasattr(item, "operator") and hasattr(item, "source"):
            stiffness.append(item.operator(target))
            source.append(item.source(target))
        else:
            unsupported.append(getattr(item, "name", type(item).__name__))
    if unsupported and getattr(model.study, "is_heat_transfer", False):
        raise ValueError(
            "Heat-transfer steps cannot consume these boundary models: "
            f"{unsupported}."
        )
    return tuple(stiffness), tuple(source)


def _thermal_expansion_is_zero(material) -> bool:
    selected = getattr(material, "thermal_expansion", 0.0)
    if hasattr(selected, "values"):
        return bool(np.all(np.asarray(selected.values, dtype=float) == 0.0))
    return float(selected or 0.0) == 0.0


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
