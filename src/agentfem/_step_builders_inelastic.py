# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Stateful inelastic and hereditary static Step builders."""

from __future__ import annotations

import numpy as np

from . import constraints as constraint_api
from . import loads as load_api


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


def _as_tuple(item) -> tuple:
    if item is None:
        return ()
    if isinstance(item, tuple):
        return item
    if isinstance(item, list):
        return tuple(item)
    return (item,)


__all__ = ()
