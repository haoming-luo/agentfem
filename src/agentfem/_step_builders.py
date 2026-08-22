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
    from .constitutive.plasticity import J2LinearIsotropicHardening
    from .constitutive.quadrature import QuadratureMaterialMap

    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    material = _quadrature_material(
        model,
        target,
        material,
        material_type=J2LinearIsotropicHardening,
        label="model.step with J2 plasticity",
    )
    if not isinstance(material, (J2LinearIsotropicHardening, QuadratureMaterialMap)):
        raise TypeError("model.step requires J2LinearIsotropicHardening here.")

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
        progress=progress,
        status_file=status_file,
        amplitude=amplitude,
        temperature=temperature,
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
    selected = tuple(loads)
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
    return tuple(item.load for item in amplitude_loads), next(iter(histories.values()))


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
