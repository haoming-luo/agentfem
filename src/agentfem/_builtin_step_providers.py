# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Built-in Step provider predicates, lowerers, and declarations.

The public provider protocol and deterministic registry live in
`step_providers`; this catalog owns only AgentFEM's built-in physics routes.
"""

from __future__ import annotations

from typing import Mapping

from ._step_provider_support import (
    COMMON_STEP_OPTIONS as _COMMON_STEP_OPTIONS,
    all_materials_support as _all_materials_support,
    has_complete_linear_system as _has_complete_linear_system,
    is_scalar_target as _is_scalar_target,
    is_vector_target as _is_vector_target,
    normalize as _normalize,
    procedure_method as _procedure_method,
    registered_materials as _registered_materials,
    selected_material as _selected_material,
    supports_axisymmetric_elasticity as _supports_axisymmetric_elasticity,
    supports_conduction as _supports_conduction,
    supports_dynamics as _supports_dynamics,
    supports_elasticity as _supports_elasticity,
    supports_heat_capacity as _supports_heat_capacity,
    target_shape as _target_shape,
)
from .step_providers import (
    StepOptionContract,
    StepProvider,
    StepRequest,
    register_step_provider,
)


def _accept_linear_static(model, request: StepRequest) -> bool:
    study = getattr(model, "study", None)
    if request.target is None:
        return False
    physics = getattr(study, "physics", None)
    if physics == "solid_mechanics":
        material_predicate = (
            _supports_axisymmetric_elasticity
            if getattr(study, "assumption", None) == "axisymmetric"
            else _supports_elasticity
        )
        return _is_vector_target(request.target) and (
            _has_complete_linear_system(request)
            or _all_materials_support(model, request, material_predicate)
        )
    if physics == "heat_transfer":
        return _is_scalar_target(request.target) and (
            _has_complete_linear_system(request)
            or _all_materials_support(model, request, _supports_conduction)
        )
    return False


def _lower_linear_static(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("material", None)
    # Completion belongs to solve_result(), not the numerical constructor.
    # The registry binds the original request in StepExecutionContext.
    options.pop("output", None)
    name = options.pop("name", None) or "linear_static"
    return _step_builders.linear_static(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _accept_transient_heat(model, request: StepRequest) -> bool:
    return (
        request.target is not None
        and _is_scalar_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "heat_transfer"
        and _all_materials_support(model, request, _supports_conduction)
        and _all_materials_support(model, request, _supports_heat_capacity)
    )


def _lower_transient_heat(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("K", None)
    options.pop("F", None)
    material = options.pop("material", None)
    options.pop("output", None)
    options.pop("history", None)
    name = options.pop("name", None) or "transient_heat"
    return _step_builders.heat_transfer(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_neo_hookean(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    study = getattr(model, "study", None)
    material = _selected_material(model, request)
    supported_kinematics = hyperelasticity.supports_hyperelastic_study(
        material,
        dimension=getattr(study, "dimension", 0),
        assumption=getattr(study, "assumption", None),
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and supported_kinematics
        and hyperelasticity.is_finite_strain_hyperelastic(material)
    )


def _accept_fabric_membrane(model, request: StepRequest) -> bool:
    from .constitutive.fabric import DecoupledFabricSurface, FabricStack

    study = getattr(model, "study", None)
    return (
        getattr(study, "analysis", None) == "nonlinear_static"
        and getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 2
        and getattr(study, "assumption", None) == "membrane"
        and _target_shape(request.target) == (2,)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, (DecoupledFabricSurface, FabricStack)),
        )
    )


def _accept_mixed_neo_hookean(model, request: StepRequest) -> bool:
    from .constitutive.hyperelasticity import MixedNeoHookeanProperties

    study = getattr(model, "study", None)
    supported_kinematics = getattr(study, "dimension", None) == 3 or (
        getattr(study, "dimension", None) == 2
        and getattr(study, "assumption", None) == "plane_strain"
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(request.target, "kind", None) == "displacement_pressure"
        and supported_kinematics
        and isinstance(
            _selected_material(model, request),
            MixedNeoHookeanProperties,
        )
    )


def _accept_j2(model, request: StepRequest) -> bool:
    from .constitutive.plasticity import (
        ChabocheCombinedHardening,
        J2LinearIsotropicHardening,
    )

    study = getattr(model, "study", None)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and (
            getattr(study, "dimension", None) == 3
            or (
                getattr(study, "dimension", None) == 2
                and getattr(study, "assumption", None) == "axisymmetric"
            )
        )
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(
                item,
                (J2LinearIsotropicHardening, ChabocheCombinedHardening),
            ),
        )
    )


def _accept_finite_strain_j2_affine(model, request: StepRequest) -> bool:
    from . import loads as load_api
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import AbaqusPeriodicConstraint, constraint_assets

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    physical_loads = load_api.load_assets(
        getattr(model, "loads", ()),
        unwrap_amplitudes=True,
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 3
        and getattr(request.target, "kind", None) != "displacement_pressure"
        and _is_vector_target(request.target)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and len(selected_constraints) == 1
        and isinstance(selected_constraints[0], AbaqusPeriodicConstraint)
        and not physical_loads
    )


def _accept_finite_strain_j2_mixed_affine(
    model,
    request: StepRequest,
) -> bool:
    from . import loads as load_api
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import AbaqusPeriodicConstraint, constraint_assets

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    physical_loads = load_api.load_assets(
        getattr(model, "loads", ()),
        unwrap_amplitudes=True,
    )
    dimension = getattr(study, "dimension", None)
    pressure_family = str(getattr(request.target, "pressure_family", "DG")).upper()
    domain = getattr(getattr(request.target, "space", None), "mesh", None)
    cell_name = None if domain is None else str(domain.topology.cell_name())
    interpolation_supported = (
        dimension == 3
        and cell_name == "tetrahedron"
        and int(getattr(request.target, "displacement_degree", -1)) == 2
        and pressure_family == "DG"
        and int(getattr(request.target, "pressure_degree", -1)) == 0
    ) or (
        dimension == 2
        and cell_name == "quadrilateral"
        and getattr(study, "assumption", None) == "plane_strain"
        and int(getattr(request.target, "displacement_degree", -1)) == 2
        and pressure_family == "DPC"
        and int(getattr(request.target, "pressure_degree", -1)) == 1
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and interpolation_supported
        and domain is not None
        and int(domain.comm.size) == 1
        and getattr(request.target, "kind", None) == "displacement_pressure"
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and len(selected_constraints) == 1
        and isinstance(selected_constraints[0], AbaqusPeriodicConstraint)
        and not physical_loads
    )


def _accept_finite_strain_j2_strong(model, request: StepRequest) -> bool:
    from . import loads as load_api
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import (
        DirichletConstraint,
        RemoteDisplacementConstraint,
        constraint_assets,
    )

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    ordinary_strong = bool(selected_constraints) and all(
        isinstance(item, (DirichletConstraint, RemoteDisplacementConstraint))
        for item in selected_constraints
    )
    physical_loads = load_api.load_assets(
        getattr(model, "loads", ()),
        unwrap_amplitudes=True,
    )
    reference_dead_loads = all(
        getattr(item, "configuration", "reference") == "reference"
        for item in physical_loads
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 3
        and getattr(request.target, "kind", None) != "displacement_pressure"
        and _is_vector_target(request.target)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and ordinary_strong
        and reference_dead_loads
        and not tuple(getattr(model, "boundary_models", ()))
    )


def _lower_finite_strain_j2(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    for legacy_name in ("K", "F"):
        legacy_value = options.pop(legacy_name, None)
        if legacy_value is not None:
            raise ValueError(
                f"Finite-strain J2 does not consume legacy {legacy_name}=; "
                "declare material, constraints and solver options explicitly."
            )
    name = options.pop("name", None) or "finite_strain_j2"
    return _step_builders.finite_strain_j2(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_j2(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = request.material
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("output", None)
    name = options.pop("name", None) or "j2_plasticity"
    return _step_builders.j2_plasticity(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_implicit_creep(model, request: StepRequest) -> bool:
    from .constitutive.creep import IsotropicPowerLawCreepMaterial

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "nonlinear_transient"
        and (
            getattr(study, "dimension", None) == 3
            or (
                getattr(study, "dimension", None) == 2
                and getattr(study, "assumption", None) == "axisymmetric"
            )
        )
        and _is_vector_target(request.target)
        and _normalize(method or "implicit_creep")
        in {"implicit_creep", "backward_euler", "backward_euler_newton"}
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, IsotropicPowerLawCreepMaterial),
        )
    )


def _lower_implicit_creep(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = request.material
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("method", None)
    options.pop("output", None)
    name = options.pop("name", None) or "implicit_creep"
    return _step_builders.creep(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_quasistatic_viscoelasticity(model, request: StepRequest) -> bool:
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "first_order_transient"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _normalize(method or "exact_generalized_maxwell_equilibrium")
        in {
            "quasistatic_viscoelasticity",
            "generalized_maxwell",
            "exact_generalized_maxwell_equilibrium",
        }
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, IsotropicGeneralizedMaxwell),
        )
    )


def _accept_harmonic_viscoelasticity(model, request: StepRequest) -> bool:
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    materials = _registered_materials(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "frequency_domain"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _normalize(method or "direct_harmonic")
        in {"direct_harmonic", "harmonic", "real_block_complex_harmonic"}
        # The current harmonic builder assembles one homogeneous constitutive
        # operator.  Do not advertise a regional multi-material model unless
        # the request explicitly selects the one material to lower.
        and len(materials) == 1
        and isinstance(materials[0], IsotropicGeneralizedMaxwell)
    )


def _accept_direct_harmonic_system(model, request: StepRequest) -> bool:
    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "frequency_domain"
        and _is_vector_target(request.target)
        and request.option("K") is not None
        and request.option("F") is not None
        and _normalize(method or "direct_harmonic")
        in {
            "direct_harmonic",
            "harmonic",
            "real_block_complex_harmonic",
            "direct_harmonic_sweep",
            "harmonic_sweep",
            "real_block_complex_harmonic_sweep",
        }
    )


def _lower_direct_harmonic_system(model, request: StepRequest):
    from . import _step_builders

    _validate_direct_harmonic_axis(request)
    options = request.lowering_options(
        "method",
        "output",
        "history",
        "progress",
        "checkpoint",
        "material",
    )
    name = options.pop("name", None) or "direct_harmonic"
    return _step_builders.direct_harmonic(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _validate_direct_harmonic_axis(request: StepRequest) -> None:
    """Keep the selected Procedure consistent with scalar or sweep input."""

    plural = any(
        request.option(name) is not None
        for name in ("frequencies", "angular_frequencies")
    )
    method = _normalize(_procedure_method(None, request) or "direct_harmonic")
    sweep = method in {
        "direct_harmonic_sweep",
        "harmonic_sweep",
        "real_block_complex_harmonic_sweep",
    }
    if plural != sweep:
        expected = "a sweep procedure" if plural else "a single-frequency procedure"
        raise ValueError(
            "Direct harmonic frequency coordinates and procedure disagree: "
            f"the request supplies {'a frequency axis' if plural else 'one frequency'} "
            f"and therefore requires {expected}."
        )
    if not plural and request.option("checkpoint") is not None:
        raise ValueError(
            "Direct harmonic checkpointing is defined for a frequency sweep; "
            "one independent frequency has no partial lifecycle to resume."
        )
    if not plural and request.option("status_file") is not None:
        raise ValueError(
            "Direct harmonic status_file is defined for a frequency sweep."
        )


def _lower_harmonic_viscoelasticity(model, request: StepRequest):
    from . import _step_builders

    options = request.lowering_options(
        "K",
        "F",
        "method",
        "output",
        "history",
        "progress",
        "checkpoint",
    )
    name = options.pop("name", None) or "harmonic_viscoelastic"
    return _step_builders.harmonic_viscoelastic(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _lower_quasistatic_viscoelasticity(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = request.material
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("method", None)
    options.pop("output", None)
    name = options.pop("name", None) or "viscoelastic"
    return _step_builders.viscoelastic(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_neo_hookean(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    name = options.pop("name", None) or "finite_strain_static"
    options.pop("K", None)
    options.pop("F", None)
    return _step_builders.hyperelastic(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_fabric_membrane(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    name = options.pop("name", None) or "fabric_membrane"
    return _step_builders.fabric_membrane(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_mixed_neo_hookean(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    name = options.pop("name", None) or "mixed_finite_strain_static"
    return _step_builders.mixed_hyperelastic(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_explicit_dynamics(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    method = _procedure_method(model, request)
    materials = _registered_materials(model, request)
    explicit_residual = request.options.get("residual") is not None
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
        and (
            (
                request.options.get("mass") is not None
                and request.options.get("residual") is not None
            )
            or _all_materials_support(model, request, _supports_dynamics)
        )
        and (
            explicit_residual
            or not any(
                hyperelasticity.is_finite_strain_hyperelastic(item)
                for item in materials
            )
        )
        and (
            method is None
            or _normalize(method) in {"explicit_dynamics", "central_difference"}
        )
    )


def _accept_finite_strain_explicit_dynamics(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    materials = _registered_materials(model, request)
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "second_order_dynamics"
        and bool(materials)
        and all(
            hyperelasticity.is_finite_strain_hyperelastic(material)
            and hyperelasticity.supports_hyperelastic_study(
                material,
                dimension=getattr(study, "dimension", 0),
                assumption=getattr(study, "assumption", None),
            )
            and material.density is not None
            for material in materials
        )
        and _normalize(method or "central_difference")
        in {"explicit_dynamics", "central_difference"}
        and request.options.get("residual") is None
    )


def _lower_finite_strain_explicit_dynamics(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    for key in ("K", "F", "solver_options", "method", "residual"):
        options.pop(key, None)
    options.pop("material", None)
    options.pop("output", None)
    options.pop("history", None)
    name = options.pop("name", None) or "finite_strain_explicit_dynamics"
    return _step_builders.finite_strain_explicit_dynamics(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_explicit_dynamics(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("solver_options", None)
    options.pop("method", None)
    options.pop("output", None)
    options.pop("history", None)
    name = options.pop("name", None) or "explicit_dynamics"
    return _step_builders.explicit_dynamics(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _accept_implicit_dynamics(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    method = _procedure_method(model, request)
    complete_system = all(
        request.options.get(item) is not None for item in ("M", "K", "F")
    )
    selected_material = _selected_material(model, request)
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
        and (
            complete_system
            or _all_materials_support(model, request, _supports_dynamics)
        )
        and (
            complete_system
            or not hyperelasticity.is_finite_strain_hyperelastic(selected_material)
        )
        and _normalize(method or "") in {"newmark", "generalized_alpha"}
    )


def _lower_implicit_dynamics(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("material", None)
    options.pop("method", None)
    options.pop("output", None)
    options.pop("history", None)
    method = _procedure_method(model, request)
    name = options.pop("name", None) or f"{method}_dynamics"
    return _step_builders.implicit_dynamics(
        model,
        target=request.target,
        method=method,
        name=name,
        **options,
    )


def _accept_modal(model, request: StepRequest) -> bool:
    complete_system = all(request.options.get(item) is not None for item in ("M", "K"))
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
        and getattr(getattr(model, "study", None), "analysis", None) == "modal"
        and (
            complete_system
            or _all_materials_support(model, request, _supports_dynamics)
        )
    )


def _lower_modal(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    for key in (
        "material",
        "F",
        "solver_options",
        "output",
        "history",
        "progress",
        "checkpoint",
    ):
        options.pop(key, None)
    name = options.pop("name", None) or "modal_analysis"
    return _step_builders.modal(model, target=request.target, name=name, **options)


def _option_contract(
    *extra: str,
    required=(),
    exactly_one_of=(),
) -> StepOptionContract:
    """Build a deterministic built-in contract without repeated core names."""

    return StepOptionContract(
        accepted=tuple(dict.fromkeys((*_COMMON_STEP_OPTIONS, *extra))),
        required=tuple(required),
        exactly_one_of=tuple(tuple(group) for group in exactly_one_of),
    )


def _accept_callable_neural_field(_model, request) -> bool:
    from .learning import NeuralFieldSpec

    executor = request.option("executor")
    return isinstance(request.target, NeuralFieldSpec) and (
        callable(executor) or callable(getattr(executor, "solve", None))
    )


def _lower_callable_neural_field(model, request):
    from .learning.execution import (
        CallableNeuralFieldStep,
        NeuralFieldExecutionRequest,
        executor_identity,
    )

    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "A neural-field executor consumes NeuralFieldSpec objectives and "
            f"conditions, not assembled FEM options: {tuple(unsupported)!r}."
        )
    executor = request.option("executor")
    executor_name, executor_version = executor_identity(
        executor,
        name=request.option("executor_name"),
        version=request.option("executor_version"),
    )
    executor_options = request.option("executor_options") or {}
    if not isinstance(executor_options, Mapping):
        raise TypeError("executor_options must be a mapping.")
    execution_request = NeuralFieldExecutionRequest(
        specification=request.target,
        model=model,
        analysis=request.analysis,
        name=request.option("name") or "neural_field",
        output_directory=request.option("output"),
        options=executor_options,
    )
    step = CallableNeuralFieldStep(
        execution_request,
        executor,
        executor_name=executor_name,
        executor_version=executor_version,
    )
    return model.add_step(step)


register_step_provider(
    StepProvider(
        name="callable_neural_field",
        analyses=(
            "linear_static",
            "nonlinear_static",
            "first_order_transient",
            "nonlinear_transient",
            "second_order_dynamics",
            "explicit_dynamics",
            "modal",
        ),
        accepts=_accept_callable_neural_field,
        lower=_lower_callable_neural_field,
        priority=450,
        description=(
            "Execute a NeuralFieldSpec through a user-owned callable while "
            "retaining the common Step and SimulationResult lifecycle."
        ),
        procedure="learning/neural_field/user_executor",
        option_contract=_option_contract(
            "executor",
            "executor_name",
            "executor_version",
            "executor_options",
            required=("executor",),
        ),
    )
)


register_step_provider(
    StepProvider(
        name="neo_hookean_finite_strain_explicit_dynamics",
        analyses=("explicit_dynamics", "second_order_dynamics"),
        accepts=_accept_finite_strain_explicit_dynamics,
        lower=_lower_finite_strain_explicit_dynamics,
        priority=120,
        description=(
            "Lower a complete finite-strain hyperelastic material partition to a "
            "current-state Total-Lagrangian residual and explicit central "
            "difference."
        ),
        procedure="explicit/central_difference/total_lagrangian",
        option_contract=_option_contract(
            "dt",
            "steps",
            "method",
            "residual",
            "state",
            "mass",
            "cohesive_force",
            "update_load",
            "save_every",
            "print_every",
            "history_every",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            "stability_safety",
            "mass_damping",
            required=("steps",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="implicit_power_law_creep",
        analyses=("nonlinear_transient",),
        accepts=_accept_implicit_creep,
        lower=_lower_implicit_creep,
        priority=120,
        description=(
            "Lower isotropic power-law creep to backward-Euler quadrature "
            "state, consistent-tangent Newton equilibrium, and cutback."
        ),
        procedure="standard/backward_euler/stateful",
        option_contract=_option_contract(
            "duration",
            "method",
            "incrementation",
            "quadrature_degree",
            "creep_strain_error_tolerance",
            "progress",
            "status_file",
            "amplitude",
            "temperature",
            required=("duration",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="direct_harmonic_system",
        analyses=("frequency_domain",),
        accepts=_accept_direct_harmonic_system,
        lower=_lower_direct_harmonic_system,
        priority=140,
        description=(
            "Lower explicit storage, loss, mass, viscous damping and force "
            "operators to a real-block direct harmonic finite-element system."
        ),
        procedure="standard/direct_harmonic/real_block_complex",
        option_contract=_option_contract(
            "M",
            "C",
            "K_loss",
            "frequency",
            "angular_frequency",
            "frequencies",
            "angular_frequencies",
            "method",
            "load_phase",
            "responses",
            "execution_order",
            "solver_options",
            "output",
            "progress",
            "checkpoint",
            "status_file",
            required=("K", "F"),
            exactly_one_of=(
                (
                    "frequency",
                    "angular_frequency",
                    "frequencies",
                    "angular_frequencies",
                ),
            ),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="harmonic_generalized_maxwell",
        analyses=("frequency_domain",),
        accepts=_accept_harmonic_viscoelasticity,
        lower=_lower_harmonic_viscoelasticity,
        priority=120,
        description=(
            "Lower isotropic generalized-Maxwell storage/loss moduli to a "
            "real-block direct harmonic finite-element system."
        ),
        procedure="standard/direct_harmonic/real_block_complex",
        option_contract=_option_contract(
            "frequency",
            "angular_frequency",
            "method",
            "density",
            "load_phase",
            "temperature",
            "solver_options",
            "output",
            exactly_one_of=(("frequency", "angular_frequency"),),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="quasistatic_generalized_maxwell",
        analyses=("first_order_transient",),
        accepts=_accept_quasistatic_viscoelasticity,
        lower=_lower_quasistatic_viscoelasticity,
        priority=120,
        description=(
            "Lower an isotropic generalized-Maxwell solid to exact quadrature "
            "history and incremental global equilibrium."
        ),
        procedure="standard/exact_generalized_maxwell/stateful",
        option_contract=_option_contract(
            "duration",
            "steps",
            "time_points",
            "incrementation",
            "time_error_tolerance",
            "method",
            "quadrature_degree",
            "progress",
            "status_file",
            "checkpoint",
            "amplitude",
            "temperature",
            required=("duration",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="implicit_euler_heat_transfer",
        analyses=("first_order_transient",),
        accepts=_accept_transient_heat,
        lower=_lower_transient_heat,
        priority=100,
        description="Lower heat capacity/conduction/source to implicit Euler.",
        procedure="standard/implicit_euler",
        option_contract=_option_contract(
            "dt",
            "steps",
            "C",
            "Q",
            "update_load",
            "save_every",
            "print_every",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            required=("dt", "steps"),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="linear_static_operators",
        analyses=("linear_static",),
        accepts=_accept_linear_static,
        lower=_lower_linear_static,
        priority=100,
        description="Lower K/F engineering operators to a linear static solve.",
        procedure="standard/linear",
        option_contract=_option_contract(),
    )
)
register_step_provider(
    StepProvider(
        name="finite_strain_j2_mixed_affine_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2_mixed_affine,
        lower=_lower_finite_strain_j2,
        priority=135,
        description=(
            "Lower logarithmic finite-strain J2 to P2/DG0 3D or Q2/DPC1 "
            "plane-strain mixed equilibrium under exact affine kinematics."
        ),
        procedure=("standard/newton/stateful/mixed_mean_kirchhoff_stress/affine_mpc"),
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "output_every",
            "progress",
            "status_file",
            "checkpoint",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="finite_strain_j2_affine_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2_affine,
        lower=_lower_finite_strain_j2,
        priority=130,
        description=(
            "Lower logarithmic finite-strain J2 to provider-owned quadrature "
            "state and exact affine/MPC Newton equilibrium."
        ),
        procedure="standard/newton/stateful/affine_mpc",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "output_every",
            "progress",
            "status_file",
            "checkpoint",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="finite_strain_j2_strong_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2_strong,
        lower=_lower_finite_strain_j2,
        priority=125,
        description=(
            "Lower logarithmic finite-strain J2 to provider-owned quadrature "
            "state and ordinary strong-boundary Newton equilibrium."
        ),
        procedure="standard/newton/stateful",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "amplitude",
            "output_every",
            "progress",
            "status_file",
            "checkpoint",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="j2_small_strain_static",
        analyses=("nonlinear_static",),
        accepts=_accept_j2,
        lower=_lower_j2,
        priority=110,
        description=(
            "Lower J2 plasticity to quadrature-state Newton equilibrium "
            "with algorithmic tangent and cutback."
        ),
        procedure="standard/newton/stateful",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "progress",
            "status_file",
            "amplitude",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="mixed_neo_hookean_constant_pressure",
        analyses=("nonlinear_static",),
        accepts=_accept_mixed_neo_hookean,
        lower=_lower_mixed_neo_hookean,
        priority=115,
        description=(
            "Lower P2 displacement and DG0 pressure to mixed finite-strain "
            "hyperelastic equilibrium."
        ),
        procedure="standard/newton/mixed_constant_pressure",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="decoupled_fabric_membrane_static",
        analyses=("nonlinear_static",),
        accepts=_accept_fabric_membrane,
        lower=_lower_fabric_membrane,
        priority=130,
        description=(
            "Lower one or more named woven layers with independent yarn-tension "
            "and trellising-shear channels to finite-kinematics in-plane "
            "membrane equilibrium."
        ),
        procedure="standard/newton/fabric_membrane",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="neo_hookean_finite_strain_static",
        analyses=("nonlinear_static",),
        accepts=_accept_neo_hookean,
        lower=_lower_neo_hookean,
        priority=100,
        description=(
            "Lower a supported displacement-based hyperelastic material to "
            "total-Lagrangian equilibrium."
        ),
        procedure="standard/newton",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="linear_structural_modes",
        analyses=("modal",),
        accepts=_accept_modal,
        lower=_lower_modal,
        priority=110,
        description="Lower K/M operators to a constrained SLEPc modal solve.",
        procedure="standard/generalized_hermitian_eigenproblem",
        option_contract=_option_contract(
            "M",
            "modes",
            "target_frequency",
            "tolerance",
            "maximum_iterations",
            "rigid_mode_tolerance",
            required=("modes",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="implicit_structural_dynamics",
        analyses=("second_order_dynamics",),
        accepts=_accept_implicit_dynamics,
        lower=_lower_implicit_dynamics,
        priority=110,
        description="Lower second-order operators to Newmark/generalized-alpha.",
        procedure="standard/newmark_or_generalized_alpha",
        option_contract=_option_contract(
            "dt",
            "steps",
            "method",
            "spectral_radius",
            "M",
            "C",
            "state",
            "update_load",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            "save_every",
            "print_every",
            required=("dt", "steps"),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="central_difference_explicit_dynamics",
        analyses=("explicit_dynamics", "second_order_dynamics"),
        accepts=_accept_explicit_dynamics,
        lower=_lower_explicit_dynamics,
        priority=100,
        description="Lower a second-order study to explicit central difference.",
        procedure="explicit/central_difference",
        option_contract=_option_contract(
            "dt",
            "steps",
            "method",
            "residual",
            "state",
            "mass",
            "cohesive_force",
            "prescribed",
            "update_load",
            "save_every",
            "print_every",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            required=("dt", "steps"),
        ),
    )
)
