"""Extensible lowering providers behind the stable :meth:`Model.step` API.

A provider decides whether it understands one analysis/material combination
and lowers that scientific request to an executable problem.  The registry
keeps case-specific choices out of user scripts and out of ``Model.step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StepRequest:
    """Normalized request passed from ``Model.step`` to a provider."""

    analysis: str
    target: object
    options: dict[str, object]

    @property
    def material(self):
        return self.options.get("material")


@dataclass(frozen=True)
class StepProvider:
    """One analysis lowering rule.

    ``accepts`` must be a read-only predicate. ``lower`` may construct and
    register an executable step on the supplied model.
    """

    name: str
    analyses: tuple[str, ...]
    accepts: Callable[[object, StepRequest], bool]
    lower: Callable[[object, StepRequest], object]
    priority: int = 0
    description: str = ""
    procedure: str | None = None

    def __post_init__(self) -> None:
        normalized = tuple(_normalize(item) for item in self.analyses)
        if not self.name:
            raise ValueError("StepProvider.name must be non-empty.")
        if not normalized:
            raise ValueError("StepProvider.analyses must be non-empty.")
        object.__setattr__(self, "analyses", normalized)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "analyses": self.analyses,
            "priority": self.priority,
            "description": self.description,
            "procedure": self.procedure,
        }


class StepProviderRegistry:
    """Ordered, inspectable collection of step-lowering providers."""

    def __init__(self):
        self._providers: dict[str, StepProvider] = {}

    def register(self, provider: StepProvider, *, replace: bool = False):
        if provider.name in self._providers and not replace:
            raise ValueError(
                f"Step provider {provider.name!r} is already registered."
            )
        self._providers[provider.name] = provider
        return provider

    def providers(self) -> tuple[StepProvider, ...]:
        return tuple(
            sorted(
                self._providers.values(),
                key=lambda item: (-item.priority, item.name),
            )
        )

    def resolve(self, model, request: StepRequest) -> StepProvider:
        analysis_candidates = [
            provider
            for provider in self.providers()
            if request.analysis in provider.analyses
        ]
        for provider in analysis_candidates:
            if provider.accepts(model, request):
                return provider
        registered_materials = [
            type(record.item).__name__
            for record in getattr(model, "materials", ())
        ]
        raise NotImplementedError(
            "No step provider accepted "
            f"analysis={request.analysis!r}, material="
            f"{type(request.material).__name__ if request.material is not None else None!r}, "
            f"registered_materials={registered_materials!r}. "
            f"Candidate providers={[item.name for item in analysis_candidates]!r}."
        )

    def lower(self, model, request: StepRequest):
        return self.resolve(model, request).lower(model, request)


_DEFAULT_REGISTRY = StepProviderRegistry()


def register_step_provider(provider: StepProvider, *, replace: bool = False):
    """Register a provider used by subsequent ``model.step(...)`` calls."""

    return _DEFAULT_REGISTRY.register(provider, replace=replace)


def step_providers() -> tuple[StepProvider, ...]:
    """Return the installed providers in deterministic selection order."""

    return _DEFAULT_REGISTRY.providers()


def step_capability(
    model,
    *,
    target=None,
    analysis: str | None = None,
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe whether the current model can be lowered without executing it.

    This is deliberately based on the same provider predicates used by
    :func:`lower_step`.  A GUI, agent, or ``model.check()`` therefore cannot
    advertise a Study/provider combination that the solver will later reject.
    """

    selected_analysis = _normalize(
        analysis or getattr(getattr(model, "study", None), "analysis", "")
    )
    selected_options = dict(options or {})
    targets = (
        (target,)
        if target is not None
        else tuple(getattr(model, "fields", ())) or (None,)
    )
    candidates = tuple(
        provider
        for provider in _DEFAULT_REGISTRY.providers()
        if selected_analysis in provider.analyses
    )
    provider = None
    selected_target = targets[0]
    for candidate_target in targets:
        request = StepRequest(
            analysis=selected_analysis,
            target=candidate_target,
            options=selected_options,
        )
        accepted = tuple(
            item for item in candidates if item.accepts(model, request)
        )
        if accepted:
            provider = accepted[0]
            selected_target = candidate_target
            break
    return {
        "analysis": selected_analysis,
        "physics": getattr(getattr(model, "study", None), "physics", None),
        "dimension": getattr(getattr(model, "study", None), "dimension", None),
        "assumption": getattr(getattr(model, "study", None), "assumption", None),
        "supported": provider is not None,
        "target": _target_summary(selected_target),
        "provider": None if provider is None else provider.summary(),
        "candidate_providers": tuple(item.name for item in candidates),
    }


def lower_step(model, *, analysis: str, target, options):
    """Normalize and lower one high-level step request."""

    request = StepRequest(
        analysis=_normalize(analysis),
        target=target,
        options=dict(options),
    )
    return _DEFAULT_REGISTRY.lower(model, request)


def _selected_material(model, request: StepRequest):
    selected = request.material
    if selected is None and len(getattr(model, "materials", ())) == 1:
        return model.materials[0].item
    return selected


def _registered_materials(model, request: StepRequest) -> tuple[object, ...]:
    selected = request.material
    if selected is not None:
        return (selected,)
    return tuple(record.item for record in getattr(model, "materials", ()))


def _target_summary(target) -> dict[str, object] | None:
    if target is None:
        return None
    shape = _target_shape(target)
    return {
        "name": getattr(target, "name", type(target).__name__),
        "kind": getattr(target, "kind", None),
        "shape": shape,
    }


def _target_shape(target) -> tuple[int, ...] | None:
    shape = getattr(target, "ufl_shape", None)
    if shape is None:
        value = getattr(target, "value", None)
        shape = getattr(value, "ufl_shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _is_scalar_target(target) -> bool:
    kind = getattr(target, "kind", None)
    if kind in {"temperature", "scalar_unknown"}:
        return True
    if kind in {"displacement", "vector_unknown"}:
        return False
    shape = _target_shape(target)
    return shape in {None, ()}


def _is_vector_target(target) -> bool:
    kind = getattr(target, "kind", None)
    if kind in {"displacement", "vector_unknown"}:
        return True
    if kind in {"temperature", "scalar_unknown"}:
        return False
    shape = _target_shape(target)
    return shape is None or len(shape) == 1


def _has_complete_linear_system(request: StepRequest) -> bool:
    return (
        request.options.get("K") is not None
        and request.options.get("F") is not None
    )


def _supports_elasticity(material) -> bool:
    return (
        hasattr(material, "stiffness_voigt")
        or (hasattr(material, "young") and hasattr(material, "poisson"))
    )


def _supports_dynamics(material) -> bool:
    return (
        _supports_elasticity(material)
        and getattr(material, "density", None) is not None
    )


def _supports_conduction(material) -> bool:
    return hasattr(material, "conductivity")


def _supports_heat_capacity(material) -> bool:
    return hasattr(material, "volumetric_heat_capacity")


def _all_materials_support(model, request: StepRequest, predicate) -> bool:
    materials = _registered_materials(model, request)
    return bool(materials) and all(predicate(item) for item in materials)


def _accept_linear_static(model, request: StepRequest) -> bool:
    study = getattr(model, "study", None)
    if request.target is None:
        return False
    physics = getattr(study, "physics", None)
    if physics == "solid_mechanics":
        return (
            getattr(study, "assumption", None) != "axisymmetric"
            and _is_vector_target(request.target)
            and (
                _has_complete_linear_system(request)
                or _all_materials_support(model, request, _supports_elasticity)
            )
        )
    if physics == "heat_transfer":
        return _is_scalar_target(request.target) and (
            _has_complete_linear_system(request)
            or _all_materials_support(model, request, _supports_conduction)
        )
    return False


def _lower_linear_static(model, request: StepRequest):
    options = dict(request.options)
    options.pop("material", None)
    name = options.pop("name", None) or "linear_static"
    return model.linear_static_step(
        target=request.target,
        name=name,
        **options,
    )


def _accept_transient_heat(model, request: StepRequest) -> bool:
    return (
        request.target is not None
        and _is_scalar_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None)
        == "heat_transfer"
        and _all_materials_support(model, request, _supports_conduction)
        and _all_materials_support(model, request, _supports_heat_capacity)
    )


def _lower_transient_heat(model, request: StepRequest):
    options = dict(request.options)
    options.pop("K", None)
    options.pop("F", None)
    material = options.pop("material", None)
    name = options.pop("name", None) or "transient_heat"
    return model.heat_transfer_step(
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_neo_hookean(model, request: StepRequest) -> bool:
    from .constitutive.hyperelasticity import NeoHookeanProperties

    study = getattr(model, "study", None)
    supported_kinematics = (
        getattr(study, "dimension", None) == 3
        or (
            getattr(study, "dimension", None) == 2
            and getattr(study, "assumption", None) == "plane_strain"
        )
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and supported_kinematics
        and isinstance(_selected_material(model, request), NeoHookeanProperties)
    )


def _accept_mixed_neo_hookean(model, request: StepRequest) -> bool:
    from .constitutive.hyperelasticity import MixedNeoHookeanProperties

    study = getattr(model, "study", None)
    supported_kinematics = (
        getattr(study, "dimension", None) == 3
        or (
            getattr(study, "dimension", None) == 2
            and getattr(study, "assumption", None) == "plane_strain"
        )
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
    from .constitutive.plasticity import J2LinearIsotropicHardening

    study = getattr(model, "study", None)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and getattr(study, "dimension", None) == 3
        and isinstance(
            _selected_material(model, request),
            J2LinearIsotropicHardening,
        )
    )


def _lower_j2(model, request: StepRequest):
    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    name = options.pop("name", None) or "j2_plasticity"
    return model.j2_plasticity_step(
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_implicit_creep(model, request: StepRequest) -> bool:
    from .constitutive.creep import IsotropicPowerLawCreepMaterial

    study = getattr(model, "study", None)
    method = request.options.get("method") or getattr(
        study,
        "preferred_procedure",
        None,
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "nonlinear_transient"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _normalize(method or "implicit_creep")
        in {"implicit_creep", "backward_euler"}
        and isinstance(
            _selected_material(model, request),
            IsotropicPowerLawCreepMaterial,
        )
    )


def _lower_implicit_creep(model, request: StepRequest):
    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("method", None)
    name = options.pop("name", None) or "implicit_creep"
    return model.creep_step(
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_neo_hookean(model, request: StepRequest):
    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    name = options.pop("name", None) or "finite_strain_static"
    options.pop("K", None)
    options.pop("F", None)
    return model.hyperelastic_step(
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_mixed_neo_hookean(model, request: StepRequest):
    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    name = options.pop("name", None) or "mixed_finite_strain_static"
    return model.mixed_hyperelastic_step(
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_explicit_dynamics(model, request: StepRequest) -> bool:
    method = request.options.get("method") or getattr(
        getattr(model, "study", None),
        "preferred_procedure",
        None,
    )
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None)
        == "solid_mechanics"
        and (
            (
                request.options.get("mass") is not None
                and request.options.get("residual") is not None
            )
            or _all_materials_support(model, request, _supports_dynamics)
        )
        and (
        method is None
        or _normalize(method) in {"explicit_dynamics", "central_difference"}
        )
    )


def _lower_explicit_dynamics(model, request: StepRequest):
    options = dict(request.options)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("solver_options", None)
    options.pop("method", None)
    name = options.pop("name", None) or "explicit_dynamics"
    return model.explicit_dynamics_step(
        target=request.target,
        name=name,
        **options,
    )


def _accept_implicit_dynamics(model, request: StepRequest) -> bool:
    method = request.options.get("method") or getattr(
        getattr(model, "study", None),
        "preferred_procedure",
        None,
    )
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None)
        == "solid_mechanics"
        and (
            all(request.options.get(item) is not None for item in ("M", "K", "F"))
            or _all_materials_support(model, request, _supports_dynamics)
        )
        and _normalize(method or "") in {"newmark", "generalized_alpha"}
    )


def _lower_implicit_dynamics(model, request: StepRequest):
    options = dict(request.options)
    options.pop("material", None)
    method = options.pop("method", None) or getattr(
        model.study,
        "preferred_procedure",
        None,
    )
    name = options.pop("name", None) or f"{method}_dynamics"
    return model.implicit_dynamics_step(
        target=request.target,
        method=method,
        name=name,
        **options,
    )


def _normalize(value: str) -> str:
    normalized = str(value).lower().replace("-", "_").strip()
    aliases = {
        "static": "linear_static",
        "hyperelastic": "nonlinear_static",
        "neo_hookean": "nonlinear_static",
        "explicit": "explicit_dynamics",
    }
    return aliases.get(normalized, normalized)


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
            "Lower P2 displacement and DG0 pressure to mixed finite-strain equilibrium."
        ),
        procedure="standard/newton/mixed_constant_pressure",
    )
)
register_step_provider(
    StepProvider(
        name="neo_hookean_finite_strain_static",
        analyses=("nonlinear_static",),
        accepts=_accept_neo_hookean,
        lower=_lower_neo_hookean,
        priority=100,
        description="Lower a Neo-Hookean material to total-Lagrangian equilibrium.",
        procedure="standard/newton",
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
    )
)
