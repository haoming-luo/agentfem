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


def _accept_linear_static(model, request: StepRequest) -> bool:
    return request.target is not None


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
        and getattr(getattr(model, "study", None), "physics", None)
        == "heat_transfer"
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

    return isinstance(_selected_material(model, request), NeoHookeanProperties)


def _accept_j2(model, request: StepRequest) -> bool:
    from .constitutive.plasticity import J2LinearIsotropicHardening

    return isinstance(
        _selected_material(model, request),
        J2LinearIsotropicHardening,
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


def _accept_explicit_dynamics(model, request: StepRequest) -> bool:
    method = request.options.get("method") or getattr(
        getattr(model, "study", None),
        "preferred_procedure",
        None,
    )
    return request.target is not None and (
        method is None
        or _normalize(method) in {"explicit_dynamics", "central_difference"}
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
    return request.target is not None and _normalize(method or "") in {
        "newmark",
        "generalized_alpha",
    }


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
