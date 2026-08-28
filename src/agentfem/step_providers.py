"""Extensible lowering providers behind the stable :meth:`Model.step` API.

A provider decides whether it understands one analysis/material combination
and lowers that scientific request to an executable problem.  The registry
keeps case-specific choices out of user scripts and out of ``Model.step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from os import PathLike
from types import MappingProxyType
from typing import Callable, Mapping


@dataclass(frozen=True)
class StepExecutionPolicy:
    """Normalized cross-cutting controls retained by every public Step.

    Users continue to pass the familiar ``model.step(...)`` keywords.  This
    object is the shared, inspectable representation consumed by result
    lifecycles, agents, GUIs, and provenance tools; it is not a second user
    configuration language.
    """

    solver: object | None = None
    output: object | None = None
    history: tuple[object, ...] = ()
    progress: object | None = None
    checkpoint: object | None = None

    def summary(self) -> dict[str, object]:
        return {
            "solver": _policy_value_summary(self.solver),
            "output": _policy_value_summary(self.output),
            "history": tuple(_policy_value_summary(item) for item in self.history),
            "progress": _policy_value_summary(self.progress),
            "checkpoint": _policy_value_summary(self.checkpoint),
        }


@dataclass(frozen=True)
class StepOptionContract:
    """Inspectable keyword contract for one public Step provider.

    ``Model.step`` deliberately remains one stable user-facing method.  The
    contract restores the precision that would otherwise be lost behind its
    extensible ``**options`` boundary: humans receive immediate repairable
    errors, while agents, IDEs, and GUIs can inspect the same option vocabulary
    that the selected provider enforces at runtime.
    """

    accepted: tuple[str, ...]
    required: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        raw = (*self.accepted, *self.required)
        invalid = tuple(item for item in raw if not isinstance(item, str) or not item)
        if invalid:
            raise ValueError("Step option names must be non-empty strings.")
        accepted = tuple(dict.fromkeys(self.accepted))
        required = tuple(dict.fromkeys(self.required))
        unknown_required = tuple(item for item in required if item not in accepted)
        if unknown_required:
            raise ValueError(
                "Required Step options must also be accepted: "
                f"{unknown_required!r}."
            )
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "required", required)

    def issues(
        self,
        options: Mapping[str, object],
        *,
        require_required: bool = True,
    ) -> tuple[dict[str, object], ...]:
        """Return stable, machine-readable option issues without raising."""

        accepted = set(self.accepted)
        issues = []
        for name in sorted(set(options).difference(accepted)):
            suggestions = tuple(get_close_matches(name, self.accepted, n=3, cutoff=0.58))
            issues.append(
                {
                    "code": "AFM-STEP-OPTION-001",
                    "option": name,
                    "message": f"Unsupported Step option {name!r}.",
                    "suggestions": suggestions,
                }
            )
        if require_required:
            for name in self.required:
                if name not in options or options[name] is None:
                    issues.append(
                        {
                            "code": "AFM-STEP-OPTION-002",
                            "option": name,
                            "message": f"Required Step option {name!r} is missing.",
                            "suggestions": (),
                        }
                    )
        return tuple(issues)

    def validate(self, options: Mapping[str, object], *, provider: str) -> None:
        """Raise one concise error for an invalid provider request."""

        issues = self.issues(options)
        if not issues:
            return
        details = []
        for issue in issues:
            message = str(issue["message"])
            suggestions = tuple(issue["suggestions"])
            if suggestions:
                message += " Did you mean " + ", ".join(repr(item) for item in suggestions) + "?"
            details.append(message)
        raise TypeError(
            f"model.step request for provider {provider!r} is invalid: "
            + " ".join(details)
            + f" Accepted options are {self.accepted!r}."
        )

    def summary(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "required": self.required,
        }


@dataclass(frozen=True)
class StepRequest:
    """Normalized request passed from ``Model.step`` to a provider."""

    analysis: str
    target: object
    options: Mapping[str, object]
    procedure: object | None = None

    def __post_init__(self) -> None:
        normalized = _normalize(self.analysis)
        selected = dict(self.options)
        if any(not isinstance(key, str) or not key for key in selected):
            raise TypeError("StepRequest option names must be non-empty strings.")
        object.__setattr__(self, "analysis", normalized)
        object.__setattr__(self, "options", MappingProxyType(selected))

    @property
    def material(self):
        return self.options.get("material")

    @property
    def method(self) -> str | None:
        """Return the requested algorithm name without inspecting the Study."""

        if self.procedure is not None:
            return getattr(self.procedure, "algorithm", None)
        selected = self.options.get("method")
        return None if selected is None else _normalize(selected)

    def option(self, name: str, default=None):
        """Read one normalized provider option."""

        return self.options.get(name, default)

    def lowering_options(self, *drop: str) -> dict[str, object]:
        """Return a mutable copy intended only for the selected lowerer."""

        selected = dict(self.options)
        for name in drop:
            selected.pop(name, None)
        return selected

    @property
    def execution_policy(self) -> StepExecutionPolicy:
        """Return the common execution controls encoded by this request."""

        history = self.option("history", ())
        if history is None:
            history = ()
        elif not isinstance(history, (tuple, list)):
            history = (history,)
        return StepExecutionPolicy(
            solver=self.option("solver_options"),
            output=self.option("output"),
            history=tuple(history),
            progress=self.option("progress"),
            checkpoint=self.option("checkpoint"),
        )

    def summary(self) -> dict[str, object]:
        return {
            "analysis": self.analysis,
            "target": _target_summary(self.target),
            "procedure": (
                None
                if self.procedure is None
                else self.procedure.summary()
            ),
            "option_names": tuple(sorted(self.options)),
            "material": (
                None
                if self.material is None
                else type(self.material).__name__
            ),
            "execution_policy": self.execution_policy.summary(),
        }


@dataclass(frozen=True)
class StepExecutionContext:
    """Public workflow assets retained by one lowered executable Step.

    Providers remain responsible only for scientific capability selection and
    lowering.  This compact context lets the common result lifecycle recover
    the owning model, target, material, and declarative output without making
    every solver-specific Step constructor depend on those workflow objects.
    """

    model: object
    target: object
    material: object | None = None
    policy: StepExecutionPolicy = StepExecutionPolicy()

    @property
    def configured_output(self):
        """Compatibility view of the declarative output policy."""

        return self.policy.output

    @property
    def configured_history(self) -> tuple[object, ...]:
        """History requests declared when the Step was constructed."""

        return self.policy.history

    @property
    def output_target(self):
        """Return the physical field used by post-processing output plans."""

        if getattr(self.target, "kind", None) == "displacement_pressure":
            return self.target.collapsed_displacement(name="U")
        return self.target

    def summary(self) -> dict[str, object]:
        declared = self.policy.summary()
        return {
            "model": getattr(self.model, "name", type(self.model).__name__),
            "target": _target_summary(self.target),
            "material": (
                None if self.material is None else type(self.material).__name__
            ),
            "configured_output": _policy_value_summary(self.configured_output),
            # ``policies`` is retained for the 0.2 result schema. New
            # consumers should use the explicit name and read resolved
            # numerical choices from ``metadata["step"]``.
            "policies": declared,
            "declared_policies": declared,
            "resolved_step_record": "metadata.step",
        }


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
    option_contract: StepOptionContract | None = None

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
            "options": (
                None
                if self.option_contract is None
                else self.option_contract.summary()
            ),
        }

    def option_issues(self, request: StepRequest) -> tuple[dict[str, object], ...]:
        """Return request issues, preserving unrestricted extension providers."""

        if self.option_contract is None:
            return ()
        return self.option_contract.issues(request.options)

    def validate_options(self, request: StepRequest) -> None:
        if self.option_contract is not None:
            self.option_contract.validate(request.options, provider=self.name)


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
        option_rejections = []
        for provider in analysis_candidates:
            if provider.accepts(model, request):
                issues = provider.option_issues(request)
                if not issues:
                    return provider
                option_rejections.append(provider)
        if option_rejections:
            # A lower-priority provider may legitimately own a distinct option
            # vocabulary for the same analysis. Only fail after every
            # scientifically compatible candidate has rejected the request.
            option_rejections[0].validate_options(request)
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
        provider = self.resolve(model, request)
        created = provider.lower(model, request)
        if request.procedure is not None and hasattr(created, "procedure"):
            actual = getattr(created, "procedure", None)
            if actual is not None and not _same_procedure(actual, request.procedure):
                raise RuntimeError(
                    f"Step provider {provider.name!r} lowered procedure "
                    f"{actual.summary()!r}, which does not match the requested "
                    f"procedure {request.procedure.summary()!r}."
                )
            created.procedure = request.procedure
        context_material = _selected_material(model, request)
        if context_material is None:
            context_material = getattr(created, "material", None)
        context = StepExecutionContext(
            model=model,
            target=request.target,
            material=context_material,
            policy=request.execution_policy,
        )
        try:
            created.execution_context = context
        except (AttributeError, TypeError):
            # Third-party providers may return frozen/slotted executables.
            # They remain valid; only the optional model-owned completion
            # context is unavailable until that provider exposes a binding.
            pass
        return created


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
    procedure=None,
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
    selected_procedure = _resolve_procedure(
        model,
        analysis=selected_analysis,
        options=selected_options,
        requested=procedure,
    )
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
    option_issues = ()
    selected_target = targets[0]
    for candidate_target in targets:
        request = StepRequest(
            analysis=selected_analysis,
            target=candidate_target,
            options=selected_options,
            procedure=selected_procedure,
        )
        accepted = tuple(
            item for item in candidates if item.accepts(model, request)
        )
        if accepted:
            rejected = []
            for candidate in accepted:
                issues = (
                    ()
                    if candidate.option_contract is None
                    else candidate.option_contract.issues(
                        request.options,
                        # Capability probes are intentionally partial:
                        # internal builders pass only the options needed to
                        # select a provider. Complete required-input
                        # enforcement happens once, at lower_step.
                        require_required=False,
                    )
                )
                if not issues:
                    provider = candidate
                    option_issues = ()
                    break
                rejected.append((candidate, issues))
            if provider is None and rejected:
                provider, option_issues = rejected[0]
            selected_target = candidate_target
            break
    return {
        "analysis": selected_analysis,
        "physics": getattr(getattr(model, "study", None), "physics", None),
        "dimension": getattr(getattr(model, "study", None), "dimension", None),
        "assumption": getattr(getattr(model, "study", None), "assumption", None),
        "supported": provider is not None and not option_issues,
        "target": _target_summary(selected_target),
        "procedure": (
            None
            if selected_procedure is None
            else selected_procedure.summary()
        ),
        "provider": None if provider is None else provider.summary(),
        "candidate_providers": tuple(item.name for item in candidates),
        "option_issues": option_issues,
    }


def lower_step(model, *, analysis: str, target, options, procedure=None):
    """Normalize and lower one high-level step request."""

    selected_analysis = _normalize(analysis)
    selected_options = dict(options)
    selected_procedure = _resolve_procedure(
        model,
        analysis=selected_analysis,
        options=selected_options,
        requested=procedure,
    )
    request = StepRequest(
        analysis=selected_analysis,
        target=target,
        options=selected_options,
        procedure=selected_procedure,
    )
    return _DEFAULT_REGISTRY.lower(model, request)


def _resolve_procedure(model, *, analysis: str, options, requested):
    """Resolve built-in procedures while leaving extension analyses open."""

    from . import procedures

    known = {
        "linear_static",
        "nonlinear_static",
        "first_order_transient",
        "nonlinear_transient",
        "second_order_dynamics",
        "explicit_dynamics",
    }
    if analysis not in known:
        if requested is not None and not isinstance(
            requested, procedures.SolutionProcedure
        ):
            raise TypeError(
                "Custom Step procedures must be SolutionProcedure objects."
            )
        return requested
    method = options.get("method")
    if requested is not None and method is not None:
        requested_name = (
            requested.algorithm
            if isinstance(requested, procedures.SolutionProcedure)
            else _normalize(requested)
        )
        if _normalize(method) != _normalize(requested_name):
            raise ValueError(
                "Pass procedure= or method=, not conflicting numerical routes."
            )
    material = options.get("material")
    registered = tuple(getattr(model, "materials", ()))
    materials = (
        (material,)
        if material is not None
        else tuple(record.item for record in registered)
    )
    stateful = analysis == "nonlinear_transient" or any(
        _supports_stateful_constitutive(selected) for selected in materials
    )
    return procedures.resolve(
        analysis=analysis,
        requested=requested if requested is not None else method,
        preferred=getattr(getattr(model, "study", None), "preferred_procedure", None),
        stateful=stateful,
    )


def _same_procedure(left, right) -> bool:
    names = (
        "family",
        "equation_order",
        "control",
        "algorithm",
        "nonlinear",
        "requires_global_solve",
        "stateful",
    )
    return all(getattr(left, name, None) == getattr(right, name, None) for name in names)


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


def _procedure_method(model, request: StepRequest) -> str | None:
    if request.procedure is not None:
        return _normalize(request.procedure.algorithm)
    if request.method is not None:
        return _normalize(request.method)
    return getattr(getattr(model, "study", None), "preferred_procedure", None)


def _target_summary(target) -> dict[str, object] | None:
    if target is None:
        return None
    shape = _target_shape(target)
    return {
        "name": getattr(target, "name", type(target).__name__),
        "kind": getattr(target, "kind", None),
        "shape": shape,
    }


def _policy_value_summary(value):
    """Describe one execution control without retaining live solver objects."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _policy_value_summary(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return tuple(_policy_value_summary(item) for item in value)
    summary = getattr(value, "summary", None)
    if callable(summary):
        return {
            "type": type(value).__name__,
            "value": _policy_value_summary(summary()),
        }
    return {"type": type(value).__name__}


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


def _supports_axisymmetric_elasticity(material) -> bool:
    """Require a constitutive record that defines the full isotropic hoop response."""

    return (
        hasattr(material, "young")
        and hasattr(material, "poisson")
        and not hasattr(material, "stiffness_voigt")
    )


def _supports_dynamics(material) -> bool:
    return (
        _supports_elasticity(material)
        and getattr(material, "density", None) is not None
    )


def _supports_conduction(material) -> bool:
    return hasattr(material, "conductivity")


def _supports_heat_capacity(material) -> bool:
    return hasattr(material, "volumetric_heat_capacity") or (
        getattr(material, "density", None) is not None
        and hasattr(material, "specific_heat")
    )


def _supports_stateful_constitutive(material) -> bool:
    """Return whether a material declares committed constitutive history.

    The protocol is intentionally structural so installed extensions can join
    procedure dispatch without teaching AgentFEM their concrete class names.
    Regional quadrature maps are stateful when every contained material makes
    the same declaration.
    """

    if bool(getattr(material, "stateful_constitutive", False)):
        return True
    regional = getattr(material, "materials", None)
    if regional is None:
        return False
    values = (
        tuple(regional.values())
        if hasattr(regional, "values")
        else tuple(regional)
    )
    return bool(values) and all(
        bool(getattr(item, "stateful_constitutive", False)) for item in values
    )


def _all_materials_support(model, request: StepRequest, predicate) -> bool:
    materials = _registered_materials(model, request)
    return bool(materials) and all(predicate(item) for item in materials)


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
        return (
            _is_vector_target(request.target)
            and (
                _has_complete_linear_system(request)
                or _all_materials_support(model, request, material_predicate)
            )
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
        and getattr(getattr(model, "study", None), "physics", None)
        == "heat_transfer"
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


def _accept_finite_strain_j2(model, request: StepRequest) -> bool:
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import AbaqusPeriodicConstraint

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    if not isinstance(selected_constraints, (tuple, list)):
        selected_constraints = (selected_constraints,)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and len(selected_constraints) == 1
        and isinstance(selected_constraints[0], AbaqusPeriodicConstraint)
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
    selected_material = _selected_material(model, request)
    explicit_residual = request.options.get("residual") is not None
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
            explicit_residual
            or not hyperelasticity.is_finite_strain_hyperelastic(selected_material)
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
    material = _selected_material(model, request)
    supported_kinematics = hyperelasticity.supports_hyperelastic_study(
        material,
        dimension=getattr(study, "dimension", 0),
        assumption=getattr(study, "assumption", None),
    )
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "second_order_dynamics"
        and supported_kinematics
        and hyperelasticity.is_finite_strain_hyperelastic(material)
        and material.density is not None
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
        and getattr(getattr(model, "study", None), "physics", None)
        == "solid_mechanics"
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


def _normalize(value: str) -> str:
    normalized = str(value).lower().replace("-", "_").strip()
    aliases = {
        "static": "linear_static",
        "hyperelastic": "nonlinear_static",
        "neo_hookean": "nonlinear_static",
        "explicit": "explicit_dynamics",
    }
    return aliases.get(normalized, normalized)


_COMMON_STEP_OPTIONS = (
    "K",
    "F",
    "constraints",
    "solver_options",
    "name",
    "material",
    "output",
)


def _option_contract(*extra: str, required=()) -> StepOptionContract:
    """Build a deterministic built-in contract without repeated core names."""

    return StepOptionContract(
        accepted=tuple(dict.fromkeys((*_COMMON_STEP_OPTIONS, *extra))),
        required=tuple(required),
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
            "Lower a supported finite-strain hyperelastic material to a "
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
        name="finite_strain_j2_affine_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2,
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
