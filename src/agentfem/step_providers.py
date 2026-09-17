# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

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

from ._step_provider_registry import ProviderSelectionRegistry
from ._step_provider_support import (
    COMMON_STEP_OPTIONS as _COMMON_STEP_OPTIONS,
    all_materials_support as _all_materials_support,
    has_complete_linear_system as _has_complete_linear_system,
    is_scalar_target as _is_scalar_target,
    is_vector_target as _is_vector_target,
    normalize as _normalize,
    procedure_method as _procedure_method,
    registered_materials as _registered_materials,
    same_procedure as _same_procedure,
    selected_material as _selected_material,
    supports_axisymmetric_elasticity as _supports_axisymmetric_elasticity,
    supports_conduction as _supports_conduction,
    supports_dynamics as _supports_dynamics,
    supports_elasticity as _supports_elasticity,
    supports_heat_capacity as _supports_heat_capacity,
    supports_stateful_constitutive as _supports_stateful_constitutive,
    target_shape as _target_shape,
    target_summary as _target_summary,
)


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
    that the selected provider enforces at runtime. ``required`` names must all
    be present. Each ``exactly_one_of`` group expresses scientific aliases for
    which precisely one coordinate must be supplied, such as frequency in Hz
    versus angular frequency in rad/s.
    """

    accepted: tuple[str, ...]
    required: tuple[str, ...] = ()
    exactly_one_of: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        raw_groups = tuple(tuple(group) for group in self.exactly_one_of)
        grouped_names = tuple(name for group in raw_groups for name in group)
        raw = (*self.accepted, *self.required, *grouped_names)
        invalid = tuple(item for item in raw if not isinstance(item, str) or not item)
        if invalid:
            raise ValueError("Step option names must be non-empty strings.")
        accepted = tuple(dict.fromkeys(self.accepted))
        required = tuple(dict.fromkeys(self.required))
        unknown_required = tuple(item for item in required if item not in accepted)
        if unknown_required:
            raise ValueError(
                f"Required Step options must also be accepted: {unknown_required!r}."
            )
        invalid_groups = tuple(
            group
            for group in raw_groups
            if len(tuple(dict.fromkeys(group))) < 2
            or any(name not in accepted for name in group)
        )
        if invalid_groups:
            raise ValueError(
                "Each exactly_one_of group must contain at least two distinct "
                "accepted Step options."
            )
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "required", required)
        object.__setattr__(
            self,
            "exactly_one_of",
            tuple(tuple(dict.fromkeys(group)) for group in raw_groups),
        )

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
            suggestions = tuple(
                get_close_matches(name, self.accepted, n=3, cutoff=0.58)
            )
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
        for group in self.exactly_one_of:
            selected = tuple(
                name for name in group if name in options and options[name] is not None
            )
            if len(selected) > 1 or (require_required and not selected):
                choices = ", ".join(repr(name) for name in group)
                issues.append(
                    {
                        "code": "AFM-STEP-OPTION-003",
                        "option": "|".join(group),
                        "options": group,
                        "selected": selected,
                        "message": f"Specify exactly one of {choices}.",
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
                message += (
                    " Did you mean "
                    + ", ".join(repr(item) for item in suggestions)
                    + "?"
                )
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
            "exactly_one_of": self.exactly_one_of,
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
            "procedure": (None if self.procedure is None else self.procedure.summary()),
            "option_names": tuple(sorted(self.options)),
            "material": (
                None if self.material is None else type(self.material).__name__
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
                None if self.option_contract is None else self.option_contract.summary()
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


class StepProviderRegistry(ProviderSelectionRegistry):
    """Compatibility facade adding lowering to the selection-only registry."""

    def lower(self, model, request: StepRequest):
        provider = self.resolve(model, request)
        created = provider.lower(model, request)
        return _bind_execution_context(model, request, provider, created)


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
    :func:`lower_step`. A GUI, agent, or ``model.check()`` therefore cannot
    advertise a Study/provider combination that the solver will later reject.
    ``supported`` answers whether the selected Study and explicit options fit
    an installed provider. ``ready`` additionally answers whether all required
    scientific inputs have been supplied.
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
    candidates = _DEFAULT_REGISTRY.candidates(selected_analysis)
    provider = None
    option_issues = ()
    readiness_issues = ()
    selected_target = targets[0]
    for candidate_target in targets:
        request = StepRequest(
            analysis=selected_analysis,
            target=candidate_target,
            options=selected_options,
            procedure=selected_procedure,
        )
        accepted = tuple(item for item in candidates if item.accepts(model, request))
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
            if provider is not None and provider.option_contract is not None:
                readiness_issues = provider.option_contract.issues(request.options)
            break
    return {
        "analysis": selected_analysis,
        "physics": getattr(getattr(model, "study", None), "physics", None),
        "dimension": getattr(getattr(model, "study", None), "dimension", None),
        "assumption": getattr(getattr(model, "study", None), "assumption", None),
        "supported": provider is not None and not option_issues,
        "ready": provider is not None and not readiness_issues,
        "target": _target_summary(selected_target),
        "procedure": (
            None if selected_procedure is None else selected_procedure.summary()
        ),
        "provider": None if provider is None else provider.summary(),
        "candidate_providers": tuple(item.name for item in candidates),
        "option_issues": option_issues,
        "readiness_issues": readiness_issues,
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


def _bind_execution_context(model, request, provider, created):
    """Bind common workflow context after provider-owned scientific lowering."""

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
        # Third-party providers may return frozen/slotted executables. They
        # remain valid; only optional model-owned completion context is absent.
        pass
    return created


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
        "modal",
        "frequency_domain",
    }
    if analysis not in known:
        if requested is not None and not isinstance(
            requested, procedures.SolutionProcedure
        ):
            raise TypeError("Custom Step procedures must be SolutionProcedure objects.")
        return requested
    method = options.get("method")
    if (
        analysis == "frequency_domain"
        and requested is None
        and method is None
        and any(
            options.get(name) is not None
            for name in ("frequencies", "angular_frequencies")
        )
    ):
        method = "direct_harmonic_sweep"
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


from . import _builtin_step_providers as _builtin_step_provider_catalog
