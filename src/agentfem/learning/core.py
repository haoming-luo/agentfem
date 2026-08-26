"""Provider-neutral scientific contracts for neural field computation.

The records in this module describe what a physics-informed or variational
neural computation means.  They deliberately do not own a neural-network
architecture, optimizer, or third-party training framework.  Those remain
replaceable execution-provider concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping

from ..surrogates.physics import FieldEncoding, PINNSpec


_OBJECTIVE_KINDS = {"residual", "energy", "data", "constraint"}
_OBJECTIVE_FORMS = {
    "strong",
    "weak",
    "discrete",
    "variational",
    "pointwise",
    "integral",
}
_MEASURES = {
    "domain",
    "boundary",
    "interface",
    "initial",
    "observation",
    "discrete",
}
_CONDITION_KINDS = {"boundary", "initial", "interface", "observation"}
_ENFORCEMENTS = {"hard", "penalty", "lagrange_multiplier", "nitsche", "data"}
_PURPOSES = {"forward", "inverse", "data_assimilation", "hybrid"}
_PARAMETER_ROLES = {"material", "load", "boundary", "source", "geometry", "state"}
_TRANSFORMS = {"identity", "log", "logit"}
_STANDARD_SAMPLING = {
    "uniform",
    "random",
    "latin_hypercube",
    "halton",
    "hammersley",
    "sobol",
    "quadrature",
    "adaptive",
    "provided",
}
_STANDARD_REPRESENTATIONS = {
    "user_module",
    "mlp",
    "fourier_feature_network",
    "rbf_network",
    "kan",
}
_INTEGRATION_ROLES = {"training", "validation", "refinement"}


@dataclass(frozen=True)
class ObjectiveTerm:
    """One named contribution to a neural-field optimization objective.

    ``coefficient`` carries the physical sign and multiplier of a contribution
    (for example ``-1`` for external work in total potential energy), while
    ``weight`` is the positive numerical weight used to balance optimization
    terms.  Keeping them separate prevents loss tuning from silently changing
    the stated mechanics.
    """

    name: str
    kind: str
    expression: str
    dependent_fields: tuple[str, ...]
    form: str
    measure: str
    unit: str | None = None
    coefficient: float = 1.0
    weight: float = 1.0
    implementation: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _name(self.name, "ObjectiveTerm.name")
        kind = _choice(self.kind, _OBJECTIVE_KINDS, "ObjectiveTerm.kind")
        form = _choice(self.form, _OBJECTIVE_FORMS, "ObjectiveTerm.form")
        measure = _choice(self.measure, _MEASURES, "ObjectiveTerm.measure")
        expression = str(self.expression).strip()
        fields = tuple(_name(item, "ObjectiveTerm.dependent_fields") for item in self.dependent_fields)
        if not expression:
            raise ValueError("ObjectiveTerm.expression must be explicit.")
        if not fields:
            raise ValueError("ObjectiveTerm requires at least one dependent field.")
        if kind == "residual" and form not in {"strong", "weak", "discrete"}:
            raise ValueError("A residual objective must use strong, weak, or discrete form.")
        if kind == "energy" and form != "variational":
            raise ValueError("An energy objective must use variational form.")
        if kind == "data" and form not in {"pointwise", "integral"}:
            raise ValueError("A data objective must use pointwise or integral form.")
        if not isfinite(float(self.coefficient)):
            raise ValueError("ObjectiveTerm.coefficient must be finite.")
        if not isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("ObjectiveTerm.weight must be finite and positive.")
        implementation = _optional_name(self.implementation)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "form", form)
        object.__setattr__(self, "measure", measure)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "dependent_fields", fields)
        object.__setattr__(self, "coefficient", float(self.coefficient))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "form": self.form,
            "measure": self.measure,
            "expression": self.expression,
            "dependent_fields": self.dependent_fields,
            "unit": self.unit,
            "coefficient": self.coefficient,
            "weight": self.weight,
            "implementation": self.implementation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConditionSpec:
    """A physical condition and the declared way it enters an objective."""

    name: str
    kind: str
    target: str
    on: str
    value: object | None = None
    enforcement: str = "penalty"
    weight: float = 1.0
    implementation: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _name(self.name, "ConditionSpec.name")
        kind = _choice(self.kind, _CONDITION_KINDS, "ConditionSpec.kind")
        target = _name(self.target, "ConditionSpec.target")
        on = _name(self.on, "ConditionSpec.on")
        enforcement = _choice(
            self.enforcement,
            _ENFORCEMENTS,
            "ConditionSpec.enforcement",
        )
        if not isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("ConditionSpec.weight must be finite and positive.")
        value = _data_value(self.value, "ConditionSpec.value")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "on", on)
        object.__setattr__(self, "enforcement", enforcement)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "implementation", _optional_name(self.implementation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "target": self.target,
            "on": self.on,
            "value": self.value,
            "enforcement": self.enforcement,
            "weight": self.weight,
            "implementation": self.implementation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SamplingPlan:
    """Inspectable coordinates or integration samples for one physical set."""

    name: str
    on: str
    strategy: str
    count: int | None = None
    region: str | None = None
    seed: int | None = None
    resample_every: int | None = None
    implementation: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _name(self.name, "SamplingPlan.name")
        on = _choice(self.on, _MEASURES, "SamplingPlan.on")
        strategy = _extensible_name(
            self.strategy,
            _STANDARD_SAMPLING,
            "SamplingPlan.strategy",
        )
        count = None if self.count is None else int(self.count)
        if count is not None and count <= 0:
            raise ValueError("SamplingPlan.count must be positive when supplied.")
        if count is None and strategy not in {"quadrature", "provided"}:
            raise ValueError(
                "SamplingPlan.count is required unless strategy is quadrature or provided."
            )
        resample_every = (
            None if self.resample_every is None else int(self.resample_every)
        )
        if resample_every is not None and resample_every <= 0:
            raise ValueError("SamplingPlan.resample_every must be positive.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "on", on)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "region", _optional_name(self.region))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "resample_every", resample_every)
        object.__setattr__(self, "implementation", _optional_name(self.implementation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "on": self.on,
            "strategy": self.strategy,
            "count": self.count,
            "region": self.region,
            "seed": self.seed,
            "resample_every": self.resample_every,
            "implementation": self.implementation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IntegrationRule:
    """One inspectable numerical-integration point set.

    The rule describes scientific identity only.  Coordinates, tensor
    creation, differentiation, and device placement remain provider concerns.
    ``independent_of`` makes held-out validation an explicit relationship
    rather than an inference from a different point count.
    """

    name: str
    role: str
    strategy: str
    count: int | None = None
    order: int | None = None
    seed: int | None = None
    independent_of: tuple[str, ...] = ()
    implementation: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _name(self.name, "IntegrationRule.name")
        role = _choice(self.role, _INTEGRATION_ROLES, "IntegrationRule.role")
        strategy = _extensible_name(
            self.strategy,
            _STANDARD_SAMPLING,
            "IntegrationRule.strategy",
        )
        count = None if self.count is None else int(self.count)
        order = None if self.order is None else int(self.order)
        if count is not None and count <= 0:
            raise ValueError("IntegrationRule.count must be positive when supplied.")
        if order is not None and order <= 0:
            raise ValueError("IntegrationRule.order must be positive when supplied.")
        if count is None and order is None and strategy != "provided":
            raise ValueError(
                "IntegrationRule requires count or order unless strategy='provided'."
            )
        independent = tuple(
            _name(item, "IntegrationRule.independent_of")
            for item in self.independent_of
        )
        if name in independent or len(set(independent)) != len(independent):
            raise ValueError(
                "IntegrationRule.independent_of must be unique and exclude itself."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "independent_of", independent)
        object.__setattr__(self, "implementation", _optional_name(self.implementation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "strategy": self.strategy,
            "count": self.count,
            "order": self.order,
            "seed": self.seed,
            "independent_of": self.independent_of,
            "implementation": self.implementation,
            "metadata": dict(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.summary())


@dataclass(frozen=True)
class IntegrationPlan:
    """Training, held-out validation, and optional refinement integration."""

    training: IntegrationRule
    validation: IntegrationRule
    refinements: tuple[IntegrationRule, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.training, IntegrationRule):
            raise TypeError("IntegrationPlan.training must be an IntegrationRule.")
        if not isinstance(self.validation, IntegrationRule):
            raise TypeError("IntegrationPlan.validation must be an IntegrationRule.")
        refinements = tuple(self.refinements)
        if any(not isinstance(item, IntegrationRule) for item in refinements):
            raise TypeError("IntegrationPlan.refinements must contain IntegrationRule records.")
        if self.training.role != "training":
            raise ValueError("IntegrationPlan.training must have role='training'.")
        if self.validation.role != "validation":
            raise ValueError("IntegrationPlan.validation must have role='validation'.")
        if any(item.role != "refinement" for item in refinements):
            raise ValueError("IntegrationPlan refinements must have role='refinement'.")
        rules = (self.training, self.validation, *refinements)
        names = [item.name for item in rules]
        if len(set(names)) != len(names):
            raise ValueError("IntegrationPlan rule names must be unique.")
        known_names = set(names)
        unknown_independence = {
            reference
            for rule in rules
            for reference in rule.independent_of
            if reference not in known_names
        }
        if unknown_independence:
            raise ValueError(
                "IntegrationPlan independence references unknown rules "
                f"{sorted(unknown_independence)!r}."
            )
        if self.training.name not in self.validation.independent_of:
            raise ValueError(
                "IntegrationPlan.validation must explicitly declare independence "
                "from the training rule."
            )
        for rule in refinements:
            required = {self.training.name, self.validation.name}
            if not required.issubset(rule.independent_of):
                raise ValueError(
                    "Every refinement rule must explicitly declare independence "
                    "from the training and validation rules."
                )
        object.__setattr__(self, "refinements", refinements)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "integration_plan",
            "schema_version": "0.1.0",
            "training": self.training.summary(),
            "validation": self.validation.summary(),
            "refinements": [item.summary() for item in self.refinements],
            "metadata": dict(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.summary())


@dataclass(frozen=True)
class IntegrationEvidence:
    """Independent objective re-integration and refinement evidence."""

    plan_fingerprint: str
    training_value: float
    validation_value: float
    refinement_values: tuple[float, ...]
    training_validation_gap: float
    refinement_gap: float | None
    balance_error: float | None
    relative_tolerance: float
    status: str
    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        numeric = (
            self.training_value,
            self.validation_value,
            self.training_validation_gap,
            self.relative_tolerance,
            *self.refinement_values,
        )
        optional = tuple(
            item for item in (self.refinement_gap, self.balance_error) if item is not None
        )
        if any(not isfinite(float(item)) for item in numeric + optional):
            raise ValueError("IntegrationEvidence values must be finite.")
        if self.relative_tolerance <= 0.0:
            raise ValueError("IntegrationEvidence.relative_tolerance must be positive.")
        if self.status not in {"accepted", "uncertain", "failed", "inconclusive"}:
            raise ValueError("IntegrationEvidence.status is not recognized.")

    def summary(self) -> dict[str, object]:
        return {
            "kind": "integration_evidence",
            "schema_version": "0.1.0",
            "plan_fingerprint": self.plan_fingerprint,
            "training_value": self.training_value,
            "validation_value": self.validation_value,
            "refinement_values": self.refinement_values,
            "training_validation_gap": self.training_validation_gap,
            "refinement_gap": self.refinement_gap,
            "balance_error": self.balance_error,
            "relative_tolerance": self.relative_tolerance,
            "status": self.status,
            "findings": self.findings,
        }


def integration_consistency_check(
    plan: IntegrationPlan,
    *,
    training_value: float,
    validation_value: float,
    refinement_values=(),
    balance_error: float | None = None,
    relative_tolerance: float = 0.05,
) -> IntegrationEvidence:
    """Compare optimized and held-out integration without trusting loss alone."""

    if not isinstance(plan, IntegrationPlan):
        raise TypeError("plan must be an IntegrationPlan.")
    training = float(training_value)
    validation = float(validation_value)
    refinements = tuple(float(item) for item in refinement_values)
    balance = None if balance_error is None else abs(float(balance_error))
    tolerance = float(relative_tolerance)
    values = (training, validation, *refinements)
    if any(not isfinite(item) for item in values):
        raise ValueError("Integrated objective values must be finite.")
    scale = max(abs(training), abs(validation), 1.0e-30)
    train_validation = abs(training - validation) / scale
    if refinements:
        sequence = (validation, *refinements)
        refinement = max(
            abs(right - left) / max(abs(left), abs(right), 1.0e-30)
            for left, right in zip(sequence, sequence[1:])
        )
    else:
        refinement = None
    findings = []
    if train_validation > tolerance:
        findings.append("training_validation_mismatch")
    if training < validation and train_validation > tolerance:
        findings.append("possible_training_quadrature_exploitation")
    if refinement is None:
        findings.append("refinement_not_evaluated")
    elif refinement > tolerance:
        findings.append("integration_refinement_not_converged")
    if balance is not None and balance > tolerance:
        findings.append("declared_physics_balance_not_closed")
    status = "accepted" if not findings else "uncertain"
    return IntegrationEvidence(
        plan_fingerprint=plan.fingerprint,
        training_value=training,
        validation_value=validation,
        refinement_values=refinements,
        training_validation_gap=train_validation,
        refinement_gap=refinement,
        balance_error=balance,
        relative_tolerance=tolerance,
        status=status,
        findings=tuple(findings),
    )


@dataclass(frozen=True)
class NeuralRepresentation:
    """How one neural function represents one or more unknown fields.

    Architecture internals remain provider owned.  The contract records which
    fields share a representation and which physical features or enrichments
    alter its approximation space, such as a signed-distance crack function or
    Williams crack-tip terms.
    """

    name: str
    fields: tuple[str, ...]
    architecture: str = "user_module"
    features: tuple[str, ...] = ("coordinates",)
    enrichments: tuple[str, ...] = ()
    implementation: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _name(self.name, "NeuralRepresentation.name")
        fields = tuple(
            _name(item, "NeuralRepresentation.fields") for item in self.fields
        )
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(
                "NeuralRepresentation.fields must be non-empty and unique."
            )
        architecture = _extensible_name(
            self.architecture,
            _STANDARD_REPRESENTATIONS,
            "NeuralRepresentation.architecture",
        )
        features = tuple(
            _extensible_feature(item, "NeuralRepresentation.features")
            for item in self.features
        )
        enrichments = tuple(
            _extensible_feature(item, "NeuralRepresentation.enrichments")
            for item in self.enrichments
        )
        if not features or len(set(features)) != len(features):
            raise ValueError(
                "NeuralRepresentation.features must be non-empty and unique."
            )
        if len(set(enrichments)) != len(enrichments):
            raise ValueError("NeuralRepresentation.enrichments must be unique.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "enrichments", enrichments)
        object.__setattr__(self, "implementation", _optional_name(self.implementation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fields": self.fields,
            "architecture": self.architecture,
            "features": self.features,
            "enrichments": self.enrichments,
            "implementation": self.implementation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrainableParameter:
    """A physical parameter inferred jointly with one or more fields."""

    name: str
    role: str
    initial: float
    bounds: tuple[float, float] | None = None
    unit: str | None = None
    transform: str = "identity"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _name(self.name, "TrainableParameter.name")
        role = _choice(self.role, _PARAMETER_ROLES, "TrainableParameter.role")
        transform = _choice(self.transform, _TRANSFORMS, "TrainableParameter.transform")
        initial = float(self.initial)
        if not isfinite(initial):
            raise ValueError("TrainableParameter.initial must be finite.")
        bounds = None
        if self.bounds is not None:
            lower, upper = (float(value) for value in self.bounds)
            if not isfinite(lower) or not isfinite(upper) or upper <= lower:
                raise ValueError("TrainableParameter.bounds must be finite and increasing.")
            if not lower <= initial <= upper:
                raise ValueError("TrainableParameter.initial must lie inside bounds.")
            bounds = (lower, upper)
        if transform == "log" and initial <= 0.0:
            raise ValueError("A log-transformed parameter must have a positive initial value.")
        if transform == "log" and bounds is not None and bounds[0] <= 0.0:
            raise ValueError("A log-transformed parameter must have positive bounds.")
        if transform == "logit":
            if bounds is None:
                raise ValueError("A logit-transformed parameter requires finite bounds.")
            if not bounds[0] < initial < bounds[1]:
                raise ValueError(
                    "A logit-transformed parameter initial value must lie strictly "
                    "inside its bounds."
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "initial": self.initial,
            "bounds": self.bounds,
            "unit": self.unit,
            "transform": self.transform,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralFieldSpec:
    """Provider-neutral contract for PINN, DEM, XDEM, and related solvers."""

    fields: tuple[FieldEncoding, ...]
    objectives: tuple[ObjectiveTerm, ...]
    conditions: tuple[ConditionSpec, ...]
    representations: tuple[NeuralRepresentation, ...]
    sampling: tuple[SamplingPlan, ...] = ()
    parameters: tuple[TrainableParameter, ...] = ()
    integration: IntegrationPlan | None = None
    purpose: str = "forward"
    required_checks: tuple[str, ...] = (
        "independent_reference_error",
        "condition_error",
        "physics_balance_or_energy_error",
        "sampling_convergence",
        "optimization_repeatability",
    )
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = field(default="0.1.0", init=False)
    status: str = field(default="declarative", init=False)

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        objectives = tuple(self.objectives)
        conditions = tuple(self.conditions)
        representations = tuple(self.representations)
        sampling = tuple(self.sampling)
        parameters = tuple(self.parameters)
        integration = self.integration
        if not fields or not objectives or not conditions or not representations:
            raise ValueError(
                "NeuralFieldSpec requires fields, objectives, explicit conditions, "
                "and neural representations."
            )
        if any(not isinstance(item, FieldEncoding) for item in fields):
            raise TypeError("NeuralFieldSpec.fields must contain FieldEncoding records.")
        if any(not isinstance(item, ObjectiveTerm) for item in objectives):
            raise TypeError("NeuralFieldSpec.objectives must contain ObjectiveTerm records.")
        if any(not isinstance(item, ConditionSpec) for item in conditions):
            raise TypeError("NeuralFieldSpec.conditions must contain ConditionSpec records.")
        if any(not isinstance(item, NeuralRepresentation) for item in representations):
            raise TypeError(
                "NeuralFieldSpec.representations must contain NeuralRepresentation records."
            )
        if any(not isinstance(item, SamplingPlan) for item in sampling):
            raise TypeError("NeuralFieldSpec.sampling must contain SamplingPlan records.")
        if any(not isinstance(item, TrainableParameter) for item in parameters):
            raise TypeError(
                "NeuralFieldSpec.parameters must contain TrainableParameter records."
            )
        if integration is not None and not isinstance(integration, IntegrationPlan):
            raise TypeError("NeuralFieldSpec.integration must be an IntegrationPlan.")
        _unique_names(fields, "fields")
        _unique_names(objectives, "objectives")
        _unique_names(conditions, "conditions")
        _unique_names(representations, "representations")
        _unique_names(sampling, "sampling plans")
        _unique_names(parameters, "trainable parameters")
        field_names = {item.name for item in fields}
        represented_fields = [
            name for representation in representations for name in representation.fields
        ]
        unknown_represented = set(represented_fields).difference(field_names)
        missing_representations = field_names.difference(represented_fields)
        repeated_representations = {
            name for name in represented_fields if represented_fields.count(name) > 1
        }
        if unknown_represented or missing_representations or repeated_representations:
            raise ValueError(
                "NeuralFieldSpec representations must cover every field exactly once; "
                f"unknown={sorted(unknown_represented)!r}, "
                f"missing={sorted(missing_representations)!r}, "
                f"repeated={sorted(repeated_representations)!r}."
            )
        for objective in objectives:
            unknown = set(objective.dependent_fields).difference(field_names)
            if unknown:
                raise ValueError(
                    f"Objective {objective.name!r} references unknown fields {sorted(unknown)!r}."
                )
        parameter_names = {item.name for item in parameters}
        for condition in conditions:
            if condition.target not in field_names | parameter_names:
                raise ValueError(
                    f"Condition {condition.name!r} targets unknown field or parameter "
                    f"{condition.target!r}."
                )
        purpose = _choice(self.purpose, _PURPOSES, "NeuralFieldSpec.purpose")
        if purpose == "forward" and parameters:
            raise ValueError(
                "A NeuralFieldSpec with trainable physical parameters must use "
                "purpose='inverse', 'data_assimilation', or 'hybrid'."
            )
        checks = tuple(_name(item, "NeuralFieldSpec.required_checks") for item in self.required_checks)
        if len(set(checks)) != len(checks):
            raise ValueError("NeuralFieldSpec.required_checks must be unique.")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "integration", integration)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "required_checks", checks)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_pinn(cls, spec: PINNSpec) -> "NeuralFieldSpec":
        """Lift the 0.2.x residual PINN contract into the general contract."""

        objectives = tuple(
            ObjectiveTerm(
                name=item.name,
                kind="residual",
                expression=item.equation,
                dependent_fields=item.dependent_fields,
                form=item.form,
                measure="discrete" if item.form == "discrete" else "domain",
                unit=item.unit,
                weight=item.weight,
                implementation=item.implementation,
            )
            for item in spec.residuals
        )
        conditions = tuple(
            ConditionSpec(
                name=item.name,
                kind=item.kind,
                target=item.target,
                on=item.location,
                value=item.value,
                enforcement="data" if item.kind == "observation" else "penalty",
                weight=item.weight,
            )
            for item in spec.conditions
        )
        purpose = {
            "inverse_or_data_physics_fusion": "hybrid",
            "forward": "forward",
            "inverse": "inverse",
            "data_assimilation": "data_assimilation",
        }.get(str(spec.purpose), "hybrid")
        return cls(
            fields=tuple(spec.fields),
            objectives=objectives,
            conditions=conditions,
            representations=(
                NeuralRepresentation(
                    name="pinn_field_network",
                    fields=tuple(item.name for item in spec.fields),
                    architecture="user_module",
                ),
            ),
            purpose=purpose,
            required_checks=tuple(spec.required_checks),
            metadata={
                "source_contract": "PINNSpec",
                "legacy_collocation_policy": dict(spec.collocation_policy),
                "autodiff_backend": spec.autodiff_backend,
            },
        )

    @property
    def objective_kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.kind for item in self.objectives))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "neural_field_spec",
            "schema_version": self.schema_version,
            "status": self.status,
            "purpose": self.purpose,
            "objective_kinds": self.objective_kinds,
            "fields": [item.summary() for item in self.fields],
            "objectives": [item.summary() for item in self.objectives],
            "conditions": [item.summary() for item in self.conditions],
            "representations": [item.summary() for item in self.representations],
            "sampling": [item.summary() for item in self.sampling],
            "parameters": [item.summary() for item in self.parameters],
            "integration": (
                None if self.integration is None else self.integration.summary()
            ),
            "required_checks": self.required_checks,
            "metadata": dict(self.metadata),
        }


def _normalize(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _name(value: object, label: str) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError(f"{label} must not be empty.")
    return selected


def _optional_name(value: object | None) -> str | None:
    return None if value is None else _name(value, "value")


def _choice(value: object, allowed: set[str], label: str) -> str:
    selected = _normalize(value)
    if selected not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)!r}.")
    return selected


def _extensible_name(value: object, standard: set[str], label: str) -> str:
    selected = _normalize(value)
    if selected in standard:
        return selected
    if ":" not in selected or any(not part for part in selected.split(":")):
        raise ValueError(
            f"{label} must be a standard value or a namespaced extension such as "
            "'provider:strategy'."
        )
    return selected


def _extensible_feature(value: object, label: str) -> str:
    selected = _normalize(value)
    if selected and (":" in selected or selected.replace("_", "").isalnum()):
        return selected
    raise ValueError(
        f"{label} must be a conventional name or a namespaced extension."
    )


def _unique_names(items: tuple[object, ...], label: str) -> None:
    names = [getattr(item, "name", None) for item in items]
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(f"NeuralFieldSpec {label} must expose non-empty names.")
    if len(set(names)) != len(names):
        raise ValueError(f"NeuralFieldSpec {label} must have unique names.")


def _data_value(value: object, label: str):
    """Return a JSON-shaped condition value without accepting live callables."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_data_value(item, label) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError(f"{label} mapping keys must be non-empty strings.")
        return {key: _data_value(item, label) for key, item in value.items()}
    raise TypeError(
        f"{label} must be JSON-shaped data or a stable reference string, not "
        f"{type(value).__name__}."
    )


def _fingerprint(record: object) -> str:
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ConditionSpec",
    "IntegrationEvidence",
    "IntegrationPlan",
    "IntegrationRule",
    "NeuralRepresentation",
    "NeuralFieldSpec",
    "ObjectiveTerm",
    "SamplingPlan",
    "TrainableParameter",
    "integration_consistency_check",
]
