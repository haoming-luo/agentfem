"""Applicability-domain guards and deterministic high-fidelity fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ..datasets import ScientificDataset
from .base import Prediction


class OutOfDomainError(ValueError):
    """Raised when an unguarded surrogate is asked to extrapolate."""


@dataclass(frozen=True)
class BoxApplicabilityDomain:
    """Axis-aligned envelope in normalized scientific parameter space."""

    parameter_space: object
    lower: np.ndarray
    upper: np.ndarray
    padding: float = 0.0

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float).reshape(-1)
        upper = np.asarray(self.upper, dtype=float).reshape(-1)
        expected = len(self.parameter_space.feature_names)
        if lower.shape != (expected,) or upper.shape != (expected,):
            raise ValueError(
                f"Applicability bounds require shape ({expected},)."
            )
        if np.any(lower > upper):
            raise ValueError("Applicability lower bounds must not exceed upper bounds.")
        if self.padding < 0.0:
            raise ValueError("Applicability padding must be non-negative.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_dataset(
        cls,
        dataset: ScientificDataset,
        *,
        padding: float = 0.0,
    ) -> "BoxApplicabilityDomain":
        X = dataset.x_matrix(normalized=True)
        return cls(
            parameter_space=dataset.parameter_space,
            lower=np.min(X, axis=0),
            upper=np.max(X, axis=0),
            padding=float(padding),
        )

    def contains(self, values: Mapping[str, object]) -> bool:
        point = self.parameter_space.encode(values)
        if not self._categorical_compatible(point):
            return False
        return bool(
            np.all(point >= self.lower - self.padding)
            and np.all(point <= self.upper + self.padding)
        )

    def margin(self, values: Mapping[str, object]) -> float:
        """Return minimum normalized distance to a box face.

        Positive values are inside; negative values quantify extrapolation
        beyond the closest violated bound.
        """

        point = self.parameter_space.encode(values)
        if not self._categorical_compatible(point):
            return -1.0
        lower_distance = point - (self.lower - self.padding)
        upper_distance = (self.upper + self.padding) - point
        if np.any(lower_distance < 0.0) or np.any(upper_distance < 0.0):
            return float(np.min(np.concatenate((lower_distance, upper_distance))))
        variable = (self.upper - self.lower) > np.finfo(float).eps
        if not np.any(variable):
            return 0.0
        return float(
            np.min(
                np.concatenate(
                    (lower_distance[variable], upper_distance[variable])
                )
            )
        )

    def _categorical_compatible(self, point: np.ndarray) -> bool:
        """Require every selected category to have appeared in training data."""

        from ..campaigns.parameters import ChoiceParameter

        offset = 0
        for parameter in self.parameter_space.parameters:
            if isinstance(parameter, ChoiceParameter):
                width = len(parameter.choices)
                block = point[offset : offset + width]
                selected = int(np.argmax(block))
                if self.upper[offset + selected] < 1.0 - 1.0e-12:
                    return False
                offset += width
            else:
                offset += 1
        return True

    def summary(self) -> dict[str, object]:
        return {
            "kind": "box_applicability_domain",
            "coordinate_system": "normalized_numeric_plus_one_hot_choices",
            "lower": self.lower,
            "upper": self.upper,
            "padding": self.padding,
            "categorical_policy": "observed_categories_only",
            "parameter_space": self.parameter_space.summary(),
        }


@dataclass
class GuardedSurrogate:
    """Use a surrogate only inside its declared applicability domain."""

    model: object
    domain: BoxApplicabilityDomain
    fallback: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None

    def __post_init__(self) -> None:
        schema = getattr(self.model, "schema", None)
        if schema is None:
            raise TypeError("GuardedSurrogate.model must expose its trained schema.")
        if (
            schema.parameter_space.summary()
            != self.domain.parameter_space.summary()
        ):
            raise ValueError(
                "Surrogate and applicability domain use different parameter spaces."
            )
        if not callable(getattr(self.model, "predict_with_uncertainty", None)):
            raise TypeError(
                "GuardedSurrogate.model must implement predict_with_uncertainty(...)."
            )
        if self.fallback is not None and not callable(self.fallback):
            raise TypeError("GuardedSurrogate.fallback must be callable or None.")

    def predict(self, values: Mapping[str, object]) -> Prediction:
        if self.domain.contains(values):
            predicted = self.model.predict_with_uncertainty(values)
            if isinstance(predicted, Prediction):
                return Prediction(
                    outputs=predicted.outputs,
                    uncertainty=predicted.uncertainty,
                    source=predicted.source,
                    in_domain=True,
                    diagnostics={
                        **predicted.diagnostics,
                        "applicability_margin": self.domain.margin(values),
                    },
                )
            return Prediction(
                outputs=predicted,
                source=getattr(self.model, "kind", "surrogate"),
                in_domain=True,
                diagnostics={"applicability_margin": self.domain.margin(values)},
            )
        if self.fallback is None:
            raise OutOfDomainError(
                "Prediction lies outside the surrogate applicability domain and "
                "no high-fidelity fallback was provided."
            )
        outputs = _validate_fallback_outputs(self.model, self.fallback(values))
        return Prediction(
            outputs=outputs,
            source="high_fidelity_fallback",
            in_domain=False,
            diagnostics={
                "applicability_margin": self.domain.margin(values),
                "fallback_trigger": "outside_applicability_domain",
            },
        )


def _validate_fallback_outputs(model, values) -> dict[str, object]:
    outputs = dict(values)
    quantities = tuple(model.schema.quantities)
    expected = {quantity.name for quantity in quantities}
    actual = set(outputs)
    if actual != expected:
        raise ValueError(
            "High-fidelity fallback outputs differ from the surrogate schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )
    return {
        quantity.name: (
            quantity.validate(outputs[quantity.name]).item()
            if quantity.shape == ()
            else quantity.validate(outputs[quantity.name])
        )
        for quantity in quantities
    }


__all__ = [
    "BoxApplicabilityDomain",
    "GuardedSurrogate",
    "OutOfDomainError",
]
