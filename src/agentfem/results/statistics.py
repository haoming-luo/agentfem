"""Source-aware weighted statistics for cell and integration-point fields.

Finite-element field summaries are meaningful only together with their
sampling location and weights.  This module therefore accepts resolved values
and physical integration weights instead of guessing weights from coefficient
arrays.  Constitutive drivers, cell-field exporters, and external providers
can all use the same solver-neutral contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WeightedFieldStatistics:
    """Global weighted distribution with explicit field semantics."""

    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    quantiles: Mapping[float, float]
    threshold_fractions: Mapping[float, float]
    total_weight: float
    sample_count: int
    location: str
    representation: str
    operation: str = "physical_weighted_distribution"

    def summary(self) -> dict[str, object]:
        return {
            "kind": "weighted_field_statistics",
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "quantiles": {
                _number_key(key): value for key, value in self.quantiles.items()
            },
            "threshold_fractions": {
                _number_key(key): value
                for key, value in self.threshold_fractions.items()
            },
            "total_weight": self.total_weight,
            "sample_count": self.sample_count,
            "location": self.location,
            "representation": self.representation,
            "operation": self.operation,
            "quantile_definition": "left-continuous weighted empirical CDF",
            "threshold_definition": "physical fraction with value > threshold",
        }


def weighted_field_statistics(
    values,
    weights,
    *,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    thresholds: Sequence[float] = (),
    location: str,
    representation: str,
    comm=None,
) -> WeightedFieldStatistics:
    """Return exact global statistics from physical sample weights.

    ``values`` must be a scalar quantity such as one stress invariant or one
    internal variable.  ``weights`` normally contain quadrature weights times
    the geometric Jacobian or cell volumes.  Distributed inputs are gathered
    only on rank zero for the exact empirical quantiles; the compact summary is
    then broadcast to all ranks.
    """

    local_values = np.asarray(values, dtype=float).reshape(-1)
    local_weights = np.asarray(weights, dtype=float).reshape(-1)
    if local_values.size != local_weights.size:
        raise ValueError("values and weights must contain the same number of samples.")
    if not np.all(np.isfinite(local_values)):
        raise ValueError("weighted field values must be finite.")
    if not np.all(np.isfinite(local_weights)) or np.any(local_weights < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    selected_quantiles = tuple(float(item) for item in quantiles)
    if any(not 0.0 <= item <= 1.0 for item in selected_quantiles):
        raise ValueError("quantiles must lie in the closed interval [0, 1].")
    if len(set(selected_quantiles)) != len(selected_quantiles):
        raise ValueError("quantiles must not contain duplicates.")
    selected_thresholds = tuple(float(item) for item in thresholds)
    if not all(np.isfinite(selected_thresholds)):
        raise ValueError("thresholds must be finite.")
    if not location.strip() or not representation.strip():
        raise ValueError("location and representation must be explicit nonempty names.")

    if comm is None or int(getattr(comm, "size", 1)) == 1:
        global_values = local_values
        global_weights = local_weights
    else:
        gathered_values = comm.gather(local_values, root=0)
        gathered_weights = comm.gather(local_weights, root=0)
        if int(comm.rank) == 0:
            global_values = np.concatenate(gathered_values)
            global_weights = np.concatenate(gathered_weights)
        else:
            global_values = None
            global_weights = None

    distributed = comm is not None and int(getattr(comm, "size", 1)) > 1
    if not distributed:
        payload = _weighted_payload(
            global_values,
            global_weights,
            quantiles=selected_quantiles,
            thresholds=selected_thresholds,
        )
    else:
        envelope = None
        if int(comm.rank) == 0:
            try:
                envelope = {
                    "payload": _weighted_payload(
                        global_values,
                        global_weights,
                        quantiles=selected_quantiles,
                        thresholds=selected_thresholds,
                    ),
                    "error": None,
                }
            except ValueError as exc:
                # Broadcast validation failure as data so non-root ranks do
                # not wait forever when the global distribution is invalid.
                envelope = {"payload": None, "error": str(exc)}
        envelope = comm.bcast(envelope, root=0)
        if envelope["error"] is not None:
            raise ValueError(envelope["error"])
        payload = envelope["payload"]
    return WeightedFieldStatistics(
        **payload,
        location=location,
        representation=representation,
    )


def _weighted_payload(values, weights, *, quantiles, thresholds):
    positive = weights > 0.0
    selected_values = values[positive]
    selected_weights = weights[positive]
    if selected_values.size == 0:
        raise ValueError("weighted statistics require positive total weight.")
    total_weight = float(np.sum(selected_weights))
    order = np.argsort(selected_values, kind="stable")
    ordered_values = selected_values[order]
    ordered_weights = selected_weights[order]
    cumulative = np.cumsum(ordered_weights)
    mean = float(np.dot(selected_weights, selected_values) / total_weight)
    variance = float(
        np.dot(selected_weights, (selected_values - mean) ** 2) / total_weight
    )
    weighted_quantiles = {
        item: float(
            ordered_values[
                min(
                    int(np.searchsorted(cumulative, item * total_weight, side="left")),
                    ordered_values.size - 1,
                )
            ]
        )
        for item in quantiles
    }
    fractions = {
        item: float(np.sum(selected_weights[selected_values > item]) / total_weight)
        for item in thresholds
    }
    return {
        "minimum": float(np.min(selected_values)),
        "maximum": float(np.max(selected_values)),
        "mean": mean,
        "standard_deviation": float(np.sqrt(max(variance, 0.0))),
        "quantiles": weighted_quantiles,
        "threshold_fractions": fractions,
        "total_weight": total_weight,
        "sample_count": int(selected_values.size),
    }


def _number_key(value: float) -> str:
    return format(float(value), ".17g")


__all__ = ["WeightedFieldStatistics", "weighted_field_statistics"]
