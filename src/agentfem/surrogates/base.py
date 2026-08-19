"""Shared prediction and validation evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..datasets import ScientificDataset


@dataclass(frozen=True)
class Prediction:
    """One named prediction with source and trust diagnostics."""

    outputs: Mapping[str, object]
    uncertainty: Mapping[str, object] | None = None
    source: str = "surrogate"
    in_domain: bool = True
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(
            self,
            "uncertainty",
            None if self.uncertainty is None else dict(self.uncertainty),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class QuantityMetrics:
    """Error evidence for one declared output quantity."""

    name: str
    rmse: float
    mae: float
    relative_l2: float | None
    r2: float | None
    sample_count: int

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rmse": self.rmse,
            "mae": self.mae,
            "relative_l2": self.relative_l2,
            "r2": self.r2,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class SurrogateValidationReport:
    """Independent validation metrics and optional acceptance decision."""

    model_kind: str
    dataset_name: str
    quantities: tuple[QuantityMetrics, ...]
    thresholds: Mapping[str, float] = field(default_factory=dict)
    accepted: bool | None = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": "surrogate_validation_report",
            "model_kind": self.model_kind,
            "dataset_name": self.dataset_name,
            "accepted": self.accepted,
            "thresholds": dict(self.thresholds),
            "quantities": [quantity.summary() for quantity in self.quantities],
        }

    def format(self) -> str:
        decision = (
            "not assessed"
            if self.accepted is None
            else ("accepted" if self.accepted else "rejected")
        )
        lines = [
            f"{self.model_kind} on {self.dataset_name}: {decision}",
        ]
        for metric in self.quantities:
            r2 = "undefined" if metric.r2 is None else f"{metric.r2:.6g}"
            relative = (
                "undefined"
                if metric.relative_l2 is None
                else f"{metric.relative_l2:.6g}"
            )
            lines.append(
                f"- {metric.name}: RMSE={metric.rmse:.6g}, "
                f"relative L2={relative}, "
                f"R2={r2}"
            )
        return "\n".join(lines)


def validate_predictions(
    *,
    model_kind: str,
    dataset: ScientificDataset,
    predictions: np.ndarray,
    thresholds: Mapping[str, float] | None = None,
) -> SurrogateValidationReport:
    """Compare flattened predictions with a dataset's declared quantities."""

    expected = dataset.y_matrix()
    selected = np.asarray(predictions, dtype=float)
    if selected.shape != expected.shape:
        raise ValueError(
            f"Prediction shape {selected.shape} differs from expected {expected.shape}."
        )
    if not np.all(np.isfinite(selected)):
        raise ValueError("Surrogate predictions contain non-finite values.")
    metrics = []
    offset = 0
    for quantity in dataset.quantities:
        truth = expected[:, offset : offset + quantity.size]
        estimate = selected[:, offset : offset + quantity.size]
        error = estimate - truth
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        denominator = float(np.linalg.norm(truth))
        error_norm = float(np.linalg.norm(error))
        relative_l2 = (
            0.0 if denominator == 0.0 and error_norm == 0.0
            else None if denominator == 0.0
            else error_norm / denominator
        )
        centered = truth - np.mean(truth, axis=0, keepdims=True)
        total = float(np.sum(centered**2))
        r2 = None if total == 0.0 else 1.0 - float(np.sum(error**2)) / total
        metrics.append(
            QuantityMetrics(
                name=quantity.name,
                rmse=rmse,
                mae=mae,
                relative_l2=relative_l2,
                r2=r2,
                sample_count=len(dataset.samples),
            )
        )
        offset += quantity.size

    try:
        selected_thresholds = {
            name: float(value) for name, value in dict(thresholds or {}).items()
        }
    except (TypeError, ValueError) as exc:
        raise TypeError("Surrogate validation thresholds must be numeric.") from exc
    accepted = None
    if selected_thresholds:
        supported = {"max_rmse", "max_relative_l2", "min_r2"}
        unknown = set(selected_thresholds) - supported
        if unknown:
            raise ValueError(f"Unknown surrogate validation thresholds: {sorted(unknown)}.")
        if not all(np.isfinite(value) for value in selected_thresholds.values()):
            raise ValueError("Surrogate validation thresholds must be finite.")
        if selected_thresholds.get("max_rmse", 0.0) < 0.0:
            raise ValueError("max_rmse must be non-negative.")
        if selected_thresholds.get("max_relative_l2", 0.0) < 0.0:
            raise ValueError("max_relative_l2 must be non-negative.")
        if selected_thresholds.get("min_r2", 1.0) > 1.0:
            raise ValueError("min_r2 must not exceed 1.")
        accepted = True
        for metric in metrics:
            if "max_rmse" in selected_thresholds:
                accepted &= metric.rmse <= selected_thresholds["max_rmse"]
            if "max_relative_l2" in selected_thresholds:
                accepted &= (
                    metric.relative_l2 is not None
                    and metric.relative_l2 <= selected_thresholds["max_relative_l2"]
                )
            if "min_r2" in selected_thresholds:
                accepted &= (
                    metric.r2 is not None and metric.r2 >= selected_thresholds["min_r2"]
                )
        accepted = bool(accepted)
    return SurrogateValidationReport(
        model_kind=model_kind,
        dataset_name=dataset.name,
        quantities=tuple(metrics),
        thresholds=selected_thresholds,
        accepted=accepted,
    )


def decode_rows(dataset: ScientificDataset, rows: Sequence[Sequence[float]]):
    return [dataset.decode_outputs(row) for row in rows]


__all__ = [
    "Prediction",
    "QuantityMetrics",
    "SurrogateValidationReport",
    "decode_rows",
    "validate_predictions",
]
