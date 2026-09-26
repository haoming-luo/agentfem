# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-independent event observations for transient scientific results."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np


EventDirection = Literal["rising", "falling", "either"]
EventStatus = Literal["observed", "left_censored", "right_censored"]


@dataclass(frozen=True)
class SolveEvent:
    """One backend-neutral observation from an analysis procedure.

    This is the shared evidence stream for progress views, status files,
    result manifests, and agent monitoring.  It lives outside the PETSc solver
    implementation so procedures and result readers can consume events without
    importing a numerical backend.
    """

    kind: str
    step_name: str
    step_number: int = 1
    increment: int = 0
    attempt: int = 0
    start_factor: float = 0.0
    target_factor: float = 0.0
    iteration: int = 0
    residual_norm: float | None = None
    step_length: float | None = None
    next_increment: float | None = None
    incrementation: str = ""
    message: str = ""
    time: float | None = None
    total_increments: int = 0
    coordinate_name: str | None = None
    coordinate_value: float | None = None
    coordinate_unit: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    display: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe, stable execution-event record."""

        def finite_or_none(value):
            if value is None:
                return None
            selected = float(value)
            return selected if isfinite(selected) else None

        return {
            "kind": self.kind,
            "step_name": self.step_name,
            "step_number": int(self.step_number),
            "increment": int(self.increment),
            "attempt": int(self.attempt),
            "start_factor": float(self.start_factor),
            "target_factor": float(self.target_factor),
            "iteration": int(self.iteration),
            "residual_norm": finite_or_none(self.residual_norm),
            "step_length": finite_or_none(self.step_length),
            "next_increment": finite_or_none(self.next_increment),
            "incrementation": self.incrementation,
            "message": self.message,
            "time": finite_or_none(self.time),
            "total_increments": int(self.total_increments),
            "coordinate_name": self.coordinate_name,
            "coordinate_value": finite_or_none(self.coordinate_value),
            "coordinate_unit": self.coordinate_unit,
            "metrics": {
                str(name): finite_or_none(value) for name, value in self.metrics.items()
            },
            "display": bool(self.display),
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "SolveEvent":
        """Restore a recorded event from a result or checkpoint manifest."""

        return cls(
            kind=str(record["kind"]),
            step_name=str(record["step_name"]),
            step_number=int(record.get("step_number", 1)),
            increment=int(record.get("increment", 0)),
            attempt=int(record.get("attempt", 0)),
            start_factor=float(record.get("start_factor", 0.0)),
            target_factor=float(record.get("target_factor", 0.0)),
            iteration=int(record.get("iteration", 0)),
            residual_norm=record.get("residual_norm"),
            step_length=record.get("step_length"),
            next_increment=record.get("next_increment"),
            incrementation=str(record.get("incrementation", "")),
            message=str(record.get("message", "")),
            time=record.get("time"),
            total_increments=int(record.get("total_increments", 0)),
            coordinate_name=record.get("coordinate_name"),
            coordinate_value=record.get("coordinate_value"),
            coordinate_unit=record.get("coordinate_unit"),
            metrics={
                str(name): float(value)
                for name, value in dict(record.get("metrics", {})).items()
                if value is not None
            },
            display=bool(record.get("display", True)),
        )


@dataclass(frozen=True)
class FirstPassageEvent:
    """One threshold event with explicit localization and censoring evidence."""

    name: str
    status: EventStatus
    threshold: float
    direction: EventDirection
    coordinate: float | None
    bracket: tuple[float, float] | None
    values: tuple[float, float] | None
    sample_index: int | None
    localization: str
    coordinate_name: str = "time"
    coordinate_unit: str | None = "s"
    value_name: str = "response"
    value_unit: str | None = None

    @property
    def observed(self) -> bool:
        return self.status == "observed"

    @property
    def censored(self) -> bool:
        return not self.observed

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.first-passage-event",
            "schema_version": "0.1.0",
            "name": self.name,
            "status": self.status,
            "observed": self.observed,
            "censored": self.censored,
            "threshold": self.threshold,
            "direction": self.direction,
            "coordinate": self.coordinate,
            "bracket": self.bracket,
            "values": self.values,
            "sample_index": self.sample_index,
            "localization": self.localization,
            "coordinate_name": self.coordinate_name,
            "coordinate_unit": self.coordinate_unit,
            "value_name": self.value_name,
            "value_unit": self.value_unit,
        }


def first_passage(
    abscissa,
    values=None,
    *,
    threshold: float,
    direction: EventDirection = "rising",
    localization: str = "linear",
    component: int | tuple[int, ...] | None = None,
    name: str = "first_passage",
    coordinate_name: str | None = None,
    coordinate_unit: str | None = None,
    value_name: str | None = None,
    value_unit: str | None = None,
) -> FirstPassageEvent:
    """Locate the first threshold crossing in a history or numeric arrays.

    ``linear`` localization records the containing sample bracket and assumes
    a continuous monitored signal inside it. For discontinuous damage or
    active-set changes, callers should retain the bracket or rerun with local
    substepping rather than treating the interpolated coordinate as exact.
    """

    history = abscissa if values is None and hasattr(abscissa, "abscissa") else None
    if history is not None:
        x = np.asarray(history.abscissa, dtype=float)
        raw = np.asarray(history.values, dtype=float)
        selected_coordinate_name = coordinate_name or history.abscissa_name
        selected_coordinate_unit = (
            history.abscissa_unit if coordinate_unit is None else coordinate_unit
        )
        selected_value_name = value_name or history.name
        selected_value_unit = history.unit if value_unit is None else value_unit
    else:
        if values is None:
            raise TypeError("first_passage requires values or a HistoryResult-like object.")
        x = np.asarray(abscissa, dtype=float)
        raw = np.asarray(values, dtype=float)
        selected_coordinate_name = coordinate_name or "time"
        selected_coordinate_unit = "s" if coordinate_unit is None else coordinate_unit
        selected_value_name = value_name or "response"
        selected_value_unit = value_unit

    y = _select_component(raw, component)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError("first_passage requires non-empty, aligned one-dimensional data.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("first_passage data must be finite.")
    if x.size > 1 and np.any(np.diff(x) <= 0.0):
        raise ValueError("first_passage abscissa must be strictly increasing.")
    selected_threshold = float(threshold)
    if not isfinite(selected_threshold):
        raise ValueError("first_passage threshold must be finite.")
    selected_direction = str(direction).strip().lower()
    if selected_direction not in {"rising", "falling", "either"}:
        raise ValueError("first_passage direction must be rising, falling, or either.")
    selected_localization = str(localization).strip().lower().replace("-", "_")
    if selected_localization not in {"linear", "sample"}:
        raise ValueError("first_passage localization must be linear or sample.")

    initial = float(y[0])
    if _past(initial, selected_threshold, selected_direction):
        exact = bool(np.isclose(initial, selected_threshold, rtol=0.0, atol=0.0))
        return FirstPassageEvent(
            name=name,
            status="observed" if exact else "left_censored",
            threshold=selected_threshold,
            direction=selected_direction,
            coordinate=float(x[0]) if exact else None,
            bracket=(float(x[0]), float(x[0])),
            values=(initial, initial),
            sample_index=0,
            localization="sample" if exact else "before_window",
            coordinate_name=selected_coordinate_name,
            coordinate_unit=selected_coordinate_unit,
            value_name=selected_value_name,
            value_unit=selected_value_unit,
        )

    for index in range(1, x.size):
        before = float(y[index - 1])
        after = float(y[index])
        if not _crossed(before, after, selected_threshold, selected_direction):
            continue
        coordinate = float(x[index])
        method = "sample"
        if selected_localization == "linear" and after != before:
            fraction = (selected_threshold - before) / (after - before)
            coordinate = float(x[index - 1] + fraction * (x[index] - x[index - 1]))
            method = "linear_within_bracket"
        return FirstPassageEvent(
            name=name,
            status="observed",
            threshold=selected_threshold,
            direction=selected_direction,
            coordinate=coordinate,
            bracket=(float(x[index - 1]), float(x[index])),
            values=(before, after),
            sample_index=index,
            localization=method,
            coordinate_name=selected_coordinate_name,
            coordinate_unit=selected_coordinate_unit,
            value_name=selected_value_name,
            value_unit=selected_value_unit,
        )

    return FirstPassageEvent(
        name=name,
        status="right_censored",
        threshold=selected_threshold,
        direction=selected_direction,
        coordinate=None,
        bracket=(float(x[-1]), float(x[-1])),
        values=(float(y[-1]), float(y[-1])),
        sample_index=None,
        localization="after_window",
        coordinate_name=selected_coordinate_name,
        coordinate_unit=selected_coordinate_unit,
        value_name=selected_value_name,
        value_unit=selected_value_unit,
    )


def _select_component(values: np.ndarray, component) -> np.ndarray:
    if values.ndim == 1:
        if component is not None:
            raise ValueError("A scalar history does not accept a component selector.")
        return values
    if component is None:
        raise ValueError("Vector/tensor histories require an explicit component selector.")
    selected = (component,) if isinstance(component, int) else tuple(component)
    return np.asarray(values[(slice(None), *selected)], dtype=float)


def _past(value: float, threshold: float, direction: str) -> bool:
    if direction == "rising":
        return value >= threshold
    if direction == "falling":
        return value <= threshold
    return value == threshold


def _crossed(before: float, after: float, threshold: float, direction: str) -> bool:
    if direction == "rising":
        return before < threshold <= after
    if direction == "falling":
        return before > threshold >= after
    return (before - threshold) * (after - threshold) <= 0.0 and before != after


__all__ = [
    "EventDirection",
    "EventStatus",
    "FirstPassageEvent",
    "SolveEvent",
    "first_passage",
]
