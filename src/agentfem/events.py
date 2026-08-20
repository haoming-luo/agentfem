"""Solver-independent event observations for transient scientific results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np


EventDirection = Literal["rising", "falling", "either"]
EventStatus = Literal["observed", "left_censored", "right_censored"]


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


__all__ = ["EventDirection", "EventStatus", "FirstPassageEvent", "first_passage"]
