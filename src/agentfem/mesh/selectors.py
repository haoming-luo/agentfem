"""Composable geometric selectors for mesh regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Selector:
    """Boolean selector evaluated on coordinate arrays.

    A selector is not a geometry object and does not generate a mesh. It is a
    reusable rule for selecting cells, facets, or dofs by coordinates.
    """

    predicate: Callable[[object], object]
    name: str = "selector"

    def __call__(self, x):
        return np.asarray(self.predicate(x), dtype=bool)

    def __and__(self, other) -> "Selector":
        other = where(other)
        return Selector(
            lambda x: self(x) & other(x),
            name=f"({self.name}&{other.name})",
        )

    def __or__(self, other) -> "Selector":
        other = where(other)
        return Selector(
            lambda x: self(x) | other(x),
            name=f"({self.name}|{other.name})",
        )

    def __invert__(self) -> "Selector":
        return Selector(lambda x: ~self(x), name=f"~{self.name}")

    def summary(self) -> dict[str, object]:
        """Return a compact selector summary."""

        return {"name": self.name, "kind": "selector"}


def where(predicate, *, name: str | None = None) -> Selector:
    """Create a selector from a vectorized coordinate predicate."""

    if isinstance(predicate, Selector):
        if name is None:
            return predicate
        return Selector(predicate.predicate, name=name)
    return Selector(predicate, name=name or getattr(predicate, "__name__", "where"))


def plane(axis: str | int, value: float, *, tolerance: float = 1.0e-12) -> Selector:
    """Select points near a coordinate plane such as ``x = 0``."""

    axis_id = _axis_id(axis)
    label = _axis_name(axis_id)
    return Selector(
        lambda x: np.isclose(x[axis_id], value, rtol=0.0, atol=tolerance),
        name=f"{label}={value:g}",
    )


def layer(axis: str | int, lower=None, upper=None) -> Selector:
    """Select points inside a coordinate interval along one axis."""

    axis_id = _axis_id(axis)
    label = _axis_name(axis_id)

    def predicate(x):
        selected = np.ones(x.shape[1], dtype=bool)
        if lower is not None:
            selected &= x[axis_id] >= float(lower)
        if upper is not None:
            selected &= x[axis_id] <= float(upper)
        return selected

    parts = []
    if lower is not None:
        parts.append(f"{lower:g}<=")
    parts.append(label)
    if upper is not None:
        parts.append(f"<={upper:g}")
    return Selector(predicate, name="".join(parts))


def box(lower, upper) -> Selector:
    """Select points inside an axis-aligned box."""

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def predicate(x):
        selected = np.ones(x.shape[1], dtype=bool)
        for axis_id in range(len(lower)):
            selected &= x[axis_id] >= lower[axis_id]
            selected &= x[axis_id] <= upper[axis_id]
        return selected

    return Selector(predicate, name="box")


def disk(center, radius: float) -> Selector:
    """Select points inside a 2D disk."""

    center = np.asarray(center, dtype=float)
    radius = float(radius)

    def predicate(x):
        return (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 <= radius**2

    return Selector(predicate, name="disk")


def ball(center, radius: float) -> Selector:
    """Select points inside a 3D ball."""

    center = np.asarray(center, dtype=float)
    radius = float(radius)

    def predicate(x):
        squared = np.zeros(x.shape[1], dtype=float)
        for axis_id in range(len(center)):
            squared += (x[axis_id] - center[axis_id]) ** 2
        return squared <= radius**2

    return Selector(predicate, name="ball")


def _axis_id(axis: str | int) -> int:
    if isinstance(axis, str):
        names = {"x": 0, "y": 1, "z": 2}
        key = axis.lower()
        if key not in names:
            raise ValueError("axis must be 'x', 'y', 'z', or an integer.")
        return names[key]
    axis_id = int(axis)
    if axis_id < 0:
        raise ValueError("axis id must be non-negative.")
    return axis_id


def _axis_name(axis_id: int) -> str:
    names = ("x", "y", "z")
    if axis_id < len(names):
        return names[axis_id]
    return f"axis{axis_id}"
