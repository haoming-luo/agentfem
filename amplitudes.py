"""Reusable time-history and scale-factor assets.

Amplitudes describe how a prescribed value changes with time. They are model
assets that can drive loads, constraints, sources, or initial-condition updates.
They are not finite-element fields and do not own spatial degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp, pi, sin

import numpy as np


@dataclass(frozen=True)
class Amplitude:
    """Named scalar history function."""

    name: str
    value: Callable[[float], float]
    kind: str = "custom"
    metadata: dict[str, object] = field(default_factory=dict)

    def __call__(self, time: float) -> float:
        """Evaluate the amplitude at ``time``."""

        return float(self.value(float(time)))

    def renamed(self, name: str) -> "Amplitude":
        """Return a copy with a model-level name."""

        return Amplitude(name=name, value=self.value, kind=self.kind, metadata=self.metadata)

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            "name": self.name,
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }


def as_amplitude(value, *, name: str = "amplitude") -> Amplitude:
    """Convert a scalar, callable, or ``Amplitude`` into an ``Amplitude``."""

    if isinstance(value, Amplitude):
        return value
    if callable(value):
        return Amplitude(name=name, value=value, kind="callable")
    return constant(value, name=name)


def constant(value: float, *, name: str = "constant") -> Amplitude:
    """Create a constant amplitude."""

    scalar = float(value)
    return Amplitude(
        name=name,
        value=lambda _time: scalar,
        kind="constant",
        metadata={"value": scalar},
    )


def ramp(
    start_value: float = 0.0,
    end_value: float = 1.0,
    *,
    start_time: float = 0.0,
    end_time: float = 1.0,
    name: str = "ramp",
) -> Amplitude:
    """Create a clipped linear ramp amplitude."""

    if end_time <= start_time:
        raise ValueError("ramp requires end_time > start_time.")
    start_value = float(start_value)
    end_value = float(end_value)
    start_time = float(start_time)
    end_time = float(end_time)

    def evaluate(time: float) -> float:
        if time <= start_time:
            return start_value
        if time >= end_time:
            return end_value
        alpha = (time - start_time) / (end_time - start_time)
        return (1.0 - alpha) * start_value + alpha * end_value

    return Amplitude(
        name=name,
        value=evaluate,
        kind="ramp",
        metadata={
            "start_value": start_value,
            "end_value": end_value,
            "start_time": start_time,
            "end_time": end_time,
        },
    )


def tabular(
    times,
    values,
    *,
    name: str = "tabular",
    left: float | None = None,
    right: float | None = None,
) -> Amplitude:
    """Create a linearly interpolated tabular amplitude."""

    time_data = np.asarray(times, dtype=float)
    value_data = np.asarray(values, dtype=float)
    if time_data.ndim != 1 or value_data.ndim != 1:
        raise ValueError("tabular requires one-dimensional times and values.")
    if len(time_data) != len(value_data):
        raise ValueError("tabular requires the same number of times and values.")
    if len(time_data) == 0:
        raise ValueError("tabular requires at least one point.")
    if np.any(np.diff(time_data) <= 0.0):
        raise ValueError("tabular times must be strictly increasing.")

    left_value = value_data[0] if left is None else float(left)
    right_value = value_data[-1] if right is None else float(right)

    def evaluate(time: float) -> float:
        return float(np.interp(time, time_data, value_data, left=left_value, right=right_value))

    return Amplitude(
        name=name,
        value=evaluate,
        kind="tabular",
        metadata={
            "points": int(len(time_data)),
            "start_time": float(time_data[0]),
            "end_time": float(time_data[-1]),
            "times": time_data.tolist(),
            "values": value_data.tolist(),
            "left": float(left_value),
            "right": float(right_value),
        },
    )


def sine(
    amplitude: float = 1.0,
    frequency: float = 1.0,
    *,
    phase: float = 0.0,
    offset: float = 0.0,
    name: str = "sine",
) -> Amplitude:
    """Create a sinusoidal amplitude."""

    amplitude = float(amplitude)
    frequency = float(frequency)
    phase = float(phase)
    offset = float(offset)

    def evaluate(time: float) -> float:
        return offset + amplitude * sin(2.0 * pi * frequency * time + phase)

    return Amplitude(
        name=name,
        value=evaluate,
        kind="sine",
        metadata={
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
            "offset": offset,
        },
    )


def gaussian_modulated_sine(
    amplitude: float,
    frequency: float,
    width: float,
    *,
    center: float | None = None,
    phase: float = 0.0,
    name: str = "gaussian_modulated_sine",
) -> Amplitude:
    """Create a Gaussian-windowed sinusoidal pulse."""

    amplitude = float(amplitude)
    frequency = float(frequency)
    width = float(width)
    if width <= 0.0:
        raise ValueError("gaussian_modulated_sine requires width > 0.")
    center_time = 3.0 * width if center is None else float(center)
    phase = float(phase)

    def evaluate(time: float) -> float:
        omega = 2.0 * pi * frequency
        envelope = exp(-((time - center_time) ** 2) / (2.0 * width**2))
        return amplitude * sin(omega * time + phase) * envelope

    return Amplitude(
        name=name,
        value=evaluate,
        kind="gaussian_modulated_sine",
        metadata={
            "amplitude": amplitude,
            "frequency": frequency,
            "width": width,
            "center": center_time,
            "phase": phase,
        },
    )
