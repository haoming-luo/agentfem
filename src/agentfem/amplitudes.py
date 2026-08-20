"""Reusable time-history and scale-factor assets.

Amplitudes describe how a prescribed value changes with time. They are model
assets that can drive loads, constraints, sources, or initial-condition updates.
They are not finite-element fields and do not own spatial degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import cos, exp, pi, sin
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Amplitude:
    """Named scalar history function."""

    name: str
    value: Callable[[float], float]
    kind: str = "custom"
    metadata: dict[str, object] = field(default_factory=dict)
    derivatives: tuple[Callable[[float], float], ...] = ()
    serializable: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Amplitude.name must not be empty.")
        if not callable(self.value):
            raise TypeError("Amplitude.value must be callable.")
        selected_derivatives = tuple(self.derivatives)
        if any(not callable(item) for item in selected_derivatives):
            raise TypeError("Amplitude derivatives must be callable.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", str(self.kind).strip() or "custom")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "derivatives", selected_derivatives)

    def __call__(self, time: float) -> float:
        """Evaluate the amplitude at ``time``."""

        return float(self.value(float(time)))

    def renamed(self, name: str) -> "Amplitude":
        """Return a copy with a model-level name."""

        return Amplitude(
            name=name,
            value=self.value,
            kind=self.kind,
            metadata=self.metadata,
            derivatives=self.derivatives,
            serializable=self.serializable,
        )

    def derivative(self, time: float, *, order: int = 1, step: float | None = None) -> float:
        """Evaluate a declared derivative or a deterministic numerical fallback."""

        selected_order = int(order)
        if selected_order not in {1, 2}:
            raise ValueError("Amplitude.derivative currently supports order 1 or 2.")
        selected_time = float(time)
        if len(self.derivatives) >= selected_order:
            return float(self.derivatives[selected_order - 1](selected_time))
        h = (
            max(1.0, abs(selected_time)) * np.finfo(float).eps ** (1.0 / 3.0)
            if step is None
            else float(step)
        )
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("Amplitude derivative step must be positive and finite.")
        if selected_order == 1:
            return (self(selected_time + h) - self(selected_time - h)) / (2.0 * h)
        return (
            self(selected_time + h) - 2.0 * self(selected_time) + self(selected_time - h)
        ) / h**2

    def velocity(self, time: float) -> float:
        return self.derivative(time, order=1)

    def acceleration(self, time: float) -> float:
        return self.derivative(time, order=2)

    def scaled(self, factor: float, *, name: str | None = None) -> "Amplitude":
        """Return a value-scaled history without changing its time coordinate."""

        selected = float(factor)
        if not np.isfinite(selected):
            raise ValueError("Amplitude scale factor must be finite.")
        return Amplitude(
            name=name or f"{self.name}_scaled",
            value=lambda time: selected * self(time),
            kind="value_scaled",
            metadata={"source": self.summary(), "factor": selected},
            derivatives=tuple(
                lambda time, order=order: selected * self.derivative(time, order=order)
                for order in (1, 2)
            ),
            serializable=self.serializable,
        )

    def time_shifted(self, delay: float, *, name: str | None = None) -> "Amplitude":
        """Return ``a(t-delay)`` with the same derivative convention."""

        selected = float(delay)
        if not np.isfinite(selected):
            raise ValueError("Amplitude time shift must be finite.")
        return Amplitude(
            name=name or f"{self.name}_shifted",
            value=lambda time: self(time - selected),
            kind="time_shifted",
            metadata={"source": self.summary(), "delay": selected},
            derivatives=tuple(
                lambda time, order=order: self.derivative(time - selected, order=order)
                for order in (1, 2)
            ),
            serializable=self.serializable,
        )

    def time_scaled(
        self,
        factor: float,
        *,
        origin: float = 0.0,
        name: str | None = None,
    ) -> "Amplitude":
        """Stretch time by ``factor`` around a declared origin."""

        selected = float(factor)
        selected_origin = float(origin)
        if not np.isfinite(selected) or selected <= 0.0:
            raise ValueError("Amplitude time-scale factor must be positive and finite.")
        if not np.isfinite(selected_origin):
            raise ValueError("Amplitude time-scale origin must be finite.")

        def source_time(time: float) -> float:
            return selected_origin + (time - selected_origin) / selected

        return Amplitude(
            name=name or f"{self.name}_time_scaled",
            value=lambda time: self(source_time(time)),
            kind="time_scaled",
            metadata={
                "source": self.summary(),
                "factor": selected,
                "origin": selected_origin,
            },
            derivatives=tuple(
                lambda time, order=order: self.derivative(
                    source_time(time), order=order
                )
                / selected**order
                for order in (1, 2)
            ),
            serializable=self.serializable,
        )

    def sample(self, times: Sequence[float], *, derivative: int = 0) -> np.ndarray:
        """Sample values or derivatives at caller-declared coordinates."""

        selected = np.asarray(times, dtype=float)
        if selected.ndim != 1 or not np.all(np.isfinite(selected)):
            raise ValueError("Amplitude sample times must be a finite one-dimensional array.")
        if int(derivative) == 0:
            return np.asarray([self(item) for item in selected], dtype=float)
        return np.asarray(
            [self.derivative(item, order=int(derivative)) for item in selected],
            dtype=float,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-safe specification when one exists."""

        if not self.serializable:
            raise ValueError(
                f"Amplitude {self.name!r} contains live callable code and cannot be serialized."
            )
        record = self._specification()
        return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))

    @property
    def fingerprint(self) -> str | None:
        if not self.serializable:
            return None
        encoded = json.dumps(
            self._specification(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{sha256(encoded).hexdigest()}"

    def audit(
        self,
        start_time: float,
        end_time: float,
        *,
        samples: int = 257,
        derivative_orders: Sequence[int] = (0, 1, 2),
    ) -> "AmplitudeAudit":
        """Inspect finiteness, ranges, and endpoint value/derivative behavior."""

        start = float(start_time)
        end = float(end_time)
        count = int(samples)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Amplitude audit requires finite end_time > start_time.")
        if count < 2:
            raise ValueError("Amplitude audit requires at least two samples.")
        orders = tuple(dict.fromkeys(int(item) for item in derivative_orders))
        if any(item not in {0, 1, 2} for item in orders):
            raise ValueError("Amplitude audit derivative orders must be 0, 1, or 2.")
        coordinates = np.linspace(start, end, count)
        records = {}
        for order in orders:
            values = self.sample(coordinates, derivative=order)
            records[str(order)] = {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "start": float(values[0]),
                "end": float(values[-1]),
                "finite": bool(np.all(np.isfinite(values))),
            }
        return AmplitudeAudit(
            amplitude=self.name,
            start_time=start,
            end_time=end,
            sample_count=count,
            derivatives=records,
            fingerprint=self.fingerprint,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            **self._specification(),
            "fingerprint": self.fingerprint,
        }

    def _specification(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "serializable": bool(self.serializable),
        }


@dataclass(frozen=True)
class AmplitudeAudit:
    """Portable endpoint and range evidence for one amplitude."""

    amplitude: str
    start_time: float
    end_time: float
    sample_count: int
    derivatives: Mapping[str, Mapping[str, object]]
    fingerprint: str | None

    @property
    def finite(self) -> bool:
        return all(bool(item["finite"]) for item in self.derivatives.values())

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.amplitude-audit",
            "schema_version": "0.1.0",
            "amplitude": self.amplitude,
            "interval": (self.start_time, self.end_time),
            "sample_count": self.sample_count,
            "finite": self.finite,
            "derivatives": {key: dict(value) for key, value in self.derivatives.items()},
            "fingerprint": self.fingerprint,
        }


def as_amplitude(value, *, name: str = "amplitude") -> Amplitude:
    """Convert a scalar, callable, or ``Amplitude`` into an ``Amplitude``."""

    if isinstance(value, Amplitude):
        return value
    if callable(value):
        return Amplitude(
            name=name,
            value=value,
            kind="callable",
            serializable=False,
        )
    return constant(value, name=name)


def constant(value: float, *, name: str = "constant") -> Amplitude:
    """Create a constant amplitude."""

    scalar = float(value)
    return Amplitude(
        name=name,
        value=lambda _time: scalar,
        kind="constant",
        metadata={"value": scalar},
        derivatives=(lambda _time: 0.0, lambda _time: 0.0),
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

    slope = (end_value - start_value) / (end_time - start_time)

    def velocity(time: float) -> float:
        return slope if start_time < time < end_time else 0.0

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
        derivatives=(velocity, lambda _time: 0.0),
    )


def smooth_step(
    start_value: float = 0.0,
    end_value: float = 1.0,
    *,
    start_time: float = 0.0,
    end_time: float = 1.0,
    name: str = "smooth_step",
) -> Amplitude:
    """Create a clipped half-cosine transition with zero endpoint slopes.

    This is useful when a prescribed motion must begin without injecting the
    velocity discontinuity of a piecewise-linear ramp.
    """

    if end_time <= start_time:
        raise ValueError("smooth_step requires end_time > start_time.")
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
        blend = 0.5 * (1.0 - cos(pi * alpha))
        return (1.0 - blend) * start_value + blend * end_value

    duration = end_time - start_time
    delta = end_value - start_value

    def velocity(time: float) -> float:
        if not start_time < time < end_time:
            return 0.0
        alpha = (time - start_time) / duration
        return delta * 0.5 * pi * sin(pi * alpha) / duration

    def acceleration(time: float) -> float:
        if not start_time < time < end_time:
            return 0.0
        alpha = (time - start_time) / duration
        return delta * 0.5 * pi**2 * cos(pi * alpha) / duration**2

    return Amplitude(
        name=name,
        value=evaluate,
        kind="smooth_step",
        metadata={
            "start_value": start_value,
            "end_value": end_value,
            "start_time": start_time,
            "end_time": end_time,
            "endpoint_slopes": "zero",
        },
        derivatives=(velocity, acceleration),
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

    slopes = np.diff(value_data) / np.diff(time_data) if len(time_data) > 1 else np.zeros(0)

    def velocity(time: float) -> float:
        if len(time_data) < 2 or time <= time_data[0] or time >= time_data[-1]:
            return 0.0
        index = int(np.searchsorted(time_data, time, side="right") - 1)
        return float(slopes[min(index, len(slopes) - 1)])

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
            "differentiability": "piecewise linear; derivatives are undefined at knots",
        },
        derivatives=(velocity,),
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

    omega = 2.0 * pi * frequency

    def velocity(time: float) -> float:
        return amplitude * omega * cos(omega * time + phase)

    def acceleration(time: float) -> float:
        return -amplitude * omega**2 * sin(omega * time + phase)

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
        derivatives=(velocity, acceleration),
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

    def velocity(time: float) -> float:
        omega = 2.0 * pi * frequency
        tau = time - center_time
        envelope = exp(-(tau**2) / (2.0 * width**2))
        return amplitude * envelope * (
            omega * cos(omega * time + phase)
            - tau * sin(omega * time + phase) / width**2
        )

    def acceleration(time: float) -> float:
        omega = 2.0 * pi * frequency
        tau = time - center_time
        angle = omega * time + phase
        envelope = exp(-(tau**2) / (2.0 * width**2))
        sine_factor = -omega**2 - 1.0 / width**2 + tau**2 / width**4
        cosine_factor = -2.0 * omega * tau / width**2
        return amplitude * envelope * (
            sine_factor * sin(angle) + cosine_factor * cos(angle)
        )

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
        derivatives=(velocity, acceleration),
    )


@dataclass(frozen=True)
class AmplitudeBasis:
    """Named, serializable loading modes with a declared coefficient order."""

    components: tuple[Amplitude, ...]
    name: str = "amplitude_basis"
    coefficient_names: tuple[str, ...] | None = None
    coordinate_name: str = "time"
    coordinate_unit: str | None = "s"
    value_unit: str | None = None

    def __post_init__(self) -> None:
        selected = tuple(self.components)
        if not selected:
            raise ValueError("AmplitudeBasis requires at least one component.")
        names = (
            tuple(item.name for item in selected)
            if self.coefficient_names is None
            else tuple(str(item).strip() for item in self.coefficient_names)
        )
        if len(names) != len(selected) or any(not item for item in names):
            raise ValueError("AmplitudeBasis coefficient names must match its components.")
        if len(set(names)) != len(names):
            raise ValueError("AmplitudeBasis coefficient names must be unique.")
        object.__setattr__(self, "components", selected)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "name", str(self.name).strip() or "amplitude_basis")

    @property
    def dimension(self) -> int:
        return len(self.components)

    def coefficients(self, values: Sequence[float] | Mapping[str, float]) -> tuple[float, ...]:
        if isinstance(values, Mapping):
            missing = [name for name in self.coefficient_names if name not in values]
            extra = [name for name in values if name not in self.coefficient_names]
            if missing or extra:
                raise ValueError(
                    f"Amplitude coefficients differ from the basis; missing={missing}, extra={extra}."
                )
            selected = tuple(float(values[name]) for name in self.coefficient_names)
        else:
            selected = tuple(float(item) for item in values)
        if len(selected) != self.dimension or not np.all(np.isfinite(selected)):
            raise ValueError(
                f"AmplitudeBasis requires {self.dimension} finite coefficients."
            )
        return selected

    def combine(
        self,
        coefficients: Sequence[float] | Mapping[str, float],
        *,
        base: float | Amplitude = 0.0,
        name: str = "combined_amplitude",
    ) -> Amplitude:
        selected = self.coefficients(coefficients)
        base_amplitude = as_amplitude(base, name="base")

        def evaluate(time: float) -> float:
            return base_amplitude(time) + sum(
                coefficient * component(time)
                for coefficient, component in zip(selected, self.components, strict=True)
            )

        def derivative(order: int):
            return lambda time: base_amplitude.derivative(time, order=order) + sum(
                coefficient * component.derivative(time, order=order)
                for coefficient, component in zip(selected, self.components, strict=True)
            )

        serializable = base_amplitude.serializable and all(
            item.serializable for item in self.components
        )
        metadata = {
            "basis": self.summary(),
            "coefficients": dict(zip(self.coefficient_names, selected, strict=True)),
            "base": base_amplitude.summary(),
        }
        return Amplitude(
            name=name,
            value=evaluate,
            kind="linear_combination",
            metadata=metadata,
            derivatives=(derivative(1), derivative(2)),
            serializable=serializable,
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.amplitude-basis",
            "schema_version": "0.1.0",
            "name": self.name,
            "dimension": self.dimension,
            "coefficient_names": self.coefficient_names,
            "coordinate_name": self.coordinate_name,
            "coordinate_unit": self.coordinate_unit,
            "value_unit": self.value_unit,
            "components": [item.summary() for item in self.components],
        }

    @property
    def fingerprint(self) -> str | None:
        if any(not item.serializable for item in self.components):
            return None
        encoded = json.dumps(
            self.summary(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{sha256(encoded).hexdigest()}"


def basis(
    *components: Amplitude,
    name: str = "amplitude_basis",
    coefficient_names: Sequence[str] | None = None,
    coordinate_name: str = "time",
    coordinate_unit: str | None = "s",
    value_unit: str | None = None,
) -> AmplitudeBasis:
    """Create a named basis for control, inverse, and transient studies."""

    return AmplitudeBasis(
        components=tuple(as_amplitude(item) for item in components),
        name=name,
        coefficient_names=(
            None if coefficient_names is None else tuple(coefficient_names)
        ),
        coordinate_name=coordinate_name,
        coordinate_unit=coordinate_unit,
        value_unit=value_unit,
    )


__all__ = [
    "Amplitude",
    "AmplitudeAudit",
    "AmplitudeBasis",
    "as_amplitude",
    "basis",
    "constant",
    "gaussian_modulated_sine",
    "ramp",
    "sine",
    "smooth_step",
    "tabular",
]
