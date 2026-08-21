"""Reusable thermal exchange boundary models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from agentfem import amplitudes
from agentfem.ir.values import describe_value
from agentfem.kernel import constants


@dataclass(frozen=True)
class ConvectionBoundary:
    """Linear convection ``-k grad(T).n = h (T - T_inf)``."""

    coefficient: object
    ambient_temperature: object
    location: object
    name: str = "convection"
    ambient_amplitude: amplitudes.Amplitude | None = None

    def __post_init__(self) -> None:
        if isinstance(self.coefficient, Real) and self.coefficient < 0.0:
            raise ValueError("Convection coefficient must be non-negative.")
        if self.location is None or not hasattr(self.location, "measure"):
            raise ValueError("Convection requires a named boundary region.")

    @property
    def measure(self):
        return self.location.measure

    def operator(self, temperature):
        from agentfem import operators

        return operators.robin_operator(
            temperature,
            self.coefficient,
            location=self.location,
        ).renamed("K_convection")

    def source(self, temperature):
        from agentfem import operators

        return operators.robin_source_vector(
            temperature,
            self.coefficient,
            self.ambient_temperature,
            location=self.location,
        ).renamed("Q_convection")

    def residual(self, temperature, test):
        """Return the convection contribution to a thermal residual."""

        return (
            self.coefficient
            * (temperature - self.ambient_temperature)
            * test
            * self.measure
        )

    def outward_heat_rate_form(self, temperature):
        """Return gross outward exchange ``h*T`` for the shared heat ledger.

        The matching ambient contribution ``h*T_inf`` is recorded as applied
        heat. Their difference is the physical net convection rate.
        """

        return self.coefficient * temperature * self.measure

    def update(self, time: float) -> float | None:
        """Update a time-dependent ambient temperature when configured."""

        if self.ambient_amplitude is None:
            return None
        value = self.ambient_amplitude(time)
        self.ambient_temperature.value = constants.scalar_value(value)
        return value

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "thermal_convection_boundary",
            "location": getattr(self.location, "name", None),
            "coefficient": describe_value(self.coefficient),
            "ambient_temperature": describe_value(self.ambient_temperature),
            "ambient_amplitude": (
                None
                if self.ambient_amplitude is None
                else self.ambient_amplitude.summary()
            ),
            "sign_convention": "positive heat transfer leaves the body when T > T_inf",
        }


def convection(
    *,
    on=None,
    location=None,
    coefficient,
    ambient_temperature,
    name: str = "convection",
) -> ConvectionBoundary:
    """Create a linear thermal convection boundary condition."""

    if on is not None and location is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    selected = location if location is not None else on
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("convection requires on= or location= with a domain.")
    ambient_amplitude = (
        amplitudes.as_amplitude(
            ambient_temperature,
            name=f"{name}_ambient_temperature",
        )
        if callable(ambient_temperature)
        else None
    )
    initial_ambient = (
        ambient_temperature
        if ambient_amplitude is None
        else ambient_amplitude(0.0)
    )
    return ConvectionBoundary(
        coefficient=constants.constant(selected.domain, coefficient),
        ambient_temperature=constants.constant(selected.domain, initial_ambient),
        location=selected,
        name=name,
        ambient_amplitude=ambient_amplitude,
    )


__all__ = ["ConvectionBoundary", "convection"]
