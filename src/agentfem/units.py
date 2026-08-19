"""Explicit consistent-unit metadata for engineering models.

Finite-element kernels operate on numbers.  This module records the unit
contract those numbers obey; it does not silently convert coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSystem:
    """Named consistent base-unit contract attached to a model."""

    length: str
    mass: str
    time: str
    temperature: str = "K"
    name: str = "consistent_units"

    def __post_init__(self) -> None:
        for field_name in ("length", "mass", "time", "temperature", "name"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"UnitSystem.{field_name} must not be empty.")
            object.__setattr__(self, field_name, value)

    @property
    def force(self) -> str:
        return f"{self.mass}*{self.length}/{self.time}^2"

    @property
    def stress(self) -> str:
        return f"{self.mass}/({self.length}*{self.time}^2)"

    @property
    def energy(self) -> str:
        return f"{self.mass}*{self.length}^2/{self.time}^2"

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "consistent_unit_system",
            "base": {
                "length": self.length,
                "mass": self.mass,
                "time": self.time,
                "temperature": self.temperature,
            },
            "derived": {
                "force": self.force,
                "stress": self.stress,
                "energy": self.energy,
            },
            "automatic_conversion": False,
        }


def consistent(*, length, mass, time, temperature="K", name="consistent_units"):
    """Declare the base units used consistently by all model inputs."""

    return UnitSystem(length, mass, time, temperature, name)


def si(*, temperature="K") -> UnitSystem:
    """Return the SI ``m-kg-s`` engineering contract."""

    return UnitSystem("m", "kg", "s", temperature, "SI")


def n_mm_mpa(*, temperature="K") -> UnitSystem:
    """Return the common ``mm-N-s-MPa`` consistent system.

    The corresponding mass unit is tonne because ``N = tonne*mm/s^2``.
    """

    return UnitSystem("mm", "tonne", "s", temperature, "N-mm-MPa")


__all__ = ["UnitSystem", "consistent", "n_mm_mpa", "si"]
