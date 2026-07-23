"""Operator-level system containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSystem:
    """Engineering-level linear system description, usually ``K x = F``."""

    stiffness: object | None = None
    force: object | None = None
    mass: object | None = None
    damping: object | None = None
    name: str = "linear_system"

    def lhs_form(self):
        """Return the stiffness-like left-hand-side form."""

        if self.stiffness is None:
            raise ValueError("LinearSystem.lhs_form requires stiffness.")
        return _expression(self.stiffness)

    def rhs_form(self):
        """Return the force-like right-hand-side form."""

        if self.force is None:
            raise ValueError("LinearSystem.rhs_form requires force.")
        return _expression(self.force)

    def summary(self) -> dict[str, object]:
        """Return a compact K/M/C/F system summary."""

        return {
            "name": self.name,
            "stiffness": _describe(self.stiffness),
            "mass": _describe(self.mass),
            "damping": _describe(self.damping),
            "force": _describe(self.force),
        }


@dataclass(frozen=True)
class SecondOrderSystem:
    """Engineering-level second-order system, ``M a + C v + K u = F``."""

    mass: object
    stiffness: object
    damping: object | None = None
    force: object | None = None
    name: str = "second_order_system"

    def summary(self) -> dict[str, object]:
        """Return a compact M/C/K/F system summary."""

        return {
            "name": self.name,
            "mass": _describe(self.mass),
            "damping": _describe(self.damping),
            "stiffness": _describe(self.stiffness),
            "force": _describe(self.force),
        }


def _expression(operator):
    return operator.expression if hasattr(operator, "expression") else operator


def _describe(operator):
    if operator is None:
        return None
    if hasattr(operator, "summary"):
        return operator.summary()
    return type(operator).__name__
