"""Inspectable operator contract for direct harmonic response."""

from __future__ import annotations

from dataclasses import dataclass

from .core import form_arity


@dataclass(frozen=True)
class DirectHarmonicSystem:
    r"""One linear steady-state harmonic system.

    With the public ``exp(+i*omega*t)`` convention, the complex dynamic
    stiffness is

    ``K_storage + i K_loss + i omega C - omega**2 M``.

    ``K_loss`` is a material loss operator while ``C`` is a viscous damping
    operator. Keeping them distinct lets the result layer report material and
    viscous dissipation separately.
    """

    storage: object
    force: object
    mass: object | None = None
    loss: object | None = None
    damping: object | None = None
    name: str = "direct_harmonic_system"
    phasor_convention: str = "exp(+i*omega*t)"

    @property
    def K(self):
        return self.storage

    @property
    def M(self):
        return self.mass

    @property
    def C(self):
        return self.damping

    @property
    def F(self):
        return self.force

    @property
    def equation(self) -> str:
        return "(K_storage + i K_loss + i omega C - omega^2 M) u_hat = F_hat"

    def validate(self):
        from ..validation import ValidationReport, issue

        issues = []
        if self.phasor_convention != "exp(+i*omega*t)":
            issues.append(
                issue(
                    "AFM-HARMONIC-OP-001",
                    f"system.{self.name}.phasor_convention",
                    "Direct harmonic systems currently require exp(+i*omega*t).",
                )
            )
        for label, operator, expected in (
            ("storage", self.storage, 2),
            ("force", self.force, 1),
            ("mass", self.mass, 2),
            ("loss", self.loss, 2),
            ("damping", self.damping, 2),
        ):
            if operator is None:
                if label in {"storage", "force"}:
                    issues.append(
                        issue(
                            "AFM-HARMONIC-OP-002",
                            f"system.{self.name}.{label}",
                            f"Required harmonic operator {label!r} is missing.",
                        )
                    )
                continue
            expression = _expression(operator)
            actual = form_arity(expression)
            if actual is not None and actual != expected:
                issues.append(
                    issue(
                        "AFM-HARMONIC-OP-003",
                        f"system.{self.name}.{label}",
                        f"Expected form arity {expected}, got {actual}.",
                        expected_arity=expected,
                        actual_arity=actual,
                    )
                )
        return ValidationReport.from_issues(
            issues,
            scope=f"harmonic_system:{self.name}",
        )

    def check(self) -> None:
        self.validate().raise_if_errors()

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "equation": self.equation,
            "phasor_convention": self.phasor_convention,
            "storage": _describe(self.storage),
            "loss": _describe(self.loss),
            "mass": _describe(self.mass),
            "damping": _describe(self.damping),
            "force": _describe(self.force),
            "dissipation_channels": {
                "material_loss": self.loss is not None,
                "viscous_damping": self.damping is not None,
            },
            "validation": self.validate().summary(),
        }

    def to_ir(self) -> dict[str, object]:
        """Expose every scientific operator to the common input fingerprint."""

        return {
            "name": self.name,
            "equation": self.equation,
            "phasor_convention": self.phasor_convention,
            "storage": self.storage,
            "loss": self.loss,
            "mass": self.mass,
            "damping": self.damping,
            "force": self.force,
        }


def direct_harmonic_system(
    K,
    F,
    *,
    M=None,
    C=None,
    K_loss=None,
    name: str = "direct_harmonic_system",
) -> DirectHarmonicSystem:
    """Create an inspectable direct harmonic ``K/M/C/F`` system."""

    system = DirectHarmonicSystem(
        storage=K,
        force=F,
        mass=M,
        loss=K_loss,
        damping=C,
        name=name,
    )
    system.check()
    return system


def _expression(operator):
    return operator.expression if hasattr(operator, "expression") else operator


def _describe(operator):
    if operator is None:
        return None
    if hasattr(operator, "summary"):
        return operator.summary()
    return type(operator).__name__


__all__ = ["DirectHarmonicSystem", "direct_harmonic_system"]
