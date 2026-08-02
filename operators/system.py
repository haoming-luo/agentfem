"""Inspectable K/M/C/F and residual-system containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSystem:
    """Engineering-level static system, usually ``K x = F``."""

    stiffness: object | None = None
    force: object | None = None
    mass: object | None = None
    damping: object | None = None
    name: str = "linear_system"

    @property
    def K(self):
        return self.stiffness

    @property
    def F(self):
        return self.force

    @property
    def equation(self) -> str:
        return "K x = F"

    def lhs_form(self):
        if self.stiffness is None:
            raise ValueError("LinearSystem.lhs_form requires stiffness.")
        return _expression(self.stiffness)

    def rhs_form(self):
        if self.force is None:
            raise ValueError("LinearSystem.rhs_form requires force.")
        return _expression(self.force)

    def validate(self):
        return _system_report(
            self.name,
            required=(("stiffness", self.stiffness, "matrix"),),
            optional=(
                ("force", self.force, "vector"),
                ("mass", self.mass, "matrix"),
                ("damping", self.damping, "matrix"),
            ),
        )

    def check(self) -> None:
        self.validate().raise_if_errors()

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "equation": self.equation,
            "stiffness": _describe(self.stiffness),
            "mass": _describe(self.mass),
            "damping": _describe(self.damping),
            "force": _describe(self.force),
            "validation": self.validate().summary(),
        }


def linear_system(K, F=None, *, name: str = "Kx_eq_F") -> LinearSystem:
    """Create a static linear system in engineering notation, ``K x = F``."""

    return LinearSystem(stiffness=K, force=F, name=name)


@dataclass(frozen=True)
class FirstOrderSystem:
    """First-order transient system, ``C x_dot + K x = F``."""

    capacity: object
    stiffness: object
    force: object | None = None
    name: str = "first_order_system"

    @property
    def C(self):
        return self.capacity

    @property
    def K(self):
        return self.stiffness

    @property
    def F(self):
        return self.force

    @property
    def equation(self) -> str:
        return "C x_dot + K x = F"

    def validate(self):
        return _system_report(
            self.name,
            required=(
                ("capacity", self.capacity, "matrix"),
                ("stiffness", self.stiffness, "matrix"),
            ),
            optional=(("force", self.force, "vector"),),
        )

    def check(self) -> None:
        self.validate().raise_if_errors()

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "equation": self.equation,
            "capacity": _describe(self.capacity),
            "stiffness": _describe(self.stiffness),
            "force": _describe(self.force),
            "validation": self.validate().summary(),
        }


def first_order_system(C, K, F=None, *, name: str = "Cxdot_plus_Kx_eq_F"):
    """Create ``C x_dot + K x = F`` for heat/diffusion-like evolution."""

    return FirstOrderSystem(capacity=C, stiffness=K, force=F, name=name)


@dataclass(frozen=True)
class SecondOrderSystem:
    """Engineering-level second-order system, ``M a + C v + K u = F``."""

    mass: object
    stiffness: object
    damping: object | None = None
    force: object | None = None
    name: str = "second_order_system"

    @property
    def M(self):
        return self.mass

    @property
    def C(self):
        return self.damping

    @property
    def K(self):
        return self.stiffness

    @property
    def F(self):
        return self.force

    @property
    def equation(self) -> str:
        return "M a + C v + K u = F"

    def validate(self):
        return _system_report(
            self.name,
            required=(
                ("mass", self.mass, "matrix"),
                ("stiffness", self.stiffness, "matrix"),
            ),
            optional=(
                ("damping", self.damping, "matrix"),
                ("force", self.force, "vector"),
            ),
        )

    def check(self) -> None:
        self.validate().raise_if_errors()

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "equation": self.equation,
            "mass": _describe(self.mass),
            "damping": _describe(self.damping),
            "stiffness": _describe(self.stiffness),
            "force": _describe(self.force),
            "validation": self.validate().summary(),
        }


def second_order_system(M, K, C=None, F=None, *, name: str = "Ma_plus_Cv_plus_Ku_eq_F"):
    """Create ``M a + C v + K u = F`` with optional damping and force."""

    return SecondOrderSystem(mass=M, stiffness=K, damping=C, force=F, name=name)


def _system_report(name, *, required, optional):
    from agentfem.validation import ValidationReport, issue

    issues = []
    for component, value, expected_role in required:
        if value is None:
            issues.append(
                issue(
                    "AFM-SYS-001",
                    f"system.{name}.{component}",
                    f"Required system component {component!r} is missing.",
                )
            )
            continue
        issues.extend(_role_issues(name, component, value, expected_role))
    for component, value, expected_role in optional:
        if value is not None:
            issues.extend(_role_issues(name, component, value, expected_role))
    return ValidationReport.from_issues(issues, scope=f"system:{name}")


def _role_issues(name, component, value, expected_role):
    from agentfem.validation import issue

    actual_role = _role(value)
    if actual_role is None or actual_role == expected_role:
        return ()
    return (
        issue(
            "AFM-SYS-002",
            f"system.{name}.{component}",
            f"Expected a {expected_role} operator, got role={actual_role!r}.",
            hint="Use OperatorForm roles or a UFL form with matching argument count.",
            expected_role=expected_role,
            actual_role=actual_role,
        ),
    )


def _role(operator):
    role = getattr(operator, "role", None)
    if role is not None:
        return role
    from .core import form_arity

    arity = form_arity(_expression(operator))
    return {0: "scalar", 1: "vector", 2: "matrix"}.get(arity)


def _expression(operator):
    return operator.expression if hasattr(operator, "expression") else operator


def _describe(operator):
    if operator is None:
        return None
    if hasattr(operator, "summary"):
        return operator.summary()
    return type(operator).__name__


__all__ = [
    "FirstOrderSystem",
    "LinearSystem",
    "SecondOrderSystem",
    "first_order_system",
    "linear_system",
    "second_order_system",
]
