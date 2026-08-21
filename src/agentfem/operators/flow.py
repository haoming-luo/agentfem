"""Reusable weak-form operators for incompressible flow."""

from __future__ import annotations

import ufl

from .core import OperatorForm


def viscous_flow_operator(
    velocity,
    test_velocity,
    viscosity,
    *,
    measure=ufl.dx,
    name: str = "K_viscous",
) -> OperatorForm:
    """Return ``nu (grad(u), grad(v))`` for incompressible momentum."""

    return OperatorForm(
        name=name,
        kind="viscous_flow_operator",
        role="matrix",
        family="incompressible_flow",
        expression=(
            viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity)) * measure
        ),
        metadata={"viscosity": str(viscosity)},
    )


def pressure_coupling_operator(
    pressure,
    test_velocity,
    *,
    measure=ufl.dx,
    name: str = "G_pressure",
) -> OperatorForm:
    """Return the pressure contribution ``-(p, div(v))``."""

    return OperatorForm(
        name=name,
        kind="pressure_coupling_operator",
        role="matrix",
        family="incompressible_flow",
        expression=-pressure * ufl.div(test_velocity) * measure,
    )


def incompressibility_operator(
    velocity,
    test_pressure,
    *,
    measure=ufl.dx,
    name: str = "D_incompressibility",
) -> OperatorForm:
    """Return the symmetric saddle-point term ``-(q, div(u))``."""

    return OperatorForm(
        name=name,
        kind="incompressibility_operator",
        role="matrix",
        family="incompressible_flow",
        expression=-test_pressure * ufl.div(velocity) * measure,
    )


def convective_momentum_operator(
    advecting_velocity,
    transported_velocity,
    test_velocity,
    *,
    measure=ufl.dx,
    name: str = "N_convection",
) -> OperatorForm:
    """Return ``((w . grad) u, v)`` for vector momentum transport."""

    return OperatorForm(
        name=name,
        kind="convective_momentum_operator",
        role="residual",
        family="incompressible_flow",
        expression=ufl.inner(
            ufl.dot(ufl.grad(transported_velocity), advecting_velocity),
            test_velocity,
        )
        * measure,
    )


__all__ = [
    "convective_momentum_operator",
    "incompressibility_operator",
    "pressure_coupling_operator",
    "viscous_flow_operator",
]
