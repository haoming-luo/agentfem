"""Transport and reaction operators for scalar continuum fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import ufl

from .core import OperatorForm


def advection_operator(
    trial,
    test,
    velocity,
    *,
    measure=ufl.dx,
    name: str = "A_advection",
) -> OperatorForm:
    """Return the Galerkin advection operator ``(v . grad(u), w)``."""

    vector = as_velocity(velocity)
    return OperatorForm(
        name=name,
        kind="advection_operator",
        role="matrix",
        family="transport",
        expression=ufl.dot(vector, ufl.grad(trial)) * test * measure,
        metadata={"velocity": _velocity_metadata(velocity)},
    )


def streamline_upwind_operator(
    strong_residual,
    test,
    velocity,
    *,
    tau=None,
    domain=None,
    measure=ufl.dx,
    name: str = "A_supg",
) -> OperatorForm:
    """Return a SUPG contribution ``tau R(u) (v . grad(w))``."""

    vector = as_velocity(velocity)
    selected_tau = intrinsic_time_scale(domain, velocity) if tau is None else tau
    return OperatorForm(
        name=name,
        kind="streamline_upwind_stabilization",
        role="operator",
        family="stabilized_transport",
        expression=(
            selected_tau * strong_residual * ufl.dot(vector, ufl.grad(test)) * measure
        ),
        metadata={
            "method": "SUPG",
            "velocity": _velocity_metadata(velocity),
        },
    )


def intrinsic_time_scale(domain, velocity):
    """Return the standard cellwise advective SUPG scale ``h/(2 |v|)``."""

    if domain is None:
        raise ValueError("intrinsic_time_scale requires domain=.")
    if isinstance(velocity, Sequence) and not isinstance(velocity, (str, bytes)):
        if np.linalg.norm(tuple(float(value) for value in velocity)) == 0.0:
            raise ValueError(
                "intrinsic_time_scale requires nonzero advection velocity."
            )
    vector = as_velocity(velocity)
    magnitude = ufl.sqrt(ufl.dot(vector, vector))
    return ufl.CellDiameter(domain) / (2.0 * magnitude)


def reaction_expression(value, law: str | Mapping[str, object], **parameters):
    """Lower a named scalar reaction law to a UFL expression.

    Laws use ``u_dot - div(epsilon grad(u)) + r(u) = f``.
    """

    if isinstance(law, Mapping):
        selected = str(law.get("type", "linear")).lower()
        values = {**dict(law), **parameters}
    else:
        selected = str(law).lower()
        values = dict(parameters)
    if selected == "linear":
        return float(values.get("alpha", 0.0)) * value
    if selected == "cubic":
        return (
            float(values.get("alpha", 0.0)) * value
            + float(values.get("beta", 1.0)) * value**3
        )
    if selected in {"allen_cahn", "allen-cahn"}:
        scale = float(values.get("lambda", 1.0))
        return scale * (value**3 - value)
    if selected == "logistic":
        scale = float(values.get("rho", 1.0))
        return scale * value * (1.0 - value)
    raise ValueError(f"Unknown scalar reaction law {selected!r}.")


def as_velocity(velocity):
    """Normalize a public velocity sequence without hiding UFL expressions."""

    if isinstance(velocity, Sequence) and not isinstance(velocity, (str, bytes)):
        return ufl.as_vector(tuple(float(value) for value in velocity))
    return velocity


def _velocity_metadata(velocity):
    if isinstance(velocity, Sequence) and not isinstance(velocity, (str, bytes)):
        try:
            return tuple(float(value) for value in velocity)
        except (TypeError, ValueError):
            pass
    return str(velocity)


__all__ = [
    "advection_operator",
    "as_velocity",
    "intrinsic_time_scale",
    "reaction_expression",
    "streamline_upwind_operator",
]
