# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

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


def burgers_convection_operator(
    advecting_scalar,
    transported_scalar,
    test,
    *,
    direction=None,
    measure=ufl.dx,
    name: str = "N_burgers",
) -> OperatorForm:
    """Return scalar Burgers transport ``u_adv (d . grad(u))``.

    The default direction contains one in every spatial direction, matching
    the common multidimensional scalar Burgers equation.  Supplying an
    explicit direction keeps the operator useful for reduced and directional
    transport models without introducing a second public API.
    """

    if direction is None:
        domain = ufl.domain.extract_unique_domain(transported_scalar)
        dimension_attribute = domain.geometric_dimension
        dimension = int(dimension_attribute() if callable(dimension_attribute) else dimension_attribute)
        selected_direction = ufl.as_vector((1.0,) * dimension)
    else:
        selected_direction = as_velocity(direction)
    return OperatorForm(
        name=name,
        kind="burgers_convection_operator",
        role="matrix",
        family="nonlinear_transport",
        expression=(
            advecting_scalar
            * ufl.dot(selected_direction, ufl.grad(transported_scalar))
            * test
            * measure
        ),
        metadata={
            "direction": _velocity_metadata("all" if direction is None else direction)
        },
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


def intrinsic_time_scale(
    domain, velocity, *, diffusivity=None, degree: int = 1, directional: bool = False,
    time_step: float | None = None
):
    """Return a cellwise SUPG scale.

    Without diffusivity, preserve the advective h/(2|v|) scale. With a
    nonnegative scalar diffusivity, blend advective and diffusive scales
    using h/degree. ``directional=True`` uses the mapped streamline length
    on full-dimensional cells (unit reference-cell convention), useful for
    stretched meshes. Symbolic diffusivity must be scalar and nonnegative.
    This bounded heuristic is not a monotonicity guarantee.
    """

    if domain is None:
        raise ValueError("intrinsic_time_scale requires domain=.")
    if isinstance(velocity, Sequence) and not isinstance(velocity, (str, bytes)):
        if np.linalg.norm(tuple(float(value) for value in velocity)) == 0.0 and diffusivity is None and time_step is None:
            raise ValueError(
                "intrinsic_time_scale requires nonzero advection velocity."
            )
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be a positive integer.")
    if time_step is not None and (not np.isfinite(time_step) or time_step <= 0):
        raise ValueError("time_step must be finite and positive.")
    vector = as_velocity(velocity)
    magnitude = ufl.sqrt(ufl.dot(vector, vector))
    h = ufl.CellDiameter(domain)
    if directional:
        # Unit-reference-cell metric: the streamwise size follows stretched
        # cells rather than their longest diagonal. Full-dimensional cells.
        mapped_velocity = ufl.dot(ufl.JacobianInverse(domain), vector)
        mapped_squared = ufl.dot(mapped_velocity, mapped_velocity)
        h = ufl.conditional(ufl.gt(mapped_squared, 0),
                            magnitude / ufl.sqrt(ufl.conditional(ufl.gt(mapped_squared, 0), mapped_squared, 1.0)), h)
    if diffusivity is None and time_step is None:
        return ufl.conditional(ufl.gt(magnitude, 0), h / (2.0 * ufl.conditional(ufl.gt(magnitude, 0), magnitude, 1.0)), 0.0)
    if diffusivity is None:
        diffusivity = 0.0
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer.")
    if np.isscalar(diffusivity):
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("diffusivity must be finite and nonnegative.")
    elif ufl.as_ufl(diffusivity).ufl_shape != ():
        raise ValueError("diffusivity must be scalar.")
    effective_h = h / degree
    inverse_squared = ((2.0 * magnitude / effective_h) ** 2
                       + (4.0 * diffusivity / effective_h**2) ** 2)
    if time_step is not None:
        inverse_squared += (2.0 / time_step) ** 2
    return ufl.conditional(ufl.gt(inverse_squared, 0),
                           1.0 / ufl.sqrt(ufl.conditional(ufl.gt(inverse_squared, 0), inverse_squared, 1.0)), 0.0)


def transient_transport_forms(
    trial,
    test,
    previous,
    source,
    previous_source,
    velocity,
    diffusivity,
    *,
    dt: float,
    theta: float = 0.5,
    tau=None,
    measure=ufl.dx,
):
    """Return ``(a, L)`` for a constant-coefficient transport theta step.

    Implements u_t + beta.grad(u) - div(kappa grad(u)) = f. The
    optional SUPG residual includes the time derivative, both endpoint
    sources, and both endpoint spatial states. Velocity and diffusivity
    must be constant in time over the step. Updating Dirichlet data at
    the new endpoint remains the caller's responsibility.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(theta) or not 0.5 <= theta <= 1.0:
        raise ValueError("theta must be between 0.5 and 1.")
    beta = as_velocity(velocity)
    state = theta * trial + (1.0 - theta) * previous
    forcing = theta * source + (1.0 - theta) * previous_source
    rate = (trial - previous) / dt
    residual = (
        (rate + ufl.dot(beta, ufl.grad(state)) - forcing) * test
        + diffusivity * ufl.inner(ufl.grad(state), ufl.grad(test))
    ) * measure
    if tau is not None:
        strong = (
            rate
            + ufl.dot(beta, ufl.grad(state))
            - ufl.div(diffusivity * ufl.grad(state))
            - forcing
        )
        residual += tau * strong * ufl.dot(beta, ufl.grad(test)) * measure
    return ufl.lhs(residual), ufl.rhs(residual)


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
    "burgers_convection_operator",
    "intrinsic_time_scale",
    "reaction_expression",
    "streamline_upwind_operator",
    "transient_transport_forms",
]
