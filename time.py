"""Time-integration kernels for finite-element transient solves."""

from __future__ import annotations

import numpy as np

from . import dofs


def central_difference_predict_displacement(u_next, u, velocity, acceleration, dt: float) -> None:
    """Predict displacement with the explicit central-difference/Newmark formula."""

    u_next.x.array[:] = (
        u.x.array + dt * velocity.x.array + 0.5 * dt**2 * acceleration.x.array
    )
    u_next.x.scatter_forward()


def acceleration_from_residual(acceleration, residual, inv_mass: np.ndarray) -> None:
    """Set acceleration from residual and inverse lumped mass.

    The conventional elastodynamic residual here is internal-force-like, so the
    explicit acceleration update is ``a = -M^{-1} r``.
    """

    dofs.assign_owned(acceleration, -residual.array * inv_mass)


def central_difference_correct_velocity(
    velocity_next, velocity, acceleration, acceleration_next, dt: float
) -> None:
    """Correct velocity with the explicit central-difference/Newmark formula."""

    velocity_next.x.array[:] = velocity.x.array + 0.5 * dt * (
        acceleration.x.array + acceleration_next.x.array
    )
    velocity_next.x.scatter_forward()
