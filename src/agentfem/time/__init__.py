"""Time-integration kernels for finite-element transient solves."""

from __future__ import annotations

import numpy as np

from ..kernel import dofs
from .runtime import ProgressPrinter, TimeStep, TimeStepper, format_duration

__all__ = [
    "ProgressPrinter",
    "TimeStep",
    "TimeStepper",
    "acceleration_from_residual",
    "central_difference_update_midstep_velocity",
    "central_difference_correct_velocity",
    "central_difference_predict_displacement",
    "central_difference_update_velocity",
    "explicit",
    "implicit",
    "GeneralizedAlphaParameters",
    "generalized_alpha",
    "newmark",
    "format_duration",
]


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


def central_difference_update_midstep_velocity(velocity_mid, velocity, acceleration, dt: float) -> None:
    """Update the central-difference mid-step velocity."""

    velocity_mid.x.array[:] = velocity.x.array + 0.5 * dt * acceleration.x.array
    velocity_mid.x.scatter_forward()


def central_difference_correct_velocity(
    velocity_next, velocity, acceleration, acceleration_next, dt: float
) -> None:
    """Correct velocity with the explicit central-difference/Newmark formula."""

    velocity_next.x.array[:] = velocity.x.array + 0.5 * dt * (
        acceleration.x.array + acceleration_next.x.array
    )
    velocity_next.x.scatter_forward()


def central_difference_update_velocity(
    velocity_next, velocity_mid, acceleration_next, dt: float
) -> None:
    """Update whole-step velocity from mid-step velocity and new acceleration."""

    velocity_next.x.array[:] = velocity_mid.x.array + 0.5 * dt * acceleration_next.x.array
    velocity_next.x.scatter_forward()


from . import explicit
from . import implicit
from .implicit import GeneralizedAlphaParameters, generalized_alpha, newmark
