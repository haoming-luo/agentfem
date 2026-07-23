"""Common UFL form building blocks."""

from __future__ import annotations

import ufl


def internal_virtual_work(stress, strain_test):
    """Internal virtual work density, ``sigma : epsilon(test)``."""

    return ufl.inner(stress, strain_test) * ufl.dx


def inertial_virtual_work(density, acceleration, test_function):
    """Inertial virtual work density, ``rho*a . test``."""

    return ufl.inner(density * acceleration, test_function) * ufl.dx


def body_force_virtual_work(force, test_function, measure=ufl.dx):
    """Body-force virtual work density, ``force . test``."""

    return ufl.inner(force, test_function) * measure


def boundary_flux_virtual_work(flux, test_function, ds_measure):
    """Boundary flux/traction virtual work, ``flux . test`` on a measure."""

    return ufl.inner(flux, test_function) * ds_measure
