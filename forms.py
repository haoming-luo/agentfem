"""Common UFL form building blocks."""

from __future__ import annotations

import ufl

from . import fields as field_api


def stiffness_form(stress, strain_test, measure=ufl.dx):
    """Internal stiffness/virtual-work form, ``sigma : epsilon(test)``."""

    return ufl.inner(stress, strain_test) * measure


def mass_form(density, trial_function, test_function, measure=ufl.dx):
    """Consistent mass form, ``rho * trial . test``."""

    trial_function = field_api.unwrap(trial_function)
    return ufl.inner(density * trial_function, test_function) * measure


def damping_form(coefficient, trial_function, test_function, measure=ufl.dx):
    """Viscous damping form, ``c * trial . test``."""

    trial_function = field_api.unwrap(trial_function)
    return ufl.inner(coefficient * trial_function, test_function) * measure


def diffusion_form(conductivity, trial_function, test_function, measure=ufl.dx):
    """Diffusion/conduction form, ``k * grad(trial) . grad(test)``."""

    trial_function = field_api.unwrap(trial_function)
    return conductivity * ufl.inner(ufl.grad(trial_function), ufl.grad(test_function)) * measure


def inertial_form(density, acceleration, test_function, measure=ufl.dx):
    """Inertial virtual-work form, ``rho * acceleration . test``."""

    acceleration = field_api.unwrap(acceleration)
    return ufl.inner(density * acceleration, test_function) * measure


def body_load_form(force, test_function, measure=ufl.dx):
    """Body-force/source virtual-work form, ``force . test``."""

    force = field_api.unwrap(force)
    if _is_scalar(test_function):
        return force * test_function * measure
    return ufl.inner(force, test_function) * measure


def boundary_load_form(load, test_function, measure):
    """Boundary flux/traction virtual-work form, ``load . test``."""

    load = field_api.unwrap(load)
    if _is_scalar(test_function):
        return load * test_function * measure
    return ufl.inner(load, test_function) * measure


def scalar_flux_form(flux, test_function, measure):
    """Scalar flux weak form, ``flux * test`` on a boundary or domain measure."""

    flux = field_api.unwrap(flux)
    return flux * test_function * measure


def robin_form(coefficient, trial_function, test_function, measure):
    """Robin/impedance bilinear form, ``coefficient * trial * test``."""

    trial_function = field_api.unwrap(trial_function)
    return coefficient * ufl.inner(trial_function, test_function) * measure


def internal_virtual_work(stress, strain_test):
    """Compatibility wrapper for ``stiffness_form``."""

    return stiffness_form(stress, strain_test)


def inertial_virtual_work(density, acceleration, test_function):
    """Compatibility wrapper for ``inertial_form``."""

    return inertial_form(density, acceleration, test_function)


def body_force_virtual_work(force, test_function, measure=ufl.dx):
    """Compatibility wrapper for ``body_load_form``."""

    return body_load_form(force, test_function, measure)


def boundary_flux_virtual_work(flux, test_function, ds_measure):
    """Compatibility wrapper for ``boundary_load_form``."""

    return boundary_load_form(flux, test_function, ds_measure)


def _is_scalar(expression) -> bool:
    return getattr(expression, "ufl_shape", ()) == ()
