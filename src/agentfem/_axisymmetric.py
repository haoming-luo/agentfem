"""Shared axisymmetric kinematics and physical integration weights.

The public modeling declaration remains ``Study(assumption="axisymmetric")``.
This private module is the single lowering point used by constitutive,
operator, result, and stateful-mechanics code.  Meridian coordinates and
displacements are ordered ``(r, z)``; embedded tensors are ordered
``(r, theta, z)``.
"""

from __future__ import annotations

from math import pi

import ufl
from ufl.domain import extract_unique_domain

from . import fields as field_api


def is_axisymmetric(study) -> bool:
    """Return whether a Study declares axisymmetric solid kinematics."""

    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 2
        and getattr(study, "assumption", None) == "axisymmetric"
    )


def strain(displacement):
    """Embed meridian displacement in the 3D axisymmetric strain tensor.

    ``u = (u_r, u_z)`` and the returned tensor uses ``(r, theta, z)``.
    Regularity requires ``u_r=0`` on a modeled symmetry axis.  Finite-element
    integration points lie inside cells, where the hoop strain is ``u_r/r``.
    """

    selected = field_api.unwrap(displacement)
    if tuple(getattr(selected, "ufl_shape", ())) != (2,):
        raise ValueError("Axisymmetric displacement must have components (u_r, u_z).")
    radius = ufl.SpatialCoordinate(extract_unique_domain(selected))[0]
    radial = selected[0]
    axial = selected[1]
    hoop = radial / radius
    shear = 0.5 * (radial.dx(1) + axial.dx(0))
    return ufl.as_tensor(
        (
            (radial.dx(0), 0.0, shear),
            (0.0, hoop, 0.0),
            (shear, 0.0, axial.dx(1)),
        )
    )


def integration_weight(field_or_domain, study, *, full_revolution: bool = True):
    """Return the cylindrical Jacobian for a meridian weak form.

    A full physical body uses ``2*pi*r``.  The constant ``2*pi`` cancels from
    equilibrium but is retained so body forces, reactions, energies, and
    resultants have their full-revolution physical meaning.
    """

    if not is_axisymmetric(study):
        return 1.0
    domain = _domain(field_or_domain)
    radius = ufl.SpatialCoordinate(domain)[0]
    return (2.0 * pi if full_revolution else 1.0) * radius


def weighted_test(test_function, study):
    """Return the test function carrying the cylindrical load measure."""

    return integration_weight(test_function, study) * test_function


def _domain(field_or_domain):
    selected = field_api.unwrap(field_or_domain)
    if hasattr(selected, "ufl_domains"):
        domains = tuple(selected.ufl_domains())
        if len(domains) == 1:
            return domains[0]
    if hasattr(selected, "ufl_cargo"):
        return selected
    if hasattr(selected, "geometry") and hasattr(selected, "topology"):
        return selected
    if hasattr(selected, "function_space"):
        return selected.function_space.mesh
    raise TypeError("Could not determine the mesh for axisymmetric integration.")


__all__ = ["integration_weight", "is_axisymmetric", "strain", "weighted_test"]
