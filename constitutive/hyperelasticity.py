"""Finite-strain compressible Neo-Hookean constitutive relations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

import numpy as np
import ufl

from agentfem import fields as field_api


@dataclass(frozen=True)
class NeoHookeanProperties:
    """Compressible Neo-Hookean parameters derived from ``E`` and ``nu``.

    The strain-energy density is

    ``psi = mu/2 (I_C - d) - mu ln(J) + lambda/2 ln(J)^2``.

    In two dimensions this is the plane-strain restriction with the
    out-of-plane stretch fixed to one.  Plane stress requires an additional
    local solve and is not implied by this model.
    """

    young: float
    poisson: float
    density: float | None = None
    name: str = "compressible Neo-Hookean"

    def __post_init__(self) -> None:
        if not isfinite(float(self.young)) or self.young <= 0.0:
            raise ValueError("NeoHookeanProperties.young must be finite and positive.")
        if not isfinite(float(self.poisson)) or not (-1.0 < self.poisson < 0.5):
            raise ValueError(
                "NeoHookeanProperties.poisson must satisfy -1 < poisson < 0.5."
            )
        if self.density is not None and (
            not isfinite(float(self.density)) or self.density <= 0.0
        ):
            raise ValueError(
                "NeoHookeanProperties.density must be finite and positive when set."
            )

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lambda_(self) -> float:
        return self.young * self.poisson / (
            (1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)
        )

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "compressible_neo_hookean",
            "kinematics": "finite_strain",
            "two_dimensional_assumption": "plane_strain",
            "young": self.young,
            "poisson": self.poisson,
            "density": self.density,
            "mu": self.mu,
            "lambda": self.lambda_,
            "bulk_modulus": self.bulk_modulus,
            "maturity": "fem_form_available",
        }


def neo_hookean(
    *,
    young: float,
    poisson: float,
    density: float | None = None,
    name: str = "compressible Neo-Hookean",
) -> NeoHookeanProperties:
    """Create a compressible Neo-Hookean material."""

    return NeoHookeanProperties(
        young=young,
        poisson=poisson,
        density=density,
        name=name,
    )


def deformation_gradient(displacement):
    """Return ``F = I + grad(u)``."""

    displacement = field_api.unwrap(displacement)
    dimension = len(displacement)
    return ufl.Identity(dimension) + ufl.grad(displacement)


def green_lagrange_strain(displacement):
    """Return the finite-strain tensor ``E = 1/2 (F.T F - I)``."""

    F = deformation_gradient(displacement)
    dimension = F.ufl_shape[0]
    return 0.5 * (F.T * F - ufl.Identity(dimension))


def strain_energy_density_from_gradient(
    F,
    properties: NeoHookeanProperties,
):
    """Return compressible Neo-Hookean energy density from ``F``."""

    dimension = F.ufl_shape[0]
    C = F.T * F
    J = ufl.det(F)
    return (
        0.5 * properties.mu * (ufl.tr(C) - dimension)
        - properties.mu * ufl.ln(J)
        + 0.5 * properties.lambda_ * ufl.ln(J) ** 2
    )


def strain_energy_density(displacement, properties: NeoHookeanProperties):
    """Return compressible Neo-Hookean energy density ``psi(u)``."""

    return strain_energy_density_from_gradient(
        deformation_gradient(displacement),
        properties,
    )


def first_piola_from_gradient(F, properties: NeoHookeanProperties):
    """Return the first Piola stress ``P = d psi / d F`` analytically."""

    inverse_transpose = ufl.inv(F).T
    J = ufl.det(F)
    return (
        properties.mu * (F - inverse_transpose)
        + properties.lambda_ * ufl.ln(J) * inverse_transpose
    )


def first_piola(displacement, properties: NeoHookeanProperties):
    """Return the first Piola stress for a displacement field."""

    return first_piola_from_gradient(
        deformation_gradient(displacement),
        properties,
    )


def cauchy_stress(displacement, properties: NeoHookeanProperties):
    """Return the Cauchy stress ``sigma = J^-1 P F^T``."""

    F = deformation_gradient(displacement)
    J = ufl.det(F)
    return (1.0 / J) * first_piola_from_gradient(F, properties) * F.T


def internal_virtual_work(
    displacement,
    test_function,
    properties: NeoHookeanProperties,
    *,
    measure=ufl.dx,
):
    """Return ``integral P : grad(v) dV``."""

    displacement = field_api.unwrap(displacement)
    return ufl.inner(
        first_piola(displacement, properties),
        ufl.grad(test_function),
    ) * measure


def residual(
    displacement,
    test_function,
    properties: NeoHookeanProperties,
    *,
    body_force=None,
    traction=None,
    measure=ufl.dx,
    boundary_measure=None,
):
    """Build total Lagrangian residual ``internal - external``.

    ``traction`` is interpreted as nominal traction per reference area.
    """

    value = internal_virtual_work(
        displacement,
        test_function,
        properties,
        measure=measure,
    )
    if body_force is not None:
        value -= ufl.inner(body_force, test_function) * measure
    if traction is not None:
        if boundary_measure is None:
            raise ValueError("hyperelastic residual traction requires boundary_measure.")
        value -= ufl.inner(traction, test_function) * boundary_measure
    return value


def tangent(residual_form, displacement, trial_function=None):
    """Differentiate a residual form with respect to displacement."""

    displacement = field_api.unwrap(displacement)
    if trial_function is None:
        trial_function = ufl.TrialFunction(displacement.function_space)
    return ufl.derivative(residual_form, displacement, trial_function)


def principal_nominal_stress(
    stretches,
    properties: NeoHookeanProperties,
) -> np.ndarray:
    """Evaluate diagonal first-Piola stresses for principal stretches.

    This numerical helper is useful for material tests and analytical
    verification.  All stretches must be finite and positive.
    """

    selected = np.asarray(stretches, dtype=float)
    if selected.ndim != 1 or selected.size not in {2, 3}:
        raise ValueError("stretches must be a vector of length two or three.")
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("principal stretches must be finite and positive.")
    J = float(np.prod(selected))
    return (
        properties.mu * (selected - 1.0 / selected)
        + properties.lambda_ * log(J) / selected
    )


def principal_energy_density(
    stretches,
    properties: NeoHookeanProperties,
) -> float:
    """Evaluate the Neo-Hookean energy for principal stretches."""

    selected = np.asarray(stretches, dtype=float)
    if selected.ndim != 1 or selected.size not in {2, 3}:
        raise ValueError("stretches must be a vector of length two or three.")
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("principal stretches must be finite and positive.")
    J = float(np.prod(selected))
    return float(
        0.5 * properties.mu * (np.dot(selected, selected) - selected.size)
        - properties.mu * log(J)
        + 0.5 * properties.lambda_ * log(J) ** 2
    )
