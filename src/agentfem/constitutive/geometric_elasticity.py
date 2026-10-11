# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Total-Lagrangian Saint-Venant--Kirchhoff elasticity.

Finite rotations are objective; the material law is appropriate for small
Green--Lagrange strains, not arbitrary large compression or rubber elasticity.
All energies are per reference volume; two-dimensional forms have unit
reference thickness.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import ufl


@dataclass(frozen=True)
class SaintVenantKirchhoffProperties:
    young: float
    poisson: float
    density: float | None = None
    plane_stress: bool = False
    name: str = "Saint-Venant--Kirchhoff"

    def __post_init__(self) -> None:
        if not isinstance(self.plane_stress, (bool, np.bool_)):
            raise TypeError("plane_stress must be a boolean.")
        if not isfinite(self.young) or self.young <= 0:
            raise ValueError(
                "Saint-Venant--Kirchhoff young must be finite and positive."
            )
        if not isfinite(self.poisson) or not -1 < self.poisson < 0.5:
            raise ValueError("Saint-Venant--Kirchhoff requires -1 < poisson < 0.5.")
        if self.density is not None and (
            not isfinite(self.density) or self.density <= 0
        ):
            raise ValueError("Density must be finite and positive.")

    @property
    def mu(self) -> float:
        return self.young / (2 * (1 + self.poisson))

    @property
    def lambda_(self) -> float:
        return self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))

    @property
    def reduced_lambda(self) -> float:
        return 2 * self.mu * self.lambda_ / (self.lambda_ + 2 * self.mu)

    def as_dict(self) -> dict[str, object]:
        return dict(
            name=self.name,
            model="saint_venant_kirchhoff",
            young=self.young,
            poisson=self.poisson,
            density=self.density,
            kinematics="finite_strain",
            stress_measure="second_piola_kirchhoff",
            strain_measure="green_lagrange",
            two_dimensional_assumption="plane_stress"
            if self.plane_stress
            else "plane_strain",
            thickness_condition="S33=0_exact_condensation"
            if self.plane_stress
            else "F33=1",
            validity="small_material_strain_finite_rotation",
            maturity="fem_form_available",
        )


def saint_venant_kirchhoff(
    *,
    young: float,
    poisson: float,
    density: float | None = None,
    plane_stress: bool = False,
    name: str = "Saint-Venant--Kirchhoff",
) -> SaintVenantKirchhoffProperties:
    """Create explicit geometrically nonlinear elastic material parameters."""
    return SaintVenantKirchhoffProperties(young, poisson, density, plane_stress, name)


def _dimension(F, material):
    shape = tuple(F.ufl_shape)
    if shape not in ((2, 2), (3, 3)) or (material.plane_stress and shape != (2, 2)):
        raise ValueError("StVK requires 2x2 or 3x3 F; plane stress requires 2x2 F.")
    return shape[0]


def thickness_squared(F, material):
    """Exact relaxed thickness squared; positive values are admissible."""
    if not material.plane_stress or _dimension(F, material) != 2:
        raise ValueError("Thickness condensation requires plane-stress StVK.")
    E = (F.T * F - ufl.Identity(2)) / 2
    return 1 - 2 * material.lambda_ / (material.lambda_ + 2 * material.mu) * ufl.tr(E)


def energy(F, material):
    dimension = _dimension(F, material)
    E = (F.T * F - ufl.Identity(dimension)) / 2
    lam = material.reduced_lambda if material.plane_stress else material.lambda_
    return lam / 2 * ufl.tr(E) ** 2 + material.mu * ufl.inner(E, E)


def first_piola(F, material):
    dimension = _dimension(F, material)
    E = (F.T * F - ufl.Identity(dimension)) / 2
    lam = material.reduced_lambda if material.plane_stress else material.lambda_
    return F * (lam * ufl.tr(E) * ufl.Identity(dimension) + 2 * material.mu * E)


def response_value(F, material: SaintVenantKirchhoffProperties) -> dict[str, object]:
    """Numerical constitutive response, including a full 3-D plane-stress embedding.

    Returns energy, in-plane/full first Piola stress and its exact tangent,
    full 3-D Cauchy stress, Green strain and relaxed thickness. Tangent indexing
    is A[i,J,k,L] = dP[i,J]/dF[k,L]. Invalid orientation/thickness is rejected.
    """
    F = np.asarray(F, dtype=float)
    if F.shape not in ((2, 2), (3, 3)) or (material.plane_stress and F.shape != (2, 2)):
        raise ValueError("StVK requires 2x2 or 3x3 F; plane stress requires 2x2 F.")
    if not np.all(np.isfinite(F)) or np.linalg.det(F) <= 0:
        raise ValueError("StVK requires finite F with J > 0.")
    d = F.shape[0]
    I = np.eye(d)
    E = (F.T @ F - I) / 2
    lam = material.reduced_lambda if material.plane_stress else material.lambda_
    S = lam * np.trace(E) * I + 2 * material.mu * E
    P = F @ S
    A = np.einsum("ik,LJ->iJkL", I, S)
    A += lam * np.einsum("iJ,kL->iJkL", F, F)
    A += material.mu * (
        np.einsum("iL,kJ->iJkL", F, F) + np.einsum("ik,JL->iJkL", F @ F.T, I)
    )
    thickness = 1.0
    if material.plane_stress:
        radicand = 1 - 2 * material.lambda_ / (
            material.lambda_ + 2 * material.mu
        ) * np.trace(E)
        if radicand <= 0:
            raise ValueError("StVK plane stress has no positive thickness solution.")
        thickness = float(np.sqrt(radicand))
    F3 = np.eye(3)
    F3[:d, :d] = F
    if d == 2:
        F3[2, 2] = thickness
    E3 = (F3.T @ F3 - np.eye(3)) / 2
    S3 = material.lambda_ * np.trace(E3) * np.eye(3) + 2 * material.mu * E3
    sigma3 = F3 @ S3 @ F3.T / np.linalg.det(F3)
    return dict(
        energy=float(lam / 2 * np.trace(E) ** 2 + material.mu * np.sum(E * E)),
        first_piola=P,
        tangent=A,
        cauchy=sigma3,
        green_strain=E3,
        thickness_stretch=thickness,
        jacobian=float(np.linalg.det(F3)),
    )


def embedded_gradient(F, material):
    """Recover the three-dimensional block gradient for 2-D stress output."""
    if _dimension(F, material) == 3:
        return F
    stretch = ufl.sqrt(thickness_squared(F, material)) if material.plane_stress else 1.0
    return ufl.as_matrix(
        ((F[0, 0], F[0, 1], 0.0), (F[1, 0], F[1, 1], 0.0), (0.0, 0.0, stretch))
    )
