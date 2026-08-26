"""Solver-neutral interaction-integral reduction for two-dimensional LEFM.

The numerical solver owns field evaluation and quadrature. AgentFEM core owns
the scientific normalization, sample validation, path evidence, and result
contract. This keeps the same extraction path available to FEM, enriched FEM,
and neural fields without making any one representation the reference truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from .fracture_geometry import (
    CrackSet2D,
    CrackTip2D,
    LinearElasticFractureMaterial2D,
    StressIntensityReport,
    stress_intensity_report,
)


@dataclass(frozen=True)
class WilliamsField2D:
    """Leading mixed-mode Williams field around one straight crack tip."""

    tip: CrackTip2D
    material: LinearElasticFractureMaterial2D
    k_i: float = 0.0
    k_ii: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.tip, CrackTip2D):
            raise TypeError("tip must be a CrackTip2D record.")
        if not isinstance(self.material, LinearElasticFractureMaterial2D):
            raise TypeError("material must be a LinearElasticFractureMaterial2D record.")
        values = (float(self.k_i), float(self.k_ii))
        if any(not isfinite(item) for item in values):
            raise ValueError("Williams stress-intensity factors must be finite.")
        object.__setattr__(self, "k_i", values[0])
        object.__setattr__(self, "k_ii", values[1])

    @property
    def shear_modulus(self) -> float:
        return self.material.young_modulus / (2.0 * (1.0 + self.material.poisson_ratio))

    @property
    def kappa(self) -> float:
        ratio = self.material.poisson_ratio
        if self.material.assumption == "plane_strain":
            return 3.0 - 4.0 * ratio
        return (3.0 - ratio) / (1.0 + ratio)

    def displacement(self, points, *, side: str | None = None):
        radius, angle, rotation = self._polar(points, side=side)
        functions, _ = self._angular_displacement(angle)
        scale = np.sqrt(radius / (2.0 * np.pi)) / (2.0 * self.shear_modulus)
        local = scale[:, None] * functions
        return np.einsum("ij,nj->ni", rotation, local)

    def displacement_gradient(self, points, *, side: str | None = None):
        radius, angle, rotation = self._polar(points, side=side)
        functions, derivatives = self._angular_displacement(angle)
        coefficient = 1.0 / (
            2.0 * self.shear_modulus * np.sqrt(2.0 * np.pi * radius)
        )
        cosine = np.cos(angle)
        sine = np.sin(angle)
        radial = 0.5 * coefficient[:, None] * functions
        angular = coefficient[:, None] * derivatives
        local = np.empty((radius.size, 2, 2), dtype=float)
        local[:, :, 0] = cosine[:, None] * radial - sine[:, None] * angular
        local[:, :, 1] = sine[:, None] * radial + cosine[:, None] * angular
        return np.einsum("ai,nij,bj->nab", rotation, local, rotation)

    def stress(self, points, *, side: str | None = None):
        radius, angle, rotation = self._polar(points, side=side)
        half = 0.5 * angle
        cosine = np.cos(half)
        sine = np.sin(half)
        cosine_three = np.cos(3.0 * half)
        sine_three = np.sin(3.0 * half)
        scale = 1.0 / np.sqrt(2.0 * np.pi * radius)
        local = np.empty((radius.size, 2, 2), dtype=float)
        local[:, 0, 0] = scale * (
            self.k_i * cosine * (1.0 - sine * sine_three)
            - self.k_ii * sine * (2.0 + cosine * cosine_three)
        )
        local[:, 1, 1] = scale * (
            self.k_i * cosine * (1.0 + sine * sine_three)
            + self.k_ii * sine * cosine * cosine_three
        )
        shear = scale * (
            self.k_i * cosine * sine * cosine_three
            + self.k_ii * cosine * (1.0 - sine * sine_three)
        )
        local[:, 0, 1] = shear
        local[:, 1, 0] = shear
        return np.einsum("ai,nij,bj->nab", rotation, local, rotation)

    def _polar(self, points, *, side):
        values = np.asarray(points, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("WilliamsField2D points must have shape (n, 2).")
        if not np.all(np.isfinite(values)):
            raise ValueError("WilliamsField2D points must be finite.")
        if side not in {None, "upper", "lower"}:
            raise ValueError("side must be None, 'upper', or 'lower'.")
        rotation = np.asarray(
            (self.tip.extension_direction, self.tip.normal), dtype=float
        ).T
        local = (values - np.asarray(self.tip.point, dtype=float)) @ rotation
        radius = np.linalg.norm(local, axis=1)
        if np.any(radius <= np.finfo(float).eps):
            raise ValueError("WilliamsField2D is singular at the crack tip.")
        angle = np.arctan2(local[:, 1], local[:, 0])
        branch = (local[:, 0] < 0.0) & (
            np.abs(local[:, 1]) <= 64.0 * np.finfo(float).eps * radius
        )
        if side == "upper":
            angle[branch] = np.pi
        elif side == "lower":
            angle[branch] = -np.pi
        return radius, angle, rotation

    def _angular_displacement(self, angle):
        half = 0.5 * angle
        cosine = np.cos(half)
        sine = np.sin(half)
        full_cosine = np.cos(angle)
        full_sine = np.sin(angle)
        kappa = self.kappa
        functions = np.empty((angle.size, 2), dtype=float)
        derivatives = np.empty_like(functions)
        functions[:, 0] = (
            self.k_i * cosine * (kappa - full_cosine)
            + self.k_ii * sine * (kappa + 2.0 + full_cosine)
        )
        functions[:, 1] = (
            self.k_i * sine * (kappa - full_cosine)
            - self.k_ii * cosine * (kappa - 2.0 + full_cosine)
        )
        derivatives[:, 0] = self.k_i * (
            -0.5 * sine * (kappa - full_cosine) + cosine * full_sine
        ) + self.k_ii * (
            0.5 * cosine * (kappa + 2.0 + full_cosine)
            - sine * full_sine
        )
        derivatives[:, 1] = self.k_i * (
            0.5 * cosine * (kappa - full_cosine) + sine * full_sine
        ) + self.k_ii * (
            0.5 * sine * (kappa - 2.0 + full_cosine)
            + cosine * full_sine
        )
        return functions, derivatives


@dataclass(frozen=True)
class InteractionIntegralSamples2D:
    """Quadrature samples for one actual/auxiliary-field interaction integral.

    Tensor entries use local crack-tip coordinates. Displacement gradients are
    stored as ``gradient[i, j] = du_i/dx_j``. The scalar weight function ``q``
    is one near the selected tip and zero on the outer integration boundary.
    ``weights`` already include quadrature weights and geometric Jacobians.
    """

    actual_stress: object
    actual_displacement_gradient: object
    auxiliary_stress: object
    auxiliary_displacement_gradient: object
    q_gradient: object
    weights: object
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        tensors = {
            "actual_stress": np.asarray(self.actual_stress, dtype=float),
            "actual_displacement_gradient": np.asarray(
                self.actual_displacement_gradient, dtype=float
            ),
            "auxiliary_stress": np.asarray(self.auxiliary_stress, dtype=float),
            "auxiliary_displacement_gradient": np.asarray(
                self.auxiliary_displacement_gradient, dtype=float
            ),
        }
        q_gradient = np.asarray(self.q_gradient, dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        count = weights.size
        if count == 0:
            raise ValueError("InteractionIntegralSamples2D requires samples.")
        for name, values in tensors.items():
            if values.shape != (count, 2, 2):
                raise ValueError(f"{name} must have shape (number_of_samples, 2, 2).")
        if q_gradient.shape != (count, 2):
            raise ValueError("q_gradient must have shape (number_of_samples, 2).")
        arrays = (*tensors.values(), q_gradient, weights)
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("Interaction-integral samples must be finite.")
        if np.any(weights <= 0.0):
            raise ValueError("Interaction-integral weights must be positive.")
        for name, values in tensors.items():
            object.__setattr__(self, name, values.copy())
        object.__setattr__(self, "q_gradient", q_gradient.copy())
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def number_of_samples(self) -> int:
        return int(self.weights.size)


def interaction_integral(samples: InteractionIntegralSamples2D) -> float:
    r"""Evaluate the straight-crack, homogeneous interaction domain integral.

    This is the traction-free, body-force-free, quasi-static form

    ``I = integral[(sigma_ij uaux_j,1 + sigmaaux_ij u_j,1
                   - sigma_jk epsilonaux_jk delta_1i) q_,i dA]``.

    Curved cracks, inhomogeneous materials, thermal eigenstrains, body forces,
    and applied crack-face tractions require additional terms and are rejected
    by higher-level adapters rather than silently folded into this expression.
    """

    if not isinstance(samples, InteractionIntegralSamples2D):
        raise TypeError("samples must be an InteractionIntegralSamples2D record.")
    actual_stress = samples.actual_stress
    actual_gradient = samples.actual_displacement_gradient
    auxiliary_stress = samples.auxiliary_stress
    auxiliary_gradient = samples.auxiliary_displacement_gradient
    auxiliary_strain = 0.5 * (
        auxiliary_gradient + np.swapaxes(auxiliary_gradient, 1, 2)
    )
    actual_derivative = actual_gradient[:, :, 0]
    auxiliary_derivative = auxiliary_gradient[:, :, 0]
    interaction_energy = np.einsum(
        "nij,nij->n", actual_stress, auxiliary_strain
    )
    flux = np.einsum("nij,nj->ni", actual_stress, auxiliary_derivative)
    flux += np.einsum("nij,nj->ni", auxiliary_stress, actual_derivative)
    flux[:, 0] -= interaction_energy
    integrand = np.einsum("ni,ni->n", flux, samples.q_gradient)
    value = float(np.dot(integrand, samples.weights))
    if not isfinite(value):
        raise ValueError("Interaction integral produced a non-finite value.")
    return value


def interaction_integral_report(
    *,
    crack: CrackSet2D,
    tip_id: str,
    integration_radii,
    mode_i_integrals,
    mode_ii_integrals,
    material: LinearElasticFractureMaterial2D,
    relative_path_tolerance: float = 0.03,
    metadata: Mapping[str, object] | None = None,
) -> StressIntensityReport:
    """Convert unit-auxiliary interaction integrals into a per-tip SIF report."""

    if not isinstance(material, LinearElasticFractureMaterial2D):
        raise TypeError("material must be a LinearElasticFractureMaterial2D record.")
    radii = tuple(float(item) for item in integration_radii)
    mode_i = tuple(float(item) for item in mode_i_integrals)
    mode_ii = tuple(float(item) for item in mode_ii_integrals)
    if len(mode_i) != len(radii) or len(mode_ii) != len(radii):
        raise ValueError("Each integration radius requires Mode-I and Mode-II values.")
    normalization = 0.5 * material.effective_modulus
    k_i = tuple(normalization * item for item in mode_i)
    k_ii = tuple(normalization * item for item in mode_ii)
    energy = tuple(
        (first**2 + second**2) / material.effective_modulus
        for first, second in zip(k_i, k_ii)
    )
    return stress_intensity_report(
        crack=crack,
        tip_id=tip_id,
        integration_radii=radii,
        k_i=k_i,
        k_ii=k_ii,
        j_integral=energy,
        extraction_method="interaction_domain_integral",
        relative_path_tolerance=relative_path_tolerance,
        metadata={
            "auxiliary_K": 1.0,
            "normalization": "K = E_prime * I / 2",
            "material": material.summary(),
            "scope": (
                "straight crack; homogeneous isotropic linear elasticity; "
                "quasi-static; no body force or applied crack-face traction"
            ),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "InteractionIntegralSamples2D",
    "WilliamsField2D",
    "interaction_integral",
    "interaction_integral_report",
]
