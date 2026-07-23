"""Linear-elastic material and strain/stress helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl


def strain(displacement):
    """Small-strain tensor, ``sym(grad(u))``."""

    return ufl.sym(ufl.grad(displacement))


def engineering_strain_voigt_2d(displacement):
    """2D engineering-strain Voigt vector: [eps_xx, eps_yy, gamma_xy]."""

    eps = strain(displacement)
    return ufl.as_vector([eps[0, 0], eps[1, 1], 2.0 * eps[0, 1]])


def stress_voigt_to_tensor_2d(stress_voigt):
    """Convert [sig_xx, sig_yy, sig_xy] to a 2D symmetric stress tensor."""

    return ufl.as_tensor(
        [
            [stress_voigt[0], stress_voigt[2]],
            [stress_voigt[2], stress_voigt[1]],
        ]
    )


def isotropic_pressure_wave_speed(young: float, poisson: float, density: float) -> float:
    """Longitudinal wave speed for a 3D isotropic elastic solid."""

    numerator = (1.0 - poisson) * young
    denominator = (1.0 + poisson) * (1.0 - 2.0 * poisson)
    return float(np.sqrt((numerator / denominator) / density))


def isotropic_shear_wave_speed(young: float, poisson: float, density: float) -> float:
    """Shear wave speed for an isotropic elastic solid."""

    mu = young / (2.0 * (1.0 + poisson))
    return float(np.sqrt(mu / density))


def estimate_elastic_wave_speeds(material) -> tuple[float, float]:
    """Return approximate ``(pressure_speed, shear_speed)`` for a material.

    For anisotropic materials this is a conservative scalar estimate, not a
    direction-dependent Christoffel analysis.
    """

    if isinstance(material, IsotropicElasticMaterial):
        return material.pressure_wave_speed, material.shear_wave_speed
    if isinstance(material, AnisotropicElasticMaterial2D):
        pressure = float(
            np.sqrt(np.max(np.linalg.eigvalsh(material.stiffness_voigt)) / material.density)
        )
        shear = float(np.sqrt(material.stiffness_voigt[2, 2] / material.density))
        return pressure, shear
    if hasattr(material, "pressure_wave_speed"):
        pressure = float(material.pressure_wave_speed)
        shear = float(getattr(material, "shear_wave_speed", pressure / np.sqrt(3.0)))
        return pressure, shear
    raise TypeError("material does not provide enough elastic data to estimate wave speeds.")


@dataclass(frozen=True)
class IsotropicElasticMaterial:
    """Small-strain isotropic linear-elastic material."""

    name: str
    young: float
    density: float
    poisson: float

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lambda_(self) -> float:
        return self.young * self.poisson / (
            (1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)
        )

    @property
    def pressure_wave_speed(self) -> float:
        return isotropic_pressure_wave_speed(self.young, self.poisson, self.density)

    @property
    def shear_wave_speed(self) -> float:
        return isotropic_shear_wave_speed(self.young, self.poisson, self.density)

    def sigma(self, displacement):
        eps = strain(displacement)
        return self.lambda_ * ufl.tr(eps) * ufl.Identity(len(displacement)) + 2.0 * self.mu * eps


@dataclass(frozen=True)
class AnisotropicElasticMaterial2D:
    """2D linear-elastic material using engineering-strain Voigt notation.

    ``stiffness_voigt`` maps [eps_xx, eps_yy, gamma_xy] to
    [sig_xx, sig_yy, sig_xy], where gamma_xy = 2 * eps_xy.
    """

    name: str
    stiffness_voigt: np.ndarray
    density: float

    def __post_init__(self) -> None:
        C = np.asarray(self.stiffness_voigt, dtype=float)
        if C.shape != (3, 3):
            raise ValueError("2D anisotropic stiffness_voigt must be a 3x3 matrix.")
        object.__setattr__(self, "stiffness_voigt", C)

    @property
    def pressure_wave_speed(self) -> float:
        return estimate_elastic_wave_speeds(self)[0]

    @property
    def shear_wave_speed(self) -> float:
        return estimate_elastic_wave_speeds(self)[1]

    def sigma(self, displacement):
        strain_voigt = engineering_strain_voigt_2d(displacement)
        stress_voigt = ufl.dot(ufl.as_matrix(self.stiffness_voigt.tolist()), strain_voigt)
        return stress_voigt_to_tensor_2d(stress_voigt)


def isotropic_elastic(
    *,
    young: float,
    density: float,
    poisson: float,
    name: str = "isotropic elastic",
) -> IsotropicElasticMaterial:
    """Create an isotropic linear-elastic material."""

    return IsotropicElasticMaterial(name=name, young=young, density=density, poisson=poisson)


def anisotropic_elastic_2d(
    *,
    stiffness_voigt,
    density: float,
    name: str = "anisotropic elastic 2D",
) -> AnisotropicElasticMaterial2D:
    """Create a 2D anisotropic linear-elastic material."""

    return AnisotropicElasticMaterial2D(
        name=name,
        stiffness_voigt=np.asarray(stiffness_voigt, dtype=float),
        density=density,
    )


def orthotropic_plane_stress_2d(
    *,
    ex: float,
    ey: float,
    nuxy: float,
    gxy: float,
    density: float,
    name: str = "orthotropic plane-stress elastic 2D",
) -> AnisotropicElasticMaterial2D:
    """Create a 2D orthotropic plane-stress material."""

    nuyx = nuxy * ey / ex
    denom = 1.0 - nuxy * nuyx
    C = np.array(
        [
            [ex / denom, nuyx * ex / denom, 0.0],
            [nuxy * ey / denom, ey / denom, 0.0],
            [0.0, 0.0, gxy],
        ],
        dtype=float,
    )
    return anisotropic_elastic_2d(stiffness_voigt=C, density=density, name=name)
