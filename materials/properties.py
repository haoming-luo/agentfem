"""Material-property containers.

These dataclasses store parameters and derived scalar quantities. Constitutive
relations such as stress-strain laws live in ``agentfem.constitutive``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class ElasticIsotropicProperties:
    """Isotropic linear-elastic material properties."""

    name: str
    young: float
    density: float
    poisson: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.young)) or self.young <= 0.0:
            raise ValueError("ElasticIsotropicProperties.young must be positive.")
        if not isfinite(float(self.density)) or self.density <= 0.0:
            raise ValueError("ElasticIsotropicProperties.density must be positive.")
        if not isfinite(float(self.poisson)) or not (-1.0 < self.poisson < 0.5):
            raise ValueError(
                "ElasticIsotropicProperties.poisson must satisfy -1 < nu < 0.5."
            )

    @property
    def lambda_(self) -> float:
        return self.young * self.poisson / (
            (1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)
        )

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def pressure_wave_speed(self) -> float:
        numerator = (1.0 - self.poisson) * self.young
        denominator = (1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)
        return float(np.sqrt((numerator / denominator) / self.density))

    @property
    def shear_wave_speed(self) -> float:
        return float(np.sqrt(self.mu / self.density))

    def as_dict(self) -> dict[str, float | str]:
        """Return constants and derived elastic quantities."""

        return {
            "name": self.name,
            "model": "isotropic_linear_elastic",
            "young": self.young,
            "poisson": self.poisson,
            "density": self.density,
            "lambda": self.lambda_,
            "mu": self.mu,
            "pressure_wave_speed": self.pressure_wave_speed,
            "shear_wave_speed": self.shear_wave_speed,
        }

    def summary(self) -> str:
        """Return a compact human-readable material-property summary."""

        return (
            f"{self.name}: isotropic linear elastic properties, "
            f"E={self.young:.6e}, nu={self.poisson:.6g}, rho={self.density:.6e}"
        )


@dataclass(frozen=True)
class ElasticAnisotropic2DProperties:
    """2D linear-elastic properties using engineering-strain Voigt notation."""

    name: str
    stiffness_voigt: np.ndarray
    density: float
    model: str = "anisotropic_linear_elastic_2d"

    def __post_init__(self) -> None:
        C = np.asarray(self.stiffness_voigt, dtype=float)
        if C.shape != (3, 3):
            raise ValueError("2D anisotropic stiffness_voigt must be a 3x3 matrix.")
        if not np.all(np.isfinite(C)):
            raise ValueError("2D anisotropic stiffness_voigt must be finite.")
        if not np.allclose(C, C.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("2D anisotropic stiffness_voigt must be symmetric.")
        if np.min(np.linalg.eigvalsh(C)) <= 0.0:
            raise ValueError(
                "2D anisotropic stiffness_voigt must be positive definite."
            )
        if not isfinite(float(self.density)) or self.density <= 0.0:
            raise ValueError("ElasticAnisotropic2DProperties.density must be positive.")
        object.__setattr__(self, "stiffness_voigt", C)

    @property
    def pressure_wave_speed(self) -> float:
        return float(np.sqrt(np.max(np.linalg.eigvalsh(self.stiffness_voigt)) / self.density))

    @property
    def shear_wave_speed(self) -> float:
        return float(np.sqrt(self.stiffness_voigt[2, 2] / self.density))

    def as_dict(self) -> dict[str, object]:
        """Return constants and scalar wave-speed estimates."""

        return {
            "name": self.name,
            "model": self.model,
            "density": self.density,
            "stiffness_voigt": self.stiffness_voigt.tolist(),
            "pressure_wave_speed_estimate": self.pressure_wave_speed,
            "shear_wave_speed_estimate": self.shear_wave_speed,
        }

    def summary(self) -> str:
        """Return a compact human-readable material-property summary."""

        return f"{self.name}: {self.model} properties, rho={self.density:.6e}"
