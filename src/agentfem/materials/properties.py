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
class ThermoElasticIsotropicProperties(ElasticIsotropicProperties):
    """Isotropic thermoelastic and heat-conduction properties.

    The constants intentionally live in one inspectable record because
    sequential heat-transfer/stress workflows consume the same material.
    Temperature-dependent tables can later implement the same property
    protocol without changing the operators.
    """

    thermal_expansion: float = 0.0
    conductivity: float = 0.0
    specific_heat: float = 0.0
    reference_temperature: float = 293.15

    def __post_init__(self) -> None:
        super().__post_init__()
        values = {
            "thermal_expansion": self.thermal_expansion,
            "conductivity": self.conductivity,
            "specific_heat": self.specific_heat,
            "reference_temperature": self.reference_temperature,
        }
        if not all(isfinite(float(value)) for value in values.values()):
            raise ValueError("Thermoelastic properties must be finite.")
        if self.thermal_expansion < 0.0:
            raise ValueError("thermal_expansion must be nonnegative.")
        if self.conductivity <= 0.0:
            raise ValueError("conductivity must be positive.")
        if self.specific_heat <= 0.0:
            raise ValueError("specific_heat must be positive.")
        if self.reference_temperature <= 0.0:
            raise ValueError("reference_temperature must be positive in kelvin.")

    @property
    def volumetric_heat_capacity(self) -> float:
        return self.density * self.specific_heat

    def as_dict(self) -> dict[str, float | str]:
        result = super().as_dict()
        result.update(
            {
                "model": "isotropic_linear_thermoelastic",
                "thermal_expansion": self.thermal_expansion,
                "conductivity": self.conductivity,
                "specific_heat": self.specific_heat,
                "reference_temperature": self.reference_temperature,
                "volumetric_heat_capacity": self.volumetric_heat_capacity,
            }
        )
        return result

    def summary(self) -> str:
        return (
            f"{self.name}: isotropic thermoelastic properties, "
            f"E={self.young:.6e}, nu={self.poisson:.6g}, "
            f"alpha={self.thermal_expansion:.6e}, k={self.conductivity:.6e}"
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
