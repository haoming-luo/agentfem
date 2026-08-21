"""Material-property containers.

These dataclasses store parameters and derived scalar quantities. Constitutive
relations such as stress-strain laws live in ``agentfem.constitutive``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class TemperaturePropertyTable:
    """One material property tabulated against absolute temperature.

    Numerical evaluation rejects out-of-range temperatures by default.
    ``extrapolation="constant"`` is an explicit endpoint-holding policy and
    is required when the table is turned into a bounded UFL expression.
    """

    temperatures: np.ndarray
    values: np.ndarray
    name: str = "property"
    unit: str | None = None
    interpolation: str = "linear"
    extrapolation: str = "error"

    def __post_init__(self) -> None:
        temperatures = np.asarray(self.temperatures, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if temperatures.ndim != 1 or values.ndim != 1:
            raise ValueError("TemperaturePropertyTable inputs must be one-dimensional.")
        if temperatures.size < 2 or values.size != temperatures.size:
            raise ValueError(
                "TemperaturePropertyTable requires equally sized arrays with at least two points."
            )
        if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(values)):
            raise ValueError("TemperaturePropertyTable values must be finite.")
        if np.any(temperatures <= 0.0) or np.any(np.diff(temperatures) <= 0.0):
            raise ValueError(
                "TemperaturePropertyTable temperatures must be positive kelvin and strictly increasing."
            )
        interpolation = str(self.interpolation).lower().replace("-", "_")
        extrapolation = str(self.extrapolation).lower().replace("-", "_")
        if interpolation != "linear":
            raise ValueError("TemperaturePropertyTable currently supports linear interpolation.")
        if extrapolation not in {"error", "constant"}:
            raise ValueError(
                "TemperaturePropertyTable extrapolation must be error or constant."
            )
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "extrapolation", extrapolation)

    def __call__(self, temperature):
        selected = np.asarray(temperature, dtype=float)
        lower = float(self.temperatures[0])
        upper = float(self.temperatures[-1])
        if self.extrapolation == "error" and (
            np.any(selected < lower) or np.any(selected > upper)
        ):
            raise ValueError(
                f"Temperature property {self.name!r} covers [{lower:g}, {upper:g}] K."
            )
        result = np.interp(selected, self.temperatures, self.values)
        return float(result) if result.ndim == 0 else result

    def ufl_value(self, temperature):
        """Return a piecewise-linear UFL coefficient with explicit bounds."""

        if self.extrapolation != "constant":
            raise ValueError(
                f"Temperature property {self.name!r} uses extrapolation='error'. "
                "Validate a finite-element temperature range first and select "
                "extrapolation='constant' explicitly for a bounded UFL coefficient."
            )
        import ufl

        expression = float(self.values[-1])
        for index in range(len(self.temperatures) - 2, -1, -1):
            left_t = float(self.temperatures[index])
            right_t = float(self.temperatures[index + 1])
            left_v = float(self.values[index])
            right_v = float(self.values[index + 1])
            segment = left_v + (right_v - left_v) * (
                (temperature - left_t) / (right_t - left_t)
            )
            expression = ufl.conditional(
                ufl.lt(temperature, right_t), segment, expression
            )
        return ufl.conditional(
            ufl.le(temperature, float(self.temperatures[0])),
            float(self.values[0]),
            expression,
        )

    def integral(self, temperature):
        """Evaluate a continuous primitive, zero at the first table point."""

        selected = np.asarray(temperature, dtype=float)
        lower = float(self.temperatures[0])
        upper = float(self.temperatures[-1])
        if self.extrapolation == "error" and (
            np.any(selected < lower) or np.any(selected > upper)
        ):
            raise ValueError(
                f"Temperature property {self.name!r} covers [{lower:g}, {upper:g}] K."
            )
        cumulative = np.zeros_like(self.temperatures)
        cumulative[1:] = np.cumsum(
            0.5
            * (self.values[:-1] + self.values[1:])
            * np.diff(self.temperatures)
        )
        clipped = np.clip(selected, lower, upper)
        indices = np.searchsorted(self.temperatures, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(self.temperatures) - 2)
        left_t = self.temperatures[indices]
        left_v = self.values[indices]
        slopes = np.diff(self.values) / np.diff(self.temperatures)
        delta = clipped - left_t
        result = cumulative[indices] + left_v * delta + 0.5 * slopes[indices] * delta**2
        result = np.where(selected < lower, self.values[0] * (selected - lower), result)
        result = np.where(
            selected > upper,
            cumulative[-1] + self.values[-1] * (selected - upper),
            result,
        )
        return float(result) if result.ndim == 0 else result

    def ufl_integral(self, temperature):
        """Return a continuous UFL primitive of the tabulated property.

        The primitive is zero at the first tabulated temperature.  Constant
        endpoint extrapolation is integrated consistently outside the table.
        This is useful for conservative state functions such as sensible
        enthalpy, where ``dH/dT`` must recover the tabulated heat capacity.
        """

        if self.extrapolation != "constant":
            raise ValueError(
                f"Temperature property {self.name!r} uses extrapolation='error'. "
                "Select extrapolation='constant' explicitly before building "
                "a bounded UFL primitive."
            )
        import ufl

        temperatures = np.asarray(self.temperatures, dtype=float)
        values = np.asarray(self.values, dtype=float)
        cumulative = np.zeros_like(temperatures)
        cumulative[1:] = np.cumsum(
            0.5 * (values[:-1] + values[1:]) * np.diff(temperatures)
        )
        expression = float(cumulative[-1]) + float(values[-1]) * (
            temperature - float(temperatures[-1])
        )
        for index in range(len(temperatures) - 2, -1, -1):
            left_t = float(temperatures[index])
            right_t = float(temperatures[index + 1])
            left_v = float(values[index])
            slope = float(values[index + 1] - values[index]) / (
                right_t - left_t
            )
            delta = temperature - left_t
            segment = float(cumulative[index]) + left_v * delta + 0.5 * slope * delta**2
            expression = ufl.conditional(
                ufl.lt(temperature, right_t), segment, expression
            )
        return ufl.conditional(
            ufl.le(temperature, float(temperatures[0])),
            float(values[0]) * (temperature - float(temperatures[0])),
            expression,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "temperature_property_table",
            "name": self.name,
            "unit": self.unit,
            "temperature_unit": "K",
            "temperatures": self.temperatures.tolist(),
            "values": self.values.tolist(),
            "interpolation": self.interpolation,
            "extrapolation": self.extrapolation,
        }


def temperature_property(temperatures, values, **kwargs) -> TemperaturePropertyTable:
    """Create an inspectable temperature-dependent material property."""

    return TemperaturePropertyTable(temperatures, values, **kwargs)


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
class TemperatureDependentThermoElasticProperties:
    """Isotropic thermoelastic properties containing constants or tables.

    Density and reference temperature remain scalar in this first contract.
    ``at_temperature`` resolves a conventional constant property record, while
    ``coefficient`` supplies a known temperature field to sequential mechanics.
    """

    name: str
    young: float | TemperaturePropertyTable
    density: float
    poisson: float | TemperaturePropertyTable
    thermal_expansion: float | TemperaturePropertyTable
    conductivity: float | TemperaturePropertyTable
    specific_heat: float | TemperaturePropertyTable
    reference_temperature: float = 293.15

    def __post_init__(self) -> None:
        if not isfinite(float(self.density)) or self.density <= 0.0:
            raise ValueError("density must be finite and positive.")
        if not isfinite(float(self.reference_temperature)) or self.reference_temperature <= 0.0:
            raise ValueError("reference_temperature must be positive in kelvin.")
        checks = {
            "young": (0.0, None),
            "poisson": (-1.0, 0.5),
            "thermal_expansion": (0.0, None),
            "conductivity": (0.0, None),
            "specific_heat": (0.0, None),
        }
        for property_name, (minimum, maximum) in checks.items():
            item = getattr(self, property_name)
            values = item.values if isinstance(item, TemperaturePropertyTable) else np.asarray([item])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{property_name} values must be finite.")
            if property_name == "thermal_expansion":
                valid_lower = np.all(values >= minimum)
            else:
                valid_lower = np.all(values > minimum)
            if not valid_lower or (maximum is not None and not np.all(values < maximum)):
                raise ValueError(f"Invalid temperature-dependent {property_name} values.")

    def coefficient(self, name: str, temperature):
        item = getattr(self, name)
        if isinstance(item, TemperaturePropertyTable):
            if isinstance(temperature, (int, float, np.ndarray, np.number)):
                return item(temperature)
            return item.ufl_value(temperature)
        return float(item)

    @property
    def state_dependent_heat_transfer(self) -> bool:
        """Whether heat transfer requires a temperature-dependent residual."""

        return isinstance(self.conductivity, TemperaturePropertyTable) or isinstance(
            self.specific_heat, TemperaturePropertyTable
        )

    def conductivity_at(self, temperature):
        """Return the thermal conductivity at a scalar or field temperature."""

        return self.coefficient("conductivity", temperature)

    def volumetric_heat_capacity_at(self, temperature):
        """Return ``rho*c_p(T)`` at a scalar or field temperature."""

        return self.density * self.coefficient("specific_heat", temperature)

    def volumetric_enthalpy(self, temperature):
        """Return a sensible-enthalpy primitive with ``dh/dT = rho*c_p(T)``."""

        if isinstance(self.specific_heat, TemperaturePropertyTable):
            if isinstance(temperature, (int, float, np.ndarray, np.number)):
                return self.density * self.specific_heat.integral(temperature)
            return self.density * self.specific_heat.ufl_integral(temperature)
        return self.density * float(self.specific_heat) * temperature

    def at_temperature(self, temperature: float) -> ThermoElasticIsotropicProperties:
        selected = float(temperature)
        return ThermoElasticIsotropicProperties(
            name=self.name,
            young=self.coefficient("young", selected),
            density=self.density,
            poisson=self.coefficient("poisson", selected),
            thermal_expansion=self.coefficient("thermal_expansion", selected),
            conductivity=self.coefficient("conductivity", selected),
            specific_heat=self.coefficient("specific_heat", selected),
            reference_temperature=self.reference_temperature,
        )

    def as_dict(self) -> dict[str, object]:
        def encoded(value):
            return value.as_dict() if hasattr(value, "as_dict") else float(value)

        return {
            "name": self.name,
            "model": "temperature_dependent_isotropic_linear_thermoelastic",
            "density": self.density,
            "reference_temperature": self.reference_temperature,
            **{
                name: encoded(getattr(self, name))
                for name in (
                    "young",
                    "poisson",
                    "thermal_expansion",
                    "conductivity",
                    "specific_heat",
                )
            },
        }

    def summary(self) -> str:
        tabulated = [
            name
            for name in (
                "young",
                "poisson",
                "thermal_expansion",
                "conductivity",
                "specific_heat",
            )
            if isinstance(getattr(self, name), TemperaturePropertyTable)
        ]
        return f"{self.name}: temperature-dependent thermoelastic ({', '.join(tabulated)})"


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
