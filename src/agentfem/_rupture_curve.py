# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reviewed rupture tables; no fitting, alloy defaults or constitutive damage."""

from dataclasses import dataclass, field, replace
from math import isclose
from numbers import Real

import numpy as np


_STRESS_SCALES = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
                  "kg/(m*s^2)": 1.0, "tonne/(mm*s^2)": 1e6}
_TIME_SCALES = {"s": 1.0, "min": 60.0, "h": 3600.0}


def _positive(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive real scalar.")
    return float(value)


@dataclass(frozen=True)
class TabulatedRuptureCurve:
    """Isothermal tensile rupture life, interpolated in log stress/log time.

    Inputs are a reviewed curve, not raw replicate observations. Stress nodes
    must increase and lives must not increase. Queries outside the stress
    interval or at another temperature are rejected. Temperature comparisons
    admit only relative 1e-12 roundoff; no temperature interpolation occurs.
    The declared stress measure is a caller-owned engineering assumption, not
    a multiaxial transformation. This curve is not a design allowable.
    """

    stresses: tuple[float, ...]
    rupture_times: tuple[float, ...]
    temperature: float
    stress_unit: str
    time_unit: str
    stress_measure: str
    material_condition: str
    failure_definition: str
    source: str
    _conversions: tuple = field(default=(), init=False, repr=False)

    def __post_init__(self):
        stresses = tuple(_positive(v, "Stress node") for v in self.stresses)
        times = tuple(_positive(v, "Rupture time") for v in self.rupture_times)
        if len(stresses) < 2 or len(stresses) != len(times):
            raise ValueError(
                "Provide at least two aligned stress and rupture-time nodes."
            )
        if np.any(np.diff(stresses) <= 0) or np.any(np.diff(times) > 0):
            raise ValueError(
                "Stress nodes must strictly increase; rupture times must not increase."
            )
        for name in (
            "stress_unit",
            "time_unit",
            "stress_measure",
            "material_condition",
            "failure_definition",
            "source",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be explicitly declared.")
        object.__setattr__(self, "stresses", stresses)
        object.__setattr__(self, "rupture_times", times)
        object.__setattr__(
            self, "temperature", _positive(self.temperature, "Temperature in kelvin")
        )

    def __call__(self, stress, temperature):
        sigma = _positive(stress, "Tensile stress")
        kelvin = _positive(temperature, "Temperature in kelvin")
        if not isclose(kelvin, self.temperature, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "Rupture curve is isothermal; temperature interpolation is disabled."
            )
        if not self.stresses[0] <= sigma <= self.stresses[-1]:
            raise ValueError(
                "Stress lies outside the rupture curve; extrapolation is disabled."
            )
        # Preserve supplied node values exactly, including endpoint lives.
        if sigma in self.stresses:
            return self.rupture_times[self.stresses.index(sigma)]
        result = float(
            np.exp(
                np.interp(
                    np.log(sigma), np.log(self.stresses), np.log(self.rupture_times)
                )
            )
        )
        if not np.isfinite(result) or result <= 0:
            raise ValueError(
                "Interpolated rupture time is outside the representable range."
            )
        return result

    def in_units(self, *, stress_unit, time_unit):
        """Return an explicitly converted table without changing its physics.

        Stress units: Pa, kPa, MPa, GPa, kg/(m*s^2), tonne/(mm*s^2).
        Time units: s, min, h. Temperature remains kelvin. Conversion evidence
        accompanies the new curve; simulation histories are never relabelled.
        """
        factors = []
        for label, source, target, scales in (
            ("stress", self.stress_unit, stress_unit, _STRESS_SCALES),
            ("time", self.time_unit, time_unit, _TIME_SCALES),
        ):
            if not isinstance(target, str) or source not in scales or target not in scales:
                raise ValueError(f"Unsupported {label} unit conversion: {source!r} to {target!r}.")
            factors.append(scales[source] / scales[target])
        converted = replace(
            self, stresses=tuple(v * factors[0] for v in self.stresses),
            rupture_times=tuple(v * factors[1] for v in self.rupture_times),
            stress_unit=stress_unit, time_unit=time_unit,
        )
        record = (self.stress_unit, stress_unit, factors[0],
                  self.time_unit, time_unit, factors[1])
        object.__setattr__(converted, "_conversions", self._conversions + (record,))
        return converted

    def validate_usage(self, *, stress_unit, time_unit, stress_measure):
        for name, value in (
            ("stress_unit", stress_unit),
            ("time_unit", time_unit),
            ("stress_measure", stress_measure),
        ):
            if value != getattr(self, name):
                raise ValueError(
                    f"Rupture curve {name} differs from the assessment; no implicit conversion."
                )

    def as_dict(self):
        record = dict(
            kind="tabulated_isothermal_rupture_curve",
            interpolation="log_log",
            stresses=list(self.stresses),
            rupture_times=list(self.rupture_times),
            temperature=self.temperature,
            temperature_unit="K",
            stress_unit=self.stress_unit,
            time_unit=self.time_unit,
            stress_measure=self.stress_measure,
            material_condition=self.material_condition,
            failure_definition=self.failure_definition,
            source=self.source,
            extrapolation=False,
            engineering_qualified=False,
        )
        if self._conversions:
            record["unit_conversions"] = [dict(zip(
                ("from_stress_unit", "to_stress_unit", "stress_factor",
                 "from_time_unit", "to_time_unit", "time_factor"), item
            )) for item in self._conversions]
        return record
