"""Power-law creep material-point relations and closed-form checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .plasticity import deviatoric, von_mises


@dataclass(frozen=True)
class CreepHistory:
    """Integrated piecewise-constant stress history."""

    time: np.ndarray
    equivalent_creep_strain: np.ndarray
    creep_strain: np.ndarray | None = None

    @property
    def final_equivalent_strain(self) -> float:
        return float(self.equivalent_creep_strain[-1])

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "creep_history",
            "time": self.time.tolist(),
            "equivalent_creep_strain": self.equivalent_creep_strain.tolist(),
            "tensor_history": self.creep_strain is not None,
        }


@dataclass(frozen=True)
class PowerLawCreep:
    """Mises time-hardening creep law.

    The equivalent creep rate is

    ``epsilon_dot = A (q / sigma_ref)^n (t / time_ref)^m``.

    With this normalized form ``A`` has units of inverse time.  Set
    ``reference_stress=1`` and use a consistent stress unit to reproduce the
    conventional dimensionful ``A q^n t^m`` notation.
    """

    coefficient: float
    stress_exponent: float
    time_exponent: float = 0.0
    reference_stress: float = 1.0
    reference_time: float = 1.0
    name: str = "Mises power-law creep"

    def __post_init__(self) -> None:
        values = (
            self.coefficient,
            self.stress_exponent,
            self.time_exponent,
            self.reference_stress,
            self.reference_time,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("PowerLawCreep parameters must be finite.")
        if self.coefficient < 0.0:
            raise ValueError("PowerLawCreep.coefficient must be nonnegative.")
        if self.stress_exponent <= 0.0:
            raise ValueError("PowerLawCreep.stress_exponent must be positive.")
        if self.time_exponent <= -1.0:
            raise ValueError("PowerLawCreep.time_exponent must be greater than -1.")
        if self.reference_stress <= 0.0 or self.reference_time <= 0.0:
            raise ValueError("PowerLawCreep reference scales must be positive.")

    def equivalent_rate(self, equivalent_stress: float, time: float) -> float:
        q = abs(float(equivalent_stress))
        selected_time = float(time)
        if selected_time < 0.0:
            raise ValueError("creep time must be nonnegative.")
        if selected_time == 0.0 and self.time_exponent < 0.0:
            return float("inf")
        time_factor = (
            1.0
            if self.time_exponent == 0.0
            else (selected_time / self.reference_time) ** self.time_exponent
        )
        return float(
            self.coefficient
            * (q / self.reference_stress) ** self.stress_exponent
            * time_factor
        )

    def constant_stress_strain(
        self,
        equivalent_stress: float,
        time: float,
    ) -> float:
        """Return the exact equivalent creep strain from zero to ``time``."""

        selected_time = float(time)
        if selected_time < 0.0:
            raise ValueError("creep time must be nonnegative.")
        q_factor = (
            abs(float(equivalent_stress)) / self.reference_stress
        ) ** self.stress_exponent
        normalized_time = selected_time / self.reference_time
        return float(
            self.coefficient
            * self.reference_time
            * q_factor
            * normalized_time ** (self.time_exponent + 1.0)
            / (self.time_exponent + 1.0)
        )

    def constant_stress_increment(
        self,
        equivalent_stress: float,
        time_start: float,
        time_end: float,
    ) -> float:
        if time_end < time_start:
            raise ValueError("time_end must be greater than or equal to time_start.")
        return self.constant_stress_strain(
            equivalent_stress,
            time_end,
        ) - self.constant_stress_strain(equivalent_stress, time_start)

    def tensor_increment(
        self,
        stress,
        time_start: float,
        time_end: float,
    ) -> np.ndarray:
        """Return an associative Mises creep-strain increment tensor."""

        selected = np.asarray(stress, dtype=float)
        q = von_mises(selected)
        if q == 0.0:
            return np.zeros((3, 3))
        equivalent_increment = self.constant_stress_increment(
            q,
            time_start,
            time_end,
        )
        return equivalent_increment * 1.5 * deviatoric(selected) / q

    def relaxation_stress(
        self,
        *,
        initial_stress: float,
        young: float,
        time: float,
    ) -> float:
        """Closed-form stress for a constant-total-strain relaxation test."""

        sigma0 = float(initial_stress)
        modulus = float(young)
        selected_time = float(time)
        if sigma0 <= 0.0 or modulus <= 0.0:
            raise ValueError("initial_stress and young must be positive.")
        if selected_time < 0.0:
            raise ValueError("time must be nonnegative.")
        exponent = self.stress_exponent
        if np.isclose(exponent, 1.0):
            power = (
                self.coefficient
                * modulus
                * self.reference_time
                / self.reference_stress
                * (selected_time / self.reference_time)
                ** (self.time_exponent + 1.0)
                / (self.time_exponent + 1.0)
            )
            return float(sigma0 * np.exp(-power))
        accumulated = (
            self.coefficient
            * modulus
            * (exponent - 1.0)
            * self.reference_time
            / self.reference_stress**exponent
            * (selected_time / self.reference_time)
            ** (self.time_exponent + 1.0)
            / (self.time_exponent + 1.0)
        )
        return float(
            (sigma0 ** (1.0 - exponent) + accumulated)
            ** (1.0 / (1.0 - exponent))
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "mises_power_law_time_hardening_creep",
            "coefficient": self.coefficient,
            "stress_exponent": self.stress_exponent,
            "time_exponent": self.time_exponent,
            "reference_stress": self.reference_stress,
            "reference_time": self.reference_time,
            "maturity": "material_point_verified",
            "fem_quadrature_driver": False,
        }


def integrate_stress_history(
    law: PowerLawCreep,
    times,
    interval_stresses,
) -> CreepHistory:
    """Integrate a piecewise-constant scalar or tensor stress history.

    ``times`` contains interval boundaries and ``interval_stresses`` contains
    one stress value per interval. Each increment uses the law's exact
    time-hardening integral, avoiding a hidden forward-Euler approximation.
    This is a verified material-point driver, not yet a global FE creep step.
    """

    selected_times = np.asarray(times, dtype=float)
    selected_stresses = np.asarray(interval_stresses, dtype=float)
    if selected_times.ndim != 1 or selected_times.size < 2:
        raise ValueError("times must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(selected_times)):
        raise ValueError("times must be finite.")
    if selected_times[0] < 0.0 or np.any(np.diff(selected_times) <= 0.0):
        raise ValueError("times must be nonnegative and strictly increasing.")
    interval_count = selected_times.size - 1
    scalar_history = selected_stresses.ndim == 1
    tensor_history = (
        selected_stresses.ndim == 3
        and selected_stresses.shape[1:] == (3, 3)
    )
    if not scalar_history and not tensor_history:
        raise ValueError(
            "interval_stresses must have shape (intervals,) or (intervals, 3, 3)."
        )
    if selected_stresses.shape[0] != interval_count:
        raise ValueError("Provide exactly one stress value per time interval.")
    if not np.all(np.isfinite(selected_stresses)):
        raise ValueError("interval_stresses must be finite.")

    equivalent = np.zeros(selected_times.size)
    tensors = (
        np.zeros((selected_times.size, 3, 3))
        if tensor_history
        else None
    )
    for index in range(interval_count):
        start = float(selected_times[index])
        end = float(selected_times[index + 1])
        if scalar_history:
            increment = law.constant_stress_increment(
                float(selected_stresses[index]),
                start,
                end,
            )
        else:
            stress = selected_stresses[index]
            q = von_mises(stress)
            increment = law.constant_stress_increment(q, start, end)
            tensors[index + 1] = (
                tensors[index] + law.tensor_increment(stress, start, end)
            )
        equivalent[index + 1] = equivalent[index] + increment
    return CreepHistory(
        time=selected_times.copy(),
        equivalent_creep_strain=equivalent,
        creep_strain=tensors,
    )
