# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Calibrated scalar strain-life assessment; never a stiffness-damage law."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log

import numpy as np

from .fatigue import rainflow_cycles


@dataclass(frozen=True)
class CoffinMansonBasquinCurve:
    """Total mechanical strain amplitude versus full cycles to failure.

    ``eps_a = sigma_f / E * (2N)**b + eps_f * (2N)**c``. Both stress
    coefficients use ``stress_unit``; strain is dimensionless, not percent.
    Calibration bounds are mandatory. No silent extrapolation or mean-stress
    correction is applied. Parameters must fit the intended temperature,
    loading mode and material condition; this object does not certify them.
    """

    young: float
    fatigue_strength_coefficient: float
    fatigue_strength_exponent: float
    fatigue_ductility_coefficient: float
    fatigue_ductility_exponent: float
    calibrated_cycles: tuple[float, float]
    stress_unit: str
    source: str

    def __post_init__(self):
        for name in ("young", "fatigue_strength_coefficient", "fatigue_ductility_coefficient"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        for name in ("fatigue_strength_exponent", "fatigue_ductility_exponent"):
            value = float(getattr(self, name))
            if not isfinite(value) or value >= 0.0:
                raise ValueError(f"{name} must be finite and negative.")
            object.__setattr__(self, name, value)
        bounds = tuple(float(x) for x in self.calibrated_cycles)
        if len(bounds) != 2 or not all(isfinite(x) and x > 0.0 for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("calibrated_cycles must be two increasing positive finite full-cycle counts.")
        object.__setattr__(self, "calibrated_cycles", bounds)
        for name in ("stress_unit", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
            object.__setattr__(self, name, value.strip())

    def _log_amplitude(self, log_cycles: float) -> float:
        log_reversals = log(2.0) + log_cycles
        return float(np.logaddexp(
            log(self.fatigue_strength_coefficient) - log(self.young)
            + self.fatigue_strength_exponent * log_reversals,
            log(self.fatigue_ductility_coefficient)
            + self.fatigue_ductility_exponent * log_reversals,
        ))

    def strain_amplitude(self, cycles_to_failure: float) -> float:
        cycles = float(cycles_to_failure)
        low, high = self.calibrated_cycles
        if not isfinite(cycles) or not low <= cycles <= high:
            raise ValueError("cycles_to_failure lies outside calibrated_cycles.")
        value = exp(self._log_amplitude(log(cycles)))
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Strain amplitude is not representable for this calibration.")
        return value

    def cycles_to_failure(self, strain_amplitude: float) -> float:
        amplitude = float(strain_amplitude)
        if not isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("strain_amplitude must be finite and positive.")
        low, high = self.calibrated_cycles
        smallest, largest = self.strain_amplitude(high), self.strain_amplitude(low)
        if not smallest <= amplitude <= largest:
            raise ValueError("strain_amplitude lies outside the calibrated range; extrapolation is disabled.")
        if amplitude == largest:
            return low
        if amplitude == smallest:
            return high
        left, right, target = log(low), log(high), log(amplitude)
        # Strictly decreasing sum of two powers; bracketed log-space inversion
        # cannot jump into negative life or silently converge to another root.
        for _ in range(100):
            middle = 0.5 * (left + right)
            if self._log_amplitude(middle) > target:
                left = middle
            else:
                right = middle
        return min(high, max(low, exp(0.5 * (left + right))))

    def as_dict(self):
        return dict(
            model="coffin_manson_basquin", maturity="postprocessor",
            young=self.young, fatigue_strength_coefficient=self.fatigue_strength_coefficient,
            fatigue_strength_exponent=self.fatigue_strength_exponent,
            fatigue_ductility_coefficient=self.fatigue_ductility_coefficient,
            fatigue_ductility_exponent=self.fatigue_ductility_exponent,
            calibrated_cycles=list(self.calibrated_cycles), stress_unit=self.stress_unit,
            strain_unit="1", life_unit="full_cycles", source=self.source,
            extrapolation="error", mean_stress_correction="none",
        )


@dataclass(frozen=True)
class StrainCycle:
    """Rainflow cycle of one signed mechanical strain component."""

    strain_range: float
    mean_strain: float
    count: float

    def __post_init__(self):
        if not isfinite(self.strain_range) or self.strain_range <= 0.0:
            raise ValueError("strain_range must be finite and positive.")
        if not isfinite(self.mean_strain) or self.count not in (0.5, 1.0):
            raise ValueError("mean_strain must be finite and count must be 0.5 or 1.0.")

    @property
    def strain_amplitude(self):
        return 0.5 * self.strain_range


@dataclass(frozen=True)
class StrainLifeAssessment:
    """Auditable uncorrected strain-life Miner sum, not an engineering approval."""

    cycles: tuple[StrainCycle, ...]
    damage: float
    curve: CoffinMansonBasquinCurve
    source: str

    @property
    def repeated_history_life(self):
        return float("inf") if self.damage == 0.0 else 1.0 / self.damage

    def as_dict(self):
        life = self.repeated_history_life
        return dict(
            kind="strain_life_fatigue_assessment", maturity="postprocessor",
            source=self.source, curve=self.curve.as_dict(), damage=self.damage,
            repeated_history_life=life if isfinite(life) else None,
            life_status=("finite" if isfinite(life) else
                         "no_counted_damage" if self.damage == 0.0 else "outside_numeric_range"),
            strain_measure="mechanical_total", strain_unit="1",
            mean_stress_correction="none", cycle_count=len(self.cycles),
            counted_repetitions=fsum(cycle.count for cycle in self.cycles),
            cycles=[dict(strain_range=c.strain_range, mean_strain=c.mean_strain, count=c.count)
                    for c in self.cycles],
        )


def assess_strain_history(history, curve: CoffinMansonBasquinCurve, *,
                          strain_measure: str, source: str,
                          mean_stress_correction: str) -> StrainLifeAssessment:
    """Count signed scalar total-mechanical-strain cycles and assess life.

    Callers explicitly declare strain semantics and accept the uncorrected
    mean-stress assumption. PEEQ/CEEQ, thermal strain and unsigned equivalent
    strain are not interchangeable inputs. Multiaxial reduction, Morrow/SWT,
    thermomechanical calibration and dwell effects are separate capabilities.
    Residual excursions count as half cycles. No FEM state is modified.
    """
    if not isinstance(curve, CoffinMansonBasquinCurve):
        raise TypeError("curve must be a CoffinMansonBasquinCurve.")
    if strain_measure != "mechanical_total":
        raise ValueError("strain_measure must be 'mechanical_total', a signed scalar component, not PEEQ/CEEQ.")
    if mean_stress_correction != "none":
        raise ValueError("Only explicit mean_stress_correction='none' is supported.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must identify the supplied mechanical strain history.")
    if np.iscomplexobj(history):
        raise ValueError("Mechanical strain history must be real, not complex.")
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Mechanical strain history must be a finite scalar array of length >= 2.")
    # Reuse the established scalar rainflow algorithm, but expose strain names
    # and metadata, never stress units or a separate cycle counting convention.
    cycles = tuple(StrainCycle(c.stress_range, c.mean_stress, c.count)
                   for c in rainflow_cycles(values))
    damage = fsum(c.count / curve.cycles_to_failure(c.strain_amplitude) for c in cycles)
    if not isfinite(damage):
        raise ValueError("Strain-life cumulative damage is not finite.")
    return StrainLifeAssessment(cycles, damage, curve, source.strip())
