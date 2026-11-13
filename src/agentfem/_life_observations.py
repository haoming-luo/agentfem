# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-condition specimen life evidence, not constitutive damage or allowables."""

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping
import warnings

import numpy as np


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


@dataclass(frozen=True)
class LifeObservation:
    """One specimen outcome. Conditions use explicit unit-bearing keys.

    ``life`` is an observed failure coordinate or a right-censoring bound,
    never a fitted life. Independent records must identify distinct specimens.
    Only exact failures and right censoring are admitted in this first slice.
    """

    specimen_id: str
    life: float
    status: str
    life_unit: str
    population: str
    batch_id: str
    failure_definition: str
    conditions: Mapping[str, str | float]
    source: str

    def __post_init__(self):
        for name in ("specimen_id", "status", "life_unit", "population", "batch_id",
                     "failure_definition", "source"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        life = _real(self.life, "life")
        if life <= 0:
            raise ValueError("life must be positive.")
        if self.status not in ("failure", "right_censored"):
            raise ValueError("status must be failure or right_censored; other censoring is not supported.")
        if self.life_unit not in ("cycles", "reversals", "s", "min", "h"):
            raise ValueError("life_unit must be cycles, reversals, s, min or h.")
        if not isinstance(self.conditions, Mapping) or not self.conditions:
            raise ValueError("conditions must be a nonempty mapping with explicit unit-bearing keys.")
        conditions = {}
        for key, value in self.conditions.items():
            name = _text(key, "condition name")
            if name in conditions:
                raise ValueError("Condition names must be unique after whitespace normalization.")
            conditions[name] = (_text(value, name) if isinstance(value, str)
                                else _real(value, name))
        object.__setattr__(self, "life", life)
        object.__setattr__(self, "conditions", MappingProxyType(dict(sorted(conditions.items()))))

    @property
    def cohort_key(self):
        """Exact grouping key: no hidden stress/temperature normalization or pooling."""
        return (self.population, self.batch_id, self.failure_definition,
                self.life_unit, tuple(self.conditions.items()))

    def as_dict(self):
        return {"specimen_id": self.specimen_id, "life": self.life,
                "status": self.status, "life_unit": self.life_unit,
                "population": self.population, "batch_id": self.batch_id,
                "failure_definition": self.failure_definition,
                "conditions": dict(self.conditions), "source": self.source}


@dataclass(frozen=True)
class LifeSurvival:
    """Kaplan–Meier estimates and pointwise, asymptotic survival confidence limits.

    Limits are not a prediction interval, simultaneous band, quantile confidence
    bound, or engineering allowable. Undefined boundary limits remain ``None``.
    """

    observations: tuple[LifeObservation, ...]
    confidence_level: float
    times: tuple[float, ...]
    survival: tuple[float, ...]
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]
    at_risk: tuple[int, ...]
    failures: tuple[int, ...]
    censored: tuple[int, ...]

    def survival_at(self, life):
        """Right-continuous estimate within observed follow-up; no extrapolation."""
        value = _real(life, "life")
        if not 0 <= value <= self.times[-1]:
            raise ValueError("Query lies outside observed follow-up; extrapolation is disabled.")
        index = int(np.searchsorted(self.times, value, side="right")) - 1
        return 1.0 if index < 0 else self.survival[index]

    def quantile(self, failure_probability):
        """First observed CDF crossing, or None if the quantile was not reached."""
        p = _real(failure_probability, "failure_probability")
        if not 0 < p < 1:
            raise ValueError("failure_probability must lie strictly between zero and one.")
        for time, probability in zip(self.times, self.survival, strict=True):
            if probability <= 1 - p:
                return time
        return None

    def as_dict(self):
        count = len(self.observations)
        failures = sum(self.failures)
        return {
            "schema": "agentfem.life-survival.v1",
            "method": "kaplan_meier",
            "life_unit": self.observations[0].life_unit,
            "specimens": count, "failure_count": failures,
            "right_censored_count": count - failures,
            "replicate_failures_available": failures >= 2,
            "confidence_level": self.confidence_level,
            "interval_kind": "pointwise_survival_confidence",
            "interval_method": "exponential_greenwood_asymptotic",
            "interval_boundary_policy": "null_at_zero_or_one_survival_or_undefined_limit",
            "independence_assumption": "distinct_independent_specimens",
            "censoring_assumption": "non_informative_right_censoring",
            "engineering_qualified": False,
            "observations": [row.as_dict() for row in self.observations],
            "stations": [dict(life=t, survival=s, lower=lo, upper=hi,
                              at_risk=r, failures=d, right_censored=c)
                         for t, s, lo, hi, r, d, c in zip(
                             self.times, self.survival, self.lower, self.upper,
                             self.at_risk, self.failures, self.censored, strict=True)],
            "median_point_estimate": self.quantile(.5),
            "limitations": [
                "No extrapolation, parametric fit, covariates, batch pooling or informative-censoring model.",
                "Small samples and few failures do not establish low-tail reliability.",
                "Point estimates of life quantiles are not confidence-qualified design lives.",
            ],
        }


def _sample(observations, confidence_level):
    level = _real(confidence_level, "confidence_level")
    if not 0 < level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    rows = tuple(observations)
    if not rows or not all(isinstance(row, LifeObservation) for row in rows):
        raise ValueError("Provide nonempty LifeObservation records.")
    if len({row.specimen_id for row in rows}) != len(rows):
        raise ValueError("Duplicate specimen_id; repeated cycles are not independent specimens.")
    if any(row.cohort_key != rows[0].cohort_key for row in rows[1:]):
        raise ValueError("Mixed population, batch, failure definition, units or test conditions; group explicitly.")
    rows = tuple(sorted(rows, key=lambda row: (row.life, row.specimen_id)))
    return rows, level


def life_survival(observations, *, independent_specimens, noninformative_censoring,
                  confidence_level=.95):
    """Analyze one declared population/batch/condition using SciPy's KM estimator.

    Both assumptions must explicitly be True. The function checks identity and
    condition consistency, not the truth of experimental independence. Invalid
    input is rejected before estimation; input observations are never mutated.
    """
    if independent_specimens is not True or noninformative_censoring is not True:
        raise ValueError("Explicit independent specimens and non-informative censoring are required.")
    rows, level = _sample(observations, confidence_level)
    from scipy.stats import CensoredData, ecdf

    lives = np.array([row.life for row in rows])
    censored = np.array([row.status == "right_censored" for row in rows])
    empirical = ecdf(CensoredData.right_censored(lives, censored)).sf
    times = empirical.quantiles
    survival = empirical.probabilities
    # Exponential Greenwood is undefined at S=0 or S=1. Retain absence, not
    # a misleading zero-width assurance interval. Do not hide other warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="The confidence interval is undefined at some observations.*")
        bounds = empirical.confidence_interval(confidence_level=level, method="log-log")
    low, high = bounds.low.probabilities, bounds.high.probabilities
    valid = (survival > 0) & (survival < 1) & np.isfinite(low) & np.isfinite(high)
    indices = np.searchsorted(times, lives)
    deaths = np.bincount(indices[~censored], minlength=len(times))
    withdrawals = np.bincount(indices[censored], minlength=len(times))
    risk = np.cumsum((deaths + withdrawals)[::-1])[::-1]
    return LifeSurvival(
        rows, level, tuple(map(float, times)), tuple(map(float, survival)),
        tuple(float(v) if ok else None for v, ok in zip(low, valid, strict=True)),
        tuple(float(v) if ok else None for v, ok in zip(high, valid, strict=True)),
        tuple(map(int, risk)), tuple(map(int, deaths)), tuple(map(int, withdrawals)),
    )


@dataclass(frozen=True)
class FixedHorizonReliability:
    """Exact two-sided binomial confidence interval at one prespecified horizon."""

    observations: tuple[LifeObservation, ...]
    horizon: float
    survivors: int
    confidence_level: float
    lower: float
    upper: float

    @property
    def reliability(self):
        return self.survivors / len(self.observations)

    def as_dict(self):
        return {
            "schema": "agentfem.fixed-horizon-reliability.v1",
            "method": "clopper_pearson_exact_binomial",
            "interval_kind": "two_sided_fixed_horizon_survival_confidence",
            "horizon": self.horizon, "life_unit": self.observations[0].life_unit,
            "specimens": len(self.observations), "survivors": self.survivors,
            "failures_by_horizon": len(self.observations) - self.survivors,
            "reliability": self.reliability, "confidence_level": self.confidence_level,
            "lower": self.lower, "upper": self.upper,
            "independence_assumption": "distinct_independent_specimens",
            "test_plan_assumption": "sample_size_and_horizon_fixed_before_outcomes",
            "survival_definition": "failure_time_strictly_greater_than_horizon",
            "engineering_qualified": False,
            "observations": [row.as_dict() for row in self.observations],
            "limitations": [
                "Not a life distribution, individual-life prediction interval or design allowable.",
                "No early censoring, adaptive stopping or post-selected horizon inference.",
                "Intervals at multiple horizons are not a simultaneous confidence band.",
            ],
        }


def fixed_horizon_reliability(observations, *, horizon, independent_specimens,
                              fixed_test_plan, confidence_level=.95):
    """Evaluate reliability at a fixed test horizon without a life-distribution fit.

    Declare independent specimens and a sample size/horizon fixed before outcomes.
    All specimens must have a known binary outcome at the horizon: failure on or
    before it, or survival through it. Early censoring is rejected, not discarded.
    Uses exact two-sided Clopper-Pearson limits; no sequential stopping correction.
    """
    if independent_specimens is not True or fixed_test_plan is not True:
        raise ValueError("Explicit independent specimens and a fixed test plan are required.")
    rows, level = _sample(observations, confidence_level)
    horizon = _real(horizon, "horizon")
    if horizon <= 0:
        raise ValueError("horizon must be positive and use the observation life_unit.")
    early = [row.specimen_id for row in rows
             if row.status == "right_censored" and row.life < horizon]
    if early:
        raise ValueError(f"Unknown outcome at horizon for early-censored specimens {early}; "
                         "use life_survival with appropriate censoring assumptions instead.")
    survivors = sum(not (row.status == "failure" and row.life <= horizon) for row in rows)
    from scipy.stats import binomtest

    bounds = binomtest(survivors, len(rows)).proportion_ci(
        confidence_level=level, method="exact")
    return FixedHorizonReliability(rows, horizon, survivors, level,
                                   float(bounds.low), float(bounds.high))
