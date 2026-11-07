"""Independent product-limit and Greenwood checks, not fitted P92 lifetimes."""

from dataclasses import FrozenInstanceError, replace
import json
import math
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pytest

from agentfem.assessments import LifeObservation, life_survival


def observation(index, life=1., status="failure", **kwargs):
    values = dict(specimen_id=str(index), life=life, status=status,
                  life_unit="h", population="synthetic", batch_id="A",
                  failure_definition="rupture", conditions={"stress_MPa": 100.},
                  source="synthetic analytic fixture")
    values.update(kwargs)
    return LifeObservation(**values)


def analyze(rows, **kwargs):
    return life_survival(rows, independent_specimens=True,
                         noninformative_censoring=True, **kwargs)


def fixture():
    return [observation(i, t, "failure" if i < 3 else "right_censored")
            for i, t in enumerate([1, 2, 3, 4, 4])]


def test_independent_product_limit_and_exponential_greenwood():
    result = analyze(fixture())
    assert result.times == (1., 2., 3., 4.)
    assert result.survival == pytest.approx((.8, .6, .4, .4))
    assert result.at_risk == (5, 4, 3, 2)
    assert result.failures == (1, 1, 1, 0)
    assert result.censored == (0, 0, 0, 2)
    greenwood = 0.
    for i, (risk, deaths) in enumerate(zip(result.at_risk, result.failures)):
        greenwood += deaths / (risk * (risk - deaths))
        s = result.survival[i]
        width = NormalDist().inv_cdf(.975) * math.sqrt(greenwood) / abs(math.log(s))
        center = math.log(-math.log(s))
        assert result.lower[i] == pytest.approx(math.exp(-math.exp(center + width)))
        assert result.upper[i] == pytest.approx(math.exp(-math.exp(center - width)))
    assert result.quantile(.5) == 3.
    assert result.quantile(.9) is None
    assert result.survival_at(0) == 1.
    assert result.survival_at(2) == pytest.approx(.6)
    assert result.survival_at(2.5) == pytest.approx(.6)


def test_tied_censoring_is_at_risk_for_same_time_failure():
    result = analyze([observation(0, 1, "right_censored"), observation(1, 2),
                      observation(2, 2, "right_censored"), observation(3, 3)])
    assert result.at_risk == (4, 3, 1)
    assert result.survival == pytest.approx((1, 2/3, 0))
    assert result.lower[0] is None and result.lower[-1] is None


@pytest.mark.parametrize("status", ["failure", "right_censored"])
def test_single_specimen_does_not_create_scatter_interval(status):
    result = analyze([observation(0, status=status)])
    assert result.lower == result.upper == (None,)
    assert not result.as_dict()["replicate_failures_available"]
    assert not result.as_dict()["engineering_qualified"]
    json.dumps(result.as_dict(), allow_nan=False)


def test_all_censored_followup_and_no_extrapolation():
    result = analyze([observation(i, i+1, "right_censored") for i in range(4)])
    assert result.survival == (1.,)*4
    assert result.quantile(.1) is None
    assert result.lower == (None,)*4
    for value in [-1, 4.1, np.nan, True]:
        with pytest.raises(ValueError):
            result.survival_at(value)


def test_units_order_and_confidence_level():
    rows = fixture()
    baseline = analyze(rows)
    scaled = analyze([replace(row, life=row.life*3600, life_unit="s") for row in rows])
    assert scaled.survival == baseline.survival
    assert scaled.lower == baseline.lower
    assert scaled.quantile(.5) == 3600 * baseline.quantile(.5)
    assert analyze(reversed(rows)).as_dict() == baseline.as_dict()
    wide = analyze(rows, confidence_level=.99)
    assert all(a < b for a, b in zip(wide.lower, baseline.lower))
    assert all(a > b for a, b in zip(wide.upper, baseline.upper))


def test_records_own_their_metadata():
    conditions = {"stress_MPa": 100.}
    row = observation(0, conditions=conditions)
    conditions["stress_MPa"] = 200.
    row.as_dict()["conditions"]["stress_MPa"] = 300.
    assert row.conditions["stress_MPa"] == 100.
    with pytest.raises(TypeError):
        row.conditions["stress_MPa"] = 500.
    with pytest.raises(FrozenInstanceError):
        row.life = 3.


@pytest.mark.parametrize("kwargs", [
    {"life": value} for value in [0, -1, True, np.nan, np.inf, 1j]
] + [{"status": "interval_censored"}, {"status": "left_censored"},
     {"life_unit": "unknown"}, {"source": " "}, {"conditions": {}},
     {"conditions": {"x": True}}, {"conditions": {"x": np.inf}},
     {"conditions": {"x": 1, " x ": 2}}])
def test_invalid_records(kwargs):
    with pytest.raises(ValueError):
        observation(0, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"life_unit": "s"}, {"population": "other"}, {"batch_id": "B"},
    {"failure_definition": "crack initiation"}, {"conditions": {"stress_MPa": 101.}},
])
def test_no_silent_pooling(kwargs):
    with pytest.raises(ValueError, match="Mixed"):
        analyze([observation(0), observation(1, **kwargs)])


def test_duplicate_and_empty_records():
    for rows in [[], [object()], [observation(0), observation(0, 2)]]:
        with pytest.raises(ValueError):
            analyze(rows)


@pytest.mark.parametrize("level", [0, 1, -.1, True, np.nan])
def test_invalid_confidence(level):
    with pytest.raises(ValueError):
        analyze(fixture(), confidence_level=level)


def test_assumptions_are_explicit():
    for independent, censoring in [(False, True), (True, False), (1, True)]:
        with pytest.raises(ValueError):
            life_survival(fixture(), independent_specimens=independent,
                          noninformative_censoring=censoring)
    for probability in [0, 1, True, np.inf]:
        with pytest.raises(ValueError):
            analyze(fixture()).quantile(probability)


def test_documented_example_runs_without_private_api():
    guide = Path(__file__).resolve().parents[1] / "docs/life_observations.md"
    example = guide.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(example, str(guide), "exec"), namespace)
    json.dumps(namespace["evidence"], allow_nan=False)
