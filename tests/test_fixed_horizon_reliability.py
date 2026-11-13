"""Exact binomial tail checks independent of SciPy's confidence implementation."""

from dataclasses import replace
import json
from math import comb
from pathlib import Path

import pytest

from agentfem.assessments import LifeObservation, fixed_horizon_reliability


def row(index, life=100, status="right_censored"):
    return LifeObservation(str(index), life, status, "cycles", "synthetic", "A",
                           "rupture", {"stress_MPa": 100}, "analytic fixture")


def assess(rows, horizon=100, **kwargs):
    return fixed_horizon_reliability(rows, horizon=horizon, independent_specimens=True,
                                     fixed_test_plan=True, **kwargs)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_zero_failure_bounds_are_not_certainty(n):
    result = assess([row(i) for i in range(n)])
    assert result.reliability == 1
    assert result.lower == pytest.approx(.025 ** (1 / n))
    assert result.upper == 1
    assert not result.as_dict()["engineering_qualified"]
    json.dumps(result.as_dict(), allow_nan=False)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_all_failures_bounds(n):
    result = assess([row(i, status="failure") for i in range(n)])
    assert result.reliability == 0
    assert result.lower == 0
    assert result.upper == pytest.approx(1 - .025 ** (1 / n))


def test_exact_tail_equations_and_known_horizon_outcomes():
    rows = [row(0, 50, "failure"), row(1, 100, "failure"), row(2, 120, "failure"),
            row(3, 100), row(4, 150)]
    result = assess(rows)
    assert result.survivors == 3
    assert result.reliability == .6
    lower_tail = sum(comb(5, k) * result.lower**k * (1-result.lower)**(5-k)
                     for k in range(3, 6))
    upper_tail = sum(comb(5, k) * result.upper**k * (1-result.upper)**(5-k)
                     for k in range(4))
    assert lower_tail == pytest.approx(.025)
    assert upper_tail == pytest.approx(.025)
    assert assess(reversed(rows)).as_dict() == result.as_dict()
    scaled = assess([replace(r, life=r.life * 2, life_unit="reversals") for r in rows], 200)
    assert (scaled.lower, scaled.upper) == (result.lower, result.upper)


def test_early_censor_is_not_dropped_even_when_other_outcomes_known():
    with pytest.raises(ValueError, match="early-censored.*missing"):
        assess([row("known", 50, "failure"), row("missing", 99)])


def test_groups_and_identity_checks_are_shared():
    for rows in [[], [row(1), row(1)], [row(1), replace(row(2), batch_id="B")],
                 [row(1), replace(row(2), conditions={"stress_MPa": 101})]]:
        with pytest.raises(ValueError):
            assess(rows)


@pytest.mark.parametrize("horizon", [0, -1, True, float("nan"), float("inf")])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        assess([row(1)], horizon)


def test_fixed_design_and_independence_must_be_declared():
    for independent, plan in [(False, True), (True, False), (True, 1)]:
        with pytest.raises(ValueError):
            fixed_horizon_reliability([row(0)], horizon=100,
                                      independent_specimens=independent, fixed_test_plan=plan)


@pytest.mark.parametrize("level", [0, 1, True, float("nan")])
def test_invalid_confidence(level):
    with pytest.raises(ValueError):
        assess([row(0)], confidence_level=level)


def test_confidence_increases_interval_width():
    rows = [row(i) for i in range(10)]
    assert assess(rows, confidence_level=.99).lower < assess(rows).lower


def test_documented_example():
    guide = Path(__file__).resolve().parents[1] / "docs/life_observations.md"
    code = guide.read_text().split("```python\n")[2].split("```", 1)[0]
    exec(compile(code, str(guide), "exec"), {})
