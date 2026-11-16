import json
import math

import pytest

from agentfem.assessments import LifeObservation, life_prediction_errors


def row(identity, life=10., status="failure", unit="h", criterion="rupture"):
    return LifeObservation(identity, life, status, unit, "fixture", "A", criterion,
                           {"stress_MPa": 100.}, "synthetic independent fixture")


def evaluate(rows, predictions):
    return life_prediction_errors(rows, predictions, life_unit="h", model_source="fixture model")


def test_exact_errors_and_censored_bounds_are_distinct():
    result = evaluate([row("a"), row("b"), row("c", status="right_censored")],
                      {"c": 1., "a": 20., "b": 5.})
    assert result["mean_log_error_on_failures"] == pytest.approx(0, abs=1e-15)
    assert result["rms_log_error_on_failures"] == pytest.approx(math.log(2))
    assert result["failure_count"] == 2
    assert result["records"][-1]["log_prediction_error"] is None
    assert result["records"][-1]["prediction_below_runout"]
    json.dumps(result, allow_nan=False)


def test_all_censored_has_no_fabricated_error_statistic():
    result = evaluate([row("a", status="right_censored")], {"a": 100})
    assert result["rms_log_error_on_failures"] is None


@pytest.mark.parametrize("predicted", [0., -1., float("inf"), float("nan"), True, 1j])
def test_invalid_predictions(predicted):
    with pytest.raises(ValueError):
        evaluate([row("a")], {"a": predicted})


@pytest.mark.parametrize("rows,predictions", [
    ([row("a")], {"b": 10}),
    ([row("a"), row("a")], {"a": 10}),
    ([row("a", unit="s")], {"a": 10}),
    ([row("a"), row("b", criterion="load drop")], {"a": 10, "b": 10}),
])
def test_identity_and_semantic_mismatches(rows, predictions):
    with pytest.raises(ValueError):
        evaluate(rows, predictions)


def test_extreme_finite_lives_remain_json_safe():
    result = evaluate([row("a", life=1.e-300)], {"a": 1.e300})
    json.dumps(result, allow_nan=False)
    assert result["records"][0]["log_prediction_error"] == pytest.approx(600*math.log(10))
