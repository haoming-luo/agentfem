from pathlib import Path

import pytest

from tools.p92_two_rate_frozen import evaluate
from tools.p92_two_rate_frozen import numerical_difference


def test_failed_or_non_early_fit_cannot_enter_frozen_replay():
    with pytest.raises(ValueError, match="early-only"):
        evaluate(dict(training_cycles=[1, 2, 3, 4]), Path("unused"))
    with pytest.raises(ValueError, match="early-only"):
        evaluate(dict(training_cycles=[1, 2, 3], conditional_fit=dict(success=False)), Path("unused"))


@pytest.mark.parametrize("component, perturbation", [(1, .001), (3, 1e-7), (8, 1e-5)])
def test_numerical_check_covers_hidden_states_not_only_plotted_fields(component, perturbation):
    from copy import deepcopy
    a = dict(segments=[dict(sampled_stress_MPa=[1.], sampled_strain=[.001], end_state=[0.]*10)])
    b = deepcopy(a)
    b["segments"][0]["end_state"][component] = perturbation
    assert not numerical_difference(a, b)["passed"]
    assert numerical_difference(a, a)["passed"]


@pytest.mark.parametrize("key,value", [("model_id", "other"), ("source_sha256", "other"),
                                       ("protocol", "SLC"), ("state_layout", ["other"] )])
def test_numerical_check_rejects_mixed_identity(key, value):
    from copy import deepcopy
    a = dict(segments=[dict(sampled_stress_MPa=[1.], sampled_strain=[.001], end_state=[0.]*10)])
    b = deepcopy(a)
    b[key] = value
    with pytest.raises(ValueError, match="identity"):
        numerical_difference(a, b)


@pytest.mark.parametrize("key,value", [("sampled_stress_MPa", [float("nan")]),
    ("sampled_strain", [.001, .002]), ("end_state", [float("inf")]*10)])
def test_numerical_check_rejects_invalid_or_broadcastable_arrays(key, value):
    from copy import deepcopy
    a = dict(segments=[dict(sampled_stress_MPa=[1.], sampled_strain=[.001], end_state=[0.]*10)])
    b = deepcopy(a)
    b["segments"][0][key] = value
    with pytest.raises(ValueError, match="finite arrays"):
        numerical_difference(a, b)


def test_numerical_check_requires_same_observations_and_nonempty_segments():
    from copy import deepcopy
    a = dict(segments=[dict(source_indices=[1], sampled_stress_MPa=[1.], sampled_strain=[.001], end_state=[0.]*10)])
    b = deepcopy(a)
    b["segments"][0]["source_indices"] = [2]
    with pytest.raises(ValueError, match="observations"):
        numerical_difference(a, b)
    with pytest.raises(ValueError, match="completed"):
        numerical_difference(dict(segments=[]), dict(segments=[]))
