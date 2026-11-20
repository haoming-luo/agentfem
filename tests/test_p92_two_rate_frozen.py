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
