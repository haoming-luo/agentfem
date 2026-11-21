from types import SimpleNamespace

import numpy as np
import pytest

from tools.p92_equal_stress_hold import observed_interval, predicted_interval


def test_observed_increment_retains_strain_drift_and_units():
    value = observed_interval([0., 1., 2.], [0., 1e-5, 2e-5], [220., 210., 200.], 100000.)
    assert value["average_nonelastic_rate_per_s"] == pytest.approx(1.1e-4)
    assert value["one_quantum_endpoint_sensitivity_per_s"] == pytest.approx(1.01e-5)


def test_equal_stress_interval_uses_predicted_crossings_not_measured_time():
    def sol(t):
        t = np.asarray(t)
        y = np.zeros((12, *t.shape))
        y[0] = 250.-2*t
        return y
    solution = SimpleNamespace(t=np.linspace(0., 100., 21), sol=sol)
    solution.y = sol(solution.t)
    actual = predicted_interval(solution, [220., 200.], 100000.)
    assert actual["times_s"] == pytest.approx([15., 25.])
    assert actual["duration_s"] == pytest.approx(10.)
    assert actual["average_nonelastic_rate_per_s"] == pytest.approx(2e-5)
    with pytest.raises(ValueError, match="crossing"):
        predicted_interval(solution, [300., 200.], 100000.)


@pytest.mark.parametrize("time,strain,stress", [
    ([0., 1.], [0., 0.], [200., 190.]),
    ([0., 1., 1.], [0., 0., 0.], [200., 190., 180.]),
    ([0., 1., 2.], [0., float("nan"), 0.], [200., 190., 180.]),
    ([0., 1., 2.], [0., 0., 0.], [180., 190., 200.]),
])
def test_reject_ambiguous_or_invalid_observations(time, strain, stress):
    with pytest.raises(ValueError):
        observed_interval(time, strain, stress, 100000.)
