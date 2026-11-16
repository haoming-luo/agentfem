import numpy as np
import pytest

from tools.p92_bounded_softening import (
    bounded_rates, bounded_jacobian, integrate_bounded, multiplier,
)
from tools.p92_published_recovery_reference import (
    published_parameters, rates, rate_jacobian, integrate_segment,
)


def test_bounded_multiplier_and_derivative():
    for p in (0., .01, 1., 100.):
        value, derivative = multiplier(p, limit=2., rate=.1)
        assert 1 <= value <= 2
        assert derivative >= 0
        if p:
            h = 1e-5
            difference = (multiplier(p+h, limit=2., rate=.1)[0]
                          - multiplier(p-h, limit=2., rate=.1)[0])/(2*h)
            assert derivative == pytest.approx(difference, abs=1e-10)


@pytest.mark.parametrize("y", [[260., 10., 20., -20., .3],
                              [-250., -15., 10., -30., .5], [0., 1., 2., -10., .2]])
def test_analytical_jacobian(y):
    p = published_parameters(600)
    y = np.array(y)
    numerical = np.empty((5, 5))
    for j in range(5):
        h = 1e-5*max(abs(y[j]), .01)
        d = np.zeros(5)
        d[j] = h
        numerical[:, j] = (bounded_rates(y+d, .001, p, limit=2., rate=.1)
                            -bounded_rates(y-d, .001, p, limit=2., rate=.1))/(2*h)
    np.testing.assert_allclose(bounded_jacobian(y, p, limit=2., rate=.1), numerical,
                               rtol=2e-6, atol=2e-6)


def test_identity_recovers_original_equations_and_integration():
    p = published_parameters(600)
    y = np.array([260., 10., 20., -20., .3])
    np.testing.assert_array_equal(bounded_rates(y, .001, p, limit=1., rate=.1), rates(y, .001, p))
    np.testing.assert_array_equal(bounded_jacobian(y, p, limit=1., rate=.1), rate_jacobian(y, p))
    reference = integrate_segment(y, 2., .001, p).y[:, -1]
    candidate = integrate_bounded(y, 2., .001, p, limit=1., rate=.1).y[:, -1]
    np.testing.assert_allclose(candidate, reference, rtol=1e-7, atol=1e-7)


def test_restart_preserves_history():
    p = published_parameters(600)
    old = np.zeros(5)
    whole = integrate_bounded(old, 8., .001, p, limit=2., rate=.1, rtol=1e-10).y[:, -1]
    half = integrate_bounded(old, 4., .001, p, limit=2., rate=.1, rtol=1e-10).y[:, -1]
    resumed = integrate_bounded(half, 4., .001, p, limit=2., rate=.1, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(whole, resumed, rtol=2e-8, atol=2e-7)
    np.testing.assert_array_equal(old, np.zeros(5))
