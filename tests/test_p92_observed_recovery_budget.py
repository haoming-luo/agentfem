import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tools.p92_observed_recovery_budget import linear_memory, window_budget


def test_constant_stress_and_initial_memory():
    t = np.array([0., .01, 1., 100.])
    k, tau, initial = np.array([100., 300.]), np.array([.2, 10.]), np.array([.1, -.2])
    actual = linear_memory(t, np.full(4, 20.), k, tau, initial)
    expected = 20/k+(initial-20/k)*np.exp(-t[:, None]/tau)
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_ramp_small_step_and_reflection():
    t = np.array([0., 1e-10, .01, 10.])
    actual = linear_memory(t, t*2, [4.], [3.], [0.])[:, 0]
    assert actual[1] == pytest.approx(2*t[1]**2/(2*4*3), rel=1e-9)
    np.testing.assert_allclose(actual[2:], 2/4*(t[2:]+3*np.expm1(-t[2:]/3)), atol=1e-15)
    np.testing.assert_allclose(linear_memory(t, -t*2, [4.], [3.], [0.])[:, 0], -actual)


def test_piecewise_linear_subdivision_and_independent_integration():
    t = np.array([0., .2, 1., 3., 10.])
    s = np.array([0., 30., -10., 5., 2.])
    k, tau, initial = np.array([100., 300.]), np.array([.2, 10.]), np.array([.1, -.2])
    actual = linear_memory(t, s, k, tau, initial)
    fine = np.sort(np.r_[t, (t[:-1]+t[1:])/2])
    np.testing.assert_allclose(linear_memory(fine, np.interp(fine, t, s), k, tau, initial)[::2], actual,
                               rtol=1e-13, atol=1e-15)
    state = initial.copy()
    for i in range(len(t)-1):
        result = solve_ivp(lambda q, a: (np.interp(q, t, s)/k-a)/tau,
                           (t[i], t[i+1]), state, method="Radau", rtol=1e-11, atol=1e-13)
        assert result.success
        state = result.y[:, -1]
        np.testing.assert_allclose(state, actual[i+1], rtol=1e-9, atol=1e-11)


def test_initial_memory_increment_bound_and_kinematic_identity():
    t = np.array([0., 3., 10.])
    s = np.array([0., 10., 20.])
    k, tau, bound = np.array([100., 300.]), np.array([.2, 10.]), np.array([.3, .1])
    zero = linear_memory(t, s, k, tau, [0., 0.])
    e = s/1000+zero.sum(axis=1)+.01*t
    record = window_budget(t, s, e, zero, 1, 2, modulus=1000., moduli=k,
                           times=tau, initial_bounds=bound)
    assert record["required_remainder_increment"] == pytest.approx(.07)
    alternate = linear_memory(t, s, k, tau, bound)
    delta = sum(alternate[2]-alternate[1]-zero[2]+zero[1])
    assert abs(delta) == pytest.approx(record["initial_memory_sensitivity"])


@pytest.mark.parametrize("t,s,k,tau,a", [
    ([0., 0.], [0., 1.], [1.], [1.], [0.]),
    ([0., 1.], [0., np.nan], [1.], [1.], [0.]),
    ([0., 1.], [0., 1.], [0.], [1.], [0.]),
    ([0., 1.], [0., 1.], [1.], [-1.], [0.]),
    ([0., 1.], [0., 1.], [1.], [1.], [1j]),
])
def test_invalid_convolution_rejected(t, s, k, tau, a):
    with pytest.raises(ValueError):
        linear_memory(t, s, k, tau, a)
