from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tools.p92_published_recovery_reference import published_parameters
from tools.p92_softening_discriminator import (
    integrate_split, isotropic_reference, split_jacobian, split_rates,
)


def test_published_rate_is_shifted_saturation_not_linear_softening():
    p = published_parameters(600)
    x = np.linspace(0., 20., 201)
    solution = solve_ivp(lambda t, R: p.b*(p.Q-R)+p.H, (0., 20.), [0.],
                         rtol=1e-10, atol=1e-11, t_eval=x)
    exact = isotropic_reference(x, Q=p.Q, b=p.b, H=p.H, mode="published_rate")
    np.testing.assert_allclose(solution.y[0], exact, rtol=1e-8, atol=1e-8)
    assert exact[-1] == pytest.approx(p.Q+p.H/p.b, abs=2e-5)
    separate = isotropic_reference(x, Q=p.Q, b=p.b, H=p.H, mode="separate_linear")
    assert separate[-1] == pytest.approx(p.Q+p.H*x[-1], abs=2e-5)
    assert abs(separate[-1]-exact[-1]) > 70


@pytest.mark.parametrize("state", [[260., 10., 20., -20., .3],
                                    [-250., -15., 10., -30., .5],
                                    [0., 1., 2., -10., .2]])
def test_split_analytical_jacobian(state):
    p = published_parameters(600)
    y = np.array(state)
    numerical = np.empty((5, 5))
    for j in range(5):
        h = 1e-5*max(abs(y[j]), .01)
        d = np.zeros(5)
        d[j] = h
        numerical[:, j] = (split_rates(y+d, .001, p)-split_rates(y-d, .001, p))/(2*h)
    np.testing.assert_allclose(split_jacobian(y, p), numerical, rtol=2e-6, atol=2e-6)


def test_split_follows_exact_saturation_plus_linear_without_recovery():
    p = replace(published_parameters(600), gamma_r1=0., gamma_r2=0.)
    old = np.zeros(5)
    solution = integrate_split(old, 8., .001, p)
    accumulated = solution.y[4]
    total = solution.y[3]+p.H*accumulated
    exact = isotropic_reference(accumulated, Q=p.Q, b=p.b, H=p.H, mode="separate_linear")
    np.testing.assert_allclose(total, exact, rtol=2e-7, atol=1e-9)
    np.testing.assert_array_equal(old, np.zeros(5))


def test_reject_nonpositive_strength_without_mutating_old():
    p = published_parameters(600)
    old = np.array([0., 0., 0., -p.k, 1.])
    frozen = old.copy()
    with pytest.raises(ValueError):
        integrate_split(old, 1., .001, p)
    np.testing.assert_array_equal(old, frozen)


def test_split_restart_and_time_partition_agree():
    p = published_parameters(600)
    old = np.zeros(5)
    whole = integrate_split(old, 8., .001, p, rtol=1e-10).y[:, -1]
    half = integrate_split(old, 4., .001, p, rtol=1e-10).y[:, -1]
    resumed = integrate_split(half.copy(), 4., .001, p, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(resumed, whole, rtol=2e-8, atol=2e-7)


def test_failed_cycle_does_not_commit_completed_segments(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    from tools import p92_softening_discriminator as module

    calls = 0

    def trial(state, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("controlled failed trial")
        values = np.column_stack([state, state + np.array([1., 0., 0., 0., .001])])
        return SimpleNamespace(y=values)

    monkeypatch.setattr(module, "integrate_split", trial)
    calibration = {"fit": {"success": True}, "initial_state": [0.] * 5,
                   "parameters": asdict(published_parameters(600))}
    result = module.run_frozen_candidate(calibration, cycles=3)
    assert result["completed_cycles"] == 1
    assert result["failure"]["cycle"] == 2
    assert result["failure"]["segment"] == 1
    assert result["last_completed_state"] == result["cycles"][0]["end_state"]
    assert result["last_completed_state"][0] == 5.
    assert calibration["initial_state"] == [0.] * 5
