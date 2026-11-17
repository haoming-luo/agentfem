from dataclasses import replace

import numpy as np
import pytest

from tools.p92_bounded_softening import bounded_rates
from tools.p92_published_recovery_reference import published_parameters
from tools.p92_ratcheting_screen import corrected_rates, segment


def test_zero_exponent_recovers_frozen_candidate_exactly():
    p = published_parameters(600)
    state = np.array([280., 10., 20., -30., .2])
    np.testing.assert_array_equal(corrected_rates(state, .001, p, exponent=0.),
                                  bounded_rates(state, .001, p, limit=2., rate=.1))


def test_subsaturation_recovery_is_reduced_not_flow_rescaled():
    p = published_parameters(600)
    state = np.array([280., 10., 20., -30., .2])
    base = corrected_rates(state, .001, p, exponent=0.)
    candidate = corrected_rates(state, .001, p, exponent=3.)
    np.testing.assert_array_equal(candidate[[0, 3, 4]], base[[0, 3, 4]])
    assert np.all(candidate[1:3] > base[1:3])


@pytest.mark.parametrize("sign", [-1., 1.])
def test_constant_flow_and_stress_control(sign):
    p = replace(published_parameters(600), C1=0., C2=0., Q=0., H=0.,
                gamma_r1=0., gamma_r2=0., gamma1=0., gamma2=0.)
    state = np.array([sign*300., 0., 0., 0., 0.])
    flow = ((300.-p.k)/p.Z)**p.n
    result = segment(state, 0., 20., 0., p, exponent=3., stress_control=True)
    np.testing.assert_allclose(result.y[:, -1], [sign*300., 0., 0., 0., 20*flow, sign*20*flow], atol=1e-9)
    assert state[4] == 0.


def test_restart_and_refinement_agree():
    p = published_parameters(600)
    state = np.array([260., 10., 20., -20., .3])
    whole = segment(state, 0., 20., 0., p, exponent=3., stress_control=True, rtol=1e-10).y[:, -1]
    half = segment(state, 0., 10., 0., p, exponent=3., stress_control=True, rtol=1e-10).y[:, -1]
    resumed = segment(half[:5], half[5], 10., 0., p, exponent=3., stress_control=True, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(whole, resumed, rtol=1e-8, atol=2e-7)


@pytest.mark.parametrize("exponent", [-1., float("nan")])
def test_invalid_exponent(exponent):
    with pytest.raises(ValueError):
        corrected_rates(np.zeros(5), 0., published_parameters(600), exponent=exponent)


def test_failed_integration_does_not_mutate_input(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_ratcheting_screen as module
    old = np.array([260., 10., 20., -20., .3])
    saved = old.copy()
    monkeypatch.setattr(module, "solve_ivp", lambda *args, **kwargs:
                        SimpleNamespace(success=False, y=np.zeros((6, 2))))
    with pytest.raises(RuntimeError, match="failed numerical"):
        segment(old, 0., 10., 0., published_parameters(600), exponent=3.)
    np.testing.assert_array_equal(old, saved)
