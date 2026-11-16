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


def test_nominal_slr_preserves_zero_strain_hold_and_cycle_state(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    from tools import p92_bounded_softening as module
    calls = []

    def integrate(old, duration, strain_rate, parameters, **kwargs):
        calls.append((old.copy(), duration, strain_rate))
        updated = old.copy()
        updated[4] += .01
        return SimpleNamespace(y=np.column_stack((old, updated)))

    monkeypatch.setattr(module, "integrate_bounded", integrate)
    calibration = dict(fit={"success": True}, parameters=asdict(published_parameters(600)),
                       initial_state=[0.]*5)
    report = module.run_candidate(calibration, cycles=2, limit=2., rate=.1, protocol="SLR")
    assert [c[1] for c in calls[:7]] == [2, 180, 2, 1800, 2, 180, 2]
    assert [c[2] for c in calls[:7]] == [-.001, 0., .001, 0., .001, 0., -.001]
    assert calls[7][0][4] == pytest.approx(.07)
    assert report["cycles"][-1]["end_state"][4] == pytest.approx(.14)
    assert calibration["initial_state"] == [0.]*5
    with pytest.raises(ValueError, match="strain-control"):
        module.run_candidate(calibration, cycles=1, limit=2., rate=.1, protocol="SLC")


def test_comparison_keeps_post_failure_and_missing_cycles_explicit():
    from tools.p92_bounded_softening import compare_candidate
    data = dict(protocol="RF", metadata={"failure_cycles": 4}, sha256="fixture",
                cycle=np.array([1, 1, 4, 4, 5, 5]), recorded_cycles=[1, 4, 5],
                stress_Pa=1e6*np.array([-100, 100, -80, 80, -70, 70]))
    report = dict(protocol="RF", cycles=[dict(cycle=i, sampled_peak_MPa=90.,
                                             sampled_valley_MPa=-90.) for i in range(1, 6)])
    compared = compare_candidate(data, report)
    assert [r["cycle"] for r in compared["records"]] == [1, 4, 5]
    assert compared["groups"]["after_source_failure"]["extrema_max_abs_error_MPa"] == 20
    assert compared["groups"]["inspected_development"]["extrema_RMSE_MPa"] == 10
    assert not compared["independent_validation"]
    with pytest.raises(ValueError, match="same protocol"):
        compare_candidate(data, {**report, "protocol": "SLR"})
