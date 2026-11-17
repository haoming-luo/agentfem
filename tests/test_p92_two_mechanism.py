from dataclasses import replace

import numpy as np
import pytest

from tools.p92_published_recovery_reference import published_parameters
from tools.p92_ratcheting_screen import segment
from tools.p92_two_mechanism import Mechanism, integrate, rates


def test_disabled_branch_is_single_mechanism_not_double_stress():
    p = published_parameters(600)
    local = np.array([260., 10., 20., -20., .3])
    initial = np.r_[local, np.zeros(4), 0.]
    two = integrate(initial, 20., 0., (Mechanism(p), Mechanism(p, enabled=False)))
    one = segment(local, 0., 20., 0., p, exponent=3.)
    np.testing.assert_allclose(two.y[:5, -1], one.y[:5, -1], atol=2e-7)
    np.testing.assert_array_equal(two.y[5:9, -1], np.zeros(4))


@pytest.mark.parametrize("sign", [-1., 1.])
def test_sum_of_two_constant_flows_matches_analytical_stress_hold(sign):
    p = replace(published_parameters(600), C1=0., C2=0., Q=0., H=0.,
                gamma1=0., gamma2=0., gamma_r1=0., gamma_r2=0.)
    q = replace(p, Z=p.Z*2)
    initial = np.zeros(10)
    initial[0] = sign*300.
    mechanisms = (Mechanism(p), Mechanism(q))
    sol = integrate(initial, 100., 0., mechanisms, stress_control=True)
    f1, f2 = [((300.-m.parameters.k)/m.parameters.Z)**p.n for m in mechanisms]
    assert sol.y[9, -1] == pytest.approx(sign*100*(f1+f2), abs=1e-10)
    np.testing.assert_allclose(sol.y[[4, 8], -1], 100*np.array([f1, f2]), atol=1e-10)
    np.testing.assert_array_equal(initial[1:], np.zeros(9))


def test_disabled_mechanisms_give_exact_elastic_limit():
    p = published_parameters(600)
    m = (Mechanism(p, enabled=False),)*2
    sol = integrate(np.zeros(10), 2., .001, m)
    assert sol.y[0, -1] == pytest.approx(p.E*.002)
    assert sol.y[9, -1] == pytest.approx(.002)


def test_mechanism_permutation_and_restart():
    p = published_parameters(600)
    m = (Mechanism(p), Mechanism(replace(p, Z=2*p.Z)))
    old = np.array([260., 10., 20., -20., .3, 5., 15., -10., .2, .001])
    full = integrate(old, 20., 0., m, stress_control=True, rtol=1e-10).y[:, -1]
    half = integrate(old, 10., 0., m, stress_control=True, rtol=1e-10).y[:, -1]
    end = integrate(half, 10., 0., m, stress_control=True, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, end, atol=2e-7, rtol=1e-8)
    permutation = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9]
    switched = integrate(old[permutation], 20., 0., m[::-1], stress_control=True, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full[permutation], switched, atol=2e-7, rtol=1e-8)


def test_shared_stress_rate_includes_both_flows():
    p = published_parameters(600)
    old = np.zeros(10)
    old[0] = 300.
    v = rates(old, .001, (Mechanism(p),)*2)
    assert v[0] == pytest.approx(p.E*(.001-v[4]-v[8]))


def test_failure_keeps_old_state(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_two_mechanism as module
    old = np.zeros(10)
    monkeypatch.setattr(module, "solve_ivp", lambda *a, **k:
                        SimpleNamespace(success=False, y=np.zeros((10, 2))))
    with pytest.raises(RuntimeError):
        integrate(old, 1., .001, (Mechanism(published_parameters(600)),)*2)
    np.testing.assert_array_equal(old, np.zeros(10))


def test_mismatched_modulus_rejected():
    p = published_parameters(600)
    with pytest.raises(ValueError):
        integrate(np.zeros(10), 1., .001, (Mechanism(p), Mechanism(replace(p, E=2*p.E))))


def test_control_switch_keeps_both_histories_and_radau_bdf_agree():
    p = published_parameters(600)
    m = (Mechanism(p), Mechanism(replace(p, Z=2*p.Z)))
    ramp = integrate(np.zeros(10), 2., .002, m, rtol=1e-10).y[:, -1]
    hold = integrate(ramp, 10., 0., m, stress_control=True, rtol=1e-10).y[:, -1]
    bdf = integrate(ramp, 10., 0., m, stress_control=True, rtol=1e-10, method="BDF").y[:, -1]
    np.testing.assert_allclose(hold, bdf, rtol=1e-7, atol=2e-6)
    assert hold[0] == pytest.approx(ramp[0])
    assert np.all(hold[[4, 8]] >= ramp[[4, 8]])
    assert hold[9] > ramp[9]
    relaxed = integrate(hold, 10., 0., m).y[:, -1]
    assert relaxed[9] == pytest.approx(hold[9])
    assert relaxed[0] < hold[0]


def test_protocol_exit_is_retained_without_partial_cycle_extrema():
    from tools.p92_two_mechanism_screen import run
    p = replace(published_parameters(600), E=155176., k=116., Z=1192.)
    report = run((Mechanism(p), Mechanism(p, enabled=False)), initial_stress=0., protocol="SLC", cycles=3)
    assert report["envelope_failure"]["cycle"] == 2
    assert len(report["cycles"]) == 1
    assert report["envelope_failure"]["strain"] > .002
    assert not report["engineering_qualified"]


@pytest.mark.parametrize("pchange", [{"Z": 0.}, {"gamma1": -1.}, {"n": .5}])
def test_invalid_mechanism_rejected(pchange):
    with pytest.raises(ValueError):
        Mechanism(replace(published_parameters(600), **pchange))
