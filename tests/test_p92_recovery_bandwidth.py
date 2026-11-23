from dataclasses import replace

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as spectrum
from tools import p92_recovery_bandwidth_reference as kernel
from tools.p92_two_rate_reference import Branch, Material


def baseline():
    b = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return spectrum.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


@pytest.mark.parametrize("spread", [0., .25, .5, .75])
def test_split_preserves_compliance_and_initial_mobility(spread):
    base = baseline()
    mat = kernel.split_slow(base, spread)
    assert mat.spectrum.moduli[0] == base.moduli[0]
    assert mat.spectrum.times[0] == base.times[0]
    h, t = mat.spectrum.moduli[1], mat.spectrum.times[1]
    assert 1/h+1/mat.extra_modulus == pytest.approx(1/base.moduli[1])
    assert 1/(h*t)+1/(mat.extra_modulus*mat.extra_time) == pytest.approx(1/(base.moduli[1]*base.times[1]))


@pytest.mark.parametrize("stress_control", [False, True])
def test_zero_spread_exact_two_mode_reduction(stress_control):
    base = baseline()
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])
    extended = np.r_[old, old[11]/2]
    extended[11] /= 2
    mat = kernel.split_slow(base, 0.)
    a = spectrum.integrate(old, 2., -.002, base, stress_control=stress_control).y[:, -1]
    b = kernel.integrate(extended, 2., -.002, mat, stress_control=stress_control).y[:, -1]
    np.testing.assert_allclose(a[:11], b[:11], atol=1e-8)
    assert a[11] == pytest.approx(sum(b[11:]), abs=1e-12)
    assert spectrum.stored_energy(a, base) == pytest.approx(kernel.stored_energy(b, mat), abs=1e-9)


@pytest.mark.parametrize("stress", [-10., 10.])
def test_signed_analytic_creep_and_unloaded_recovery(stress):
    mat = kernel.split_slow(baseline(), .5)
    h = np.r_[mat.spectrum.moduli, mat.extra_modulus]
    tau = np.r_[mat.spectrum.times, mat.extra_time]
    old = np.zeros(13)
    old[0], old[7] = stress, stress/mat.spectrum.backbone.modulus
    end = kernel.integrate(old, 7., 0., mat, stress_control=True).y[:, -1]
    expected = stress/h*(1-np.exp(-7/tau))
    np.testing.assert_allclose(end[10:], expected, atol=1e-12)
    np.testing.assert_array_equal(end[[3, 6]], 0.)
    assert end[7] == pytest.approx(old[7]+sum(expected), abs=1e-12)
    # An explicitly imposed instantaneous elastic unloading prepares a new
    # synthetic initial state. No experimental history is reset by this test.
    unloaded = end.copy()
    unloaded[0] = 0.
    unloaded[7] -= stress/mat.spectrum.backbone.modulus
    back = kernel.integrate(unloaded, 9., 0., mat, stress_control=True).y[:, -1]
    np.testing.assert_allclose(back[10:], expected*np.exp(-9/tau), atol=1e-12)
    assert back[7] == pytest.approx(sum(back[10:]), abs=1e-12)


@pytest.mark.parametrize("stress_control", [False, True])
def test_power_sign_and_independent_split_integration(stress_control):
    mat = kernel.split_slow(baseline(), .5)
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, -.0002])
    before = old.copy()
    rate = kernel.rates(old, -.002, mat, stress_control=stress_control)
    eps = 1e-5
    de = (kernel.stored_energy(old+eps*rate, mat)-kernel.stored_energy(old-eps*rate, mat))/(2*eps)
    assert rate[8] == pytest.approx(de+rate[9], abs=1e-9)
    assert rate[9] >= 0.
    sign = np.ones(13)
    sign[[0, 1, 2, 4, 5, 7, 10, 11, 12]] = -1
    np.testing.assert_allclose(kernel.rates(old*sign, .002, mat, stress_control=stress_control), rate*sign)
    full = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control).y[:, -1]
    mid = kernel.integrate(old, .7, -.002, mat, stress_control=stress_control).y[:, -1]
    split = kernel.integrate(mid, 1.3, -.002, mat, stress_control=stress_control).y[:, -1]
    other = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control, method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, split, atol=2e-8, rtol=1e-7)
    np.testing.assert_allclose(full, other, atol=2e-8, rtol=1e-7)
    np.testing.assert_array_equal(old, before)


def test_invalid_parameters_and_incompatible_evolution_rejected():
    with pytest.raises(ValueError, match="between"):
        kernel.Material(baseline(), 40000., .01)
    for spread in (-.1, 1., np.nan):
        with pytest.raises(ValueError):
            kernel.split_slow(baseline(), spread)
    for base in (replace(baseline(), recoverable_powers=(2., 1.)),
                 replace(baseline(), static_ratios=(.5, 1.)),
                 replace(baseline(), static_powers=(1., 0.)),
                 replace(baseline(), directional_recovery=True)):
        with pytest.raises(ValueError, match="isolated"):
            kernel.split_slow(base, .5)
    old = np.zeros(13)
    old[3] = -.01
    with pytest.raises(ValueError):
        kernel.integrate(old, 1., 0., kernel.split_slow(baseline(), .5))
    assert old[3] == -.01


def test_declared_probe_arms_and_intermediate_creep_inequality():
    from tools.p92_recovery_bandwidth_probe import candidates
    base = baseline()
    arms = candidates(base)
    assert [s for s, _ in arms] == [0., .25, .5, .75]
    t = np.geomspace(1e-4, 1e4, 100)
    reference = (1-np.exp(-t/base.times[1]))/base.moduli[1]
    for _, mat in arms:
        split = ((1-np.exp(-t/mat.spectrum.times[1]))/mat.spectrum.moduli[1]
                 +(1-np.exp(-t/mat.extra_time))/mat.extra_modulus)
        assert np.all(split <= reference+1e-18)


def test_failed_integrator_preserves_committed_state(monkeypatch):
    from types import SimpleNamespace
    old = np.zeros(13)
    before = old.copy()

    def fail(fun, interval, initial, **kwargs):
        initial[:] = 123.
        return SimpleNamespace(success=False)

    monkeypatch.setattr(kernel, "solve_ivp", fail)
    with pytest.raises(RuntimeError, match="input retained"):
        kernel.integrate(old, 1., 0., kernel.split_slow(baseline(), .5))
    np.testing.assert_array_equal(old, before)
