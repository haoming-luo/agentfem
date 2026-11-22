from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from tools import p92_inviscid_branch_reference as kernel
from tools.p92_recoverable_spectrum import Material as Spectrum
from tools.p92_two_rate_reference import Branch, Material as Backbone


def material():
    a = Branch(1000., .001, 3., (100., 200.), (10., 20.), (0., 0.))
    b = Branch(50., .001, 3., (300., 400.), (100., 200.), (0., 0.), ratcheting=(0., 0.))
    return kernel.Material(Spectrum(Backbone(10000., (a, b)), (1e30, 1e30), (.5, 10.)))


def test_exact_monotonic_armstrong_frederick_limit_and_yield_event():
    m, y = material(), np.zeros(12)
    sol = kernel.integrate(y, 10., .001, m)
    b = m.spectrum.backbone.branches[1]
    stress = lambda p: b.strength+sum(np.array(b.saturation)*(-np.expm1(-np.array(b.hardening)/b.saturation*p)))
    p = brentq(lambda p: stress(p)/10000.+p-.01, 0., .01, xtol=1e-15)
    assert sol.y[6, -1] == pytest.approx(p, abs=1e-10)
    assert sol.y[0, -1] == pytest.approx(stress(p), abs=1e-6)
    assert sol.transitions[0][0] == pytest.approx(5., abs=1e-9)
    assert sol.transitions[0][1] == 1.
    np.testing.assert_array_equal(y, np.zeros(12))
    np.testing.assert_allclose(sol.sol([1., 2., 3.])[0], [10., 20., 30.], atol=1e-7)
    assert sol.sol([]).shape == (12, 0)


def test_rate_independent_limit_time_scaling():
    m, y = material(), np.zeros(12)
    a = kernel.integrate(y, 10., .001, m)
    b = kernel.integrate(y, 100., .0001, m)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-8, rtol=1e-7)


def test_reverse_loading_independent_integration_and_split():
    m = material()
    old = kernel.integrate(np.zeros(12), 10., .001, m).y[:, -1]
    saved = old.copy()
    a = kernel.integrate(old, 20., -.001, m)
    b = kernel.integrate(old, 20., -.001, m, method="BDF", rtol=1e-10)
    mid = kernel.integrate(old, 7., -.001, m).y[:, -1]
    c = kernel.integrate(mid, 13., -.001, m)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(a.y[:, -1], c.y[:, -1], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(old, saved)
    assert a.y[0, -1] < -50.


@pytest.mark.parametrize("stress_control", [False, True])
def test_active_consistency_power_reflection(stress_control):
    m = material()
    a, b = m.spectrum.backbone.branches
    m = kernel.Material(replace(m.spectrum, backbone=replace(m.spectrum.backbone,
        branches=(a, replace(b, static_recovery=(.01, .02)))), moduli=(20000., 30000.)))
    y = np.zeros(12)
    y[[0, 4, 5, 10, 11]] = [80., 10., 20., .001, .002]
    r = kernel.rates(y, .01, m, stress_control=stress_control, active=1.)
    assert r[6] > 0
    assert r[0]-r[4]-r[5] == pytest.approx(0., abs=1e-12)
    energy_rate = y[0]*r[0]/10000.+y[4]*r[4]/300.+y[5]*r[5]/400.
    energy_rate += sum(np.array(m.spectrum.moduli)*y[10:]*r[10:])
    assert r[8] == pytest.approx(energy_rate+r[9], abs=1e-12)
    assert r[9] >= 0
    signs = np.ones(12)
    signs[[0, 1, 2, 4, 5, 7, 10, 11]] = -1.
    np.testing.assert_allclose(kernel.rates(y*signs, -.01, m,
        stress_control=stress_control, active=-1.), r*signs, atol=1e-12)


def test_removed_viscous_constants_cannot_affect_new_law():
    m = material()
    a, b = m.spectrum.backbone.branches
    alternate = kernel.Material(replace(m.spectrum, backbone=replace(m.spectrum.backbone,
        branches=(a, replace(b, reference_flow=99., exponent=33.)))))
    y = np.zeros(12)
    y[0] = 50.
    np.testing.assert_array_equal(kernel.rates(y, .01, m, active=1.),
                                  kernel.rates(y, .01, alternate, active=1.))
    mapped = kernel.mapped(m.spectrum)
    assert mapped.spectrum.backbone.branches[1].strength == 170.


def test_inadmissible_initial_yield_state_rejected_without_clipping():
    y = np.zeros(12)
    y[0] = 51.
    saved = y.copy()
    with pytest.raises(ValueError, match="yield"):
        kernel.integrate(y, 1., 0., material())
    np.testing.assert_array_equal(y, saved)
