from dataclasses import replace

import numpy as np
import pytest

from tools import p92_activated_flow_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(10., 1e-4, 3., (1000., 2000.), (20., 30.), (.1, .01))
    return kernel.Material(spectrum.Material(Material(10000., (b, b)),
                                            (20000., 30000.), (.5, 10.)))


@pytest.mark.parametrize("n", [1., 1.+1e-9, 1.1, 2.8, 5., 10.])
def test_original_reference_rate_and_tangent_are_matched(n):
    b = replace(material().spectrum.backbone.branches[1], exponent=n)
    assert kernel.speed(120., b) == pytest.approx(b.reference_flow, rel=1e-12)
    delta = .0001
    tangent = (kernel.speed(120.+delta, b)-kernel.speed(120.-delta, b))/(2*delta)
    assert tangent == pytest.approx(n*b.reference_flow/120., rel=1e-8)
    assert kernel.speed(0., b) == kernel.speed(-1., b) == 0.
    assert kernel.speed(1., b) > 0.


@pytest.mark.parametrize("control", [False, True])
def test_positive_power_and_energy(control):
    mat = material()
    state = np.zeros(12)
    state[:3], state[4:6], state[10:] = [50., -3., 4.], [5., -6.], [.0003, -.0002]
    rate = kernel.rates(state, .001, mat, stress_control=control)
    h = 1e-6
    derivative = (kernel.stored_energy(state+h*rate, mat)-kernel.stored_energy(state-h*rate, mat))/(2*h)
    assert derivative+rate[9] == pytest.approx(rate[8], abs=1e-8)
    assert rate[9] >= 0.
    disabled = replace(mat, activated=False)
    np.testing.assert_array_equal(kernel.rates(state, .001, disabled, stress_control=control),
                                  spectrum.rates(state, .001, mat.spectrum, stress_control=control))


def test_split_sign_independent_integrator_and_recovery():
    mat, old = material(), np.zeros(12)
    full = kernel.integrate(old, 3., .002, mat)
    bdf = kernel.integrate(old, 3., .002, mat, method="BDF")
    half = kernel.integrate(old, 1., .002, mat)
    split = kernel.integrate(half.y[:, -1], 2., .002, mat)
    negative = kernel.integrate(old, 3., -.002, mat)
    np.testing.assert_allclose(bdf.y[:, -1], full.y[:, -1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(split.y[:, -1], full.y[:, -1], atol=1e-7, rtol=1e-7)
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(negative.y[:, -1], sign*full.y[:, -1], atol=1e-9)
    old[10:] = (.0003, -.0002)
    recovery = kernel.integrate(old, 2., 0., mat, stress_control=True)
    np.testing.assert_allclose(recovery.y[10:, -1], old[10:]*np.exp(-2./np.asarray(mat.spectrum.times)), atol=1e-12)


def test_overflow_rejects_instead_of_clipping():
    with pytest.raises(ValueError, match="numerical range"):
        kernel.speed(1e10, material().spectrum.backbone.branches[1])


@pytest.mark.parametrize("n", [1., 1.1, 2.8, 5.])
def test_tangent_match_never_reduces_fixed_state_flow(n):
    b = replace(material().spectrum.backbone.branches[1], exponent=n)
    for u in np.geomspace(1e-5, 10., 100):
        assert kernel.speed(120*u, b) >= b.reference_flow*u**n*(1.-1e-12)


def test_failure_retains_old_state(monkeypatch):
    from types import SimpleNamespace
    old = np.zeros(12)
    def fail(fun, time, y, **kwargs):
        y[0] = 999.
        return SimpleNamespace(success=False)
    monkeypatch.setattr(kernel, "solve_ivp", fail)
    with pytest.raises(RuntimeError):
        kernel.integrate(old, 1., .001, material())
    assert not old.any()
