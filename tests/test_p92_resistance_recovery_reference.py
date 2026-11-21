from dataclasses import replace

import numpy as np
import pytest

from tools import p92_resistance_recovery_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(kind="strength", minimum=.9):
    b = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    base = spectrum.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))
    return kernel.Material(base, kind, minimum, 250., 1/180.)


@pytest.mark.parametrize("kind", ["strength", "mobility"])
def test_zero_amplitude_identity(kind):
    m = material(kind, 1.)
    y = np.zeros(14); y[0] = 70.
    np.testing.assert_array_equal(kernel.rates(y, .002, m)[:12], spectrum.rates(y[:12], .002, m.spectrum))
    end = kernel.integrate(y, 2., -.002, m).y[:, -1]
    np.testing.assert_allclose(end[:12], spectrum.integrate(y[:12], 2., -.002, m.spectrum).y[:, -1], atol=1e-8)
    np.testing.assert_array_equal(end[12:], 0.)


def test_anchor_equivalence_but_different_activation():
    a, b = material(), material("mobility")
    y = np.zeros(14); y[12:] = .05; y[0] = 140.
    np.testing.assert_allclose(kernel.rates(y, 0., a)[[3, 6]], kernel.rates(y, 0., b)[[3, 6]])
    y[0] = 19.5
    assert kernel.rates(y, 0., a)[3] > 0
    assert kernel.rates(y, 0., b)[3] == 0


@pytest.mark.parametrize("kind", ["strength", "mobility"])
@pytest.mark.parametrize("stress_control", [False, True])
def test_power_symmetry_and_continuation(kind, stress_control):
    m = material(kind)
    y = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, .03, .04])
    old = y.copy()
    v = kernel.rates(y, .003, m, stress_control=stress_control)
    h = 1e-5
    psi_rate = (kernel.stored_energy(y+h*v, m)-kernel.stored_energy(y-h*v, m))/(2*h)
    assert v[8] == pytest.approx(psi_rate+v[9], abs=1e-9)
    assert v[9] >= 0
    sign = np.array([-1., -1., -1., 1., -1., -1., 1., -1., 1., 1., -1., -1., 1., 1.])
    np.testing.assert_allclose(kernel.rates(y*sign, -.003, m, stress_control=stress_control), sign*v)
    end = kernel.integrate(y, 2., -.002, m, stress_control=stress_control).y[:, -1]
    bdf = kernel.integrate(y, 2., -.002, m, stress_control=stress_control, method="BDF", rtol=1e-10).y[:, -1]
    mid = kernel.integrate(y, .7, -.002, m, stress_control=stress_control).y[:, -1]
    split = kernel.integrate(mid, 1.3, -.002, m, stress_control=stress_control).y[:, -1]
    np.testing.assert_allclose(end, bdf, atol=2e-8, rtol=1e-7)
    np.testing.assert_allclose(end, split, atol=2e-8, rtol=1e-7)
    np.testing.assert_array_equal(y, old)


def test_inactive_recovery_and_inward_boundaries():
    m = material(); y = np.zeros(14); y[12:] = .05
    end = kernel.integrate(y, 180., 0., m).y[:, -1]
    np.testing.assert_allclose(end[12:], .05/np.e, atol=1e-10)
    y[0] = 100.; y[12:] = 0
    assert np.all(kernel.rates(y, 0., m)[12:] >= 0)
    y[12:] = 1-m.minimum
    assert np.all(kernel.rates(y, 0., m)[12:] <= 0)


@pytest.mark.parametrize("changes", [dict(kind="unknown"), dict(minimum=0.),
    dict(minimum=1.1), dict(dynamic=-1.), dict(recovery=np.nan), dict(anchor_MPa=0.)])
def test_invalid_parameters(changes):
    with pytest.raises(ValueError):
        replace(material(), **changes)


def test_reject_double_evolution_and_invalid_state():
    m = material()
    with pytest.raises(ValueError):
        replace(m, spectrum=replace(m.spectrum, strength_ratios=(.9, 1.)))
    for y in [np.zeros(13), np.full(14, np.nan), np.full(14, 1j)]:
        with pytest.raises(ValueError):
            kernel.integrate(y, 1., 0., m)
    y = np.zeros(14); y[12] = -.01
    with pytest.raises(ValueError):
        kernel.integrate(y, 1., 0., m)
    assert y[12] == -.01
