from dataclasses import replace

import numpy as np
import pytest

from tools import p92_resistance_recovery_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(minimum=.9):
    branch = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    base = spectrum.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))
    return kernel.Material(base, "strength", minimum, 250., 1/180.,
                           evolution="static_softening", history="shared")


def test_inactive_static_softening_closed_form_and_zero_amplitude():
    mat = material()
    old = np.zeros(14)
    out = kernel.integrate(old, 360., 0., mat, stress_control=True).y[:, -1]
    np.testing.assert_allclose(out[12:], .1*(1-np.exp(-2.)), atol=1e-10)
    np.testing.assert_array_equal(out[:12], 0.)
    assert kernel.stored_energy(out, mat) == 0.
    zero = material(1.)
    old[0] = 70.
    actual = kernel.integrate(old, 2., -.002, zero).y[:, -1]
    expected = spectrum.integrate(old[:12], 2., -.002, zero.spectrum).y[:, -1]
    np.testing.assert_allclose(actual[:12], expected, atol=1e-8)
    np.testing.assert_array_equal(actual[12:], 0.)


def test_shared_rehardening_uses_both_irreversible_rates_not_recoverable():
    mat = material()
    y = np.zeros(14)
    y[0], y[10:], y[12:] = 70., [0., 0., 0., 0.], [.03, .03]
    rate = kernel.rates(y, 0., mat)
    expected = mat.recovery*(.1-.03)-mat.dynamic*(rate[3]+rate[6])*.03
    np.testing.assert_allclose(rate[12:], expected)
    other = y.copy()
    other[10:12] = [.001, -.002]
    np.testing.assert_array_equal(kernel.rates(other, 0., mat)[12:], rate[12:])
    for loss in (0., .1):
        y[12:] = loss
        bound_rate = kernel.rates(y, 0., mat)[12:]
        assert np.all(bound_rate >= -1e-15) if loss == 0. else np.all(bound_rate <= 1e-15)


@pytest.mark.parametrize("stress_control", [False, True])
def test_mechanical_power_reflection_and_continuation(stress_control):
    mat = material()
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, .03, .03])
    before = old.copy()
    rate = kernel.rates(old, -.002, mat, stress_control=stress_control)
    eps = 1e-5
    energy_rate = (kernel.stored_energy(old+eps*rate, mat)-kernel.stored_energy(old-eps*rate, mat))/(2*eps)
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-9)
    assert rate[9] >= 0.
    sign = np.ones(14)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(kernel.rates(old*sign, .002, mat, stress_control=stress_control), rate*sign)
    full = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control).y[:, -1]
    first = kernel.integrate(old, .7, -.002, mat, stress_control=stress_control).y[:, -1]
    split = kernel.integrate(first, 1.3, -.002, mat, stress_control=stress_control).y[:, -1]
    other = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control, method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, split, atol=2e-8, rtol=1e-7)
    np.testing.assert_allclose(full, other, atol=2e-8, rtol=1e-7)
    np.testing.assert_array_equal(old, before)
    assert full[12] == pytest.approx(full[13])
    assert np.all((full[12:] >= 0.) & (full[12:] <= .1))


def test_reject_unknown_evolution_or_history():
    for update in (dict(evolution="protocol-dependent"), dict(history="recoverable")):
        with pytest.raises(ValueError):
            replace(material(), **update)
