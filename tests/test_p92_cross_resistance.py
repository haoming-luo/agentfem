from dataclasses import replace

import numpy as np
import pytest

from tools import p92_cross_resistance_reference as kernel
from tools.p92_recoverable_spectrum import Material as Spectrum
from tools.p92_two_rate_reference import Branch, Material as Backbone


def material(threshold=10.):
    a = Branch(threshold, .001, 3., (100., 200.), (10., 20.), (.01, .02))
    b = replace(a, hardening=(300., 400.), static_recovery=(.04, .08))
    return kernel.Material(Spectrum(Backbone(10000., (a, b)), (20000., 40000.), (.5, 10.)))


@pytest.mark.parametrize("stress_control", [False, True])
def test_power_reflection_and_invariants(stress_control):
    m = material()
    rng = np.random.default_rng(42)
    signs = np.ones(12)
    signs[[0, 1, 2, 4, 7, 10, 11]] = -1
    h, bounds = np.array([100., 200., 300., 400.]), np.array([10., 20., 10., 20.])
    for _ in range(50):
        y = np.zeros(12)
        y[0] = rng.normal()*100.
        y[[1, 2, 4]] = rng.uniform(-1., 1., 3)*bounds[:3]
        y[5] = rng.uniform(0., 20.)
        y[10:] = rng.normal(size=2)*.001
        r = kernel.rates(y, .01, m, stress_control=stress_control)
        energy_rate = y[0]*r[0]/10000.+sum(y[[1, 2, 4, 5]]*r[[1, 2, 4, 5]]/h)
        energy_rate += sum(np.array(m.spectrum.moduli)*y[10:]*r[10:])
        assert r[8] == pytest.approx(energy_rate+r[9], abs=1e-11)
        assert r[9] >= 0
        np.testing.assert_allclose(kernel.rates(y*signs, -.01, m, stress_control=stress_control), r*signs)
        for value, sign in ((0., -1), (20., 1)):
            test = y.copy()
            test[5] = value
            assert sign*kernel.rates(test, .01, m, stress_control=stress_control)[5] <= 0
        for i, a in zip((1, 2, 4), bounds[:3], strict=True):
            for sign in (-1., 1.):
                test = y.copy()
                test[i] = sign*a
                assert sign*kernel.rates(test, .01, m, stress_control=stress_control)[i] <= 1e-10


def test_inactive_analytic_recovery_and_removed_exponent():
    m = material(threshold=1000.)
    y = np.zeros(12)
    y[[0, 1, 2, 4, 5]] = [10., 2., -3., 4., 5.]
    result = kernel.integrate(y, 2., 0., m, stress_control=True).y[:, -1]
    np.testing.assert_allclose(result[[1, 2, 4, 5]], y[[1, 2, 4, 5]]*np.exp(-2*np.array([.01, .02, .04, .08])), rtol=1e-9)
    np.testing.assert_array_equal(result[[3, 6]], [0., 0.])
    a, b = m.spectrum.backbone.branches
    alternate = kernel.Material(replace(m.spectrum, backbone=replace(m.spectrum.backbone,
        branches=(a, replace(b, ratcheting=(3., 99.))))))
    y[0] = 2000.
    np.testing.assert_array_equal(kernel.rates(y, .01, m), kernel.rates(y, .01, alternate))


@pytest.mark.parametrize("stress_control", [False, True])
def test_independent_and_split_integration(stress_control):
    m, y = material(), np.zeros(12)
    y[0] = 100.
    original = y.copy()
    a = kernel.integrate(y, 2., .001, m, stress_control=stress_control)
    b = kernel.integrate(y, 2., .001, m, stress_control=stress_control, method="BDF", rtol=1e-10)
    c = kernel.integrate(y, .8, .001, m, stress_control=stress_control)
    d = kernel.integrate(c.y[:, -1], 1.2, .001, m, stress_control=stress_control)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(a.y[:, -1], d.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(y, original)


def test_negative_resistance_and_combined_hypotheses_rejected():
    y = np.zeros(12)
    y[5] = -.001
    with pytest.raises(ValueError):
        kernel.integrate(y, 1., 0., material())
    assert y[5] == -.001
    with pytest.raises(ValueError):
        kernel.Material(replace(material().spectrum, directional_recovery=True))
