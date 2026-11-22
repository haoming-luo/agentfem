from dataclasses import replace

import numpy as np
import pytest

from tools import p92_common_memory_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material as Backbone


def material(shared=True, threshold=10.):
    a = Branch(threshold, .001, 3., (100., 200.), (10., 20.), (.01, .02))
    b = replace(a, hardening=(300., 400.), static_recovery=(.04, .08))
    return kernel.Material(spectrum.Material(Backbone(10000., (a, b)), (20000., 40000.), (.5, 10.)), shared)


def test_control_identity_and_zero_memory_flow_identity():
    y = np.zeros(12)
    y[0] = 100.
    m = material(False)
    np.testing.assert_array_equal(kernel.integrate(y, 2., 0., m).y,
                                  spectrum.integrate(y, 2., 0., m.spectrum).y)
    shared = kernel.rates(y, .01, material())
    original = kernel.rates(y, .01, m)
    np.testing.assert_allclose(shared[[0, 3, 6, 7, 8, 9, 10, 11]],
                               original[[0, 3, 6, 7, 8, 9, 10, 11]], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("stress_control", [False, True])
@pytest.mark.parametrize("shared", [True, "slow"])
def test_power_reflection_and_boundary_invariance(stress_control, shared):
    m = material(shared)
    rng = np.random.default_rng(15)
    h, bounds = np.array([100., 200., 300., 400.]), np.array([10., 20., 10., 20.])
    signs = np.ones(12)
    signs[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    for _ in range(30):
        y = np.zeros(12)
        y[0] = rng.normal()*100.
        y[[1, 2, 4, 5]] = rng.uniform(-1, 1, 4)*bounds
        y[10:] = rng.normal(size=2)*.001
        r = kernel.rates(y, .01, m, stress_control=stress_control)
        energy_rate = y[0]*r[0]/10000.+sum(y[[1, 2, 4, 5]]*r[[1, 2, 4, 5]]/h)
        energy_rate += sum(np.array(m.spectrum.moduli)*y[10:]*r[10:])
        assert r[8] == pytest.approx(energy_rate+r[9], abs=1e-11)
        assert r[9] >= 0
        np.testing.assert_allclose(kernel.rates(y*signs, -.01, m, stress_control=stress_control), r*signs)
        for idx, bound in zip([1, 2, 4, 5], bounds, strict=True):
            for direction in (-1., 1.):
                boundary = y.copy()
                boundary[idx] = direction*bound
                assert direction*kernel.rates(boundary, .01, m, stress_control=stress_control)[idx] <= 1e-10


def test_inactive_recovery_and_kelvin_analytic_solution():
    m = material(threshold=1000.)
    y = np.zeros(12)
    y[[0, 1, 2, 4, 5]] = [10., 2., -3., 4., -5.]
    y[10:] = [.001, -.001]
    result = kernel.integrate(y, 2., 0., m, stress_control=True).y[:, -1]
    np.testing.assert_allclose(result[[1, 2, 4, 5]], y[[1, 2, 4, 5]]*np.exp(-2*np.array([.01, .02, .04, .08])), rtol=1e-9)
    eq = y[0]/np.array(m.spectrum.moduli)
    np.testing.assert_allclose(result[10:], eq+(y[10:]-eq)*np.exp(-2/np.array(m.spectrum.times)), atol=1e-12)
    np.testing.assert_array_equal(result[[3, 6]], [0., 0.])


@pytest.mark.parametrize("stress_control", [False, True])
@pytest.mark.parametrize("shared", [True, "slow"])
def test_continuation_independent_and_input_preservation(stress_control, shared):
    y = np.zeros(12)
    y[0] = 100.
    old = y.copy()
    m = material(shared)
    a = kernel.integrate(y, 2., .001, m, stress_control=stress_control)
    b = kernel.integrate(y, 2., .001, m, stress_control=stress_control, method="BDF", rtol=1e-10)
    c = kernel.integrate(y, .8, .001, m, stress_control=stress_control)
    d = kernel.integrate(c.y[:, -1], 1.2, .001, m, stress_control=stress_control)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(a.y[:, -1], d.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(y, old)


def test_hypothesis_combinations_rejected():
    for options in ({"directional_recovery": True}, {"recoverable_powers": (1., 2.)},
                    {"static_powers": (1., 0.)}, {"hardening_ratios": (.9, 1.)}):
        with pytest.raises(ValueError):
            kernel.Material(replace(material().spectrum, **options))


def test_invalid_state_rejected_without_modification():
    y = np.zeros(12)
    y[1] = 11.
    with pytest.raises(ValueError):
        kernel.integrate(y, 1., 0., material())
    assert y[1] == 11.
