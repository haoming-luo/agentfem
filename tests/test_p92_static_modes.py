from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from tools import p92_static_modes_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material as Backbone


def material(mode="diagonal", threshold=10.):
    a = Branch(threshold, .001, 3., (100., 200.), (10., 20.), (.01, .02))
    b = replace(a, hardening=(300., 400.), static_recovery=(.04, .08))
    return kernel.Material(spectrum.Material(Backbone(10000., (a, b)), (20000., 40000.), (.5, 10.)), mode)


@pytest.mark.parametrize("mode", ["diagonal", "shared_slow", "difference_slow", "exchanged_slow"])
def test_eigenvalues_and_inactive_flow_exact_solution(mode):
    m = material(mode, threshold=1000.)
    np.testing.assert_allclose(np.linalg.eigvalsh(kernel.matrix(m)), [.02, .08], atol=1e-15)
    y = np.zeros(12)
    y[[2, 5]] = [5., 10.]
    a = kernel.integrate(y, 2., 0., m, stress_control=True)
    sqrt_h = np.sqrt([200., 400.])
    expected = sqrt_h*(expm(-2.*kernel.matrix(m))@(y[[2, 5]]/sqrt_h))
    np.testing.assert_allclose(a.y[[2, 5], -1], expected, rtol=1e-9)
    assert a.y[9, -1] == pytest.approx(kernel.stored_energy(y, m)-kernel.stored_energy(a.y[:, -1], m), abs=1e-10)


def test_original_arm_is_exact_and_keeps_recoverability():
    m = material()
    y = np.zeros(12)
    y[0] = 100.
    a = kernel.integrate(y, 2., 0., m)
    b = spectrum.integrate(y, 2., 0., m.spectrum)
    np.testing.assert_array_equal(a.y, b.y)
    assert np.all(a.y[10:, -1] > 0)


@pytest.mark.parametrize("mode", ["shared_slow", "difference_slow", "exchanged_slow", "recoverable_shared", "recoverable_difference"])
def test_power_reflection_and_continuation(mode):
    m = material(mode)
    y = np.zeros(12)
    y[0] = 100.
    y[[1, 2, 4, 5]] = [2., 5., 2., 5.]
    r = kernel.rates(y, .01, m)
    h = np.array([100., 200., 300., 400.])
    energy_rate = y[0]*r[0]/10000.+sum(y[[1, 2, 4, 5]]*r[[1, 2, 4, 5]]/h)
    energy_rate += sum(np.array(m.spectrum.moduli)*y[10:]*r[10:])
    assert r[8] == pytest.approx(energy_rate+r[9], abs=1e-13)
    assert r[9] >= 0
    signs = np.ones(12)
    signs[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(kernel.rates(y*signs, -.01, m), r*signs)
    a = kernel.integrate(y, 2., .001, m)
    b = kernel.integrate(y, 2., .001, m, method="BDF", rtol=1e-10)
    c = kernel.integrate(y, .8, .001, m)
    d = kernel.integrate(c.y[:, -1], 1.2, .001, m)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(a.y[:, -1], d.y[:, -1], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("mode", ["recoverable_shared", "recoverable_difference", "recoverable_exchanged"])
def test_recoverable_mode_analytic_creep_and_unloaded_recovery(mode):
    m = material(mode, threshold=1000.)
    y = np.zeros(12)
    y[0], y[5], y[11] = 100., 5., .0001
    scales = np.sqrt([400., 40000.])
    z = scales*np.array([y[5]/400., y[11]])
    equilibrium = np.array([0., 100./scales[1]])
    a = kernel.integrate(y, 2., 0., m, stress_control=True)
    expected = equilibrium+expm(-2*kernel.matrix(m))@(z-equilibrium)
    np.testing.assert_allclose(a.y[[5, 11], -1], expected*np.array([scales[0], 1/scales[1]]), atol=1e-9)
    np.testing.assert_allclose(np.linalg.eigvalsh(kernel.matrix(m)), [.08, .1], atol=1e-15)
    unloaded = y.copy()
    unloaded[0] = 0.
    end = kernel.integrate(unloaded, 1000., 0., m, stress_control=True).y[:, -1]
    np.testing.assert_allclose(end[[5, 11]], 0., atol=1e-8)


@pytest.mark.parametrize("mode", ["shared_slow", "difference_slow", "exchanged_slow", "recoverable_shared", "recoverable_difference"])
def test_dissipation_for_signed_internal_states(mode):
    m = material(mode)
    rng = np.random.default_rng(11)
    for _ in range(40):
        y = rng.normal(size=12)
        y[0] *= 100.
        y[[1, 2, 4, 5]] *= 20.
        y[[3, 6, 9]] = 0.
        y[10:] *= .001
        assert kernel.rates(y, 0., m)[9] >= -1e-12


def test_other_hypotheses_are_not_combined():
    with pytest.raises(ValueError, match="Isolate"):
        kernel.Material(replace(material().spectrum, directional_recovery=True))
    with pytest.raises(ValueError, match="Unknown"):
        material("arbitrary_angle")


def test_exchange_is_exactly_original_equations_with_permuted_rates():
    m = material("exchanged_slow")
    a, b = m.spectrum.backbone.branches
    other = replace(m.spectrum, backbone=replace(m.spectrum.backbone, branches=(
        replace(a, static_recovery=(a.static_recovery[0], b.static_recovery[1])),
        replace(b, static_recovery=(b.static_recovery[0], a.static_recovery[1])))))
    y = np.zeros(12)
    y[[0, 1, 2, 4, 5]] = [80., 3., -4., 5., -6.]
    np.testing.assert_allclose(kernel.rates(y, .01, m), spectrum.rates(y, .01, other), atol=1e-14)
    first = kernel.integrate(y, 5., .001, m)
    second = spectrum.integrate(y, 5., .001, other)
    np.testing.assert_allclose(first.y[:, -1], second.y[:, -1], atol=1e-8)


def test_kelvin_exchange_preserves_equations_with_reassigned_time_scales():
    m = material("recoverable_exchanged")
    a, b = m.spectrum.backbone.branches
    other = replace(m.spectrum, backbone=replace(m.spectrum.backbone, branches=(a,
        replace(b, static_recovery=(b.static_recovery[0], 1/m.spectrum.times[1])))),
        times=(m.spectrum.times[0], 1/b.static_recovery[1]))
    y = np.zeros(12)
    y[[0, 1, 2, 4, 5, 10, 11]] = [80., 3., -4., 5., -6., .001, -.0005]
    for controlled in (True, False):
        np.testing.assert_allclose(kernel.rates(y, .01, m, stress_control=controlled),
            spectrum.rates(y, .01, other, stress_control=controlled), atol=1e-13)
        first = kernel.integrate(y, 5., .001, m, stress_control=controlled)
        second = spectrum.integrate(y, 5., .001, other, stress_control=controlled)
        np.testing.assert_allclose(first.y[:, -1], second.y[:, -1], atol=1e-8)


@pytest.mark.parametrize("mode", ["recoverable_shared", "recoverable_difference", "recoverable_exchanged"])
def test_recoverable_reassignment_rejects_zero_decay_eigenvalue(mode):
    s = material().spectrum
    a, b = s.backbone.branches
    s = replace(s, backbone=replace(s.backbone, branches=(a, replace(b, static_recovery=(.04, 0.)))))
    with pytest.raises(ValueError, match="positive decay"):
        kernel.Material(s, mode)
