from dataclasses import replace

import numpy as np
import pytest

from tools import p92_critical_recovery_reference as kernel
from tools.p92_recoverable_spectrum import Material as Spectrum
from tools.p92_two_rate_reference import Branch, Material as Backbone


def material(recovery=.01):
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (recovery, recovery))
    return kernel.Material(Spectrum(Backbone(10000., (b, b)), (20000., 40000.), (.5, 10.)))


def test_power_identity_and_reflection():
    m = material()
    y = np.zeros(12)
    y[0] = 100.
    y[kernel.INDICES] = [10., 5., -10., -5.]
    rate = kernel.rates(y, .01, m)
    h, _, _ = kernel.coefficients(m)
    energy_rate = y[0]*rate[0]/m.spectrum.backbone.modulus
    energy_rate += sum(y[kernel.INDICES]*rate[kernel.INDICES]/h)
    energy_rate += sum(np.asarray(m.spectrum.moduli)*y[10:]*rate[10:])
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-13)
    assert rate[9] >= 0
    signs = np.ones(12)
    signs[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(kernel.rates(y*signs, -.01, m), rate*signs)


def test_zero_flow_static_decay():
    m = material(.1)
    y = np.zeros(12)
    y[kernel.INDICES] = [1., 2., 1., 2.]
    a = kernel.integrate(y, 2., 0., m, stress_control=True)
    np.testing.assert_allclose(a.y[kernel.INDICES, -1], y[kernel.INDICES]*np.exp(-.2), rtol=1e-9)
    assert not a.transitions


def test_saturation_reversal_and_independent_continuation():
    m = material(0.)
    y = np.zeros(12)
    y[0] = 500.
    before = y.copy()
    a = kernel.integrate(y, 5., 0., m, stress_control=True)
    assert len(a.transitions) >= 1
    _, bounds, _ = kernel.coefficients(m)
    np.testing.assert_allclose(a.y[kernel.INDICES, -1], bounds, atol=1e-8)
    b = kernel.integrate(y, 5., 0., m, stress_control=True, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-7, rtol=1e-7)
    c = kernel.integrate(y, .8, 0., m, stress_control=True)
    d = kernel.integrate(c.y[:, -1], 4.2, 0., m, stress_control=True)
    np.testing.assert_allclose(a.y[:, -1], d.y[:, -1], atol=1e-7, rtol=1e-7)
    reverse = a.y[:, -1].copy()
    reverse[0] *= -1
    rate = kernel.rates(reverse, 0., m, stress_control=True)
    assert np.all(rate[kernel.INDICES] < 0)
    np.testing.assert_array_equal(y, before)
    np.testing.assert_allclose(a.sol(a.t), a.y, atol=1e-8)
    assert a.sol(np.array([])).shape == (12, 0)
    np.testing.assert_allclose(a.sol(5.), a.y[:, -1])


def test_static_recovery_can_leave_boundary():
    m = material(.1)
    y = np.zeros(12)
    y[kernel.INDICES] = [10., 20., 10., 20.]
    y[0] = 30.
    a = kernel.integrate(y, .1, 0., m, stress_control=True)
    assert np.all(a.y[kernel.INDICES, -1] < y[kernel.INDICES])


def test_saturated_constant_stress_has_analytic_flow_and_kelvin_response():
    m = material(0.)
    y = np.zeros(12)
    y[0] = 500.
    _, bounds, _ = kernel.coefficients(m)
    y[kernel.INDICES] = bounds
    duration = 2.
    flow = .001*((500.-30.-10.)/120.)**3
    result = kernel.integrate(y, duration, 0., m, stress_control=True)
    end = result.y[:, -1]
    np.testing.assert_allclose(end[kernel.INDICES], bounds, atol=1e-10)
    np.testing.assert_allclose(end[[3, 6]], duration*flow, atol=1e-11)
    kelvin = 500./np.array(m.spectrum.moduli)*(1-np.exp(-duration/np.array(m.spectrum.times)))
    np.testing.assert_allclose(end[10:], kelvin, atol=1e-11)
    assert end[7] == pytest.approx(2*duration*flow+sum(kelvin), abs=1e-10)
    assert not result.transitions


def test_integrated_reflection_and_release_during_relaxation():
    m = material(.1)
    y = np.zeros(12)
    y[0] = 500.
    y[kernel.INDICES] = kernel.coefficients(m)[1]
    signs = np.ones(12)
    signs[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    a = kernel.integrate(y, 20., 0., m)
    b = kernel.integrate(y*signs, 20., 0., m)
    assert any(kind == "release" for _, kind, _ in a.transitions)
    np.testing.assert_allclose(a.y[:, -1]*signs, b.y[:, -1], rtol=1e-8, atol=1e-8)
    independent = kernel.integrate(y, 20., 0., m, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(a.y[:, -1], independent.y[:, -1], rtol=1e-7, atol=1e-7)


def test_model_identity_roundtrip_and_fit_isolation():
    from dataclasses import asdict
    from pathlib import Path
    from tools.p92_recoverable_frozen import decode_material
    from tools.p92_spectrum_recovery_joint import identify
    from tools.p92_recoverable_spectrum import MODEL_ID
    m = material()
    module, restored = decode_material(asdict(m), kernel.MODEL_ID)
    assert module is kernel and restored == m
    with pytest.raises(ValueError, match="isolate"):
        identify({"model_id": MODEL_ID}, Path("unused"), (0., 0.),
                 critical_recovery=True, directional=True)


@pytest.mark.parametrize("duration,rate", [(0., 0.), (-1., 0.), (1., np.nan)])
def test_invalid_controls(duration, rate):
    with pytest.raises(ValueError):
        kernel.integrate(np.zeros(12), duration, rate, material())


def test_incompatible_hypothesis_and_invalid_state():
    with pytest.raises(ValueError, match="Isolate"):
        kernel.Material(replace(material().spectrum, directional_recovery=True))
    y = np.zeros(12)
    y[1] = 100.
    with pytest.raises(ValueError, match="domain"):
        kernel.integrate(y, 1., 0., material())
