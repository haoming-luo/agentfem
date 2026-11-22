from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from tools import p92_primary_creep_reference as kernel
from tools.p92_recoverable_frozen import decode_material
from tools.p92_spectrum_recovery_joint import identify
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    return kernel.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


@pytest.mark.parametrize("stress", [-40., 0., 40.])
def test_constant_stress_analytic_and_continuation(stress):
    m = material(); old = np.zeros(12); old[0] = stress
    before = old.copy()
    a = kernel.integrate(old, 12., 0., m, stress_control=True)
    h, tau = m.moduli[1], m.times[1]
    exact = 120/h*np.log1p(abs(stress)*12/(120*tau))
    assert a.y[11, -1] == pytest.approx(exact, abs=1e-12)
    b = kernel.integrate(old, 12., 0., m, stress_control=True, method="BDF", rtol=1e-10)
    c = kernel.integrate(old, 4., 0., m, stress_control=True)
    d = kernel.integrate(c.y[:, -1], 8., 0., m, stress_control=True)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(a.y[:, -1], d.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(old, before)


def test_reflection_memory_and_power():
    m = material(); y = np.array([-40., 5., 10., .01, 5., 10., .02, 0., 0., 0., .001, .003])
    a = kernel.rates(y, .01, m)
    energy_rate = y[0]*a[0]/m.backbone.modulus + m.moduli[0]*y[10]*a[10]
    for offset, b in zip((1, 4), m.backbone.branches):
        energy_rate += sum(y[offset+i]*a[offset+i]/h for i, h in enumerate(b.hardening))
    assert a[8] == pytest.approx(energy_rate+a[9], abs=1e-12)
    assert a[9] >= 0 and a[11] > 0
    signs = np.ones(12); signs[[0, 1, 2, 4, 5, 7, 10]] = -1
    np.testing.assert_allclose(kernel.rates(signs*y, -.01, m), signs*a, atol=1e-14)
    assert kernel.primary_flow(0., y[11], m) == 0
    assert abs(kernel.primary_flow(40., .01, m)) < abs(kernel.primary_flow(40., 0., m))


def test_distinct_identity_and_no_combination():
    m = material()
    module, decoded = decode_material(asdict(m), kernel.MODEL_ID)
    assert module is kernel and decoded == m
    assert kernel.STATE_LAYOUT[11] == "accumulated_primary_strain"
    with pytest.raises(ValueError, match="isolate"):
        kernel.integrate(np.zeros(12), 1., 0., replace(m, directional_recovery=True))
    y = np.zeros(12); y[11] = -.001
    with pytest.raises(ValueError, match="state"):
        kernel.integrate(y, 1., 0., m)
    with pytest.raises(ValueError, match="isolate"):
        identify(dict(model_id="research.scalar-two-rate-recoverable-spectrum.v1"),
                 Path("unused"), (0., 0.), primary=True, directional=True)


def test_zero_stress_holds_primary_history_without_reset():
    m = material(); old = np.zeros(12); old[11] = .003
    result = kernel.integrate(old, 1800., 0., m, stress_control=True)
    np.testing.assert_array_equal(result.y[11], np.full(result.y.shape[1], .003))
    np.testing.assert_array_equal(result.y[:11], np.zeros_like(result.y[:11]))


def test_reversal_retains_primary_hardening_and_no_new_parameters():
    from tools import p92_recoverable_spectrum as original
    from tools.p92_spectrum_recovery_joint import decode
    m = material(); old = np.zeros(12); old[0] = 40.
    held = kernel.integrate(old, 180., 0., m, stress_control=True).y[:, -1]
    rate = abs(kernel.primary_flow(-40., held[11], m))
    assert rate < abs(kernel.primary_flow(-40., 0., m))
    assert kernel.stored_energy(held, m) < original.stored_energy(held, m)
    card = asdict(m)
    decoded = decode(card, np.zeros(22), (0., 0.))
    assert asdict(decoded) == card
    assert len(card) == len(asdict(original.Material(m.backbone, m.moduli, m.times)))


@pytest.mark.parametrize("rate", [-.002, .002])
def test_strain_control_continuation_and_independent_solution(rate):
    m = material(); old = np.zeros(12)
    whole = kernel.integrate(old, 3., rate, m)
    independent = kernel.integrate(old, 3., rate, m, method="BDF", rtol=1e-10)
    first = kernel.integrate(old, 1., rate, m)
    rest = kernel.integrate(first.y[:, -1], 2., rate, m)
    np.testing.assert_allclose(whole.y[:, -1], independent.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(whole.y[:, -1], rest.y[:, -1], atol=1e-7, rtol=1e-7)
