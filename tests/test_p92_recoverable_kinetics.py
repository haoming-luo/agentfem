"""Independent checks of the research nonlinear recoverable dashpots."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_two_rate_reference import Branch, Material


def material(powers=(1., 1.)):
    branch = Branch(1000., 1e-5, 3., (100., 200.), (10., 20.), (0., 0.))
    return kernel.Material(Material(10000., (branch, branch)),
                           (20000., 40000.), (.5, 10.), recoverable_powers=powers)


def exact_force(force, tau, power, time):
    if power == 1.:
        return force*np.exp(-time/tau)
    return force/(1+(power-1)*abs(force/120.)**(power-1)*time/tau)**(1/(power-1))


@pytest.mark.parametrize("powers", [(1., 1.), (2., 1.), (1., 2.), (2., 2.), (1.3, 2.7)])
@pytest.mark.parametrize("stress", [-100., 100.])
def test_creep_and_zero_stress_recovery_against_closed_form(powers, stress):
    mat = material(powers)
    old = np.zeros(12)
    old[0], old[7] = stress, stress/mat.backbone.modulus
    before = old.copy()
    out = kernel.integrate(old, 7., 0., mat, stress_control=True).y[:, -1]
    expected = np.array([(stress-exact_force(stress, t, m, 7.))/h
                         for h, t, m in zip(mat.moduli, mat.times, powers)])
    np.testing.assert_allclose(out[10:], expected, atol=2e-11, rtol=1e-8)
    np.testing.assert_array_equal(old, before)
    out[0] = 0.
    recovered = kernel.integrate(out, 15., 0., mat, stress_control=True).y[:, -1]
    expected_recovery = [-exact_force(-h*a, t, m, 15.)/h
                         for h, t, m, a in zip(mat.moduli, mat.times, powers, expected)]
    np.testing.assert_allclose(recovered[10:], expected_recovery, atol=2e-11, rtol=1e-8)
    assert np.all(abs(recovered[10:]) < abs(out[10:]))


def test_power_balance_reflection_and_split_with_active_plastic_flow():
    mat = material((1.5, 2.))
    mat = replace(mat, backbone=replace(mat.backbone,
        branches=tuple(replace(b, strength=10.) for b in mat.backbone.branches)))
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])
    rate = kernel.rates(old, -.002, mat)
    energy_rate = old[0]*rate[0]/mat.backbone.modulus
    energy_rate += sum(h*a*v for h, a, v in zip(mat.moduli, old[10:], rate[10:]))
    energy_rate += sum(old[o+i]*rate[o+i]/h for o, b in zip((1, 4), mat.backbone.branches)
                       for i, h in enumerate(b.hardening))
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-12)
    assert rate[9] >= 0
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(kernel.rates(old*sign, .002, mat), rate*sign, atol=1e-12)
    full = kernel.integrate(old, 2., -.002, mat).y[:, -1]
    first = kernel.integrate(old, .7, -.002, mat).y[:, -1]
    split = kernel.integrate(first, 1.3, -.002, mat).y[:, -1]
    other = kernel.integrate(old, 2., -.002, mat, method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, split, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(full, other, rtol=1e-7, atol=1e-7)


def test_zero_force_limits_validation_and_explicit_decoder():
    from tools.p92_recoverable_frozen import decode_material
    from tools.p92_primary_creep_reference import validate_material
    mat = material((2., 1.5))
    old = np.zeros(12)
    old[0], old[10:] = 100., 100/np.asarray(mat.moduli)
    np.testing.assert_array_equal(kernel.rates(old, 0., mat)[10:], [0., 0.])
    _, decoded = decode_material(asdict(mat), kernel.MODEL_ID)
    assert decoded.recoverable_powers == mat.recoverable_powers
    with pytest.raises(ValueError, match="isolate"):
        validate_material(mat)
    for powers in ((.9, 1.), (1., np.inf), (1.,)):
        with pytest.raises(ValueError):
            material(powers)


def test_parameter_substitution_freezes_original_weak_directions():
    from tools.p92_spectrum_recovery_joint import decode, decode_kinetics
    card = asdict(material())
    z = np.zeros(22)
    z[7:9] = [.1, -.2]
    decoded = decode_kinetics(card, z)
    assert decoded.backbone.branches == material().backbone.branches
    np.testing.assert_allclose(decoded.recoverable_powers, 1.+10.**z[7:9])
    assert card["recoverable_powers"] == (1., 1.)
    with pytest.raises(ValueError, match="baseline"):
        decode(asdict(decoded), np.zeros(22), (0., 0.))
