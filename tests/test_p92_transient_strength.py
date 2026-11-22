from dataclasses import asdict, replace

import numpy as np
import pytest

from tools import p92_transient_strength_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(maximum=.1):
    branch = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    base = spectrum.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))
    return kernel.Material(base, maximum, 250., 1/180.)


def test_zero_flow_exact_recovery_and_unchanged_zero_amplitude():
    mat = material()
    old = np.zeros(13)
    old[12] = .08
    end = kernel.integrate(old, 360., 0., mat, stress_control=True).y[:, -1]
    assert end[12] == pytest.approx(.08*np.exp(-2.), abs=1e-10)
    np.testing.assert_array_equal(end[:12], 0.)
    old[0], old[12] = 70., 0.
    zero = material(0.)
    actual = kernel.integrate(old, 2., -.002, zero).y[:, -1]
    expected = spectrum.integrate(old[:12], 2., -.002, zero.spectrum).y[:, -1]
    np.testing.assert_allclose(actual[:12], expected, atol=1e-8)
    assert actual[12] == 0.


def test_shared_irreversible_driver_and_invariant_boundaries():
    mat = material()
    y = np.zeros(13)
    y[0], y[12] = 70., .03
    r = kernel.rates(y, 0., mat)
    assert r[12] == pytest.approx(mat.build*(r[3]+r[6])*.07-mat.recovery*.03)
    other = y.copy()
    other[10:12] = [.001, -.002]
    assert kernel.rates(other, 0., mat)[12] == r[12]
    for h in (0., .1):
        y[12] = h
        rate = kernel.rates(y, 0., mat)[12]
        assert rate >= 0. if h == 0. else rate <= 0.


def test_unloaded_recoverable_strain_does_not_create_permanent_creep_or_strength():
    mat = material()
    old = np.zeros(13)
    old[10:12] = [.001, -.0003]
    old[7] = sum(old[10:12])
    old[12] = .04
    duration = 3.
    end = kernel.integrate(old, duration, 0., mat, stress_control=True).y[:, -1]
    recovered = old[10:12]*np.exp(-duration/np.asarray(mat.spectrum.times))
    np.testing.assert_allclose(end[10:12], recovered, atol=1e-12, rtol=1e-8)
    assert end[7] == pytest.approx(sum(recovered), abs=1e-12)
    np.testing.assert_array_equal(end[[0, 3, 6, 8]], 0.)
    assert end[12] == pytest.approx(old[12]*np.exp(-mat.recovery*duration), abs=1e-12)
    assert end[9] == pytest.approx(kernel.stored_energy(old, mat)-kernel.stored_energy(end, mat),
                                  abs=1e-10)


@pytest.mark.parametrize("stress_control", [False, True])
def test_power_reflection_split_and_independent_integration(stress_control):
    mat = material()
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, .03])
    before = old.copy()
    rate = kernel.rates(old, -.002, mat, stress_control=stress_control)
    eps = 1e-5
    energy_rate = (kernel.stored_energy(old+eps*rate, mat)-kernel.stored_energy(old-eps*rate, mat))/(2*eps)
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-9)
    assert rate[9] >= 0.
    sign = np.ones(13)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(kernel.rates(old*sign, .002, mat, stress_control=stress_control), rate*sign)
    full = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control).y[:, -1]
    first = kernel.integrate(old, .7, -.002, mat, stress_control=stress_control).y[:, -1]
    split = kernel.integrate(first, 1.3, -.002, mat, stress_control=stress_control).y[:, -1]
    other = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control, method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, split, atol=2e-8, rtol=1e-7)
    np.testing.assert_allclose(full, other, atol=2e-8, rtol=1e-7)
    np.testing.assert_array_equal(old, before)
    assert 0 <= full[12] <= .1


def test_reject_invalid_material_and_history_without_mutation():
    for kwargs in (dict(maximum=-1.), dict(build=-1.), dict(recovery=np.nan),
                   dict(spectrum=replace(material().spectrum, recoverable_powers=(2., 1.))),
                   dict(spectrum=replace(material().spectrum, static_powers=(1., 0.))),
                   dict(spectrum=replace(material().spectrum, static_ratios=(.5, 1.)))):
        with pytest.raises(ValueError):
            replace(material(), **kwargs)
    old = np.zeros(13)
    old[12] = -.01
    before = old.copy()
    with pytest.raises(ValueError):
        kernel.integrate(old, 1., 0., material())
    np.testing.assert_array_equal(old, before)


def test_matched_decoder_replaces_two_coordinates_without_fit_dimension_growth():
    from tools.p92_spectrum_recovery_joint import decode_transient, identify
    card = asdict(material().spectrum)
    base = decode_transient(card, np.zeros(22))
    z = np.zeros(22)
    z[7:9] = [.1, -.2]
    changed = decode_transient(card, z)
    assert base.spectrum == changed.spectrum
    assert changed.maximum == base.maximum == .1
    assert changed.build/base.build == pytest.approx(10**.1)
    assert changed.recovery/base.recovery == pytest.approx(10**-.2)
    assert changed.spectrum.backbone.branches[0].static_recovery == material().spectrum.backbone.branches[0].static_recovery
    for z in (np.zeros(23), np.full(22, np.nan)):
        with pytest.raises(ValueError):
            decode_transient(card, z)
    with pytest.raises(ValueError, match="isolate"):
        identify(dict(model_id=spectrum.MODEL_ID), None, (0., 0.),
                 recoverable_kinetics=True, transient_strength=True)


def test_frozen_material_identity_round_trip():
    from tools.p92_recoverable_frozen import decode_material
    expected = material()
    module, actual = decode_material(asdict(expected), kernel.MODEL_ID)
    assert module is kernel
    assert actual == expected


def test_frozen_comparison_includes_dimensionless_strength_history():
    from copy import deepcopy
    from tools.p92_two_rate_frozen import numerical_difference
    a = dict(segments=[dict(sampled_stress_MPa=[1.], sampled_strain=[.001], end_state=[0.]*13)])
    b = deepcopy(a)
    b["segments"][0]["end_state"][12] = 1e-6
    result = numerical_difference(a, b, extra_strain_indices=(10, 11), dimensionless_indices=(12,))
    assert not result["passed"]
    assert result["dimensionless_state_difference"] == 1e-6
    assert result["strain_state_difference"] == 0.


def test_failed_integrator_cannot_mutate_the_committed_input(monkeypatch):
    from types import SimpleNamespace
    old = np.zeros(13)
    before = old.copy()

    def fail(fun, interval, initial, **kwargs):
        initial[:] = 123.
        return SimpleNamespace(success=False)

    monkeypatch.setattr(kernel, "solve_ivp", fail)
    with pytest.raises(RuntimeError, match="input retained"):
        kernel.integrate(old, 1., 0., material())
    np.testing.assert_array_equal(old, before)
