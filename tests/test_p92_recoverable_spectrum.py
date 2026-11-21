from dataclasses import replace

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as spectrum
from tools import p92_recoverable_reference as single
from tools.p92_two_rate_reference import Branch, Material


def material():
    branch = Branch(1000., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return spectrum.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))


def test_two_time_creep_recovery_exact_and_input_retained():
    mat = material()
    old = np.zeros(12)
    old[0], old[7] = 100., .01
    before = old.copy()
    solution = spectrum.integrate(old, 20., 0., mat, stress_control=True)
    expected = 100/np.asarray(mat.moduli)*(1-np.exp(-20/np.asarray(mat.times)))
    np.testing.assert_allclose(solution.y[10:, -1], expected, atol=1e-11)
    assert solution.y[7, -1] == pytest.approx(.01+sum(expected), abs=1e-11)
    np.testing.assert_array_equal(old, before)
    unloaded = solution.y[:, -1].copy()
    unloaded[0] = 0.
    recovery = spectrum.integrate(unloaded, 20., 0., mat, stress_control=True)
    np.testing.assert_allclose(recovery.y[10:, -1], expected*np.exp(-20/np.asarray(mat.times)), atol=1e-11)


def test_equal_times_reduce_to_single_compliance_and_independent_integration():
    mat = replace(material(), times=(.5, .5))
    one = single.Material(mat.backbone, 1/sum(1/np.asarray(mat.moduli)), .5)
    a = spectrum.integrate(np.zeros(12), 2., .002, mat)
    b = single.integrate(np.zeros(11), 2., .002, one)
    np.testing.assert_allclose(a.y[:10, -1], b.y[:10, -1], rtol=1e-7, atol=1e-9)
    assert sum(a.y[10:, -1]) == pytest.approx(b.y[10, -1], abs=1e-11)
    c = spectrum.integrate(np.zeros(12), 2., .002, mat, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(a.y[:, -1], c.y[:, -1], rtol=1e-7, atol=1e-8)


def test_power_identity_with_active_flow_and_split_continuation():
    mat = material()
    mat = replace(mat, backbone=replace(mat.backbone,
        branches=tuple(replace(b, strength=10.) for b in mat.backbone.branches)))
    state = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])
    rate = spectrum.rates(state, .003, mat)
    energy_rate = state[0]*rate[0]/mat.backbone.modulus+sum(
        h*a*v for h, a, v in zip(mat.moduli, state[10:], rate[10:], strict=True))
    for offset, branch in zip((1, 4), mat.backbone.branches):
        energy_rate += sum(state[offset+i]*rate[offset+i]/h for i, h in enumerate(branch.hardening))
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-12)
    assert rate[9] >= 0
    full = spectrum.integrate(state, 2., -.002, mat)
    first = spectrum.integrate(state, .7, -.002, mat)
    second = spectrum.integrate(first.y[:, -1], 1.3, -.002, mat)
    np.testing.assert_allclose(full.y[:, -1], second.y[:, -1], atol=1e-7, rtol=1e-7)


def test_reject_wrong_state_identity_and_invalid_spectrum():
    with pytest.raises(ValueError, match="state"):
        spectrum.integrate(np.zeros(11), 1., 0., material())
    for changes in (dict(moduli=(0., 1.)), dict(times=(2., 1.)), dict(times=(1., np.nan))):
        with pytest.raises(ValueError, match="positive"):
            replace(material(), **changes)


def test_frozen_numerical_check_covers_slow_recoverable_state():
    from copy import deepcopy
    from tools.p92_two_rate_frozen import numerical_difference
    a = dict(segments=[dict(sampled_stress_MPa=[0.], sampled_strain=[0.], end_state=[0.]*12)])
    b = deepcopy(a)
    b["segments"][0]["end_state"][11] = 1e-5
    assert not numerical_difference(a, b, extra_strain_indices=(10, 11))["passed"]


def test_constant_strain_relaxation_matches_matrix_exponential():
    from scipy.linalg import expm
    mat = material()
    h, tau = np.asarray(mat.moduli), np.asarray(mat.times)
    matrix = np.zeros((3, 3))
    matrix[1:, 0] = 1/(h*tau)
    matrix[1:, 1:] = -np.diag(1/tau)
    matrix[0] = -mat.backbone.modulus*matrix[1:].sum(axis=0)
    old = np.zeros(12)
    old[0], old[7] = 100., .01
    actual = spectrum.integrate(old, 7., 0., mat).y[:, -1]
    expected = expm(matrix*7.)@np.array([100., 0., 0.])
    np.testing.assert_allclose(actual[[0, 10, 11]], expected, atol=1e-8, rtol=1e-9)
    assert actual[7] == pytest.approx(.01)


def test_signed_loading_symmetry_does_not_reverse_accumulation_or_dissipation():
    mat = material()
    mat = replace(mat, backbone=replace(mat.backbone,
        branches=tuple(replace(b, strength=10.) for b in mat.backbone.branches)))
    state = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(spectrum.rates(state*sign, -.003, mat),
                               spectrum.rates(state, .003, mat)*sign, atol=1e-12)
    assert spectrum.stored_energy(state*sign, mat) == spectrum.stored_energy(state, mat)


def test_research_material_decoder_requires_explicit_spectrum_identity():
    from dataclasses import asdict
    from tools.p92_recoverable_frozen import decode_material
    card = asdict(material())
    module, decoded = decode_material(card, spectrum.MODEL_ID)
    assert module is spectrum
    np.testing.assert_allclose(decoded.moduli, material().moduli)
    assert len(module.STATE_LAYOUT) == 12
    with pytest.raises(ValueError, match="identity"):
        decode_material(card, "unrecognized")


def test_branch_resistance_memory_preserves_dissipation_and_zero_limit():
    mat = material()
    mat = replace(mat, backbone=replace(mat.backbone,
        branches=tuple(replace(b, strength=20.) for b in mat.backbone.branches)))
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])
    unchanged = replace(mat, strength_ratios=(.5, .7), strength_rates=(0., 0.))
    np.testing.assert_array_equal(spectrum.rates(old, .003, mat), spectrum.rates(old, .003, unchanged))
    evolved = replace(mat, strength_ratios=(.5, .7), strength_rates=(100., 100.))
    rate = spectrum.rates(old, .003, evolved)
    stored_rate = old[0]*rate[0]/mat.backbone.modulus+sum(
        h*a*v for h, a, v in zip(mat.moduli, old[10:], rate[10:], strict=True))
    stored_rate += sum(old[o+i]*rate[o+i]/h for o, b in zip((1, 4), mat.backbone.branches)
                       for i, h in enumerate(b.hardening))
    assert rate[8] == pytest.approx(stored_rate+rate[9], abs=1e-12)
    assert rate[9] >= 0
    full = spectrum.integrate(old, 1., -.002, evolved)
    bdf = spectrum.integrate(old, 1., -.002, evolved, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(full.y[:, -1], bdf.y[:, -1], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("stress_control", [False, True])
def test_evolving_modulus_energy_derivative_and_symmetry(stress_control):
    mat = material()
    mat = replace(mat, backbone=replace(mat.backbone,
        branches=tuple(replace(b, strength=10.) for b in mat.backbone.branches)),
        hardening_ratios=(.5, .8), hardening_rates=(100., 50.))
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])
    derivative = spectrum.rates(old, -.003, mat, stress_control=stress_control)
    step = 1e-5
    energy_rate = (spectrum.stored_energy(old+step*derivative, mat)
                   - spectrum.stored_energy(old-step*derivative, mat))/(2*step)
    assert derivative[8] == pytest.approx(energy_rate+derivative[9], abs=1e-10)
    assert derivative[9] >= 0
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(spectrum.rates(old*sign, .003, mat, stress_control=stress_control),
                               derivative*sign, atol=1e-12)


def test_evolving_modulus_zero_limit_split_and_independent_integration():
    mat = material()
    mat = replace(mat, backbone=replace(mat.backbone,
        branches=tuple(replace(b, strength=10.) for b in mat.backbone.branches)))
    old = np.zeros(12)
    inactive = replace(mat, hardening_ratios=(.2, .6), hardening_rates=(0., 0.))
    np.testing.assert_array_equal(spectrum.rates(old, .03, mat), spectrum.rates(old, .03, inactive))
    evolved = replace(mat, hardening_ratios=(.5, .7), hardening_rates=(100., 60.))
    first = spectrum.integrate(old, 2., .03, evolved)
    current = first.y[:, -1]
    full = spectrum.integrate(current, 2., -.03, evolved)
    a = spectrum.integrate(current, .7, -.03, evolved)
    b = spectrum.integrate(a.y[:, -1], 1.3, -.03, evolved)
    independent = spectrum.integrate(current, 2., -.03, evolved, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(full.y[:, -1], b.y[:, -1], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(full.y[:, -1], independent.y[:, -1], rtol=1e-7, atol=1e-7)
    assert min(current[[3, 6]]) > 0
    for updates in (dict(hardening_ratios=(1.1, .5)), dict(hardening_rates=(-1., 0.))):
        with pytest.raises(ValueError):
            replace(mat, **updates)
