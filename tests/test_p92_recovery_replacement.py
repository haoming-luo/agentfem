from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools import p92_recovery_replacement_reference as kernel
from tools.p92_recoverable_spectrum import Material as Spectrum
from tools.p92_two_rate_reference import Branch, Material


def baseline():
    branch = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return Spectrum(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))


def material():
    return kernel.Material(10000., baseline().backbone.branches[1],
                           (20000., 30000., 40000.), (.5, 3., 10.))


def test_mapping_replaces_coefficients_and_matches_reference_secant_flow():
    old = baseline()
    actual = kernel.replace_low_stress_flow(old)
    assert actual.branch == old.backbone.branches[1]
    assert actual.moduli[-1] == sum(old.backbone.branches[0].hardening)
    state = np.zeros(10)
    state[0] = 120.
    expected = old.backbone.branches[0].reference_flow*((120.-20.)/120.)**3
    assert kernel.rates(state, 0., actual)[6] == pytest.approx(expected)
    assert "branches" not in asdict(actual) and "spectrum" not in asdict(actual)
    assert len(actual.moduli)+len(actual.times) == 6
    bad = replace(old, backbone=replace(old.backbone,
        branches=(replace(old.backbone.branches[0], strength=120.), old.backbone.branches[1])))
    with pytest.raises(ValueError, match="activate"):
        kernel.replace_low_stress_flow(bad)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_subthreshold_creep_and_recovery_are_analytical_and_not_permanent(sign):
    mat = material()
    old = np.zeros(10)
    old[0] = sign*10.
    old[7] = old[0]/mat.modulus
    end = kernel.integrate(old, 20., 0., mat, stress_control=True).y[:, -1]
    expected = old[0]/np.asarray(mat.moduli)*(1-np.exp(-20./np.asarray(mat.times)))
    np.testing.assert_allclose(end[4:7], expected, atol=1e-12, rtol=1e-8)
    assert end[7] == pytest.approx(old[7]+sum(expected), abs=1e-12)
    assert end[3] == 0.
    unloaded = end.copy()
    unloaded[0], unloaded[7], unloaded[8], unloaded[9] = 0., sum(expected), 0., 0.
    recovered = kernel.integrate(unloaded, 30., 0., mat, stress_control=True).y[:, -1]
    np.testing.assert_allclose(recovered[4:7], expected*np.exp(-30./np.asarray(mat.times)), atol=1e-12)
    assert recovered[7] == pytest.approx(sum(recovered[4:7]), abs=1e-12)
    assert recovered[3] == 0.
    assert recovered[9] == pytest.approx(kernel.stored_energy(unloaded, mat)
                                         -kernel.stored_energy(recovered, mat), abs=1e-11)


@pytest.mark.parametrize("stress_control", [False, True])
def test_power_symmetry_split_and_independent_integration(stress_control):
    mat = material()
    old = np.array([70., 1., -2., .01, -.001, .002, -.0003, .01, 0., 0.])
    before = old.copy()
    rate = kernel.rates(old, -.002, mat, stress_control=stress_control)
    eps = 1e-5
    e = (kernel.stored_energy(old+eps*rate, mat)-kernel.stored_energy(old-eps*rate, mat))/(2*eps)
    assert rate[8] == pytest.approx(e+rate[9], abs=1e-9)
    assert rate[9] >= 0.
    sign = np.ones(10)
    sign[[0, 1, 2, 4, 5, 6, 7]] = -1
    np.testing.assert_allclose(kernel.rates(old*sign, .002, mat, stress_control=stress_control), rate*sign)
    full = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control).y[:, -1]
    mid = kernel.integrate(old, .7, -.002, mat, stress_control=stress_control).y[:, -1]
    split = kernel.integrate(mid, 1.3, -.002, mat, stress_control=stress_control).y[:, -1]
    other = kernel.integrate(old, 2., -.002, mat, stress_control=stress_control, method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, split, atol=2e-8, rtol=1e-7)
    np.testing.assert_allclose(full, other, atol=2e-8, rtol=1e-7)
    np.testing.assert_array_equal(old, before)


def test_failure_and_bad_state_do_not_mutate_input(monkeypatch):
    old = np.zeros(10)
    before = old.copy()

    def fail(fun, interval, initial, **kwargs):
        initial[:] = 123.
        return SimpleNamespace(success=False)

    monkeypatch.setattr(kernel, "solve_ivp", fail)
    with pytest.raises(RuntimeError, match="input retained"):
        kernel.integrate(old, 1., 0., material())
    np.testing.assert_array_equal(old, before)
    with pytest.raises(ValueError):
        kernel.integrate(np.zeros(12), 1., 0., material())
    for change in (dict(modulus=-1.), dict(times=(1., 2.)), dict(moduli=(1., 2., np.nan)),
                   dict(branch=replace(material().branch, saturation_ratio=.5))):
        with pytest.raises(ValueError):
            replace(material(), **change)


def test_replay_keeps_new_state_identity_and_reference_observables():
    from tools.p92_two_rate_controls import replay
    path = dict(initial_stress=0., initial_strain=0., source_sha256="synthetic",
        segments=[dict(cycle=1, duration=1., stress_control=False, target=.002,
                       sample_times=[.5, 1.], measured_stress=[0., 0.],
                       measured_strain=[.001, .002], source_indices=[0, 1])])
    _, report = replay(path, material(), kernel=kernel)
    assert report["model_id"] == kernel.MODEL_ID
    assert report["state_layout"] == kernel.STATE_LAYOUT
    assert len(report["segments"][0]["end_state"]) == 10
    assert abs(report["mechanical_balance_MPa"]) < 1e-8


def test_frozen_decoder_and_strain_slots_are_not_old_backstresses():
    from copy import deepcopy
    from tools.p92_recoverable_frozen import decode_material
    from tools.p92_two_rate_frozen import numerical_difference
    module, actual = decode_material(asdict(material()), kernel.MODEL_ID)
    assert module is kernel and actual == material()
    a = dict(segments=[dict(sampled_stress_MPa=[1.], sampled_strain=[.001], end_state=[0.]*10)])
    b = deepcopy(a)
    b["segments"][0]["end_state"][4] = 1e-5
    result = numerical_difference(a, b, stress_state_indices=(0, 1, 2), extra_strain_indices=(4, 5))
    assert result["strain_state_difference"] == 1e-5
    assert result["stress_state_difference_MPa"] == 0.
    assert not result["passed"]
