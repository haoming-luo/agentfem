import numpy as np
import pytest

from tools import p92_recoverable_reference as model
from tools.p92_two_rate_reference import Branch, Material


def material():
    branch = Branch(1000., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return model.Material(Material(10000., (branch, branch)), 20000., 2.)


def test_constant_stress_creep_and_zero_stress_recovery():
    mat = material()
    old = np.zeros(11)
    old[0], old[7] = 100., .01
    result = model.integrate(old, 4., 0., mat, stress_control=True)
    expected = .005*(1-np.exp(-2.))
    assert result.y[10, -1] == pytest.approx(expected, abs=1e-11)
    assert result.y[7, -1] == pytest.approx(.01+expected, abs=1e-11)
    assert old[10] == 0.
    unload = result.y[:, -1].copy()
    unload[0] = 0.
    unload[7] -= .01  # explicit elastic unload, not a history reset
    recovered = model.integrate(unload, 4., 0., mat, stress_control=True)
    assert recovered.y[10, -1] == pytest.approx(expected*np.exp(-2.), abs=1e-11)
    assert recovered.y[7, -1] == pytest.approx(recovered.y[10, -1], abs=1e-11)


def test_power_identity_and_nonnegative_dissipation():
    from dataclasses import replace
    mat = material()
    mat = replace(mat, backbone=Material(mat.backbone.modulus,
        tuple(replace(b, strength=10.) for b in mat.backbone.branches)))
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001])
    rate = model.rates(old, .003, mat)
    energy_rate = old[0]*rate[0]/mat.backbone.modulus+mat.recoverable_modulus*old[10]*rate[10]
    for offset, branch in zip((1, 4), mat.backbone.branches):
        energy_rate += sum(old[offset+i]*rate[offset+i]/h for i, h in enumerate(branch.hardening))
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-12)
    assert rate[9] >= 0


def test_constant_strain_relaxation_matches_series_analytic_solution():
    mat = material()
    old = np.zeros(11)
    old[0], old[7] = 100., .01
    result = model.integrate(old, 3., 0., mat)
    equilibrium = .01/(1/mat.backbone.modulus+1/mat.recoverable_modulus)
    decay = (1+mat.backbone.modulus/mat.recoverable_modulus)/mat.relaxation_time
    expected = equilibrium+(100-equilibrium)*np.exp(-3*decay)
    assert result.y[0, -1] == pytest.approx(expected, abs=1e-7)
    assert result.y[7, -1] == .01


def test_continuation_and_independent_integration():
    mat = material()
    old = np.zeros(11)
    full = model.integrate(old, 2., .001, mat).y[:, -1]
    split = model.integrate(old, 1., .001, mat).y[:, -1]
    split = model.integrate(split, 1., .001, mat).y[:, -1]
    bdf = model.integrate(old, 2., .001, mat, method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full, split, atol=1e-9)
    np.testing.assert_allclose(full, bdf, atol=1e-8)
    negative = model.integrate(old, 2., -.001, mat).y[:, -1]
    np.testing.assert_allclose(full[[0, 7, 10]], -negative[[0, 7, 10]], atol=1e-10)


@pytest.mark.parametrize("h,t", [(0., 1.), (1., 0.), (-1., 1.), (np.inf, 1.)])
def test_invalid_material(h, t):
    with pytest.raises(ValueError, match="positive"):
        model.Material(material().backbone, h, t)


def test_joint_coordinates_and_invalid_initial_backstress():
    from tools.p92_recoverable_joint import decode
    mat = material()
    candidate = decode(mat.backbone, np.zeros(8))
    assert candidate.recoverable_modulus == 20*mat.backbone.modulus
    assert candidate.relaxation_time == .2
    with pytest.raises(ValueError, match="eight"):
        decode(mat.backbone, np.zeros(6))
    old = np.zeros(11)
    old[1] = 100.
    with pytest.raises(ValueError, match="invariant"):
        model.integrate(old, 1., 0., mat)


def test_frozen_check_includes_recoverable_state():
    from copy import deepcopy
    from tools.p92_two_rate_frozen import numerical_difference
    from tools.p92_recoverable_frozen import evaluate
    from pathlib import Path
    a = dict(segments=[dict(sampled_stress_MPa=[0.], sampled_strain=[0.], end_state=[0.]*11)])
    b = deepcopy(a)
    b["segments"][0]["end_state"][10] = 1e-5
    assert not numerical_difference(a, b, extra_strain_indices=(10,))["passed"]
    with pytest.raises(ValueError, match="early-only"):
        evaluate(dict(training_cycles=[1, 2, 3], fit=dict(success=False)), Path("unused"))


def test_dissipative_strength_softening_retains_power_identity():
    from dataclasses import replace
    mat = material()
    mat = replace(mat, backbone=Material(mat.backbone.modulus,
        tuple(replace(b, strength=10.) for b in mat.backbone.branches)),
        strength_ratio=.3, strength_softening_rate=100.)
    old = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001])
    rate = model.rates(old, .003, mat)
    energy_rate = old[0]*rate[0]/mat.backbone.modulus+mat.recoverable_modulus*old[10]*rate[10]
    for offset, branch in zip((1, 4), mat.backbone.branches):
        energy_rate += sum(old[offset+i]*rate[offset+i]/h for i, h in enumerate(branch.hardening))
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-12)
    assert rate[9] >= 0
