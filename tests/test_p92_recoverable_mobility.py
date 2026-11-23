from dataclasses import replace

import numpy as np
import pytest

from tools import p92_recoverable_mobility_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(feedback="flow"):
    b = Branch(10., 1e-4, 3., (1000., 2000.), (20., 30.), (.1, .01))
    return kernel.Material(spectrum.Material(Material(10000., (b, b)),
                                            (20000., 30000.), (.5, 10.)), feedback)


@pytest.mark.parametrize("feedback", ["none", "flow", "static_recovery", "strength"])
@pytest.mark.parametrize("control", [False, True])
def test_power_identity_with_unchanged_energy(feedback, control):
    mat = material(feedback)
    state = np.zeros(12)
    state[:3], state[4:6], state[10:] = [50., -3., 4.], [5., -6.], [.0003, -.0002]
    rate = kernel.rates(state, .001, mat, stress_control=control)
    h = 1e-6
    derivative = (kernel.stored_energy(state+h*rate, mat)-kernel.stored_energy(state-h*rate, mat))/(2*h)
    assert derivative+rate[9] == pytest.approx(rate[8], abs=1e-8)
    assert rate[9] >= 0.


def test_zero_memory_and_no_feedback_are_original():
    state = np.zeros(12)
    state[0] = 40.
    for feedback in ("flow", "static_recovery", "strength"):
        mat = material(feedback)
        np.testing.assert_array_equal(kernel.rates(state, .001, mat), spectrum.rates(state, .001, mat.spectrum))
    mat = material("none")
    state[11] = .0003
    np.testing.assert_array_equal(kernel.rates(state, .001, mat), spectrum.rates(state, .001, mat.spectrum))


@pytest.mark.parametrize("feedback", ["flow", "static_recovery", "strength"])
def test_sign_split_and_independent_integrator(feedback):
    mat, old = material(feedback), np.zeros(12)
    full = kernel.integrate(old, 3., .002, mat)
    bdf = kernel.integrate(old, 3., .002, mat, method="BDF")
    half = kernel.integrate(old, 1., .002, mat)
    split = kernel.integrate(half.y[:, -1], 2., .002, mat)
    negative = kernel.integrate(old, 3., -.002, mat)
    np.testing.assert_allclose(bdf.y[:, -1], full.y[:, -1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(split.y[:, -1], full.y[:, -1], atol=1e-7, rtol=1e-7)
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(negative.y[:, -1], sign*full.y[:, -1], atol=1e-9)


def test_parameter_neutral_and_bounded_symmetric_factor():
    mat = material()
    assert set(vars(mat)) == {"spectrum", "feedback"}
    state = np.zeros(12)
    for a in (0., .0001, .001, 1.):
        state[11] = a
        positive = kernel.factor(state, mat)
        state[11] = -a
        assert 0 < positive <= 1.
        assert kernel.factor(state, mat) == positive
    with pytest.raises(ValueError):
        replace(mat, feedback="both")
    with pytest.raises(ValueError, match="Isolate"):
        replace(mat, spectrum=replace(mat.spectrum, directional_recovery=True))


def test_failure_does_not_mutate_input(monkeypatch):
    from types import SimpleNamespace
    old = np.zeros(12)
    def fail(fun, time, y, **kwargs):
        y[0] = 999.
        return SimpleNamespace(success=False)
    monkeypatch.setattr(kernel, "solve_ivp", fail)
    with pytest.raises(RuntimeError):
        kernel.integrate(old, 1., .001, material())
    assert not old.any()


@pytest.mark.parametrize("feedback", ["flow", "static_recovery", "strength"])
def test_offload_keeps_both_recoverable_modes(feedback):
    mat = material(feedback)
    old = np.zeros(12)
    old[10:] = (.0003, -.0002)
    old[7] = old[10:].sum()
    duration = 2.
    result = kernel.integrate(old, duration, 0., mat, stress_control=True)
    expected = old[10:]*np.exp(-duration/np.asarray(mat.spectrum.times))
    np.testing.assert_allclose(result.y[10:, -1], expected, atol=1e-12)
    assert result.y[7, -1] == pytest.approx(expected.sum(), abs=1e-12)
    assert result.y[3, -1] == result.y[6, -1] == 0.


def test_frozen_identity_roundtrip_and_isolated_fit_flags():
    from dataclasses import asdict
    from pathlib import Path
    from tools.p92_recoverable_frozen import decode_material
    from tools.p92_spectrum_recovery_joint import identify
    mat = material("static_recovery")
    module, restored = decode_material(asdict(mat), kernel.MODEL_ID)
    assert module is kernel and restored == mat
    seed = dict(model_id=spectrum.MODEL_ID)
    for conflicting in ("primary", "directional", "shared_history", "recoverable_kinetics", "transient_strength"):
        with pytest.raises(ValueError, match="isolate"):
            identify(seed, Path("unused"), (0., 0.), recoverable_mobility=True, **{conflicting: True})


def test_equal_stress_attribution_retains_recoverable_signs():
    from tools.p92_equal_stress_hold import mechanism_partition
    mat = material("static_recovery")
    old = kernel.integrate(np.zeros(12), 3., .002, mat).y[:, -1]
    solution = kernel.integrate(old, 5., 0., mat)
    report = mechanism_partition(solution, dict(times_s=[.1, 4.]), mat, module=kernel)
    assert abs(report["strain_closure_residual"]) < 1e-9
    assert report["names"] == ["inelastic_A", "inelastic_B", "recoverable_fast", "recoverable_slow"]


def test_strength_feedback_is_a_threshold_not_a_flow_multiplier():
    from tools.p92_recoverable_mobility_probe import strength_candidates
    mat = material("strength")
    state = np.zeros(12)
    state[0] = 11.
    state[11] = .0002  # Threshold hypot(10, 6) exceeds effective stress 11.
    assert kernel.rates(state, 0., mat)[6] == 0.
    assert kernel.rates(state, 0., replace(mat, feedback="flow"))[6] > 0.
    state[11] *= -1
    assert kernel.rates(state, 0., mat)[6] == 0.
    assert [name for name, _ in strength_candidates(mat.spectrum)] == ["none", "strength"]
