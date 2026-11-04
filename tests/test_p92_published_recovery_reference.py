"""Numerical/source-transcription gates, not experimental acceptance."""

import numpy as np
import pytest

from tools.p92_published_recovery_reference import (
    implicit_update, integrate_segment, published_parameters, rate_jacobian, rates, run_cycles,
)


def test_table_and_no_temperature_interpolation():
    cold, hot = published_parameters(600), published_parameters(650)
    assert cold.E == 161933. and hot.E == 134573.
    assert cold.Q == -111.06 and hot.Qr == -86.67
    assert cold.gamma2 == 7.47e-7 and hot.gamma_r2 == 9.22e-8
    with pytest.raises(ValueError):
        published_parameters(620)


def test_recovery_exists_without_viscoplastic_flow():
    p = published_parameters(600)
    state = np.array([20., 10., 5., -10., .2])
    derivative = rates(state, 0., p)
    assert derivative[0] == derivative[4] == 0.
    assert derivative[1] < 0 and derivative[2] < 0 and derivative[3] < 0
    np.testing.assert_array_equal(state, [20., 10., 5., -10., .2])


def test_signed_symmetry_of_uniaxial_equations():
    p = published_parameters(600)
    state = np.array([300., 25., 50., -10., .02])
    reverse = state * np.array([-1., -1., -1., 1., 1.])
    forward = rates(state, .001, p)
    backward = rates(reverse, -.001, p)
    np.testing.assert_allclose(backward, forward*np.array([-1., -1., -1., 1., 1.]))


@pytest.mark.parametrize("stress", [-350., 20., 350.])
@pytest.mark.parametrize("temperature", [600, 650])
def test_rate_jacobian_against_independent_differences(stress, temperature):
    p = published_parameters(temperature)
    state = np.array([stress, 20., -15., -10., .2])
    analytical = rate_jacobian(state, p)
    columns = []
    for i in range(5):
        perturbation = np.zeros(5)
        # The history derivative is O(0.1) beside O(1e3) flow terms;
        # 1e-7 suffers cancellation. This central step keeps O(h^2) < 1e-6.
        perturbation[i] = 1.e-5 if i < 4 else 1.e-4
        columns.append((rates(state+perturbation, .001, p)-rates(state-perturbation, .001, p))
                       / (2*perturbation[i]))
    np.testing.assert_allclose(analytical, np.column_stack(columns), rtol=1.e-6, atol=1.e-7)


@pytest.mark.parametrize("increment", [-.001, 0., .001])
def test_implicit_tangent_after_inherited_state(increment):
    p = published_parameters(600)
    old = integrate_segment(np.zeros(5), 4., .001, p).y[:, -1].copy()
    before = old.copy()
    state, tangent = implicit_update(old, increment, .2, p)
    h = 1.e-8
    numerical = (implicit_update(old, increment+h, .2, p)[0][0]
                 - implicit_update(old, increment-h, .2, p)[0][0])/(2*h)
    assert tangent == pytest.approx(numerical, rel=2.e-5, abs=1.e-3)
    np.testing.assert_array_equal(old, before)
    assert state[4] >= old[4]


def test_implicit_hold_refines_to_continuous_reference():
    p = published_parameters(600)
    old = integrate_segment(np.zeros(5), 4., .001, p).y[:, -1].copy()
    exact = integrate_segment(old, 60., 0., p).y[:, -1]
    errors = []
    for count in (60, 120, 240):
        current = old.copy()
        for _ in range(count):
            current, _ = implicit_update(current, 0., 60./count, p)
        errors.append(np.max(np.abs(current[:4]-exact[:4])))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .3


def test_implicit_complete_loading_reversal_and_hold_refinement():
    p = published_parameters(600)
    exact = np.array(run_cycles(cycles=2, dwell=60.)["final_state"])
    errors = []
    for count in (32, 64, 128):
        current = np.zeros(5)
        for duration, rate in ((4., .001), (60., 0.), (8., -.001), (8., .001), (60., 0.)):
            dt = duration/count
            for _ in range(count):
                current, _ = implicit_update(current, rate*dt, dt, p)
        errors.append(np.max(np.abs(current[:4]-exact[:4])))
    assert errors[2] < .7*errors[1] < .49*errors[0]
    assert errors[2] < 1.


@pytest.mark.parametrize("temperature", [600, 650])
@pytest.mark.parametrize("dwell", [60., 600.])
def test_distinct_stiff_integrators_agree(temperature, dwell):
    radau = run_cycles(temperature_C=temperature, dwell=dwell, cycles=2, method="Radau")
    bdf = run_cycles(temperature_C=temperature, dwell=dwell, cycles=2, method="BDF")
    np.testing.assert_allclose(radau["final_state"][:4], bdf["final_state"][:4], rtol=1.e-5, atol=1.e-4)
    assert radau["final_state"][4] == pytest.approx(bdf["final_state"][4], rel=1.e-5, abs=1.e-8)
    for first, second in zip(radau["cycles"], bdf["cycles"], strict=True):
        np.testing.assert_allclose(first["hold_stress_samples_MPa"], second["hold_stress_samples_MPa"], atol=1.e-3)
        assert first["relaxed_stress_MPa"] > 0
