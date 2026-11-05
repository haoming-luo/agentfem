# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent numerical checks of the public-data candidate screen."""
import numpy as np
import pytest

from tools.screen_p92_sinh import (
    SinhParameters, continuous_response, fem_path, fit_profile, integrate_segment,
)
from tools.validate_p92_relaxation import Hold


@pytest.mark.parametrize("exponent", [1., 2., 4., 8.])
def test_normalized_rate_matches_existing_material_and_differentiates(exponent):
    from agentfem.constitutive import SinhCreep
    parameters = SinhParameters(1.5, 40., exponent)
    young = 150_000.
    law = SinhCreep(parameters.coefficient(young), 40., exponent)
    for stress in (-300., -10., 0., 10., 200., 300.):
        rate, slope = parameters.rate_slope(stress)
        assert rate == pytest.approx(np.sign(stress) * young * law.equivalent_rate(stress), rel=1.e-12, abs=1.e-10)
        h = 1.e-4
        numerical = (parameters.rate_slope(stress + h)[0] - parameters.rate_slope(stress - h)[0]) / (2 * h)
        assert slope == pytest.approx(numerical, rel=1.e-5, abs=1.e-9)
    assert parameters.rate_slope(200.)[0] == pytest.approx(1.5)


def test_sinh_hold_matches_exact_n1_solution_and_sign_symmetry():
    parameters = SinhParameters(2., 60.)
    times = np.linspace(0., 200., 41)
    coefficient = parameters.rate_at_200_MPa_per_s / np.sinh(200. / 60.)
    for initial in (-180., 180.):
        exact = 120. * np.arctanh(np.tanh(initial / 120.) * np.exp(-coefficient * times / 60.))
        response = integrate_segment(times, initial, 0., parameters)
        np.testing.assert_allclose(response(times)[0], exact, atol=1.e-6, rtol=1.e-9)


def test_fit_is_rf_only_and_recovers_synthetic_hold():
    parameters = SinhParameters(2., 60.)
    times = np.linspace(0., 180., 21)
    exact = 120. * np.arctanh(np.tanh(200. / 120.) * np.exp(
        -2. / np.sinh(200. / 60.) * times / 60.))
    hold = Hold("RF", times, exact, np.zeros_like(times), np.full_like(times, 893.15), "synthetic")
    report = fit_profile(hold, exponents=(1.,))
    candidate = report["candidates"][0]
    assert candidate["RF_fit_rmse_MPa"] < 1.e-6
    assert candidate["parameters"]["stress_scale_MPa"] == pytest.approx(60., rel=1.e-6)
    assert not report["SLR_used_for_calibration"]
    with pytest.raises(ValueError, match="RF only"):
        fit_profile(Hold("SLR", times, exact, hold.strain, hold.temperature, "synthetic"))


def test_parameter_and_path_rejection():
    for values in ((0., 60., 1.), (1., 0., 1.), (1., 60., .5), (np.nan, 60., 1.), (1j, 60., 1.)):
        with pytest.raises(ValueError):
            SinhParameters(*values)
    parameters = SinhParameters(2., 60.)
    with pytest.raises(ValueError, match="start at zero"):
        continuous_response([0., 0.], [0., .001], 200_000., parameters)
    with pytest.raises(ValueError, match="real"):
        continuous_response([0., 1.], [0., .001j], 200_000., parameters)
    response = continuous_response([0., 1., 2.], [0., .001, 0.], 200_000., parameters)
    with pytest.raises(ValueError, match="extrapolate"):
        response([2.1])
    assert response([0.])[0] == 0.
    assert response([2.])[0] < 0.  # inherited creep strain, not a reset


def test_sinh_full_path_fem_convergence_and_restart(tmp_path):
    times = np.array([0., 1., 11., 13., 23.])
    strain = np.array([0., .001, .001, -.001, -.001])
    young, parameters = 200_000., SinhParameters(10., 100., 2.)
    reference = continuous_response(times, strain, young, parameters)
    errors = []
    for refinement in (8, 16, 32):
        full = fem_path(times, strain, young, parameters, refinement=refinement)
        result = full.solve_result()
        stress = result.histories["gauge.S"]
        errors.append(float(np.max(np.abs(stress.values[:, 0, 0] - reference(stress.abscissa)))))
        np.testing.assert_allclose(stress.values[:, 1:, :], 0., atol=1.e-7)
        np.testing.assert_allclose([item.generalized_reaction for item in full.energy_history],
                                   stress.values[:, 0, 0], atol=1.e-6)
    assert errors[0] > errors[1] > errors[2]
    assert 1.7 < errors[1] / errors[2] < 2.2
    partial = fem_path(times, strain, young, parameters, refinement=32)
    partial.solve(until=11.)
    checkpoint = partial.save_checkpoint(tmp_path / "reversal")
    resumed = fem_path(times, strain, young, parameters, refinement=32)
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    for actual, expected in zip(resumed.material_point_histories, full.material_point_histories, strict=True):
        np.testing.assert_allclose(actual.abscissa, expected.abscissa, atol=1.e-12)
        np.testing.assert_allclose(actual.values, expected.values, atol=1.e-7)
