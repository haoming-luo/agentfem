# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline protocol and structural verification; no experimental data fixture."""
import numpy as np
import pytest

from tools.validate_p92_loading_path import (
    compare_path, continuous_response, estimate_modulus, fem_path,
    low_stress_diagnostic, nominal_path, read_cycle, refined_times, validate_path,
)
from tools.validate_p92_relaxation import Hold

FIT = {"rate_at_200_MPa_per_s": 10., "exponent": 1.}


def test_scalar_ramp_hold_reversal_has_independent_linear_exact_solution():
    times = np.array([0., 1., 3., 5., 7.])
    strain = np.array([0., .001, .001, -.001, -.001])
    young, rate = 200_000., FIT["rate_at_200_MPa_per_s"] / 200.
    query = np.linspace(0., 7., 71)
    exact = np.zeros_like(query)
    previous = 0.
    for i, (a, b) in enumerate(zip(times[:-1], times[1:], strict=True)):
        loading = young * (strain[i + 1] - strain[i]) / (b - a)
        selected = (query >= a) & (query <= b)
        elapsed = query[selected] - a
        exact[selected] = previous * np.exp(-rate * elapsed) - loading / rate * np.expm1(-rate * elapsed)
        previous = previous * np.exp(-rate * (b - a)) - loading / rate * np.expm1(-rate * (b - a))
    response = continuous_response(times, strain, young, FIT)
    np.testing.assert_allclose(response(query), exact, atol=1.e-6)
    assert response(np.array([3.]))[0] != 0.  # history is not reset at reversal
    with pytest.raises(ValueError, match="extrapolate"):
        response([7.1])


@pytest.mark.parametrize("family,duration", [("RF", 368.), ("SLR", 2168.)])
def test_reviewed_nominal_paths_and_refinement_preserve_knots(family, duration):
    times, strain = nominal_path(family)
    assert times[-1] == duration
    assert strain[0] == strain[-1] == 0.
    assert strain.min() == -.002 and strain.max() == .002
    finer = refined_times(times, 4)
    assert np.all(np.diff(finer) > 0)
    np.testing.assert_array_equal(finer[3::4], times[1:])
    with pytest.raises(ValueError, match="reviewed"):
        nominal_path("SLC")


def test_source_and_path_rejections(tmp_path):
    path = tmp_path / "Data_RF.lis"
    path.write_bytes(b"wrong data")
    with pytest.raises(ValueError, match="checksum"):
        read_cycle(path, "RF")
    for times, strain in (([0., 0., 1.], [0., 0., .01]),
                           ([0., 1.], [.001, .01]), ([0., 1.], [0., np.nan])):
        with pytest.raises(ValueError, match="Path"):
            validate_path(times, strain, 100_000., FIT)
    for refinement in (True, 0, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            refined_times([0., 1.], refinement)
    with pytest.raises(ValueError, match="real"):
        validate_path([0., 1.], [0., .001 + .001j], 100_000., FIT)


def test_modulus_uses_rf_only_and_does_not_import_stress_intercept():
    times = np.linspace(0., 1., 101)
    strain = times * .001
    cycle = Hold("RF", times, 150_000. * strain - 3., strain, times * 0 + 893.15, "synthetic")
    modulus = estimate_modulus(cycle)
    assert modulus["selected_MPa"] == pytest.approx(150_000.)
    assert all(w["intercept_MPa"] == pytest.approx(-3.) for w in modulus["windows"])
    other = Hold("SLR", times, cycle.stress, strain, cycle.temperature, "other")
    with pytest.raises(ValueError, match="Only RF"):
        estimate_modulus(other)
    # Deliberately wrong measured stress is comparison-only, never initialization.
    response = continuous_response(times, strain, modulus["selected_MPa"], FIT)
    assert response([0.])[0] == 0.
    comparison = compare_path(other, response, 1.)
    assert not comparison["initial_stress_conditioned_on_measurement"]


def test_conditional_discriminator_is_separate_from_full_path_prediction():
    time = np.linspace(194., 1980., 101)
    hold = Hold("SLR", time, np.full(101, 100.), np.zeros(101),
                np.full(101, 893.15), "synthetic")
    before = hold.stress.copy()
    report = low_stress_diagnostic(hold, FIT)
    assert report["result"]["initial_stress_conditioned_on_measurement"]
    assert "post-hoc" in report["role"]
    np.testing.assert_array_equal(hold.stress, before)


def test_full_structural_reversal_refines_and_restarts_without_reset(tmp_path):
    times = np.array([0., 1., 3., 5., 7.])
    strain = np.array([0., .001, .001, -.001, -.001])
    exact = continuous_response(times, strain, 200_000., FIT)
    errors = []
    for refinement in (4, 8, 16):
        full = fem_path(times, strain, 200_000., FIT, refinement=refinement)
        result = full.solve_result()
        stress = result.histories["gauge.S"]
        errors.append(np.max(np.abs(stress.values[:, 0, 0] - exact(stress.abscissa))))
        np.testing.assert_allclose(stress.values[:, 1:, :], 0., atol=1.e-7)
        # Unit face, unit BC multiplier: generalized reaction is axial force.
        np.testing.assert_allclose([f.generalized_reaction for f in full.energy_history],
                                   stress.values[:, 0, 0], atol=1.e-6)
    assert 1.8 < errors[0] / errors[1] < 2.1
    assert 1.8 < errors[1] / errors[2] < 2.1
    partial = fem_path(times, strain, 200_000., FIT, refinement=16)
    partial.solve(until=3.)
    checkpoint = partial.save_checkpoint(tmp_path / "reversal")
    resumed = fem_path(times, strain, 200_000., FIT, refinement=16)
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    for actual, expected in zip(resumed.material_point_histories, full.material_point_histories, strict=True):
        np.testing.assert_allclose(actual.abscissa, expected.abscissa, atol=1.e-12)
        np.testing.assert_allclose(actual.values, expected.values, atol=1.e-7)
