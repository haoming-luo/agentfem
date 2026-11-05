# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic offline tests of the experiment protocol, not P92 observations."""
import numpy as np
import pytest

from tools.validate_p92_relaxation import (
    Hold, _parse_lis, comparison, fem_step, fit_relaxation, read_hold, relaxation, time_weights,
)


def test_bam_conversion_keeps_same_rows_and_does_not_fill_sparse_cycles():
    raw = ("[Data]\nTime\tCycle\tStress\tStrain\tTemperature\n"
           "s\tNo\tMPa\t%\t°C\n0 1 100 0.2 620\n1 1 90 0.2 620\n"
           "1000 10 80 0.2 619\n").encode("cp1252")
    data = _parse_lis(raw)
    assert data.shape == (3, 5)
    np.testing.assert_array_equal(data[:, 1], [1, 1, 10])
    np.testing.assert_allclose(data[:, 3], .002)
    np.testing.assert_allclose(data[:, 4], [893.15, 893.15, 892.15])
    with pytest.raises(ValueError, match="units"):
        _parse_lis(raw.replace(b"MPa", b"Pa"))
    with pytest.raises(ValueError, match="nonfinite"):
        _parse_lis(raw.replace(b"1000 10 80", b"1000 10 nan"))
    with pytest.raises(ValueError, match="units"):
        _parse_lis(b"[Data]")


def test_source_pin_rejects_same_named_unreviewed_data(tmp_path):
    path = tmp_path / "Data_RF.lis"
    path.write_bytes(b"not the public source")
    with pytest.raises(ValueError, match="checksum"):
        read_hold(path, "RF")


def test_time_weighting_integrates_time_not_sample_density():
    t = np.array([0., .01, .02, 1., 10.])
    assert np.dot(time_weights(t), t) == pytest.approx(5.)
    with pytest.raises(ValueError, match="strictly"):
        time_weights([0., 0., 1.])


@pytest.mark.parametrize("n", [1., 3., 12.])
def test_relaxation_sign_symmetry_and_semigroup(n):
    times = np.linspace(0., 30., 11)
    positive = relaxation(times, 220., 1., n)
    np.testing.assert_allclose(relaxation(times, -220., 1., n), -positive)
    np.testing.assert_allclose(relaxation(times + 30., 220., 1., n),
                               relaxation(times, positive[-1], 1., n), rtol=1.e-13)
    assert np.all(np.diff(positive) < 0)


def test_calibration_recovers_synthetic_parameters_without_holdout_values():
    t = np.linspace(0., 170., 65)
    hold = Hold("synthetic", t, relaxation(t, 210., 1.4, 9.),
                np.full(len(t), .002), np.full(len(t), 893.15), "synthetic")
    fit = fit_relaxation(hold)
    assert fit["rate_at_200_MPa_per_s"] == pytest.approx(1.4, rel=1.e-6)
    assert fit["exponent"] == pytest.approx(9., rel=1.e-6)
    bad = Hold("held_out", t, np.full(len(t), -210.), -hold.strain, hold.temperature, "other")
    metrics = comparison(bad, fit)
    assert metrics["time_weighted_rmse_MPa"] > 10
    assert metrics["acceptance"].startswith("not_assigned")


def test_fem_realizes_effective_relaxation_and_restart(tmp_path):
    fit = {"rate_at_200_MPa_per_s": 1.5, "exponent": 9.}
    full = fem_step(fit, initial=-210., duration=20., count=16)
    full.solve()
    partial = fem_step(fit, initial=-210., duration=20., count=16)
    partial.solve(until=10.)
    path = partial.save_checkpoint(tmp_path / "state")
    resumed = fem_step(fit, initial=-210., duration=20., count=16)
    resumed.load_checkpoint(path)
    result = resumed.solve_result()
    for a, b in zip(full.material_point_histories, resumed.material_point_histories, strict=True):
        np.testing.assert_allclose(a.values, b.values, rtol=1.e-9, atol=1.e-9)
    stress = result.histories["gauge.S"]
    assert stress.values[-1, 0, 0] == pytest.approx(
        full.energy_history[-1].generalized_reaction / (-210. / 100_000.), abs=1.e-7)
    np.testing.assert_allclose(stress.values[:, 1:, :], 0., atol=1.e-7)
    assert np.all(np.diff(stress.values[:, 0, 0]) > 0)
    # Changing E while holding E*A and initial stress fixed changes strain but
    # not this uniaxial relaxation: E is not identified by this experiment.
    other = fem_step(fit, initial=-210., duration=20., count=16, young=200_000.)
    other_result = other.solve_result()
    np.testing.assert_allclose(other_result.histories["gauge.S"].values, stress.values, atol=1.e-7)
