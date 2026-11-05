import numpy as np
import pytest

from tools.p92_structural_cycle_screen import waveform, run


@pytest.mark.parametrize("cycles", [1, 2, 20])
def test_waveform_preserves_published_rate_and_visits_hold_boundaries(cycles):
    knots, strain, accepted, peaks, ends = waveform(cycles=cycles, dwell=60.)
    assert knots[-1] == 4 + 60 * cycles + 16 * (cycles - 1)
    assert len(peaks) == len(ends) == cycles
    rates = np.diff(strain) / np.diff(knots)
    np.testing.assert_allclose(np.unique(np.abs(rates)), [0., .001])
    assert set(knots[1:]).issubset(set(accepted))
    assert accepted[-1] == ends[-1]
    assert np.all(np.diff(accepted) > 0)


def test_bounded_structural_smoke_and_restart(tmp_path):
    coarse = run(cycles=2, dwell=5.)
    report = run(cycles=2, dwell=5., refinement=2, checkpoint_dir=tmp_path / "restart")
    assert report["restart_histories_exact"]
    assert report["maximum_lateral_stress_MPa"] < 1e-6
    assert report["maximum_prescribed_strain_error"] < 1e-10
    assert report["maximum_stress_error_MPa"] < 15.
    assert report["maximum_stress_error_MPa"] < .7 * coarse["maximum_stress_error_MPa"]
    assert not report["engineering_qualified"]
