"""Continuous-state replay verification, not experimental validation."""

import numpy as np
import pytest

from tools.p92_bam_cyclic_screen import replay
from tools.p92_published_recovery_reference import published_parameters


def test_split_restart_matches_uninterrupted_measured_input():
    t = np.linspace(0, 10, 41)
    e = .004*np.sin(t)
    old = np.zeros(5)
    p = published_parameters(600)
    whole = replay(t, e, p, old)
    first = replay(t[:21], e[:21], p, old)
    second = replay(t[20:], e[20:], p, first[-1])
    np.testing.assert_array_equal(np.vstack((first, second[1:])), whole)
    np.testing.assert_array_equal(old, np.zeros(5))
    assert whole[-1, 4] > 0
    assert np.all(np.diff(whole[:, 4]) >= 0)


def test_refinement_approaches_independent_ode():
    t = np.array([0., 1., 2., 3., 5.])
    e = np.array([0., .002, .004, -.002, -.002])
    p = published_parameters(600)
    exact = replay(t, e, p, np.zeros(5), method="Radau")
    coarse = replay(t, e, p, np.zeros(5), subdivisions=8)
    fine = replay(t, e, p, np.zeros(5), subdivisions=32)
    assert np.max(abs(fine[:, 0]-exact[:, 0])) < np.max(abs(coarse[:, 0]-exact[:, 0]))


def test_measured_starting_stress_is_not_reset():
    old = np.array([10., 0., 0., 0., 0.])
    result = replay([5., 6.], [.001, .001], published_parameters(600), old)
    np.testing.assert_array_equal(result[0], old)
    assert result[-1, 0] == 10


@pytest.mark.parametrize("times,strains", [([0, 0], [0, 1]), ([0, 1], [0]),
                                         ([0, np.nan], [0, 1])])
def test_bad_input_never_advances_state(times, strains):
    old = np.zeros(5)
    with pytest.raises(ValueError):
        replay(times, strains, published_parameters(600), old)
    np.testing.assert_array_equal(old, np.zeros(5))


def test_screen_never_sends_holdout_observations_to_calibration(monkeypatch):
    from tools import p92_bam_cyclic_screen as module
    received = []
    p = published_parameters(600)
    def frozen_fit(times, strains, observed, initial_state, **kwargs):
        received.append((times.copy(), strains.copy(), observed.copy(), initial_state.copy()))
        return p, {"success": True, "test_fixture": True}
    monkeypatch.setattr(module, "calibrate", frozen_fit)
    t = np.arange(101, dtype=float)
    strain = .003*np.sin(t)
    cycles = np.minimum(t.astype(int)//10+1, 10)
    measured = replay(t, strain, p, np.zeros(5))[:, 0]*1e6
    data = {"protocol": "RF", "time_s": t, "strain": strain, "stress_Pa": measured,
            "cycle": cycles, "temperature_K": np.full(len(t), 893.15), "sha256": "synthetic"}
    first = module.screen(data)
    changed = dict(data, stress_Pa=measured.copy())
    changed["stress_Pa"][cycles > 3] += 1e8
    second = module.screen(changed)
    for a, b in zip(received[0], received[1], strict=True):
        np.testing.assert_array_equal(a, b)
    assert received[0][0][-1] == 29
    assert first["parameters"] == second["parameters"]
    assert first["final_state"] == second["final_state"]
    assert second["cycles"][-1]["stress_errors"]["coordinate_weighted_rms_error"] > 90
