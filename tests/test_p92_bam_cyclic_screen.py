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


def test_independent_protocol_transfer_never_fits_observations():
    from dataclasses import asdict
    from tools.p92_bam_cyclic_screen import transfer_screen
    t = np.arange(21, dtype=float)
    cycles = np.minimum(t.astype(int)//2+1, 10)
    data = {"protocol": "SLR", "time_s": t, "cycle": cycles, "strain": .001*np.sin(t),
            "stress_Pa": np.zeros(len(t)), "temperature_K": np.full(len(t), 893.15),
            "sha256": "synthetic-SLR"}
    calibration = {"schema": "agentfem.research.p92-bam-cyclic.v1", "fit": {"success": True},
                   "parameters": asdict(published_parameters(600)), "source_sha256": "synthetic-RF"}
    first = transfer_screen(data, calibration)
    data["stress_Pa"][1:] = 1e8
    second = transfer_screen(data, calibration)
    assert first["parameters"] == second["parameters"] == calibration["parameters"]
    assert first["final_state"] == second["final_state"]
    assert first["calibration_record_sha256"] == second["calibration_record_sha256"]
    assert first["cycles"][-1]["stress_errors"] != second["cycles"][-1]["stress_errors"]
    calibration["fit"]["success"] = False
    with pytest.raises(ValueError, match="converge"):
        transfer_screen(data, calibration)


def test_roundoff_stagnation_in_elastic_recovery_is_checked_by_equations():
    from dataclasses import replace
    from tools.p92_published_recovery_reference import implicit_update, rates
    p = replace(published_parameters(600), E=146410.23798190424,
                k=108.23149217400956, Z=1499.9999999999977)
    old = np.array([-63.09640233221885, 0., 0., -.010139502168748472, 0.])
    dt, de = .01999999999999999, -2.2499999999999978e-5
    new, tangent = implicit_update(old, de, dt, p)
    np.testing.assert_allclose(new-old-dt*rates(new, de/dt, p), 0., atol=1e-12)
    assert new[0] == pytest.approx(old[0]+p.E*de)
    assert tangent == pytest.approx(p.E)


def test_bad_root_with_success_status_is_still_rejected(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_published_recovery_reference as reference
    monkeypatch.setattr(reference, "root", lambda *a, **k:
                        SimpleNamespace(success=True, x=np.zeros(5), message="mock success"))
    with pytest.raises(RuntimeError, match="failed"):
        reference.implicit_update(np.zeros(5), .001, .1, published_parameters(600))


@pytest.mark.parametrize("times,strains,stress", [
    ([0., 1.], [0., .001], [[0.], [10.]]),
    ([0., 0.], [0., .001], [0., 10.]),
    ([0., 1.], [0., np.nan], [0., 10.]),
    ([0., 1.], [0., .001], [0., complex(10, 1)]),
])
def test_invalid_calibration_cannot_enter_optimizer(monkeypatch, times, strains, stress):
    from tools import p92_bam_cyclic_screen as module
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid histories reached the optimizer")
    monkeypatch.setattr(module, "least_squares", forbidden)
    with pytest.raises(ValueError):
        module.calibrate(times, strains, stress, np.zeros(5))


def test_unconverged_fit_cannot_publish_holdout(monkeypatch):
    from tools import p92_bam_cyclic_screen as module
    monkeypatch.setattr(module, "calibrate", lambda *a, **k:
                        (published_parameters(600), {"success": False, "message": "evaluation limit"}))
    t = np.arange(101, dtype=float)
    data = {"protocol": "RF", "time_s": t, "strain": np.zeros_like(t),
            "stress_Pa": np.zeros_like(t), "cycle": np.minimum(t.astype(int)//10+1, 10),
            "temperature_K": np.full(len(t), 893.15), "sha256": "synthetic"}
    with pytest.raises(RuntimeError, match="no holdout prediction"):
        module.screen(data)


def test_sensitivity_freezes_state_and_never_uses_measured_stress(monkeypatch):
    from dataclasses import asdict
    from tools import p92_bam_cyclic_screen as module
    t = np.arange(21, dtype=float)
    data = {"protocol": "RF", "time_s": t, "cycle": np.minimum(t.astype(int)//2+1, 10),
            "strain": .001*np.sin(t), "stress_Pa": np.zeros(len(t)),
            "temperature_K": np.full(len(t), 893.15), "sha256": "synthetic"}
    calibration = {"schema": "agentfem.research.p92-bam-cyclic.v1", "fit": {"success": True},
                   "parameters": asdict(published_parameters(600)), "source_sha256": "synthetic",
                   "initial_state": [0., 0., 0., 0., 0.]}
    def forbidden(*args, **kwargs):
        pytest.fail("Sensitivity must not refit")
    monkeypatch.setattr(module, "calibrate", forbidden)
    first = module.sensitivity_screen(data, calibration)
    data["stress_Pa"][:] = 1e8
    assert first == module.sensitivity_screen(data, calibration)
    assert len(first["records"]) == 20
    assert not first["engineering_qualified"]
    calibration["source_sha256"] = "different-source"
    with pytest.raises(ValueError, match="match"):
        module.sensitivity_screen(data, calibration)
