from dataclasses import asdict, replace

import numpy as np
import pytest

from tools.p92_mixed_control import stress_hold, run
from tools.p92_published_recovery_reference import published_parameters


@pytest.mark.parametrize("sign", [1., -1.])
def test_stress_control_matches_constant_flow_solution_without_state_mutation(sign):
    p = replace(published_parameters(600), Q=0., H=0., gamma_r1=0., gamma_r2=0.,
                C1=0., C2=0., gamma1=0., gamma2=0.)
    old = np.array([sign*300., 0., 0., 0., 0.])
    flow = ((300-p.k)/p.Z)**p.n
    result = stress_hold(old, .002, 100., p, limit=1., rate=.1)
    np.testing.assert_allclose(result.y[:, -1], [sign*300., 0., 0., 0., 100*flow,
                                                .002+sign*100*flow], atol=1e-10)
    np.testing.assert_array_equal(old, [sign*300., 0., 0., 0., 0.])


def test_stress_hold_subdivision_preserves_history_and_creep_strain():
    p = published_parameters(600)
    old = np.array([260., 10., 20., -20., .3])
    whole = stress_hold(old, 0., 20., p, limit=2., rate=.1, rtol=1e-10).y[:, -1]
    half = stress_hold(old, 0., 10., p, limit=2., rate=.1, rtol=1e-10).y[:, -1]
    resumed = stress_hold(half[:5], half[5], 10., p,
                          limit=2., rate=.1, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(whole, resumed, rtol=1e-8, atol=2e-7)


def test_nominal_slc_carries_stress_and_strain_through_control_switches():
    calibration = dict(fit={"success": True}, initial_state=[0.]*5,
                       parameters=asdict(replace(published_parameters(600),
                                                 E=155176., k=116., Z=1192.)))
    report = run(calibration, cycles=2)
    first, second = report["cycles"]
    ends = first["segment_endpoints"]
    assert ends[3]["control"] == "stress"
    assert ends[3]["state"][0] == pytest.approx(ends[2]["state"][0], abs=1e-8)
    assert ends[3]["strain"] > ends[2]["strain"]
    assert ends[4]["strain"] == .002 and ends[-1]["strain"] == 0.
    assert ends[4]["duration_s"] == pytest.approx(abs(.002-ends[3]["strain"])/.001)
    assert second["end_state"][4] > first["end_state"][4] > 0
    assert report["protocol_envelope_exceeded"]
    assert report["first_envelope_exit_cycle"] == 2
    assert not report["life_qualified"]
    assert calibration["initial_state"] == [0.]*5
    alternate = run(calibration, cycles=1, ramp_mode="table_duration")
    assert alternate["cycles"][0]["segment_endpoints"][4]["duration_s"] == 2.


def test_invalid_hold_rejected_before_returning_state():
    with pytest.raises(ValueError):
        stress_hold([0.]*5, 0., -1., published_parameters(600), limit=2., rate=.1)


def test_failed_local_solve_does_not_publish_a_candidate(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_mixed_control as module
    old = np.array([260., 10., 20., -20., .3])
    saved = old.copy()
    monkeypatch.setattr(module, "solve_ivp", lambda *args, **kwargs:
                        SimpleNamespace(success=False, y=np.zeros((6, 2))))
    with pytest.raises(RuntimeError, match="failed numerical"):
        stress_hold(old, 0., 10., published_parameters(600), limit=2., rate=.1)
    np.testing.assert_array_equal(old, saved)


def test_real_mixed_control_failure_is_not_promoted():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "evidence/creep/2026-11-16-mixed-control-screen.json"
    record = json.loads(path.read_text())
    assert not record["model_admitted"] and not record["life_qualified"]
    assert record["first_envelope_exit_cycle"] == 2
    assert record["second_hold_predicted_strain"] > record["positive_strain_target"]
    assert all(r["cycle10_peak_relative_error"] < -.46
               for r in record["diagnostic_continuations"])
