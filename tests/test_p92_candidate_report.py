import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.p92_candidate_report import summarize


def fixture():
    data = dict(protocol="RF", metadata={"failure_cycles": 4}, sha256="fixture",
                cycle=np.array([1, 1]), recorded_cycles=[1], stress_Pa=np.array([-100e6, 100e6]))
    coarse = dict(parameters={"E": 1000.}, limit=2., rate=.1, rtol=1e-8,
                  protocol="RF", calibration_sha256="fixture", segment_end_times_s=[2., 182.], cycles=[
                      dict(cycle=1, sampled_peak_MPa=100., sampled_valley_MPa=-100.,
                           end_state=[0.]*5)])
    fine = copy.deepcopy(coarse)
    fine["rtol"] = 1e-10
    return coarse, fine, data


def test_numerical_gate_does_not_promote_science():
    a, b, data = fixture()
    report = summarize(a, b, data)
    assert report["numerical_refinement_passed"]
    assert not report["engineering_qualified"] and not report["life_qualified"]
    b["cycles"][0]["end_state"][0] = 1.
    assert not summarize(a, b, data)["numerical_refinement_passed"]


def test_different_parameters_and_cycles_cannot_be_called_refinement():
    for field, value in (("parameters", {"E": 2000}), ("rate", .2), ("rtol", 1e-7),
                         ("protocol", "SLR"), ("calibration_sha256", "different-initial-state"),
                         ("segment_end_times_s", [2., 200.])):
        a, b, data = fixture()
        b[field] = value
        with pytest.raises(ValueError):
            summarize(a, b, data)
    a, b, data = fixture()
    b["cycles"][0]["cycle"] = 2
    with pytest.raises(ValueError):
        summarize(a, b, data)


def test_valleys_and_accumulated_strain_cannot_escape_numerical_gate():
    a, b, data = fixture()
    b["cycles"][0]["sampled_valley_MPa"] += .01
    assert not summarize(a, b, data)["numerical_refinement_passed"]
    a, b, data = fixture()
    b["cycles"][0]["end_state"][4] = 1.e-5
    assert not summarize(a, b, data)["numerical_refinement_passed"]
    b["cycles"][0]["sampled_valley_MPa"] = float("nan")
    with pytest.raises(ValueError, match="extrema must be finite"):
        summarize(a, b, data)


def test_incomplete_legacy_artifact_requires_regeneration_not_invented_identity():
    a, b, data = fixture()
    del a["calibration_sha256"]
    with pytest.raises(ValueError, match="regenerate"):
        summarize(a, b, data)


def test_archived_real_screen_preserves_failed_transfer_and_all_observations():
    root = Path(__file__).resolve().parents[1] / "evidence/creep"
    rf = json.loads((root / "2026-11-16-bounded-recovery-screen.json").read_text())
    slr = json.loads((root / "2026-11-16-bounded-recovery-slr.json").read_text())
    assert rf["accepted_cycles"] == 1000 and rf["numerical_refinement_passed"]
    assert len(rf["comparison"]["records"]) == 118
    assert rf["peak_refinement_max_MPa"] < 1e-6
    assert slr["comparison"]["records"][-1]["peak_relative_error"] < -.12
    assert not rf["engineering_qualified"] and not slr["engineering_qualified"]
