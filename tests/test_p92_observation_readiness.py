import numpy as np
import pytest
import json
from pathlib import Path

from tools.p92_observation_readiness import secant_support
from tools.p92_response_screen import POLICY


def test_resolution_gate_is_independent_of_a_material_prediction():
    result = secant_support([0., 10.], [0., .00002], 1e-5)
    assert result["status"] == "unresolved"
    assert result["resolution_units"] == 2.
    assert result["required_signal"] == POLICY["minimum_resolution_units"]*1e-5
    assert secant_support([0., 10.], [0., .00006], 1e-5)["status"] == "resolved"


def test_dense_samples_do_not_invent_independent_replicates():
    sparse = secant_support([0., 10.], [0., .00002], 1e-5)
    dense = secant_support(np.linspace(0, 10, 101), np.linspace(0, .00002, 101), 1e-5)
    assert dense["status"] == sparse["status"]
    assert dense["resolution_units"] == sparse["resolution_units"]


@pytest.mark.parametrize("t,y,q", [([0., 0.], [0., 1.], .01), ([1., 0.], [0., 1.], .01),
    ([0., 1.], [[0.], [1.]], .01), ([0., 1.], [0., np.nan], .01),
    ([0., 1j], [0., 1.], .01), ([0., 1.], [0., 1.], 0.),
    ([0., 1.], [0., 1.], [.01]), ([0., 1.], [0., 1.], .01j),
    ([0., 1.], [0., 1.], True), ([0., 1.], [0., 1.], "0.01"),
    ([0., 1.], ["0", "1"], .01)])
def test_invalid_observations_rejected(t, y, q):
    with pytest.raises(ValueError, match="ordered real"):
        secant_support(t, y, q)


def test_missing_window_does_not_become_a_zero_rate():
    result = secant_support([], [], .01)
    assert result["status"] == "unresolved"
    assert "observed_secant_per_s" not in result


def test_archived_resolution_limit_is_not_a_material_pass():
    path = Path(__file__).resolve().parents[1]/"evidence/creep/2026-11-22-observation-readiness.json"
    records = json.loads(path.read_text())
    assert records["RF"]["all_secants_resolved"]
    assert records["SLR"]["all_secants_resolved"]
    slc = records["SLC"]
    assert not slc["all_secants_resolved"]
    unresolved = [r for r in slc["records"] if r["status"] == "unresolved"]
    assert len(unresolved) == 10
    assert {r["cycle"] for r in unresolved} == set(range(1, 11))
    assert all(r["window"] == "late" and r["channel"] == "strain" for r in unresolved)
    for record in records.values():
        assert record["response_policy"] == POLICY
        assert not record["material_solved"] and not record["engineering_qualified"]


def test_archived_temperature_span_is_not_assigned_to_every_hold():
    path = Path(__file__).resolve().parents[1]/"evidence/creep/2026-11-22-observation-readiness.json"
    records = json.loads(path.read_text())
    expected_spans = {"RF": 2.7, "SLR": 9.6, "SLC": 11.1}
    for name, report in records.items():
        lo, hi = report["recorded_temperature_range_K"]
        assert hi-lo == pytest.approx(expected_spans[name], abs=1e-10)
        for row in report["records"]:
            thermal = row["temperature_context"]
            assert thermal["samples"] == thermal["source_indices"][1]-thermal["source_indices"][0]+1
            assert lo <= thermal["minimum_K"] <= thermal["maximum_K"] <= hi
            assert thermal["end_K"]-thermal["start_K"] == thermal["change_K"]
    for name, change in (("SLR", -.4), ("SLC", .1)):
        first = next(r for r in records[name]["records"]
                     if r["segment"] == 5 and r["window"] == "early")
        thermal = first["temperature_context"]
        assert thermal["change_K"] == pytest.approx(change, abs=1e-10)
        assert thermal["maximum_K"]-thermal["minimum_K"] < 1.5
