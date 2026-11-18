import copy
import json

import pytest

from agentfem.assessments import life_survival
from tools.grade92_creep_observations import SOURCE, observations, screen


def test_real_rupture_and_runout_counts():
    report = screen()
    assert (report["specimens"], report["failures"], report["right_censored"]) == (25, 22, 3)
    assert len(report["groups"]) == 22
    assert not report["engineering_qualified"]
    assert not report["bam_p92_holdout"]
    groups = [g for g in report["groups"] if g["right_censored_count"]]
    assert len(groups) == 3
    assert all(g["failure_count"] == 0 and g["median_point_estimate"] is None for g in groups)
    assert sorted(g["stations"][0]["life"] for g in groups) == [22246, 28988, 39863]
    assert all(g["stations"][0]["survival"] == 1 for g in groups)
    assert all(g["stations"][0]["lower"] is None for g in groups)
    json.dumps(report, allow_nan=False)


def test_replicates_preserve_heat_and_plate():
    report = screen()
    replicates = [g for g in report["groups"] if g["specimens"] > 1]
    assert sorted(g["specimens"] for g in replicates) == [2, 3]
    three = next(g for g in replicates if g["specimens"] == 3)
    assert three["median_point_estimate"] == 23143.8
    assert {r["batch_id"] for r in three["observations"]} == {"011448"}
    assert {r["conditions"]["plate"] for r in three["observations"]} == {"48A2"}
    rows = observations()
    same_condition_different_heats = [r for r in rows if r.conditions["stress_MPa"] == 90]
    with pytest.raises(ValueError, match="Mixed population"):
        life_survival(same_condition_different_heats, independent_specimens=True,
                      noninformative_censoring=True)


def test_processing_and_nonmonotonic_data_not_rewritten():
    rows = {r.specimen_id: r for r in observations()}
    assert rows["64-2-05"].life == 344.8
    assert "Re-normalized" in rows["49NT01"].conditions["processing"]
    assert "Re-normalized" not in rows["48A1R-1"].conditions["processing"]
    assert rows["49NT01"].status == "right_censored"


def test_duplicate_specimens_and_reordered_columns_rejected():
    data = json.loads(SOURCE.read_text())
    duplicate = copy.deepcopy(data)
    duplicate["observations"].append(duplicate["observations"][0])
    with pytest.raises(ValueError, match="Duplicate specimen"):
        observations(duplicate)
    data["columns"][0:2] = data["columns"][1::-1]
    with pytest.raises(ValueError, match="columns changed"):
        observations(data)
