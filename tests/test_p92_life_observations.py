import json

import pytest

from agentfem.assessments import life_survival
from tools.p92_life_observations import SOURCE, observations, screen


def test_published_records_keep_units_conditions_and_source():
    data = json.loads(SOURCE.read_text())
    rows = observations(data)
    assert len(rows) == 18
    assert [r.life for r in rows[:12]] == [r[2] for r in data["creep"]]
    assert [r.life for r in rows[12:]] == [r[3] for r in data["fatigue"]+data["dwell"]]
    assert {r.life_unit for r in rows[:12]} == {"h"}
    assert {r.life_unit for r in rows[12:]} == {"cycles"}
    assert rows[12].conditions["peak_strain"] == .003
    assert rows[15].conditions["dwell_s"] == 600
    assert all(r.status == "failure" for r in rows)
    assert all(r.source.startswith(data["source"]) for r in rows)
    with pytest.raises(ValueError, match="Mixed"):
        life_survival(rows, independent_specimens=True, noninformative_censoring=True)


def test_p92_is_not_a_fictitious_18_specimen_scatter_population():
    report = screen()
    json.dumps(report, allow_nan=False)
    assert report["observation_count"] == report["condition_groups"] == 18
    assert report["reported_right_censored_count"] == 0
    assert not report["fitted_parameters"] and not report["engineering_qualified"]
    for group in report["groups"]:
        assert group["specimens"] == group["failure_count"] == 1
        assert not group["replicate_failures_available"]
        assert group["stations"][0]["lower"] is None
        assert group["stations"][0]["upper"] is None
