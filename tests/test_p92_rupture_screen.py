import json
from dataclasses import replace

import pytest

from agentfem import assessments
from tools.p92_rupture_screen import screen


def test_published_nodes_remain_distinct_and_reproduce_time_fraction():
    report = screen()
    assert not report["engineering_qualified"]
    assert len(report["excluded_interrupted_test_ids"]) == 7
    assert [r["curve"]["rupture_times"] for r in report["records"]] == [
        [807.0, 7.2], [330.0, 7.4]]
    for record in report["records"]:
        for check in record["checks"]:
            assert check["recovered_rupture_hours"] == check["supplied_rupture_hours"]
            assert check["half_life_time_fraction"] == pytest.approx(.5)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("condition", [0, 1])
def test_explicit_time_conversion_and_unadmitted_queries(condition):
    record = screen()["records"][condition]["curve"]
    keys = ("stresses", "rupture_times", "temperature", "stress_unit", "time_unit",
            "stress_measure", "material_condition", "failure_definition", "source")
    curve = assessments.TabulatedRuptureCurve(**{key: record[key] for key in keys})
    seconds = replace(curve, rupture_times=tuple(v * 3600 for v in curve.rupture_times), time_unit="s")
    for stress in (120., 140., 170.):
        assert seconds(stress, 923.15) / 3600 == pytest.approx(curve(stress, 923.15))
    for stress, temperature in ((119., 923.15), (171., 923.15), (-120., 923.15), (120., 873.15)):
        with pytest.raises(ValueError):
            curve(stress, temperature)
