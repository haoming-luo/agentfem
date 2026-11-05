"""Cyclic figure parsing and scope, without a costly campaign in CI."""

import copy
import json
import numpy as np
import pytest
from tools.p92_cyclic_figure_screen import SOURCE, SOURCE_SHA256, screen


def synthetic_campaign():
    data = json.loads(SOURCE.read_text())
    return {"runs": [dict(
        temperature_C=p["temperature_C"], strain_range=p["strain_range"],
        tensile_dwell_s=p["dwell_s"], source_pdf_sha256=SOURCE_SHA256,
        reported_life_used_to_select_run_length=p["reported_life_cycles"],
        method="synthetic_test",
        cycles=[dict(cycle=i, peak_stress_MPa=300.) for i in range(1, 701)],
    ) for p in data["panels"]]}


def test_axes_and_report_do_not_turn_observed_life_into_prediction():
    report = screen(synthetic_campaign())
    assert report["reported_lives_are_inputs_not_predictions"]
    assert not report["engineering_qualified"]
    assert [c["count"] for c in report["comparisons"]] == [5, 5]
    for comparison in report["comparisons"]:
        for row in comparison["samples"]:
            assert row["predicted_peak_MPa"] == 300.
            low, high = row["reading_error_interval_MPa"]
            np.testing.assert_allclose((low+high)/2, 300-row["observed_peak_MPa"])


@pytest.mark.parametrize("fault", ["source", "normalization", "duplicate", "truncated", "nan", "order"])
def test_inconsistent_campaign_rejected(fault):
    data = synthetic_campaign()
    run = data["runs"][0]
    if fault == "source":
        run["source_pdf_sha256"] = "other"
    elif fault == "normalization":
        run["reported_life_used_to_select_run_length"] += 1
    elif fault == "duplicate":
        data["runs"].append(copy.deepcopy(run))
    elif fault == "truncated":
        run["cycles"] = run["cycles"][:100]
    elif fault == "nan":
        run["cycles"][0]["peak_stress_MPa"] = float("nan")
    else:
        run["cycles"] = run["cycles"][::-1]
    with pytest.raises(ValueError):
        screen(data)
