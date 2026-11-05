"""Reproducibility and no-leakage checks; not additional material evidence."""

import copy
import numpy as np
import pytest
from tools.p92_life_screen import calibrate, features, load_table, screen


def test_published_table_and_explicit_failure_definition():
    data = load_table()
    assert [r["cycles"] for r in data["rows"]] == [
        1303,
        1190,
        675,
        541,
        870,
        853,
        625,
        482,
    ]
    assert data["strain_measure"] == "total axial strain range, not amplitude"
    assert "10 percent" in data["failure_definition"]


def test_held_out_life_does_not_change_calibration():
    data = load_table()
    changed = copy.deepcopy(data)
    for row in changed["rows"]:
        if row["id"] in changed["validation_ids"]:
            row["cycles"] *= 100
    np.testing.assert_array_equal(calibrate(data), calibrate(changed))


def test_cross_publication_overlap_is_not_external_validation():
    report = screen()
    audit = report["source_overlap_audit"]
    assert audit["matching_ids"] == [1, 2, 3]
    assert "exclude from independent validation" in audit["decision"]
    assert "Within-source" in report["scope"]


def test_fixed_baseline_and_reported_holdout_error():
    report = screen()
    held = [r for r in report["predictions"] if r["role"] == "validation"]
    np.testing.assert_allclose(
        [r["predicted_cycles"] for r in held],
        [961.8917203253, 788.4844593101],
        rtol=1e-9,
    )
    assert not report["engineering_qualified"]
    assert report["calibration_ids"] == [1, 3, 4, 5, 7, 8]
    assert report["validation_ids"] == [2, 6]
    assert max(r["error_factor"] for r in held) < 1.24


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature_C", 620),
        ("temperature_C", True),
        ("strain_range", 0.004),
        ("strain_range", 0.02),
        ("dwell_s", 0),
        ("dwell_s", 1800),
        ("dwell_s", float("nan")),
        ("dwell_s", 60 + 0j),
    ],
)
def test_no_interpolation_or_extrapolation(field, value):
    row = dict(temperature_C=600, strain_range=0.008, dwell_s=180)
    row[field] = value
    with pytest.raises(ValueError):
        features(row)


def test_missing_calibration_information_rejected():
    data = load_table()
    data["calibration_ids"] = [1, 3, 5, 7]
    data["validation_ids"] = [2, 4, 6, 8]
    with pytest.raises(ValueError, match="rank"):
        calibrate(data)


def test_baseline_monotone_within_declared_loading_family():
    coefficients = calibrate(load_table())
    assert np.all(coefficients[1:] <= 0)


@pytest.mark.parametrize("mutation", ["overlap", "duplicate", "missing", "empty"])
def test_in_memory_input_cannot_bypass_partition_validation(mutation):
    data = load_table()
    if mutation == "overlap":
        data["calibration_ids"].append(2)
    elif mutation == "duplicate":
        data["validation_ids"].append(2)
    elif mutation == "missing":
        data["validation_ids"].pop()
    else:
        data["calibration_ids"] += data["validation_ids"]
        data["validation_ids"] = []
    with pytest.raises(ValueError, match="partition"):
        screen(data)


@pytest.mark.parametrize("cycles", [True, 0, -1, np.inf, np.nan, 100 + 0j])
def test_in_memory_life_values_are_validated(cycles):
    data = load_table()
    data["rows"][0]["cycles"] = cycles
    with pytest.raises(ValueError, match="cycles"):
        calibrate(data)
