"""Frozen research evidence consistency; never an engineering accuracy gate."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]/"evidence/creep"


def read(name):
    return json.loads((ROOT/name).read_text())


def test_transfer_uses_exact_constrained_calibration_record():
    rf = read("2026-11-16-bam-positive-strength.json")
    slr = read("2026-11-16-bam-positive-strength-slr.json")
    assert slr["calibration_record_sha256"] == sha256(
        json.dumps(rf, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert slr["parameters"] == rf["parameters"]
    assert rf["fit"]["active_bounds"] == [0, -1, 0]
    assert rf["fit"]["sufficient_strength_lower_bound_MPa"] > 0
    assert not rf["engineering_qualified"] and not slr["engineering_qualified"]


def test_nominal_long_path_numerical_agreement_is_separate_from_accuracy():
    reports = read("2026-11-16-bam-positive-strength-long-cycles.json")["reports"]
    assert [r["method"] for r in reports] == ["Radau", "BDF"]
    for report in reports:
        assert report["completed_cycles"] == 1000 and report["failure"] is None
        assert not report["engineering_qualified"]
    for first, second in zip(reports[0]["cycles"], reports[1]["cycles"], strict=True):
        assert first["cycle"] == second["cycle"]
        assert first["sampled_peak_MPa"] == pytest.approx(second["sampled_peak_MPa"], abs=2e-6, rel=0)
        np.testing.assert_allclose(first["end_state"], second["end_state"], atol=3e-6, rtol=0)
    # Good numerical agreement does not erase the measured peak discrepancy.
    assert abs(reports[0]["cycles"][-1]["sampled_peak_MPa"]/181.74-1) > .09
