import numpy as np
import pytest
import json
from pathlib import Path

from tools.p92_residual_conflicts import split_cost


def test_partition_conserves_cost_and_finite_change():
    old = np.array([1., -2., 3., 4., -5.])
    new = old+np.array([.1, .2, -.3, -.4, .5])
    rows = split_cost(old, new, [0., 1., 20., 80., 100.], duration=100., hold=True, scale=2.)
    assert sum(r["samples"] for r in rows) == len(old)
    assert sum(r["baseline_cost"] for r in rows) == pytest.approx(old@old/8)
    assert sum(r["candidate_cost"] for r in rows) == pytest.approx(new@new/8)
    for row in rows:
        assert row["candidate_cost"]-row["baseline_cost"] == pytest.approx(
            row["linear_change"]+row["quadratic_change"])
        for label in ("baseline", "candidate"):
            assert row[f"{label}_cost"] == pytest.approx(
                row[f"{label}_offset_cost"]+row[f"{label}_shape_cost"]
                +row[f"{label}_offset_shape_cross"])


def test_anchor_is_first_observation_not_a_fabricated_control_switch():
    rows = split_cost([4., 4., 4.], [4., 2., -2.], [1., 5., 90.],
                      duration=100., hold=True, scale=1.)
    assert all(r["anchor_time"] == 1. for r in rows)
    assert all(r["baseline_shape_rms"] == 0. for r in rows)
    assert rows[-1]["candidate_shape_rms"] == 6.
    assert sum(r["candidate_cost"] for r in rows) == 12.


def test_ramps_are_not_mislabelled_as_hold_windows():
    rows = split_cost([1., 2.], [1., 2.], [0., 1.], duration=1., hold=False, scale=1.)
    assert len(rows) == 1 and rows[0]["window"] == "whole"
    assert rows[0]["linear_change"] == rows[0]["quadratic_change"] == 0.


@pytest.mark.parametrize("times", ([1., 0.], [0., 2.], [0., np.nan]))
def test_invalid_observation_support_rejected(times):
    with pytest.raises(ValueError):
        split_cost([1., 2.], [1., 2.], times, duration=1., hold=True, scale=1.)


def test_archive_reproduces_matched_cost_without_changing_acceptance():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    report = json.loads((root/"2026-11-22-early-residual-conflicts.json").read_text())
    for label, variant in (("baseline", "baseline"), ("candidate", "shared")):
        fit = json.loads((root/f"2026-11-21-static-memory-matched-{variant}.json").read_text())
        assert report[f"{label}_cost"] == pytest.approx(fit["fit"]["cost"], abs=1e-16)
        assert report["source_sha256"] == fit["source_sha256"]
        assert sum(r[f"{label}_cost"] for r in report["records"]) == pytest.approx(
            report[f"{label}_cost"], abs=1e-16)
    assert not report["fitted"] and not report["engineering_qualified"]
    assert {r["cycle"] for r in report["records"]} == {1, 2, 3}
    for r in report["records"]:
        for label in ("baseline", "candidate"):
            assert r[f"{label}_cost"] == pytest.approx(
                sum(r[f"{label}_{key}"] for key in
                    ("offset_cost", "shape_cost", "offset_shape_cross")), abs=1e-16)
