# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit digitization and reference calculation, not material admission."""

import copy
import json
import sys

import numpy as np
import pytest

from tools.p92_dwell_figure_screen import axis_map, load_observations, main, run_screen


def test_axis_conversion_preserves_coordinates_and_reports_fit_residual():
    slope, offset, residual = axis_map([[100, 0], [300, 10], [500, 20]])
    assert slope == pytest.approx(.05)
    assert offset == pytest.approx(-5.)
    assert residual < 1.e-10
    inverted = axis_map([[100, 400], [700, 200]])
    assert inverted[0] == pytest.approx(-1/3)


@pytest.mark.parametrize("ticks", [[], [[1, 2]], [[1, 2], [1, 3]], [[1, 2], [2, 2]],
                                  [[1, 2], [np.nan, 3]]])
def test_invalid_axis_calibration(ticks):
    with pytest.raises(ValueError):
        axis_map(ticks)


@pytest.mark.parametrize("mutation", ["source", "duplicate", "cycle", "time", "orientation", "bound", "pixel"])
def test_changed_or_invalid_observation_contract_is_rejected(tmp_path, mutation):
    data = copy.deepcopy(load_observations())
    panel = data["panels"][0]
    if mutation == "source":
        data["source_pdf_sha256"] = "0"*64
    elif mutation == "duplicate":
        data["panels"].append(copy.deepcopy(panel))
    elif mutation == "cycle":
        panel["cycle"] = 20
    elif mutation == "time":
        panel["experimental_square_centres_px"][0][0] = 3000.
    elif mutation == "orientation":
        for tick in panel["y_ticks_pixel_stress_MPa"]:
            tick[1] *= -1
    elif mutation == "bound":
        data["coordinate_reading_bound_px"] = 0.
    elif mutation == "pixel":
        panel["experimental_square_centres_px"][0][1] = -1.
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_observations(path)


def test_distinct_integrators_agree_and_keep_experiment_separate_from_model_plot():
    radau = run_screen(method="Radau")
    bdf = run_screen(method="BDF")
    assert radau["source_pdf_sha256"] == bdf["source_pdf_sha256"]
    assert radau["observation_sha256"] == bdf["observation_sha256"]
    assert not radau["material_admitted"]
    assert not radau["experimental_uncertainty_known"]
    assert not radau["metric_is_article_one_second_grid_MRE"]
    for a, b in zip(radau["panels"], bdf["panels"], strict=True):
        assert a["experimental_markers"]["count"] == 8
        for key in ("experimental_markers", "published_model_dashes"):
            np.testing.assert_allclose([s["predicted_MPa"] for s in a[key]["samples"]],
                                       [s["predicted_MPa"] for s in b[key]["samples"]], atol=1.e-3)
        # A regression guard on recorded disagreement, not an accuracy target.
        assert not a["experimental_markers"]["all_error_intervals_include_zero"]
        assert a["experimental_markers"]["maximum_absolute_error_MPa"] > 10.
    plotted_model = radau["panels"][0]["published_model_dashes"]
    assert plotted_model["count"] == 6
    assert plotted_model["all_error_intervals_include_zero"]


def test_cli_never_overwrites_scientific_evidence(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["screen", "--output", str(output)])
    assert main() == 0
    before = output.read_bytes()
    report = json.loads(before)
    assert report["parameters"].startswith("published Table 4, unchanged")
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes() == before
