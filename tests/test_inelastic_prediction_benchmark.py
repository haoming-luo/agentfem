# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical failures must not turn into successful performance evidence."""

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("phase", ["setup", "solve"])
def test_failed_prediction_benchmark_retains_source_and_failure(monkeypatch, tmp_path, phase):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    functions = runpy.run_path(str(root / "tools/benchmark_inelastic_prediction.py"))
    main = functions["main"]
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])

    def fail(**kwargs):
        raise RuntimeError("injected numerical failure")

    def make(**kwargs):
        if phase == "setup":
            fail()
        return SimpleNamespace(solve=fail)

    monkeypatch.setitem(main.__globals__, "make_step", make)
    with pytest.raises(RuntimeError, match="injected numerical failure"):
        main()
    report = json.loads(output.read_text())
    assert len(report["source_sha256"]) == 64
    assert len(report["tool_sha256"]) == 64
    assert report["git_commit"]
    assert report["reference_is_exact"] is False
    assert report["records"][0]["status"] == "failed"
    assert report["records"][0]["error_type"] == "RuntimeError"
    assert "paired_timing" not in report
    assert "source_unchanged" not in report


def test_zero_reference_has_absolute_difference_not_false_relative_accuracy(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    functions = runpy.run_path(str(root / "tools/benchmark_inelastic_prediction.py"))
    assert functions["difference"]([1., 0.], [0., 0.]) == {
        "absolute_l2": 1., "relative_l2": None,
    }


def test_limit_refinement_cannot_publish_an_abba_speed_claim(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    main = runpy.run_path(str(root / "tools/benchmark_inelastic_prediction.py"))["main"]
    output = tmp_path / "refinement.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--refine-limit", "--output", str(output)])
    limits = []
    zeros = np.zeros(2)

    def make(**kwargs):
        limits.append(getattr(kwargs["incrementation"], "maximum_inelastic_increment", None))
        history = []
        def solve(*, until):
            history.append(SimpleNamespace(load_factor=until, converged=True, iterations=1,
                                           checks={"maximum_plastic_increment": 0.}))
        return SimpleNamespace(
            solve=solve, accepted_increments=history, attempted_increments=history,
            solution=SimpleNamespace(x=SimpleNamespace(array=zeros)),
            response=SimpleNamespace(state=SimpleNamespace(committed_state_vectors=lambda: zeros),
                                     first_piola_stress=SimpleNamespace(values=zeros),
                                     tangent=SimpleNamespace(values=zeros)),
        )

    def observe(*args):
        return dict.fromkeys(("force_on_structure", "pressure_area_mean", "maximum_pressure",
                              "active_reference_area", "peeq_reference_volume_mean", "maximum_peeq",
                              "displacement_l2", "mpc_equation_mismatch"), 0.)

    monkeypatch.setitem(main.__globals__, "make_step", make)
    monkeypatch.setitem(main.__globals__, "observe_penalty", observe)
    main()
    report = json.loads(output.read_text())
    assert limits == [None, .00075, .000375]
    assert report["status"] == "completed_limit_refinement"
    assert report["source_unchanged"]
    assert "paired_timing" not in report
