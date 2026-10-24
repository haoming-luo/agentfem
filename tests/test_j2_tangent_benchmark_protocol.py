# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Timing evidence is never successful if a correctness comparison failed."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


def entry(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    return runpy.run_path(str(tools / "benchmark_j2_tangent_algebra.py"))["main"]


def test_tangent_benchmark_keeps_failed_comparison(tmp_path, monkeypatch):
    main = entry(monkeypatch)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    monkeypatch.setitem(main.__globals__, "reference_tangent",
                        lambda: (lambda *a, **k: None, "test-reference"))
    def fail(*args, **kwargs):
        raise AssertionError("injected derivative mismatch")
    monkeypatch.setitem(main.__globals__, "paired", fail)
    with pytest.raises(AssertionError, match="derivative mismatch"):
        main()
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "AssertionError"
    assert report["kernels"] == [] and report["paths"] == []
    assert "global_abba" not in report
    assert report["runtime_source_sha256"]


def test_tangent_benchmark_never_overwrites_evidence(tmp_path, monkeypatch):
    main = entry(monkeypatch)
    output = tmp_path / "retained.json"
    output.write_text("keep previous evidence")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text() == "keep previous evidence"


def test_tangent_comparison_uses_each_points_stiffness_scale(monkeypatch):
    compare = entry(monkeypatch).__globals__["compare_tangent_tensors"]
    reference = np.full((2, 3, 3, 3, 3), 1000.)
    reference[:, 0, 0, 1, 2] = 0.
    actual = reference.copy()
    actual[0, 0, 0, 1, 2] = 2.3e-9
    assert compare(actual, reference) < 1e-12
    # A real stiffness change is not hidden by a global norm dominated by
    # another point, nor by a free absolute tolerance on zero reference data.
    reference[1] *= 1e-8
    actual = reference.copy()
    actual[1, 0, 0, 0, 0] *= 1.00001
    with pytest.raises(AssertionError, match="tensor error"):
        compare(actual, reference)
    with pytest.raises(AssertionError, match="tensor error"):
        compare(np.ones((1, 9, 9)), np.zeros((1, 9, 9)))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_tangent_comparison_rejects_nonfinite(monkeypatch, value):
    compare = entry(monkeypatch).__globals__["compare_tangent_tensors"]
    with pytest.raises(AssertionError, match="Invalid"):
        compare(np.full((1, 9, 9), value), np.ones((1, 9, 9)))
