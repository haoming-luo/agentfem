# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Timing evidence is never successful if a correctness comparison failed."""
import json
from pathlib import Path
import runpy
import sys

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
