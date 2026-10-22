# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A rejected performance candidate must leave evidence and release its scope."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("phase", ["setup", "solve"])
def test_failed_benchmark_preserves_record_and_restores_workspace(monkeypatch, tmp_path, phase):
    functions = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                  / "tools/benchmark_retained_material_trial.py"))
    augmentation = functions["AffinePlaneNormalAugmentation"]
    original = augmentation._linear_workspace_scope
    output = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--mechanism", "combined", "--output", str(output)])

    def fail(*args, **kwargs):
        raise RuntimeError("injected numerical failure")

    def make(**kwargs):
        if phase == "setup":
            fail()
        return SimpleNamespace(output_every=1, state_transaction=SimpleNamespace(), solve=fail)

    monkeypatch.setattr(functions["runpy"], "run_path", lambda *args: {"make_step": make})
    with pytest.raises(RuntimeError, match="injected"):
        functions["main"]()
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["mechanism"] == "combined"
    assert report["runs"][0]["status"] == "failed_" + phase
    assert report["runs"][0]["error_type"] == "RuntimeError"
    assert len(report["tool_sha256"]) == 64
    assert "observed_elapsed_reduction" not in report
    assert augmentation._linear_workspace_scope is original
