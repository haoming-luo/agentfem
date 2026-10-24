# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A rejected performance candidate must leave evidence and release its scope."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
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
    assert report["source_sha256"] and report["platform"] and report["python"]
    assert all(len(value) == 64 for value in report["source_sha256"].values())
    assert "observed_elapsed_reduction" not in report
    assert augmentation._linear_workspace_scope is original


def test_changed_source_cannot_promote_timing(monkeypatch, tmp_path):
    functions = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                  / "tools/benchmark_retained_material_trial.py"))
    main = functions["main"]
    snapshots = iter(({"fixture": "before"}, {"fixture": "after"}))
    monkeypatch.setitem(main.__globals__, "source_fingerprints", lambda: next(snapshots))
    output = tmp_path / "changed-source.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--mechanism", "combined", "--output", str(output)])
    zeros = np.zeros(2)

    def make(**kwargs):
        return SimpleNamespace(
            output_every=None, state_transaction=SimpleNamespace(), solve=lambda **kw: None,
            solution=SimpleNamespace(x=SimpleNamespace(array=zeros)),
            response=SimpleNamespace(
                state=SimpleNamespace(committed_state_vectors=lambda: zeros),
                first_piola_stress=SimpleNamespace(values=zeros), tangent=SimpleNamespace(values=zeros)),
            contact_provider=SimpleNamespace(accepted_point_samples=lambda: {"normal_pressures": zeros}),
            accepted_history_recorders={"normal_contact_work:interior_plane":
                                        SimpleNamespace(records=[{"work": 0.}])},
            accepted_increments=[SimpleNamespace(iterations=0, checks={})])

    monkeypatch.setattr(functions["runpy"], "run_path", lambda *args: {"make_step": make})
    with pytest.raises(RuntimeError, match="source changed"):
        main()
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert len(report["runs"]) == 4
    assert all(row["accepted_by_declared_comparison"] for row in report["runs"])
    assert report["source_sha256"] == {"fixture": "before"}
    assert report["source_sha256_after"] == {"fixture": "after"}
    assert "observed_elapsed_reduction" not in report
