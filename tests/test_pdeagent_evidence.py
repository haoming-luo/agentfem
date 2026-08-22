from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.freeze_pdeagent_bench_evidence import freeze_evidence, verify_evidence


ROOT = Path(__file__).parents[1]


def _summary(path: Path, family: str, case_id: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "agent_name": "runner-label",
                "equation_type_summary": {family: {"cases": 1, "passed": 1}},
                "results": [
                    {
                        "case_id": case_id,
                        "equation_type": family,
                        "status": "PASS",
                        "error": 1.0e-6,
                        "time": 0.1,
                        "gate_breakdown": {"final_pass": True},
                    }
                ],
            }
        )
    )
    return path


def test_fixed_adapter_evidence_is_self_contained_and_unambiguous(tmp_path):
    first = _summary(tmp_path / "first.json", "poisson", "case-a")
    second = _summary(tmp_path / "second.json", "stokes", "case-b")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "id": "case-a",
                "pde_classification": {"dim": 2},
                "output": {"grid": {"bbox": [0.0, 1.0, 0.0, 1.0]}},
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "case-b",
                "pde_classification": {},
                "oracle_config": {
                    "output": {"grid": {"bbox": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}}
                },
            }
        )
        + "\n"
    )

    manifest_path = freeze_evidence(
        (first, second),
        catalog=catalog,
        output=tmp_path / "evidence",
        repository=Path(__file__).parents[1],
        runner_labels=("runner-label",),
        development_agent="test agent",
    )

    manifest = json.loads(manifest_path.read_text())
    report = json.loads((manifest_path.parent / "report.json").read_text())
    assert manifest["evaluation_mode"] == "fixed_adapter"
    assert manifest["model_called_during_evaluation"] is False
    assert manifest["runner_labels"] == ["runner-label"]
    assert manifest["result"]["passed"] == 2
    assert manifest["result"]["total"] == 2
    assert report["macro_pass_rate"] == pytest.approx(1.0)
    assert report["minimum_family_pass_rate"] == pytest.approx(1.0)
    assert report["by_dimension"]["3"]["total"] == 1
    assert report["source"] == [
        "raw/01-poisson.json",
        "raw/02-stokes.json",
    ]
    assert set(manifest["artifacts"]) == {
        "raw/01-poisson.json",
        "raw/02-stokes.json",
        "report.json",
        "report.md",
    }
    verified = verify_evidence(manifest_path.parent)
    assert verified["agentfem_commit"] == manifest["agentfem_commit"]

    (manifest_path.parent / "report.md").write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_evidence(manifest_path.parent)


def test_fixed_adapter_verification_rejects_manifest_report_drift(tmp_path):
    summary = _summary(tmp_path / "summary.json", "poisson", "case-a")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps({"id": "case-a", "pde_classification": {"dim": 2}}) + "\n"
    )
    manifest_path = freeze_evidence(
        (summary,),
        catalog=catalog,
        output=tmp_path / "evidence",
        repository=Path(__file__).parents[1],
        runner_labels=("runner-label",),
        development_agent="test agent",
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["result"]["by_family"]["poisson"]["passed"] = 0
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="disagree for by_family"):
        verify_evidence(manifest_path.parent)


@pytest.mark.parametrize(
    "script",
    (
        "freeze_pdeagent_bench_evidence.py",
        "pdeagent_bench_report.py",
        "pdeagent_bench_solver.py",
    ),
)
def test_repository_evidence_tools_prefer_checkout_over_installed_package(
    script,
    tmp_path,
):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
