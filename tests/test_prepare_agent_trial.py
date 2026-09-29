from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_agent_trial.py"
SPEC = importlib.util.spec_from_file_location("prepare_agent_trial", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_agent_trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_agent_trial)


def test_prepare_agent_trial_binds_task_to_exact_candidate(tmp_path):
    wheel = tmp_path / "agentfem-0.3.0-py3-none-any.whl"
    wheel.write_bytes(b"immutable candidate")
    output = tmp_path / "trial"

    contract = prepare_agent_trial.prepare(
        wheel=wheel,
        output=output,
        source_commit="a" * 40,
        agentfem_version="0.3.0",
    )

    persisted = json.loads((output / "trial-contract.json").read_text())
    assert persisted == contract
    assert persisted["schema_version"] == "0.2.0"
    assert persisted["source_commit"] == "a" * 40
    assert len(persisted["wheel_sha256"]) == 64
    assert (output / wheel.name).read_bytes() == b"immutable candidate"
    assert (output / "project").is_dir()
    assert "plane-strain" in (output / "TASK.md").read_text()
    assert "complete fresh-task transcript" in (output / "REVIEW.md").read_text()
    assert persisted["task_sha256"] == prepare_agent_trial._sha256(
        output / "TASK.md"
    )
    assert persisted["review_sha256"] == prepare_agent_trial._sha256(
        output / "REVIEW.md"
    )
    assert persisted["required_sequence"] == [
        "doctor",
        "capabilities",
        "init",
        "check",
        "run",
        "inspect",
        "verify",
    ]


def test_trial_version_comes_from_candidate_wheel_not_installed_runtime(tmp_path):
    wheel = tmp_path / "agentfem-0.3.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "agentfem-0.3.1.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: agentfem\nVersion: 0.3.1\n",
        )

    assert prepare_agent_trial._wheel_version(wheel) == "0.3.1"


def test_trial_bundle_refuses_a_dirty_candidate_checkout(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    source = tmp_path / "candidate.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean source checkout"):
        prepare_agent_trial._require_clean_checkout(tmp_path)

    subprocess.run(["git", "add", "candidate.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "candidate",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    prepare_agent_trial._require_clean_checkout(tmp_path)
