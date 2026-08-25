from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
    assert persisted["source_commit"] == "a" * 40
    assert len(persisted["wheel_sha256"]) == 64
    assert (output / wheel.name).read_bytes() == b"immutable candidate"
    assert (output / "project").is_dir()
    assert "plane-strain" in (output / "TASK.md").read_text()
    assert "complete fresh-task transcript" in (output / "REVIEW.md").read_text()
