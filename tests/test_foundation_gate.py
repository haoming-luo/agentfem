# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foundation_gate


def _write(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _evidence(tmp_path: Path):
    wheel = tmp_path / "agentfem-0.4.0-py3-none-any.whl"
    wheel.write_bytes(b"candidate wheel")
    wheel_digest = foundation_gate._sha256(wheel)
    agent = _write(
        tmp_path / "agent.json",
        {
            "schema": "agentfem.agent-acceptance",
            "agentfem_version": "0.4.0",
            "runtime": "passed",
            "capability_discovery": "passed",
            "declared_maturity_evidence": "passed",
            "templates": {"static": {"provenance": "verified"}},
            "mpi_smoke": {"status": "passed", "rank_count": 2},
        },
    )
    extension = _write(
        tmp_path / "extension.json",
        {
            "schema": "agentfem.extension-acceptance",
            "status": "passed",
            "agentfem_version": "0.4.0",
            "core_commit": "a" * 40,
            "core_wheel_sha256": wheel_digest,
            "extension": "reference-material",
            "extension_distribution": "agentfem-reference-material",
            "extension_version": "1.0.0",
            "installed_wheel": True,
            "isolated_from_source_checkout": True,
            "core_modified": False,
            "simulation_result": "passed",
            "verification": "passed",
            "trust_level": "verified",
        },
    )
    return wheel, agent, extension


def test_foundation_acceptance_binds_all_release_stages(tmp_path):
    wheel, agent, extension = _evidence(tmp_path)

    record = foundation_gate.create_foundation_acceptance(
        wheel=wheel,
        agent_acceptance_path=agent,
        extension_acceptance_path=extension,
        source_commit="a" * 40,
        source_dirty=False,
        validated=tuple(sorted(foundation_gate.REQUIRED_VALIDATIONS)),
    )

    assert record["status"] == "passed"
    assert record["wheel_sha256"] == foundation_gate._sha256(wheel)
    assert set(record["representative_mpi"]) == {
        "state",
        "nonlinear",
        "output",
        "checkpoint",
    }
    assert record["external_provider"]["name"] == "reference-material"
    assert record["evidence"]["validated_stages"] == tuple(
        sorted(foundation_gate.REQUIRED_VALIDATIONS)
    )


def test_foundation_acceptance_fails_closed_on_missing_stage(tmp_path):
    wheel, agent, extension = _evidence(tmp_path)

    with pytest.raises(ValueError, match="missing="):
        foundation_gate.create_foundation_acceptance(
            wheel=wheel,
            agent_acceptance_path=agent,
            extension_acceptance_path=extension,
            source_commit="a" * 40,
            source_dirty=False,
            validated=("complete_serial",),
        )


def test_foundation_acceptance_rejects_dirty_or_mismatched_candidate(tmp_path):
    wheel, agent, extension = _evidence(tmp_path)
    validated = tuple(sorted(foundation_gate.REQUIRED_VALIDATIONS))

    with pytest.raises(RuntimeError, match="clean source"):
        foundation_gate.create_foundation_acceptance(
            wheel=wheel,
            agent_acceptance_path=agent,
            extension_acceptance_path=extension,
            source_commit="a" * 40,
            source_dirty=True,
            validated=validated,
        )

    payload = json.loads(extension.read_text(encoding="utf-8"))
    payload["core_wheel_sha256"] = "f" * 64
    _write(extension, payload)
    with pytest.raises(RuntimeError, match="not bound"):
        foundation_gate.create_foundation_acceptance(
            wheel=wheel,
            agent_acceptance_path=agent,
            extension_acceptance_path=extension,
            source_commit="a" * 40,
            source_dirty=False,
            validated=validated,
        )
