from __future__ import annotations

import json
from pathlib import Path

import agentfem
import promotion_gate


def _write(tmp_path, name, record):
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _agent_trial(*, version: str, commit: str, agent: str = "fresh-test-agent"):
    return {
        "schema": "agentfem.agent-trial-acceptance",
        "schema_version": "0.2.0",
        "status": "passed",
        "agent": agent,
        "agentfem_version": version,
        "source_commit": commit,
        "installed_wheel": True,
        "fresh_context": True,
        "human_interventions": 0,
        "runtime": "passed",
        "capability_discovery": "passed",
        "project_check": "passed",
        "simulation_result": "passed",
        "verification": "passed",
        "scientific_explanation": "reviewed",
        "candidate_identity_verified": True,
        "path_binding_verified": True,
        "output_contract_verified": True,
        "sequence_contract_verified": True,
        "result_trust_level": "verified",
        "wheel_sha256": "4" * 64,
        "trial_contract_sha256": "7" * 64,
        "task_sha256": "8" * 64,
        "review_sha256": "9" * 64,
        "transcript_sha256": "5" * 64,
        "explanation_sha256": "6" * 64,
        "output_records": {
            path: {
                "present": True,
                "sha256": "a" * 64,
            }
            for path in (
                "project/agentfem.toml",
                "project/case.py",
                "project/result.json",
                "project/explanation.md",
                "agent-transcript.md",
            )
        },
    }


def test_candidate_identity_is_bound_to_the_current_checkout():
    version, commit = promotion_gate._candidate_identity()

    expected_package = (promotion_gate.SOURCE_ROOT / "agentfem").resolve()
    assert Path(agentfem.__file__).resolve().is_relative_to(expected_package)
    assert version == agentfem.__version__
    assert commit is None or len(commit) == 40


def test_core_promotion_gates_are_executable_and_external_gaps_stay_visible():
    report = promotion_gate.evaluate()

    assert report["schema"] == "agentfem.platform-promotion"
    assert [item["gate"] for item in report["gates"]] == [
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
    ]
    assert all(item["passed"] for item in report["gates"][:4])
    assert all(not item["passed"] for item in report["gates"][4:])
    assert report["status"] == "incomplete"


def test_04_foundation_gate_separates_internal_readiness_from_release_evidence():
    report = promotion_gate.evaluate(target="0.4-foundation")

    assert report["target"] == "0.4-foundation"
    assert [item["gate"] for item in report["gates"]] == [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
    ]
    assert all(item["passed"] for item in report["gates"][:5])
    assert all(not item["passed"] for item in report["gates"][5:])
    assert report["status"] == "incomplete"


def test_04_foundation_acceptance_is_candidate_bound_and_fail_closed(tmp_path):
    version = "0.4.0"
    commit = "a" * 40
    record = {
        "schema": "agentfem.foundation-acceptance",
        "schema_version": "0.1.0",
        "status": "passed",
        "agentfem_version": version,
        "source_commit": commit,
        "source_dirty": False,
        "complete_serial": "passed",
        "mpi_rank_count": 2,
        "representative_mpi": {
            "state": "passed",
            "nonlinear": "passed",
            "output": "passed",
            "checkpoint": "passed",
        },
        "installed_wheel": "passed",
        "public_examples": "passed",
        "compatibility_imports": "passed",
        "wheel_sha256": "2" * 64,
    }
    extension_record = {
        "schema": "agentfem.extension-acceptance",
        "status": "passed",
        "extension": "reference-extension",
        "extension_distribution": "agentfem-reference-extension",
        "extension_version": "1.0.0",
        "entry_point": "reference_extension:extension",
        "entry_point_discovered": True,
        "entry_point_activated": True,
        "installed_wheel": True,
        "isolated_from_source_checkout": True,
        "core_modified": False,
        "simulation_result": "passed",
        "verification": "passed",
        "trust_level": "verified",
        "agentfem_version": version,
        "core_commit": commit,
        "core_wheel_sha256": "2" * 64,
        "extension_wheel_sha256": "3" * 64,
        "core_installation_sha256_before": "4" * 64,
        "core_installation_sha256_after": "4" * 64,
        "result_manifest_sha256": "5" * 64,
    }
    extension = _write(tmp_path, "extension.json", extension_record)
    agent_record = {
        "schema": "agentfem.agent-acceptance",
        "agentfem_version": version,
        "runtime": "passed",
        "capability_discovery": "passed",
        "declared_maturity_evidence": "passed",
    }
    agent = _write(tmp_path, "agent.json", agent_record)
    record["evidence"] = {
        "agent_acceptance_sha256": promotion_gate._record_sha256(agent_record),
        "extension_acceptance_sha256": promotion_gate._record_sha256(
            extension_record
        ),
        "validated_stages": [
            "complete_serial",
            "compatibility_imports",
            "mpi_checkpoint",
            "mpi_nonlinear",
            "mpi_output",
            "mpi_state",
        ],
    }
    evidence = _write(tmp_path, "foundation.json", record)

    report = promotion_gate.evaluate(
        target="0.4-foundation",
        evidence=(evidence, extension, agent),
        candidate_version=version,
        candidate_commit=commit,
    )
    gate = next(item for item in report["gates"] if item["gate"] == "F6")
    assert gate["passed"] is True

    record["representative_mpi"]["checkpoint"] = "missing"
    incomplete = _write(tmp_path, "incomplete-foundation.json", record)
    report = promotion_gate.evaluate(
        target="0.4-foundation",
        evidence=(incomplete, extension, agent),
        candidate_version=version,
        candidate_commit=commit,
    )
    gate = next(item for item in report["gates"] if item["gate"] == "F6")
    assert gate["passed"] is False


def test_external_evidence_can_complete_platform_extension_and_agent_gates(tmp_path):
    version = "0.3.0"
    commit = "a" * 40
    evidence = []
    for platform in ("linux", "macos"):
        evidence.append(
            _write(
                tmp_path,
                f"{platform}.json",
                {
                    "schema": "agentfem.platform-acceptance",
                    "platform_id": platform,
                    "status": "passed",
                    "installed_wheel": True,
                    "release_smoke": "passed",
                    "agentfem_version": version,
                    "source_commit": commit,
                    "source_dirty": False,
                    "wheel_sha256": "1" * 64,
                },
            )
        )
    evidence.append(
        _write(
            tmp_path,
            "extension.json",
            {
                "schema": "agentfem.extension-acceptance",
                "extension": "agentfem-learning.xdem",
                "status": "passed",
                "installed_wheel": True,
                "isolated_from_source_checkout": True,
                "core_modified": False,
                "simulation_result": "passed",
                "verification": "passed",
                "trust_level": "verified",
                "agentfem_version": version,
                "core_commit": commit,
                "extension_distribution": "agentfem-learning",
                "extension_version": "0.1.0",
                "entry_point": "agentfem_learning:extension",
                "entry_point_discovered": True,
                "entry_point_activated": True,
                "core_wheel_sha256": "2" * 64,
                "extension_wheel_sha256": "3" * 64,
                "core_installation_sha256_before": "4" * 64,
                "core_installation_sha256_after": "4" * 64,
                "result_manifest_sha256": "5" * 64,
            },
        )
    )
    evidence.append(
        _write(
            tmp_path,
            "agent.json",
            _agent_trial(version=version, commit=commit),
        )
    )

    report = promotion_gate.evaluate(
        evidence=evidence,
        candidate_version=version,
        candidate_commit=commit,
    )

    assert report["status"] == "passed"
    assert report["passed"] == report["required"] == 7


def test_wsl2_is_supported_evidence_but_not_a_promotion_blocker(tmp_path):
    version = "0.3.0"
    commit = "a" * 40
    evidence = []
    for platform in ("linux", "macos"):
        evidence.append(
            _write(
                tmp_path,
                f"{platform}.json",
                {
                    "schema": "agentfem.platform-acceptance",
                    "platform_id": platform,
                    "status": "passed",
                    "installed_wheel": True,
                    "release_smoke": "passed",
                    "agentfem_version": version,
                    "source_commit": commit,
                    "source_dirty": False,
                    "wheel_sha256": "1" * 64,
                },
            )
        )

    report = promotion_gate.evaluate(
        evidence=evidence,
        candidate_version=version,
        candidate_commit=commit,
    )
    gate = next(item for item in report["gates"] if item["gate"] == "G5")

    assert gate["passed"] is True
    assert gate["gaps"] == ()


def test_deterministic_entrypoint_smoke_cannot_impersonate_fresh_agent(tmp_path):
    record = _write(
        tmp_path,
        "automatic.json",
        {
            "schema": "agentfem.agent-acceptance",
            "runtime": "passed",
            "capability_discovery": "passed",
            "declared_maturity_evidence": "passed",
            "templates": {"static-solid": {"run": "completed"}},
        },
    )

    report = promotion_gate.evaluate(evidence=(record,))

    gate = next(item for item in report["gates"] if item["gate"] == "G7")
    assert gate["passed"] is False
    assert "fresh-agent" in gate["gaps"][0]


def test_old_or_different_commit_evidence_cannot_promote_current_candidate(tmp_path):
    platform = _write(
        tmp_path,
        "old-linux.json",
        {
            "schema": "agentfem.platform-acceptance",
            "platform_id": "linux",
            "status": "passed",
            "installed_wheel": True,
            "release_smoke": "passed",
            "agentfem_version": "0.2.2",
            "source_commit": "b" * 40,
        },
    )

    report = promotion_gate.evaluate(
        evidence=(platform,),
        candidate_version="0.3.0",
        candidate_commit="a" * 40,
    )

    gate = next(item for item in report["gates"] if item["gate"] == "G5")
    assert gate["evidence"] == ()
    assert any("linux" in item for item in gate["gaps"])


def test_behavior_equivalent_bridge_reuses_but_does_not_rewrite_agent_trial(tmp_path):
    source_commit = "b" * 40
    target_commit = "a" * 40
    source = _agent_trial(version="0.2.6", commit=source_commit, agent="Codex")
    source_path = _write(tmp_path, "source-agent.json", source)
    bridge_path = _write(
        tmp_path,
        "bridge.json",
        {
            "schema": "agentfem.agent-trial-promotion",
            "status": "passed",
            "allowed_changes_only": True,
            "behavior_equivalent": True,
            "source_acceptance_sha256": promotion_gate._record_sha256(source),
            "source_agentfem_version": "0.2.6",
            "source_commit": source_commit,
            "source_wheel_sha256": "4" * 64,
            "target_agentfem_version": "0.3.0",
            "target_commit": target_commit,
            "target_wheel_sha256": "7" * 64,
            "protected_runtime_tree_source": "8" * 64,
            "protected_runtime_tree_target": "8" * 64,
        },
    )

    report = promotion_gate.evaluate(
        evidence=(source_path, bridge_path),
        candidate_version="0.3.0",
        candidate_commit=target_commit,
    )
    gate = next(item for item in report["gates"] if item["gate"] == "G7")

    assert gate["passed"] is True
    assert gate["evidence"] == ("Codex:0.2.6->0.3.0:behavior-equivalent",)


def test_behavior_bridge_without_original_agent_trial_is_rejected(tmp_path):
    bridge_path = _write(
        tmp_path,
        "bridge-only.json",
        {
            "schema": "agentfem.agent-trial-promotion",
            "status": "passed",
            "allowed_changes_only": True,
            "behavior_equivalent": True,
            "source_acceptance_sha256": "1" * 64,
            "source_agentfem_version": "0.2.6",
            "source_commit": "b" * 40,
            "source_wheel_sha256": "4" * 64,
            "target_agentfem_version": "0.3.0",
            "target_commit": "a" * 40,
            "target_wheel_sha256": "7" * 64,
            "protected_runtime_tree_source": "8" * 64,
            "protected_runtime_tree_target": "8" * 64,
        },
    )

    report = promotion_gate.evaluate(
        evidence=(bridge_path,),
        candidate_version="0.3.0",
        candidate_commit="a" * 40,
    )
    gate = next(item for item in report["gates"] if item["gate"] == "G7")

    assert gate["passed"] is False
