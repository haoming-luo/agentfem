from __future__ import annotations

import json

import promotion_gate


def _write(tmp_path, name, record):
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_core_promotion_gates_are_executable_and_external_gaps_stay_visible():
    report = promotion_gate.evaluate()

    assert report["schema"] == "agentfem.platform-promotion"
    assert [item["gate"] for item in report["gates"]] == [
        "G1", "G2", "G3", "G4", "G5", "G6", "G7"
    ]
    assert all(item["passed"] for item in report["gates"][:4])
    assert all(not item["passed"] for item in report["gates"][4:])
    assert report["status"] == "incomplete"


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
                "core_modified": False,
                "simulation_result": "passed",
                "agentfem_version": version,
                "core_commit": commit,
                "companion_commit": "b" * 40,
                "core_wheel_sha256": "2" * 64,
                "extension_wheel_sha256": "3" * 64,
            },
        )
    )
    evidence.append(
        _write(
            tmp_path,
            "agent.json",
            {
                "schema": "agentfem.agent-trial-acceptance",
                "status": "passed",
                "agent": "fresh-test-agent",
                "agentfem_version": "0.3.0",
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
                "wheel_sha256": "4" * 64,
                "transcript_sha256": "5" * 64,
                "explanation_sha256": "6" * 64,
            },
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
