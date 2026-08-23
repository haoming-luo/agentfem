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
    evidence = []
    for platform in ("linux", "macos", "wsl2"):
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
            },
        )
    )
    evidence.append(
        _write(
            tmp_path,
            "agent.json",
            {
                "schema": "agentfem.agent-acceptance",
                "agentfem_version": "0.3.0",
                "runtime": "passed",
                "capability_discovery": "passed",
                "declared_maturity_evidence": "passed",
                "templates": {"static-solid": {"run": "completed"}},
            },
        )
    )

    report = promotion_gate.evaluate(evidence=evidence)

    assert report["status"] == "passed"
    assert report["passed"] == report["required"] == 7
