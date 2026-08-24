from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "agent_trial_acceptance.py"
)
SPEC = importlib.util.spec_from_file_location("agent_trial_acceptance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
agent_trial_acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent_trial_acceptance)


def test_agent_trial_requires_installed_fresh_zero_intervention_evidence(
    tmp_path, monkeypatch
):
    transcript = tmp_path / "transcript.md"
    explanation = tmp_path / "explanation.md"
    transcript.write_text("fresh task transcript", encoding="utf-8")
    explanation.write_text("reviewed scientific explanation", encoding="utf-8")

    records = {
        "doctor": {
            "schema": "agentfem.runtime-report",
            "packages": {"agentfem": "0.3.0"},
            "execution": {
                "mode": "installed_distribution",
                "distribution_mismatch": False,
            },
        },
        "capabilities": {"schema": "agentfem.capabilities"},
        "check": {"status": "passed"},
        "inspect": {"schema": "agentfem.simulation-result"},
        "verify": {"status": "verified"},
    }

    def fake_cli(command, *_arguments, cwd):
        assert cwd == tmp_path
        return records[command]

    monkeypatch.setattr(agent_trial_acceptance, "_cli", fake_cli)

    report = agent_trial_acceptance.evaluate(
        tmp_path,
        agent="fresh-agent",
        transcript=transcript,
        explanation=explanation,
        fresh_context=True,
        human_interventions=0,
        explanation_reviewed=True,
    )

    assert report["status"] == "passed"
    assert report["installed_wheel"] is True
    assert report["gaps"] == []

    repaired = agent_trial_acceptance.evaluate(
        tmp_path,
        agent="fresh-agent",
        transcript=transcript,
        explanation=explanation,
        fresh_context=True,
        human_interventions=1,
        explanation_reviewed=True,
    )
    assert repaired["status"] == "failed"
    assert any("intervention" in item for item in repaired["gaps"])
