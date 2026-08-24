"""Record a fresh AI-agent trial without confusing it with deterministic CI.

The AI agent runs AgentFEM in a clean task first.  This recorder then checks
the installed runtime, project, structured result and artifact integrity.  A
human reviewer only confirms whether the saved explanation is scientifically
adequate; any repair or prompting intervention remains visible and prevents a
zero-intervention promotion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _cli(*arguments: str, cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "agentfem.cli", *arguments, "--json"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "stdout": completed.stdout,
        }
    return json.loads(completed.stdout)


def evaluate(
    project: Path,
    *,
    agent: str,
    transcript: Path,
    explanation: Path,
    fresh_context: bool,
    human_interventions: int,
    explanation_reviewed: bool,
) -> dict[str, object]:
    root = Path(project).resolve()
    transcript_path = Path(transcript).resolve()
    explanation_path = Path(explanation).resolve()
    doctor = _cli("doctor", cwd=root)
    capabilities = _cli("capabilities", cwd=root)
    check = _cli("check", "--project", str(root), cwd=root)
    inspect = _cli("inspect", "--project", str(root), cwd=root)
    verify = _cli("verify", "--project", str(root), cwd=root)

    execution = doctor.get("execution", {})
    installed_wheel = (
        execution.get("mode") == "installed_distribution"
        and execution.get("distribution_mismatch") is False
    )
    transcript_ok = transcript_path.is_file() and bool(
        transcript_path.read_text(encoding="utf-8").strip()
    )
    explanation_ok = explanation_path.is_file() and bool(
        explanation_path.read_text(encoding="utf-8").strip()
    )
    runtime = "passed" if doctor.get("schema") == "agentfem.runtime-report" else "failed"
    capability_discovery = (
        "passed"
        if capabilities.get("schema") == "agentfem.capabilities"
        else "failed"
    )
    project_check = "passed" if check.get("status") == "passed" else "failed"
    simulation_result = (
        "passed"
        if inspect.get("schema") == "agentfem.simulation-result"
        or inspect.get("status") in {"completed", "verified"}
        else "failed"
    )
    verification = "passed" if verify.get("status") == "verified" else "failed"
    scientific_explanation = (
        "reviewed" if explanation_ok and explanation_reviewed else "unreviewed"
    )
    gaps = []
    if not installed_wheel:
        gaps.append("trial did not execute one consistent installed wheel")
    if not fresh_context:
        gaps.append("agent task inherited project-specific history")
    if int(human_interventions) != 0:
        gaps.append("agent required human repair or redirect intervention")
    if not transcript_ok:
        gaps.append("trial transcript is missing or empty")
    for name, value in (
        ("runtime", runtime),
        ("capability discovery", capability_discovery),
        ("project check", project_check),
        ("simulation result", simulation_result),
        ("verification", verification),
    ):
        if value != "passed":
            gaps.append(f"{name} did not pass")
    if scientific_explanation != "reviewed":
        gaps.append("scientific explanation has not been reviewed")

    return {
        "schema": "agentfem.agent-trial-acceptance",
        "schema_version": "0.1.0",
        "status": "passed" if not gaps else "failed",
        "agent": str(agent),
        "agentfem_version": doctor.get("packages", {}).get("agentfem"),
        "installed_wheel": installed_wheel,
        "fresh_context": bool(fresh_context),
        "human_interventions": int(human_interventions),
        "runtime": runtime,
        "capability_discovery": capability_discovery,
        "project_check": project_check,
        "simulation_result": simulation_result,
        "verification": verification,
        "scientific_explanation": scientific_explanation,
        "project": str(root),
        "transcript": str(transcript_path),
        "explanation": str(explanation_path),
        "runtime_fingerprint": doctor,
        "gaps": gaps,
    }


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--explanation", type=Path, required=True)
    parser.add_argument("--fresh-context", action="store_true")
    parser.add_argument("--human-interventions", type=int, default=0)
    parser.add_argument("--reviewed-explanation", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args()
    report = evaluate(
        options.project,
        agent=options.agent,
        transcript=options.transcript,
        explanation=options.explanation,
        fresh_context=options.fresh_context,
        human_interventions=options.human_interventions,
        explanation_reviewed=options.reviewed_explanation,
    )
    _write(options.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
