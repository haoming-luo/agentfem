"""Record a fresh AI-agent trial without confusing it with deterministic CI.

The AI agent runs AgentFEM in a clean task first.  This recorder then checks
the installed runtime, project, structured result and artifact integrity.  A
human reviewer only confirms whether the saved explanation is scientifically
adequate; any repair or prompting intervention remains visible and prevents a
zero-intervention promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


_REQUIRED_SEQUENCE = (
    "doctor",
    "capabilities",
    "init",
    "check",
    "run",
    "inspect",
    "verify",
)
_REQUIRED_OUTPUTS = {
    "project/agentfem.toml",
    "project/case.py",
    "project/result.json",
    "project/explanation.md",
    "agent-transcript.md",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_member(bundle: Path, value: object) -> Path | None:
    """Resolve one contract-owned path without allowing bundle escape."""

    if not isinstance(value, str) or not value.strip():
        return None
    candidate = (bundle / value).resolve()
    if not candidate.is_relative_to(bundle):
        return None
    return candidate


def _bound_file(
    bundle: Path,
    contract: dict[str, object],
    *,
    path_field: str,
    hash_field: str,
) -> tuple[Path | None, bool]:
    path = _contract_member(bundle, contract.get(path_field))
    expected = contract.get(hash_field)
    valid = bool(
        path is not None
        and path.is_file()
        and isinstance(expected, str)
        and len(expected) == 64
        and _sha256(path) == expected
    )
    return path, valid


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
    source_commit: str | None = None,
    wheel: Path | None = None,
    contract: Path | None = None,
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
    wheel_path = None if wheel is None else Path(wheel).resolve()
    wheel_ok = wheel_path is not None and wheel_path.is_file()
    wheel_sha256 = _sha256(wheel_path) if wheel_ok else None
    contract_path = None if contract is None else Path(contract).resolve()
    contract_record: dict[str, object] = {}
    if contract_path is not None and contract_path.is_file():
        loaded = json.loads(contract_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            contract_record = loaded
    bundle = contract_path.parent if contract_path is not None else root.parent
    task_path, task_ok = _bound_file(
        bundle,
        contract_record,
        path_field="task",
        hash_field="task_sha256",
    )
    review_path, review_ok = _bound_file(
        bundle,
        contract_record,
        path_field="review",
        hash_field="review_sha256",
    )
    expected_project = _contract_member(
        bundle, contract_record.get("project_directory")
    )
    required_outputs = contract_record.get("required_outputs", ())
    output_records: dict[str, dict[str, object]] = {}
    output_contract_verified = bool(
        isinstance(required_outputs, list)
        and set(required_outputs) == _REQUIRED_OUTPUTS
    )
    if isinstance(required_outputs, list):
        for logical_path in required_outputs:
            path = _contract_member(bundle, logical_path)
            present = bool(
                path is not None and path.is_file() and path.stat().st_size > 0
            )
            output_records[str(logical_path)] = {
                "path": None if path is None else str(path),
                "present": present,
                "sha256": _sha256(path) if present else None,
            }
            output_contract_verified = output_contract_verified and present
    else:
        output_contract_verified = False
    required_sequence = contract_record.get("required_sequence")
    sequence_contract_verified = bool(
        isinstance(required_sequence, list)
        and tuple(required_sequence) == _REQUIRED_SEQUENCE
    )

    expected_transcript = _contract_member(bundle, "agent-transcript.md")
    expected_wheel = _contract_member(bundle, contract_record.get("wheel"))
    expected_explanation = (
        None
        if expected_project is None
        else (expected_project / "explanation.md").resolve()
    )
    path_binding_verified = bool(
        expected_project == root
        and expected_wheel == wheel_path
        and expected_transcript == transcript_path
        and expected_explanation == explanation_path
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
        and inspect.get("trust_level") in {"verified", "validated"}
        else "failed"
    )
    verification = "passed" if verify.get("status") == "verified" else "failed"
    scientific_explanation = (
        "reviewed" if explanation_ok and explanation_reviewed else "unreviewed"
    )
    gaps = []
    if not installed_wheel:
        gaps.append("trial did not execute one consistent installed wheel")
    if not source_commit:
        gaps.append("trial does not identify the exact AgentFEM source commit")
    if not wheel_ok:
        gaps.append("trial does not retain the exact installed wheel candidate")
    candidate_identity_verified = bool(
        contract_record.get("schema") == "agentfem.agent-trial-contract"
        and contract_record.get("schema_version") == "0.2.0"
        and contract_record.get("agentfem_version")
        == doctor.get("packages", {}).get("agentfem")
        and contract_record.get("source_commit") == source_commit
        and contract_record.get("wheel")
        == (None if wheel_path is None else wheel_path.name)
        and contract_record.get("wheel_sha256") == wheel_sha256
        and task_ok
        and review_ok
        and path_binding_verified
        and output_contract_verified
        and sequence_contract_verified
    )
    if not candidate_identity_verified:
        gaps.append("trial contract does not match the installed candidate identity")
    if not fresh_context:
        gaps.append("agent task inherited project-specific history")
    if int(human_interventions) != 0:
        gaps.append("agent required human repair or redirect intervention")
    if not transcript_ok:
        gaps.append("trial transcript is missing or empty")
    if not task_ok:
        gaps.append("trial task is missing or differs from the immutable contract")
    if not review_ok:
        gaps.append("review instructions differ from the immutable contract")
    if not path_binding_verified:
        gaps.append("project, transcript, or explanation is outside its contract path")
    if not output_contract_verified:
        gaps.append("one or more contract-required outputs are missing")
    if not sequence_contract_verified:
        gaps.append("trial command sequence differs from the product contract")
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
        "schema_version": "0.2.0",
        "status": "passed" if not gaps else "failed",
        "agent": str(agent),
        "agentfem_version": doctor.get("packages", {}).get("agentfem"),
        "source_commit": source_commit,
        "wheel": None if wheel_path is None else str(wheel_path),
        "wheel_sha256": wheel_sha256,
        "trial_contract": None if contract_path is None else str(contract_path),
        "trial_contract_sha256": (
            _sha256(contract_path)
            if contract_path is not None and contract_path.is_file()
            else None
        ),
        "candidate_identity_verified": candidate_identity_verified,
        "task": None if task_path is None else str(task_path),
        "task_sha256": _sha256(task_path) if task_ok else None,
        "review": None if review_path is None else str(review_path),
        "review_sha256": _sha256(review_path) if review_ok else None,
        "path_binding_verified": path_binding_verified,
        "output_contract_verified": output_contract_verified,
        "sequence_contract_verified": sequence_contract_verified,
        "output_records": output_records,
        "installed_wheel": installed_wheel,
        "fresh_context": bool(fresh_context),
        "human_interventions": int(human_interventions),
        "runtime": runtime,
        "capability_discovery": capability_discovery,
        "project_check": project_check,
        "simulation_result": simulation_result,
        "result_trust_level": inspect.get("trust_level"),
        "verification": verification,
        "scientific_explanation": scientific_explanation,
        "project": str(root),
        "transcript": str(transcript_path),
        "transcript_sha256": _sha256(transcript_path) if transcript_ok else None,
        "explanation": str(explanation_path),
        "explanation_sha256": _sha256(explanation_path) if explanation_ok else None,
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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
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
        source_commit=options.source_commit,
        wheel=options.wheel,
        contract=options.contract,
    )
    _write(options.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
