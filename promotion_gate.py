"""Executable platform-promotion audit for the AgentFEM 0.3 contract.

The ordinary release gate proves that one package artifact is internally
consistent.  This audit answers the broader question needed before 0.3:
whether the public language, provider boundary, result lifecycle, scientific
evidence, platform routes, external-extension seam, and agent entrypoints have
all produced evidence.  Missing external evidence is reported, never guessed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Iterable


TARGET = "0.3"


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _record_sha256(record: dict[str, object]) -> str:
    """Hash one evidence record independently of JSON whitespace."""

    payload = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_agent_trial(record: dict[str, object]) -> bool:
    """Return whether *record* is a complete fresh-agent acceptance."""

    return bool(
        record.get("schema") == "agentfem.agent-trial-acceptance"
        and record.get("status") == "passed"
        and record.get("installed_wheel") is True
        and record.get("fresh_context") is True
        and record.get("human_interventions") == 0
        and record.get("runtime") == "passed"
        and record.get("capability_discovery") == "passed"
        and record.get("project_check") == "passed"
        and record.get("simulation_result") == "passed"
        and record.get("verification") == "passed"
        and record.get("scientific_explanation") == "reviewed"
        and record.get("candidate_identity_verified") is True
        and _is_sha256(record.get("wheel_sha256"))
        and _is_sha256(record.get("transcript_sha256"))
        and _is_sha256(record.get("explanation_sha256"))
    )


def _candidate_identity() -> tuple[str, str | None]:
    """Return the core version and exact checkout commit under audit."""

    from agentfem import __version__

    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip() or None
    return str(__version__), commit


def _matches_candidate(
    record: dict[str, object],
    *,
    version: str,
    commit_field: str,
    commit: str | None,
) -> bool:
    if str(record.get("agentfem_version", "")) != version:
        return False
    if not commit or str(record.get(commit_field, "")) != commit:
        return False
    return True


@dataclass(frozen=True)
class GateResult:
    """One addressable promotion decision."""

    gate: str
    title: str
    status: str
    evidence: tuple[str, ...]
    gaps: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.gaps

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "evidence": self.evidence,
            "gaps": self.gaps,
        }


def _read_evidence(paths: Iterable[Path]) -> tuple[dict[str, object], ...]:
    records = []
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not isinstance(record.get("schema"), str):
            raise ValueError(f"Promotion evidence {path} has no schema.")
        records.append(record)
    return tuple(records)


def _gate_public_language() -> GateResult:
    from agentfem import _api_contract
    from agentfem._architecture_contract import OWNERSHIP_BOUNDARIES

    core = _api_contract.model_methods("core")
    compatibility = _api_contract.model_methods("compatibility")
    stages = _api_contract.WORKFLOW_STAGES
    gaps = []
    if "step" not in core or "stage" not in core:
        gaps.append("core Model language omits stage or step")
    leaked = tuple(name for name in compatibility if name in core)
    if leaked:
        gaps.append(f"compatibility methods leaked into core: {leaked}")
    expected = (
        "study",
        "model",
        "mesh_and_regions",
        "fields",
        "materials",
        "loads_and_constraints",
        "step",
        "solve",
        "result_and_verification",
    )
    if stages != expected:
        gaps.append("machine workflow stages differ from the recommended grammar")
    ownership = tuple(item.name for item in OWNERSHIP_BOUNDARIES)
    expected_ownership = (
        "model",
        "constitutive",
        "state",
        "operator",
        "procedure",
        "backend",
        "result_verification",
    )
    if ownership != expected_ownership:
        gaps.append("ownership boundaries differ from the stable middle-layer contract")
    return GateResult(
        "G1",
        "one public scientific language",
        "passed" if not gaps else "blocked",
        (
            "dependency-free _api_contract owns module, Model, command, and stage discovery",
            "dependency-free _architecture_contract owns scientific responsibility boundaries",
            "model.step is the recommended construction boundary",
            "historical material/procedure step methods are compatibility-only",
        ),
        tuple(gaps),
    )


def _gate_lowering() -> GateResult:
    from agentfem.step_providers import StepOptionContract, step_providers

    providers = step_providers()
    gaps = []
    names = tuple(item.name for item in providers)
    if len(set(names)) != len(names):
        gaps.append("step provider names are not unique")
    for provider in providers:
        if not isinstance(provider.option_contract, StepOptionContract):
            gaps.append(f"{provider.name} has no StepOptionContract")
        if not callable(provider.lower):
            gaps.append(f"{provider.name} has no callable lowerer")
        if "output" not in provider.option_contract.accepted:
            gaps.append(f"{provider.name} does not accept the shared output policy")
    return GateResult(
        "G2",
        "one provider-owned lowering architecture",
        "passed" if not gaps else "blocked",
        tuple(f"provider:{name}" for name in names),
        tuple(gaps),
    )


def _gate_result_lifecycle() -> GateResult:
    from agentfem import state
    from agentfem.results import SimulationResult
    from agentfem.step_providers import StepExecutionPolicy

    policy = StepExecutionPolicy().summary()
    expected = {"solver", "output", "history", "progress", "checkpoint"}
    gaps = []
    if set(policy) != expected:
        gaps.append(f"execution policy keys are {tuple(policy)}, expected {tuple(sorted(expected))}")
    for name in ("verify", "write_manifest", "add_checkpoint", "add_history"):
        if not callable(getattr(SimulationResult, name, None)):
            gaps.append(f"SimulationResult omits {name}()")
    for state_class in (state.TransientState, state.SecondOrderDynamicsState):
        missing = tuple(
            name
            for name in ("commit", "rollback", "snapshot", "restore")
            if not callable(getattr(state_class, name, None))
        )
        if missing:
            gaps.append(f"{state_class.__name__} omits state transaction methods {missing}")
    return GateResult(
        "G3",
        "one execution and evidence lifecycle",
        "passed" if not gaps else "blocked",
        (
            "StepExecutionPolicy normalizes solver/output/history/progress/checkpoint",
            "state protocols separate restart and atomic trial replacement",
            "SimulationResult owns verification, histories, checkpoints, and manifests",
            "SolveEvent traces retain accepted and failed attempts",
        ),
        tuple(gaps),
    )


def _gate_scientific_evidence() -> GateResult:
    from agentfem.benchmarks import audit_capability_evidence

    reports = audit_capability_evidence()
    gaps = tuple(
        f"{item.capability}: {item.gaps}"
        for item in reports
        if not item.meets_declared_maturity
    )
    return GateResult(
        "G4",
        "capability claims stay within executable evidence",
        "passed" if not gaps else "blocked",
        tuple(
            f"{item.capability}:{item.maturity}:{len(item.benchmark_ids)} benchmark(s)"
            for item in reports
        ),
        gaps,
    )


def _gate_platforms(
    records: tuple[dict[str, object], ...],
    *,
    version: str,
    commit: str | None,
) -> GateResult:
    required = {"linux", "macos"}
    accepted = {
        str(record.get("platform_id", "")).lower()
        for record in records
        if record.get("schema") == "agentfem.platform-acceptance"
        and record.get("status") == "passed"
        and record.get("installed_wheel") is True
        and record.get("release_smoke") == "passed"
        and record.get("source_dirty") is False
        and _is_sha256(record.get("wheel_sha256"))
        and _matches_candidate(
            record,
            version=version,
            commit_field="source_commit",
            commit=commit,
        )
    }
    missing = tuple(sorted(required - accepted))
    return GateResult(
        "G5",
        "clean installed use on Linux and macOS; WSL2 tracked separately",
        "passed" if not missing else "external_evidence_required",
        tuple(f"platform:{name}" for name in sorted(accepted)),
        tuple(f"missing installed-wheel acceptance for {name}" for name in missing),
    )


def _gate_extension(
    records: tuple[dict[str, object], ...],
    *,
    version: str,
    commit: str | None,
) -> GateResult:
    accepted = [
        record
        for record in records
        if record.get("schema") == "agentfem.extension-acceptance"
        and record.get("status") == "passed"
        and record.get("installed_wheel") is True
        and record.get("core_modified") is False
        and record.get("simulation_result") == "passed"
        and bool(record.get("companion_commit"))
        and _is_sha256(record.get("core_wheel_sha256"))
        and _is_sha256(record.get("extension_wheel_sha256"))
        and _matches_candidate(
            record,
            version=version,
            commit_field="core_commit",
            commit=commit,
        )
    ]
    gaps = () if accepted else (
        "missing installed companion/third-party provider acceptance",
    )
    return GateResult(
        "G6",
        "external provider without core modification",
        "passed" if accepted else "external_evidence_required",
        tuple(str(item.get("extension")) for item in accepted),
        gaps,
    )


def _gate_agent(
    records: tuple[dict[str, object], ...],
    *,
    version: str,
    commit: str | None,
) -> GateResult:
    accepted = [
        record
        for record in records
        if _valid_agent_trial(record)
        and _matches_candidate(
            record,
            version=version,
            commit_field="source_commit",
            commit=commit,
        )
    ]
    source_trials = [record for record in records if _valid_agent_trial(record)]
    promoted = []
    for bridge in records:
        if not (
            bridge.get("schema") == "agentfem.agent-trial-promotion"
            and bridge.get("status") == "passed"
            and bridge.get("allowed_changes_only") is True
            and bridge.get("behavior_equivalent") is True
            and bridge.get("protected_runtime_tree_source")
            == bridge.get("protected_runtime_tree_target")
            and _is_sha256(bridge.get("protected_runtime_tree_source"))
            and _is_sha256(bridge.get("target_wheel_sha256"))
            and _is_sha256(bridge.get("source_acceptance_sha256"))
            and str(bridge.get("target_agentfem_version", "")) == version
            and commit
            and str(bridge.get("target_commit", "")) == commit
        ):
            continue
        source = next(
            (
                trial
                for trial in source_trials
                if _record_sha256(trial)
                == bridge.get("source_acceptance_sha256")
                and trial.get("agentfem_version")
                == bridge.get("source_agentfem_version")
                and trial.get("source_commit") == bridge.get("source_commit")
                and trial.get("wheel_sha256")
                == bridge.get("source_wheel_sha256")
            ),
            None,
        )
        if source is not None:
            promoted.append((bridge, source))

    gaps = () if accepted or promoted else (
        "missing zero-intervention fresh-agent trial from an installed wheel",
    )
    evidence = [
        f"{item.get('agent')}:{item.get('agentfem_version')}"
        for item in accepted
    ]
    evidence.extend(
        f"{source.get('agent')}:{bridge.get('source_agentfem_version')}"
        f"->{bridge.get('target_agentfem_version')}:behavior-equivalent"
        for bridge, source in promoted
    )
    return GateResult(
        "G7",
        "a fresh AI agent can inspect, build, run, verify, and explain",
        "passed" if accepted or promoted else "external_evidence_required",
        tuple(evidence),
        gaps,
    )


def evaluate(
    *,
    evidence: Iterable[Path] = (),
    candidate_version: str | None = None,
    candidate_commit: str | None = None,
) -> dict[str, object]:
    """Evaluate G1--G7 and return a stable JSON-safe report."""

    records = _read_evidence(tuple(Path(item) for item in evidence))
    detected_version, detected_commit = _candidate_identity()
    version = detected_version if candidate_version is None else str(candidate_version)
    commit = detected_commit if candidate_commit is None else str(candidate_commit)
    gates = (
        _gate_public_language(),
        _gate_lowering(),
        _gate_result_lifecycle(),
        _gate_scientific_evidence(),
        _gate_platforms(records, version=version, commit=commit),
        _gate_extension(records, version=version, commit=commit),
        _gate_agent(records, version=version, commit=commit),
    )
    return {
        "schema": "agentfem.platform-promotion",
        "schema_version": "0.1.0",
        "target": TARGET,
        "candidate_version": version,
        "candidate_commit": commit,
        "status": "passed" if all(item.passed for item in gates) else "incomplete",
        "passed": sum(item.passed for item in gates),
        "required": len(gates),
        "gates": [item.as_dict() for item in gates],
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
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument(
        "--evidence-directory",
        type=Path,
        action="append",
        default=[],
        help="Recursively consume JSON acceptance records from this directory.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    options = parser.parse_args()
    evidence = list(options.evidence)
    for directory in options.evidence_directory:
        if not directory.is_dir():
            parser.error(f"evidence directory does not exist: {directory}")
        evidence.extend(sorted(directory.rglob("*.json")))
    # Artifact downloads can contain the same record more than once. Preserve
    # stable order while preventing duplicate evidence from inflating a gate.
    evidence = list(dict.fromkeys(Path(item).resolve() for item in evidence))
    report = evaluate(evidence=evidence)
    report["evidence_files"] = [str(item) for item in evidence]
    if options.report is not None:
        _write(options.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if options.require_complete and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
