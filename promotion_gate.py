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
import json
from pathlib import Path
from typing import Callable, Iterable


TARGET = "0.3"


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
    return GateResult(
        "G1",
        "one public scientific language",
        "passed" if not gaps else "blocked",
        (
            "dependency-free _api_contract owns module, Model, command, and stage discovery",
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
    return GateResult(
        "G3",
        "one execution and evidence lifecycle",
        "passed" if not gaps else "blocked",
        (
            "StepExecutionPolicy normalizes solver/output/history/progress/checkpoint",
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


def _gate_platforms(records: tuple[dict[str, object], ...]) -> GateResult:
    required = {"linux", "macos", "wsl2"}
    accepted = {
        str(record.get("platform_id", "")).lower()
        for record in records
        if record.get("schema") == "agentfem.platform-acceptance"
        and record.get("status") == "passed"
        and record.get("installed_wheel") is True
        and record.get("release_smoke") == "passed"
    }
    missing = tuple(sorted(required - accepted))
    return GateResult(
        "G5",
        "clean installed use on Linux, macOS, and WSL2",
        "passed" if not missing else "external_evidence_required",
        tuple(f"platform:{name}" for name in sorted(accepted)),
        tuple(f"missing installed-wheel acceptance for {name}" for name in missing),
    )


def _gate_extension(records: tuple[dict[str, object], ...]) -> GateResult:
    accepted = [
        record
        for record in records
        if record.get("schema") == "agentfem.extension-acceptance"
        and record.get("status") == "passed"
        and record.get("installed_wheel") is True
        and record.get("core_modified") is False
        and record.get("simulation_result") == "passed"
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


def _gate_agent(records: tuple[dict[str, object], ...]) -> GateResult:
    accepted = [
        record
        for record in records
        if record.get("schema") == "agentfem.agent-acceptance"
        and record.get("runtime") == "passed"
        and record.get("capability_discovery") == "passed"
        and record.get("declared_maturity_evidence") == "passed"
        and record.get("templates")
    ]
    gaps = () if accepted else (
        "missing installed agent acceptance with at least one completed template",
    )
    return GateResult(
        "G7",
        "an unfamiliar agent can inspect, build, run, verify, and explain",
        "passed" if accepted else "external_evidence_required",
        tuple(f"agentfem:{item.get('agentfem_version')}" for item in accepted),
        gaps,
    )


def evaluate(*, evidence: Iterable[Path] = ()) -> dict[str, object]:
    """Evaluate G1--G7 and return a stable JSON-safe report."""

    records = _read_evidence(tuple(Path(item) for item in evidence))
    gates = (
        _gate_public_language(),
        _gate_lowering(),
        _gate_result_lifecycle(),
        _gate_scientific_evidence(),
        _gate_platforms(records),
        _gate_extension(records),
        _gate_agent(records),
    )
    return {
        "schema": "agentfem.platform-promotion",
        "schema_version": "0.1.0",
        "target": TARGET,
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
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    options = parser.parse_args()
    report = evaluate(evidence=options.evidence)
    if options.report is not None:
        _write(options.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if options.require_complete and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
