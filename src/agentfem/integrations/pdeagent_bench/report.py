"""Failure-aware summaries for official PDEAgent-Bench output."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path


FAILURE_STAGES = (
    "PASS",
    "SCHEMA_FAIL",
    "GEOMETRY_FAIL",
    "SOLVER_FAIL",
    "OUTPUT_FAIL",
    "ACCURACY_FAIL",
    "TIME_FAIL",
    "EXECUTION_FAIL",
)


@dataclass(frozen=True)
class BenchmarkReport:
    """Compact scientific and product evidence from an official summary."""

    source: Path
    total: int
    passed: int
    by_family: dict[str, dict[str, object]]
    by_dimension: dict[str, dict[str, object]]
    failures: dict[str, int]
    cases: tuple[dict[str, object], ...]

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.total == 0 else self.passed / self.total

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.pdeagent-bench-report.v1",
            "source": str(self.source),
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "by_family": self.by_family,
            "by_dimension": self.by_dimension,
            "failures": self.failures,
            "cases": list(self.cases),
        }

    def markdown(self) -> str:
        lines = [
            "# AgentFEM PDEAgent-Bench report",
            "",
            f"- Cases: **{self.total}**",
            f"- Passed: **{self.passed} ({self.pass_rate:.1%})**",
            "",
            "## Equation families",
            "",
            "| Family | Passed | Total | Pass rate | Median error | Median time |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for family, summary in sorted(self.by_family.items()):
            lines.append(
                f"| {family} | {summary['passed']} | {summary['total']} | "
                f"{summary['pass_rate']:.1%} | {summary['median_error']:.3e} | "
                f"{summary['median_time']:.3f} s |"
            )
        if self.by_dimension:
            lines.extend(
                [
                    "",
                    "## Spatial dimensions",
                    "",
                    "| Dimension | Passed | Total | Pass rate |",
                    "|---|---:|---:|---:|",
                ]
            )
            for dimension, summary in sorted(self.by_dimension.items()):
                lines.append(
                    f"| {dimension}D | {summary['passed']} | {summary['total']} | "
                    f"{summary['pass_rate']:.1%} |"
                )
        lines.extend(["", "## Failure taxonomy", ""])
        if self.failures:
            for stage, count in sorted(self.failures.items()):
                lines.append(f"- {stage}: {count}")
        else:
            lines.append("- No failures.")
        return "\n".join(lines) + "\n"


def read_official_summary(
    path: str | Path,
    *,
    case_catalog: str | Path | None = None,
) -> BenchmarkReport:
    """Read one official ``summary.json`` and normalize failure evidence."""

    selected = Path(path)
    payload = json.loads(selected.read_text())
    results = tuple(dict(item) for item in payload.get("results", ()))
    dimensions = _case_dimensions(case_catalog)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_by_dimension: dict[str, list[dict[str, object]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    normalized_cases = []
    for item in results:
        family = str(item.get("equation_type", "unknown"))
        stage = classify_official_result(item)
        normalized = {
            "case_id": item.get("case_id"),
            "family": family,
            "dimension": dimensions.get(str(item.get("case_id"))),
            "status": "PASS" if stage == "PASS" else "FAIL",
            "failure_stage": stage,
            "error": item.get("error"),
            "time": item.get("time"),
            "target_error": item.get("target_error"),
            "target_time": item.get("target_time"),
        }
        normalized_cases.append(normalized)
        grouped[family].append(normalized)
        if normalized["dimension"] is not None:
            grouped_by_dimension[str(normalized["dimension"])].append(normalized)
        if stage != "PASS":
            failure_counts[stage] += 1

    by_family = {}
    for family, entries in grouped.items():
        passed = sum(entry["status"] == "PASS" for entry in entries)
        errors = [float(entry["error"]) for entry in entries if entry["error"] is not None]
        times = [float(entry["time"]) for entry in entries if entry["time"] is not None]
        by_family[family] = {
            "total": len(entries),
            "passed": passed,
            "pass_rate": passed / len(entries),
            "median_error": _median(errors),
            "median_time": _median(times),
        }
    by_dimension = {
        dimension: _compact_summary(entries)
        for dimension, entries in grouped_by_dimension.items()
    }
    return BenchmarkReport(
        source=selected,
        total=len(results),
        passed=sum(item["status"] == "PASS" for item in normalized_cases),
        by_family=by_family,
        by_dimension=by_dimension,
        failures=dict(failure_counts),
        cases=tuple(normalized_cases),
    )


def _case_dimensions(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    selected = Path(path)
    dimensions = {}
    if selected.is_dir():
        cases = (
            json.loads(case_path.read_text())
            for case_path in selected.glob("*/agent_output/case_spec.json")
        )
    else:
        cases = (
            json.loads(line)
            for line in selected.read_text().splitlines()
            if line.strip()
        )
    for case in cases:
        classification = case.get("pde_classification") or {}
        dimension = classification.get("dim")
        if dimension is None:
            output = case.get("output") or {}
            bbox = (output.get("grid") or {}).get("bbox") or ()
            if len(bbox) in {4, 6}:
                dimension = len(bbox) // 2
        if dimension is not None:
            dimensions[str(case.get("id"))] = int(dimension)
    return dimensions


def _compact_summary(entries: list[dict[str, object]]) -> dict[str, object]:
    passed = sum(entry["status"] == "PASS" for entry in entries)
    return {
        "total": len(entries),
        "passed": passed,
        "pass_rate": passed / len(entries),
    }


def classify_official_result(item: dict[str, object]) -> str:
    """Map official gate data and AgentFEM diagnostics to a stable taxonomy."""

    gates = item.get("gate_breakdown") or {}
    if bool(gates.get("final_pass", item.get("status") == "PASS")):
        return "PASS"
    reason = str(
        gates.get("failure_reason")
        or item.get("fail_reason")
        or item.get("error_message")
        or ""
    )
    upper = reason.upper()
    if "AFM-PDEB-001" in upper or "AFM-PDEB-002" in upper or "AFM-PDEB-003" in upper:
        return "SCHEMA_FAIL"
    if "AFM-PDEB-004" in upper:
        return "GEOMETRY_FAIL"
    if "AFM-PDEB-006" in upper or "AFM-PDEB-007" in upper or "SHAPE" in upper:
        return "OUTPUT_FAIL"
    stage = str(gates.get("failure_stage") or "").lower()
    if stage == "accuracy":
        return "ACCURACY_FAIL"
    if stage == "time":
        return "TIME_FAIL"
    if "KSP" in upper or "SOLVE" in upper or "AFM-PDEB-009" in upper:
        return "SOLVER_FAIL"
    return "EXECUTION_FAIL"


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])
