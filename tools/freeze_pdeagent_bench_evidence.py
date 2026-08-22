"""Freeze fixed-adapter PDEAgent-Bench results as an auditable evidence bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
source_root = str(SOURCE_ROOT)
if SOURCE_ROOT.is_dir() and source_root not in sys.path:
    # Repository tools must audit the checkout, not an older installed wheel.
    sys.path.insert(0, source_root)

from agentfem.integrations.pdeagent_bench.report import combine_official_summaries
from agentfem.integrations.pdeagent_bench.schema import BENCHMARK_COMMIT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=repository, text=True
    ).strip()


def _family_slug(summary: Path) -> str:
    payload = json.loads(summary.read_text())
    families = sorted((payload.get("equation_type_summary") or {}).keys())
    return "-".join(families) if families else summary.stem


def freeze_evidence(
    summaries: tuple[Path, ...],
    *,
    catalog: Path,
    output: Path,
    repository: Path,
    runner_labels: tuple[str, ...] | None,
    development_agent: str,
) -> Path:
    """Copy raw summaries and seal their normalized result and provenance."""

    if not summaries:
        raise ValueError("At least one official summary is required.")
    commit = _git_value(repository, "rev-parse", "HEAD")
    dirty = bool(_git_value(repository, "status", "--porcelain"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evidence directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_directory = output / "raw"
    raw_directory.mkdir()

    labels = {
        str(json.loads(path.read_text()).get("agent_name", "")) for path in summaries
    }
    if runner_labels is not None and labels != set(runner_labels):
        raise ValueError(
            f"Runner label mismatch: summaries contain {sorted(labels)!r}, "
            f"expected {sorted(runner_labels)!r}."
        )

    frozen_sources = []
    artifact_hashes: dict[str, str] = {}
    for index, source in enumerate(summaries, start=1):
        destination = raw_directory / f"{index:02d}-{_family_slug(source)}.json"
        shutil.copy2(source, destination)
        relative = destination.relative_to(output).as_posix()
        frozen_sources.append(relative)
        artifact_hashes[relative] = _sha256(destination)

    report = combine_official_summaries(
        summaries,
        case_catalogs=(catalog,),
    )
    normalized = report.as_dict()
    normalized["source"] = frozen_sources
    report_json = output / "report.json"
    report_markdown = output / "report.md"
    report_json.write_text(json.dumps(normalized, indent=2) + "\n")
    report_markdown.write_text(report.markdown())
    artifact_hashes[report_json.name] = _sha256(report_json)
    artifact_hashes[report_markdown.name] = _sha256(report_markdown)

    adapter = repository / "src/agentfem/integrations/pdeagent_bench/adapter.py"
    manifest = {
        "schema": "agentfem.pdeagent-bench-evidence.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": "fixed_adapter",
        "model_called_during_evaluation": False,
        "runner_labels": sorted(labels),
        "runner_label_note": (
            "Labels retained from upstream or repair-run summaries; none is "
            "the identity of a model evaluated by this fixed-adapter run."
        ),
        "development_agent": development_agent,
        "agentfem_commit": commit,
        "agentfem_source_dirty": dirty,
        "benchmark_commit": BENCHMARK_COMMIT,
        "adapter_sha256": _sha256(adapter),
        "catalog_sha256": _sha256(catalog),
        "result": {
            "total": report.total,
            "passed": report.passed,
            "pass_rate": report.pass_rate,
            "by_family": report.by_family,
            "failures": report.failures,
        },
        "artifacts": artifact_hashes,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def verify_evidence(directory: Path) -> dict[str, object]:
    """Verify every sealed artifact and the normalized result totals."""

    root = directory.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "agentfem.pdeagent-bench-evidence.v1":
        raise ValueError(f"Unsupported evidence schema in {manifest_path}.")
    if manifest.get("evaluation_mode") != "fixed_adapter":
        raise ValueError("Evidence bundle is not a fixed-adapter evaluation.")
    if manifest.get("model_called_during_evaluation") is not False:
        raise ValueError(
            "Fixed-adapter evidence must state whether model inference occurred."
        )
    for relative, expected in (manifest.get("artifacts") or {}).items():
        artifact = (root / str(relative)).resolve()
        if not artifact.is_relative_to(root):
            raise ValueError(f"Artifact escapes evidence directory: {relative}")
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing evidence artifact: {relative}")
        observed = _sha256(artifact)
        if observed != expected:
            raise ValueError(
                f"Evidence hash mismatch for {relative}: {observed} != {expected}"
            )
    report = json.loads((root / "report.json").read_text())
    result = manifest.get("result") or {}
    for key in ("passed", "total", "pass_rate", "by_family", "failures"):
        if report.get(key) != result.get(key):
            raise ValueError(
                f"Manifest and normalized report disagree for {key}."
            )
    families = report.get("by_family") or {}
    if sum(int(item["total"]) for item in families.values()) != report.get("total"):
        raise ValueError("Family totals do not cover the normalized report.")
    if sum(int(item["passed"]) for item in families.values()) != report.get("passed"):
        raise ValueError("Family pass counts do not match the normalized report.")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="*", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runner-label",
        action="append",
        help="Expected summary label; repeat when a bundle contains several labels.",
    )
    parser.add_argument(
        "--development-agent",
        default="Codex (GPT-5.6-sol) used in the AgentFEM development workflow",
    )
    arguments = parser.parse_args()
    if arguments.verify is not None:
        if arguments.summary:
            parser.error("--verify does not accept summary inputs")
        manifest = verify_evidence(arguments.verify)
        result = manifest["result"]
        print(
            f"Verified {result['passed']}/{result['total']} fixed-adapter cases "
            f"in {arguments.verify}"
        )
        return 0
    if not arguments.summary or arguments.catalog is None or arguments.output is None:
        parser.error("freezing requires summaries, --catalog, and --output")
    path = freeze_evidence(
        tuple(arguments.summary),
        catalog=arguments.catalog,
        output=arguments.output,
        repository=arguments.repository,
        runner_labels=(
            None if arguments.runner_label is None else tuple(arguments.runner_label)
        ),
        development_agent=arguments.development_agent,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
