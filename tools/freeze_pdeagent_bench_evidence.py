"""Freeze fixed-adapter PDEAgent-Bench results as an auditable evidence bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="+", type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runner-label",
        action="append",
        help="Expected summary label; repeat when a bundle contains several labels.",
    )
    parser.add_argument(
        "--development-agent",
        default="Codex (GPT-5.6-sol), collaboratively directed by the project author",
    )
    arguments = parser.parse_args()
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
