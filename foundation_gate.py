# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Create candidate-bound acceptance for the AgentFEM 0.4 foundation.

This recorder does not run another test matrix.  It is invoked only after the
release job has completed its serial, MPI, installed-wheel, public-example,
compatibility-import, and external-extension checks.  It validates the durable
artifacts from those checks and binds the successful workflow stages to one
source commit and wheel digest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


SCHEMA = "agentfem.foundation-acceptance"
SCHEMA_VERSION = "0.1.0"
REQUIRED_VALIDATIONS = frozenset(
    {
        "complete_serial",
        "compatibility_imports",
        "mpi_checkpoint",
        "mpi_nonlinear",
        "mpi_output",
        "mpi_state",
    }
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_sha256(record: dict[str, object]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _read_record(
    path: Path,
    *,
    schema: str,
    require_passed_status: bool = True,
) -> dict[str, object]:
    selected = Path(path)
    record = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != schema:
        raise ValueError(f"{selected} is not a {schema} record.")
    if require_passed_status and record.get("status") != "passed":
        raise RuntimeError(f"{selected} did not pass: {record.get('status')!r}.")
    return record


def _source_identity(repository: Path) -> tuple[str, bool]:
    root = Path(repository).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("Foundation acceptance requires a full source commit.")
    return commit, bool(status)


def create_foundation_acceptance(
    *,
    wheel: Path,
    agent_acceptance_path: Path,
    extension_acceptance_path: Path,
    source_commit: str,
    source_dirty: bool,
    validated: tuple[str, ...],
) -> dict[str, object]:
    """Validate prior release artifacts and create one foundation record."""

    selected_wheel = Path(wheel).resolve()
    if not selected_wheel.is_file() or selected_wheel.suffix != ".whl":
        raise FileNotFoundError(f"Candidate wheel not found: {selected_wheel}")
    if len(str(source_commit)) != 40:
        raise ValueError("source_commit must be a full 40-character Git commit.")
    if source_dirty:
        raise RuntimeError("Foundation acceptance requires a clean source checkout.")

    declared = frozenset(str(item) for item in validated)
    missing = tuple(sorted(REQUIRED_VALIDATIONS - declared))
    unknown = tuple(sorted(declared - REQUIRED_VALIDATIONS))
    if missing or unknown:
        raise ValueError(
            "Foundation validation stages are incomplete or unknown: "
            f"missing={missing}, unknown={unknown}."
        )

    agent_path = Path(agent_acceptance_path).resolve()
    extension_path = Path(extension_acceptance_path).resolve()
    agent = _read_record(
        agent_path,
        schema="agentfem.agent-acceptance",
        require_passed_status=False,
    )
    extension = _read_record(
        extension_path,
        schema="agentfem.extension-acceptance",
    )
    version = str(agent.get("agentfem_version", ""))
    templates = agent.get("templates")
    mpi_smoke = agent.get("mpi_smoke")
    if not (
        version
        and agent.get("runtime") == "passed"
        and agent.get("capability_discovery") == "passed"
        and agent.get("declared_maturity_evidence") == "passed"
        and isinstance(templates, dict)
        and templates
        and all(
            isinstance(item, dict) and item.get("provenance") == "verified"
            for item in templates.values()
        )
        and isinstance(mpi_smoke, dict)
        and mpi_smoke.get("status") == "passed"
        and int(mpi_smoke.get("rank_count", 0)) >= 2
    ):
        raise RuntimeError(
            "Installed agent acceptance does not prove runtime, capability, "
            "public-template, and two-rank smoke contracts."
        )

    wheel_digest = _sha256(selected_wheel)
    if not (
        extension.get("agentfem_version") == version
        and extension.get("core_commit") == source_commit
        and extension.get("installed_wheel") is True
        and extension.get("isolated_from_source_checkout") is True
        and extension.get("core_modified") is False
        and extension.get("core_wheel_sha256") == wheel_digest
        and extension.get("simulation_result") == "passed"
        and extension.get("verification") == "passed"
        and extension.get("trust_level") in {"verified", "validated"}
    ):
        raise RuntimeError(
            "External-provider acceptance is not bound to this clean candidate "
            "wheel and source commit."
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "agentfem_version": version,
        "source_commit": source_commit,
        "source_dirty": False,
        "wheel_sha256": wheel_digest,
        "complete_serial": "passed",
        "mpi_rank_count": int(mpi_smoke["rank_count"]),
        "representative_mpi": {
            "state": "passed",
            "nonlinear": "passed",
            "output": "passed",
            "checkpoint": "passed",
        },
        "installed_wheel": "passed",
        "public_examples": "passed",
        "compatibility_imports": "passed",
        "external_provider": {
            "name": extension.get("extension"),
            "distribution": extension.get("extension_distribution"),
            "version": extension.get("extension_version"),
        },
        "evidence": {
            "agent_acceptance_sha256": _record_sha256(agent),
            "extension_acceptance_sha256": _record_sha256(extension),
            "validated_stages": tuple(sorted(declared)),
        },
        "workflow": {
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "job": os.environ.get("GITHUB_JOB"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--agent-acceptance", type=Path, required=True)
    parser.add_argument("--extension-acceptance", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--validated",
        action="append",
        choices=tuple(sorted(REQUIRED_VALIDATIONS)),
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()

    actual_commit, source_dirty = _source_identity(arguments.repository)
    if actual_commit != arguments.source_commit:
        raise RuntimeError(
            f"Requested source commit {arguments.source_commit} does not match "
            f"the checkout {actual_commit}."
        )
    record = create_foundation_acceptance(
        wheel=arguments.wheel,
        agent_acceptance_path=arguments.agent_acceptance,
        extension_acceptance_path=arguments.extension_acceptance,
        source_commit=actual_commit,
        source_dirty=source_dirty,
        validated=tuple(arguments.validated),
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(arguments.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.report)
    print(
        "AgentFEM 0.4 foundation acceptance "
        f"| {record['status'].upper()} | evidence={arguments.report}"
    )


if __name__ == "__main__":
    main()
