#!/usr/bin/env python3
"""Exercise a freshly installed AgentFEM runtime and write acceptance evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def execute(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str],
    timeout: int = 900,
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--prefix",
        default=str(Path.home() / "Library" / "AgentFEMRuntime"),
    )
    result.add_argument("--output", required=True)
    result.add_argument("--profile", choices=("core", "complete"), required=True)
    result.add_argument("--skip-mpi", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    prefix = Path(args.prefix).expanduser().resolve()
    executable = prefix / "bin" / "agentfem"
    python = prefix / "bin" / "python"
    if not executable.is_file() or not python.is_file():
        raise SystemExit(f"AgentFEM runtime is incomplete at {prefix}")
    environment = os.environ.copy()
    environment["PATH"] = f"{prefix / 'bin'}:{environment.get('PATH', '')}"
    # A fresh cache makes the static model exercise the FFCx/CFFI JIT rather
    # than accepting a warm-cache import as runtime evidence.
    with tempfile.TemporaryDirectory(prefix="agentfem-runtime-acceptance-") as raw:
        root = Path(raw)
        environment["XDG_CACHE_HOME"] = str(root / "cache")
        checks: list[dict[str, Any]] = []
        checks.append(execute([str(executable), "doctor", "--json"], env=environment))
        checks.append(execute([str(executable), "--version"], env=environment))
        if args.profile == "complete":
            checks.append(
                execute(
                    [
                        str(python),
                        "-c",
                        (
                            "import gmsh; gmsh.initialize(); "
                            "print(gmsh.__version__); gmsh.finalize()"
                        ),
                    ],
                    env=environment,
                )
            )
        serial_project = root / "serial"
        checks.append(
            execute(
                [str(executable), "init", "--template", "static-solid", str(serial_project)],
                env=environment,
            )
        )
        checks.append(
            execute(
                [str(executable), "check", "--project", str(serial_project), "--json"],
                env=environment,
            )
        )
        checks.append(
            execute(
                [str(executable), "run", "--project", str(serial_project), "--json"],
                env=environment,
            )
        )
        checks.append(
            execute(
                [str(executable), "verify", "--project", str(serial_project), "--json"],
                env=environment,
            )
        )
        if not args.skip_mpi:
            mpi_project = root / "mpi"
            checks.append(
                execute(
                    [str(executable), "init", "--template", "static-solid", str(mpi_project)],
                    env=environment,
                )
            )
            checks.append(
                execute(
                    [
                        str(executable),
                        "run",
                        "--project",
                        str(mpi_project),
                        "--mpi",
                        "2",
                        "--json",
                    ],
                    env=environment,
                )
            )
            checks.append(
                execute(
                    [str(executable), "verify", "--project", str(mpi_project), "--json"],
                    env=environment,
                )
            )
    runtime_record = prefix / "share" / "agentfem" / "runtime-release.json"
    checks.append(
        {
            "command": ["runtime-record", str(runtime_record)],
            "returncode": 0 if runtime_record.is_file() else 1,
            "elapsed_seconds": 0.0,
            "stdout": (
                "runtime release identity is embedded\n"
                if runtime_record.is_file()
                else ""
            ),
            "stderr": (
                "" if runtime_record.is_file() else "runtime release identity is missing\n"
            ),
            "passed": runtime_record.is_file(),
        }
    )
    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "schema": "agentfem.runtime-acceptance",
        "schema_version": 1,
        "status": "accepted" if passed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "prefix": str(prefix),
        "profile": args.profile,
        "runtime_record": (
            {"path": str(runtime_record), "sha256": digest(runtime_record)}
            if runtime_record.is_file()
            else None
        ),
        "checks": checks,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
