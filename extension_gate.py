# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed-wheel acceptance for an external AgentFEM extension.

The gate deliberately runs outside the source checkout.  It installs the core
and extension wheels into a temporary environment, discovers the standard
entry point without importing it, executes an ordinary AgentFEM project, and
checks that the result and its verification evidence retain the activated
extension identity.  A hash of the installed core package before and after the
run proves that the extension did not patch AgentFEM in place.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _run_json(command: list[str], *, cwd: Path, environment: dict[str, str]):
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Command did not return JSON: {command!r}\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        ) from exc


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def accept_extension(
    *,
    core_wheel: Path,
    extension_wheel: Path,
    project: Path,
    extension: str,
    core_commit: str | None = None,
) -> dict[str, object]:
    """Run one external extension through the installed product lifecycle."""

    core_wheel = core_wheel.expanduser().resolve()
    extension_wheel = extension_wheel.expanduser().resolve()
    project = project.expanduser().resolve()
    for artifact in (core_wheel, extension_wheel):
        if not artifact.is_file() or artifact.suffix != ".whl":
            raise FileNotFoundError(f"Wheel not found: {artifact}")
    if not (project / "agentfem.toml").is_file():
        raise FileNotFoundError(f"AgentFEM project not found: {project}")

    with tempfile.TemporaryDirectory(prefix="agentfem-extension-gate-") as raw:
        work = Path(raw)
        installation = work / "site-packages"
        python = Path(sys.executable).resolve()
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(installation),
                str(core_wheel),
                str(extension_wheel),
            ],
            cwd=work,
            check=True,
        )
        isolated_project = work / "project"
        shutil.copytree(project, isolated_project)
        environment = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(installation),
        }

        identity = _run_json(
            [
                str(python),
                "-c",
                (
                    "import json, pathlib, agentfem; "
                    "print(json.dumps({'version': agentfem.__version__, "
                    "'package': str(pathlib.Path(agentfem.__file__).resolve().parent)}))"
                ),
            ],
            cwd=work,
            environment=environment,
        )
        package = Path(identity["package"])
        if package.is_relative_to(ROOT):
            raise RuntimeError(
                "Extension acceptance imported AgentFEM from the source checkout."
            )
        core_before = _tree_sha256(package)

        inventory = _run_json(
            [str(python), "-m", "agentfem.cli", "extensions", "--json"],
            cwd=work,
            environment=environment,
        )
        descriptors = [
            item for item in inventory["installed"] if item["name"] == extension
        ]
        if len(descriptors) != 1 or descriptors[0]["loaded"]:
            raise RuntimeError(
                "The external extension must be discovered exactly once and remain "
                "unloaded before project execution."
            )

        check = _run_json(
            [
                str(python),
                "-m",
                "agentfem.cli",
                "check",
                "--project",
                str(isolated_project),
                "--json",
            ],
            cwd=work,
            environment=environment,
        )
        if check.get("status") != "passed":
            raise RuntimeError(f"External project preflight failed: {check!r}")
        run = _run_json(
            [
                str(python),
                "-m",
                "agentfem.cli",
                "run",
                "--project",
                str(isolated_project),
                "--run-id",
                "gate-6-external-provider",
                "--json",
            ],
            cwd=work,
            environment=environment,
        )
        if run.get("status") != "completed":
            raise RuntimeError(f"External project did not complete: {run!r}")
        execution_path = Path(run["execution_record"])
        result_path = Path(run["result_manifest"])
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        verification = _run_json(
            [
                str(python),
                "-m",
                "agentfem.cli",
                "verify",
                str(result_path),
                "--json",
            ],
            cwd=work,
            environment=environment,
        )

        activated = [
            item for item in execution.get("extensions", ())
            if item.get("name") == extension
        ]
        result_extensions = (
            result.get("metadata", {}).get("run", {}).get("extensions", ())
        )
        retained = [
            item for item in result_extensions if item.get("name") == extension
        ]
        if len(activated) != 1 or len(retained) != 1:
            raise RuntimeError(
                "Activated extension identity was not retained by execution and result."
            )
        if result.get("trust_level") not in {"verified", "validated"}:
            raise RuntimeError(
                f"External project trust level is {result.get('trust_level')!r}."
            )
        if verification.get("status") != "verified":
            raise RuntimeError(f"Result provenance verification failed: {verification!r}")

        core_after = _tree_sha256(package)
        if core_before != core_after:
            raise RuntimeError("External execution modified the installed AgentFEM core.")
        descriptor = descriptors[0]
        loaded = activated[0]
        return {
            "schema": "agentfem.extension-acceptance",
            "schema_version": "0.2.0",
            "status": "passed",
            "extension": extension,
            "extension_distribution": descriptor.get("distribution"),
            "extension_version": descriptor.get("distribution_version"),
            "entry_point": descriptor.get("entry_point"),
            "entry_point_discovered": True,
            "entry_point_activated": True,
            "registered_capabilities": loaded.get("registrations", {}),
            "installed_wheel": True,
            "isolated_from_source_checkout": True,
            "core_modified": False,
            "core_installation_sha256_before": core_before,
            "core_installation_sha256_after": core_after,
            "simulation_result": "passed",
            "verification": "passed",
            "trust_level": result["trust_level"],
            "agentfem_version": identity["version"],
            "core_commit": core_commit or _git_commit(),
            "core_wheel_sha256": _sha256(core_wheel),
            "extension_wheel_sha256": _sha256(extension_wheel),
            "result_manifest_sha256": _sha256(result_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--extension-wheel", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--core-commit")
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args()
    record = accept_extension(
        core_wheel=options.core_wheel,
        extension_wheel=options.extension_wheel,
        project=options.project,
        extension=options.extension,
        core_commit=options.core_commit,
    )
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
