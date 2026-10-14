# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Check a trusted frozen project's files and declared runtime before a solve.

No simulation, installation, network access or checkpoint migration. Metadata
checks use only the standard library. --probe explicitly imports trusted code
and loads declared native libraries in a bounded child process. A passing probe
does not establish model accuracy, performance or cross-platform validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate metadata key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite metadata value: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=invalid, parse_float=finite_float)


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Project paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path escapes the project: {relative}")
    return path


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def host_identity():
    return {"system": platform.system(), "machine": platform.machine(),
            "python_cache_tag": sys.implementation.cache_tag}


_PROBE = r'''
import ctypes, importlib, json, pathlib, sys, traceback
sys.dont_write_bytecode = True
root = pathlib.Path(sys.argv[1]).resolve()
site = (root / sys.argv[2]).resolve()
sys.path[:0] = [str(site), str(root)]
request = json.loads(sys.argv[3])
stage = "imports"
try:
    modules = {}
    for name in request["imports"]:
        stage = "import:" + name
        module = importlib.import_module(name)
        modules[name] = str(getattr(module, "__version__", "loaded"))
    import agentfem
    if pathlib.Path(agentfem.__file__).resolve().parent != site / "agentfem":
        raise RuntimeError("Frozen AgentFEM was shadowed by another installation")
    libraries = []
    for relative in request["native_libraries"]:
        stage = "native:" + relative
        libraries.append(ctypes.CDLL(str(root / relative)))
    print(json.dumps({"passed": True, "imports": modules,
                      "native_libraries": len(libraries)}))
except BaseException as error:
    print(json.dumps({"passed": False, "code": "runtime_probe_failed",
                      "stage": stage, "type": type(error).__name__,
                      "message": str(error)}))
    sys.exit(2)
'''


def inspect_project(root, requirements, *, probe=False, timeout=30, host=None):
    root = Path(root).resolve()
    report = {"schema": "agentfem.frozen-project-preflight.v1",
              "project": str(root), "host": host or host_identity(),
              "payload_intact": False, "platform_compatible": False,
              "runtime_ready": False, "files_checked": 0, "issues": [],
              "probe": {"status": "not_requested"},
              "scientifically_validated": False,
              "scope": "artifact/runtime admission; no model solve or performance claim"}
    try:
        if not root.is_dir():
            raise ValueError("Project directory does not exist")
        manifest_path = root / "runtime-manifest.json"
        manifest = _json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("schema") != "agentfem.frozen-core-project.v2":
            raise ValueError("Unsupported frozen-project manifest")
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict) or not manifest_files:
            raise ValueError("Frozen manifest must list nonempty file hashes")
        site = _inside(root, manifest["installed_site"])
        if not site.is_dir():
            raise ValueError("Frozen installed site is missing")
        for relative, expected in manifest_files.items():
            path = _inside(root, relative)
            if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid SHA-256: {relative}")
            if not path.is_file():
                report["issues"].append({"code": "missing_artifact", "path": relative})
            elif _sha256(path) != expected:
                report["issues"].append({"code": "artifact_hash_mismatch", "path": relative})
            report["files_checked"] += 1
        report["manifest_sha256"] = _sha256(manifest_path)
        report["payload_intact"] = not report["issues"]
        spec = _json(requirements)
        report["requirements_sha256"] = _sha256(Path(requirements))
        if not isinstance(spec, dict) or spec.get("schema") != "agentfem.frozen-project-requirements.v1":
            raise ValueError("Unsupported runtime requirements")
        expected = spec["artifact_platform"]
        if (not isinstance(expected, dict) or set(expected) != set(report["host"])
                or not all(isinstance(v, str) and v for v in expected.values())):
            raise ValueError("Artifact platform must identify system, machine and Python ABI")
        mismatches = {k: {"expected": v, "actual": report["host"][k]}
                      for k, v in expected.items() if report["host"][k] != v}
        report["platform_compatible"] = not mismatches
        if mismatches:
            report["issues"].append({"code": "artifact_platform_mismatch", "details": mismatches,
                                    "action": "Rebuild/revalidate this artifact for the recipient; do not change the physical Model"})
        imports, native = spec["imports"], spec["native_libraries"]
        if (not isinstance(imports, list) or not imports or "agentfem" not in imports
                or not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", v) for v in imports)):
            raise ValueError("Declare agentfem and the required importable module names")
        if not isinstance(native, list):
            raise ValueError("native_libraries must be a list")
        for relative in native:
            _inside(root, relative)
            if relative not in manifest_files:
                raise ValueError(f"Native library is not pinned by the manifest: {relative}")
        pinned_core = site / "agentfem" / "__init__.py"
        if pinned_core.relative_to(root).as_posix() not in manifest_files:
            raise ValueError("AgentFEM import entry is not pinned by the manifest")
        if probe and not report["issues"]:
            if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Probe timeout must be finite and positive")
            environment = dict(os.environ, PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1",
                               OMP_NUM_THREADS="1")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", _PROBE, str(root), manifest["installed_site"],
                     json.dumps({"imports": imports, "native_libraries": native})],
                    cwd=root, env=environment, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=timeout,
                )
                lines = completed.stdout.strip().splitlines()
                try:
                    outcome = json.loads(lines[-1]) if lines else None
                except ValueError:
                    outcome = None
                if not isinstance(outcome, dict):
                    outcome = {"passed": False, "code": "invalid_probe_report"}
                report["probe"] = {"status": "completed", "returncode": completed.returncode,
                                   "outcome": outcome, "stderr": completed.stderr[-4000:]}
                report["runtime_ready"] = completed.returncode == 0 and outcome.get("passed") is True
                if not report["runtime_ready"]:
                    report["issues"].append({"code": "runtime_probe_failed"})
            except subprocess.TimeoutExpired:
                report["probe"] = {"status": "timeout", "timeout_seconds": timeout}
                report["issues"].append({"code": "runtime_probe_timeout"})
        elif probe:
            report["probe"] = {"status": "skipped_due_to_failed_admission"}
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["issues"].append({"code": "invalid_project_metadata", "message": str(error)})
    report["passed"] = not report["issues"] and (report["runtime_ready"] if probe else report["payload_intact"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="Import/load the trusted project in a bounded child; never run its model")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, help="New report file; existing evidence is never overwritten")
    args = parser.parse_args()
    report = inspect_project(args.project, args.requirements, probe=args.probe, timeout=args.timeout)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except OSError as error:
            report["passed"] = False
            report["issues"].append({"code": "report_write_failed", "message": str(error)})
    print(json.dumps(report, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
