# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Installed readiness must not silently become a simulation campaign."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "packaging/runtime/verify_installed_runtime.py"
spec = importlib.util.spec_from_file_location("runtime_readiness", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.mark.parametrize("readiness_only", [True, False])
@pytest.mark.parametrize("health", ['{"solver_ready":true}', '{"solver_ready":false}', '{}', '[]', 'invalid'])
def test_readiness_and_explicit_developer_solve_have_distinct_evidence(tmp_path, monkeypatch, readiness_only, health):
    prefix = tmp_path / "runtime"
    (prefix / "bin").mkdir(parents=True)
    for name in ("agentfem", "python"):
        (prefix / "bin" / name).touch()
    record = prefix / "share/agentfem/runtime-release.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"version":"test"}')
    output = tmp_path / "report.json"
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return dict(command=command, passed=True, returncode=0,
                    stdout=health if "doctor" in command else "", stderr="")
    monkeypatch.setattr(runtime, "execute", execute)
    monkeypatch.setattr(runtime.shutil, "which", lambda name, **kw: str(prefix / "bin" / name))
    args = [str(SCRIPT), "--prefix", str(prefix), "--output", str(output), "--profile", "core"]
    if readiness_only:
        args.append("--readiness-only")
    monkeypatch.setattr(sys, "argv", args)
    valid = health == '{"solver_ready":true}'
    assert runtime.main() == (0 if valid else 1)
    report = json.loads(output.read_text())
    assert report["scope"] == ("installation_readiness" if readiness_only else "first_solve")
    assert report["toolchain"]["fresh_jit_exercised"] is (not readiness_only and valid)
    solves = [cmd for cmd in calls if len(cmd) > 1 and cmd[1] == "run"]
    assert len(solves) == (0 if readiness_only else 2)
    if readiness_only:
        assert not any(set(cmd) & {"init", "check", "run", "verify", "--mpi"} for cmd in calls)
