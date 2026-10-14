# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Delivery preflight separates frozen artifacts from transferable Model inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/check_frozen_project.py"
_SPEC = importlib.util.spec_from_file_location("frozen_preflight", _PATH)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


@pytest.fixture
def project(tmp_path):
    site = tmp_path / "runtime/core/agentfem"
    site.mkdir(parents=True)
    entry = site / "__init__.py"
    entry.write_text("# harmless test package\n")
    manifest = {"schema": "agentfem.frozen-core-project.v2", "installed_site": "runtime/core",
                "files": {entry.relative_to(tmp_path).as_posix(): hashlib.sha256(entry.read_bytes()).hexdigest()}}
    (tmp_path / "runtime-manifest.json").write_text(json.dumps(manifest))
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps({"schema": "agentfem.frozen-project-requirements.v1",
                                       "artifact_platform": checker.host_identity(),
                                       "imports": ["agentfem"], "native_libraries": []}))
    return tmp_path, requirements


def change_manifest(root, action):
    path = root / "runtime-manifest.json"
    data = json.loads(path.read_text())
    action(data)
    path.write_text(json.dumps(data))


def test_metadata_check_is_not_runtime_or_physics_acceptance(project):
    report = checker.inspect_project(*project)
    assert report["passed"] and report["payload_intact"]
    assert not report["runtime_ready"] and not report["scientifically_validated"]
    assert report["files_checked"] == 1
    assert report["probe"]["status"] == "not_requested"


def test_probe_uses_pinned_installation_and_never_case(project):
    (project[0] / "case.py").write_text("raise RuntimeError('Never execute the case during preflight')\n")
    report = checker.inspect_project(*project, probe=True)
    assert report["passed"] and report["runtime_ready"]
    assert not report["scientifically_validated"]


@pytest.mark.parametrize("problem", ["missing", "corrupt", "escape", "untracked_native", "untracked_core", "duplicate", "bad_site"])
def test_rejects_bad_artifacts_before_loading(project, monkeypatch, problem):
    root, requirements = project
    if problem == "missing":
        (root / "runtime/core/agentfem/__init__.py").unlink()
    elif problem == "corrupt":
        (root / "runtime/core/agentfem/__init__.py").write_text("tampered")
    elif problem == "escape":
        change_manifest(root, lambda m: m["files"].update({"../escape": "0" * 64}))
    elif problem == "untracked_core":
        change_manifest(root, lambda m: m["files"].clear())
    elif problem == "duplicate":
        (root / "runtime-manifest.json").write_text('{"schema": 1, "schema": 2}')
    elif problem == "bad_site":
        change_manifest(root, lambda m: m.update(installed_site="../elsewhere"))
    else:
        data = json.loads(requirements.read_text())
        data["native_libraries"] = ["untracked.so"]
        requirements.write_text(json.dumps(data))
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: pytest.fail("Failed admission must not execute code"))
    report = checker.inspect_project(*project, probe=True)
    assert not report["passed"] and not report["runtime_ready"] and report["issues"]


def test_platform_mismatch_is_an_artifact_boundary_not_model_failure(project, monkeypatch):
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: pytest.fail("Incompatible artifact must not be loaded"))
    host = dict(checker.host_identity(), machine="different-architecture")
    report = checker.inspect_project(*project, host=host, probe=True)
    assert report["payload_intact"] and not report["platform_compatible"]
    assert report["issues"][0]["code"] == "artifact_platform_mismatch"


def test_probe_timeout_is_addressable(project, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(checker.subprocess, "run", timeout)
    report = checker.inspect_project(*project, probe=True)
    assert not report["passed"]
    assert report["issues"] == [{"code": "runtime_probe_timeout"}]


def test_native_dependency_failure_does_not_become_ready(project, monkeypatch):
    outcome = {"passed": False, "stage": "native:provider.so", "message": "Missing libgfortran"}
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=2, stdout=json.dumps(outcome), stderr=""))
    report = checker.inspect_project(*project, probe=True)
    assert not report["passed"] and not report["runtime_ready"]
    assert report["probe"]["outcome"] == outcome


def test_missing_project_is_structured(tmp_path):
    report = checker.inspect_project(tmp_path / "missing", tmp_path / "missing.json")
    assert not report["passed"] and report["issues"][0]["code"] == "invalid_project_metadata"


@pytest.mark.parametrize("name", ["runtime-manifest.json", "requirements.json"])
@pytest.mark.parametrize("payload", ["[]", "null", "42", '{"x": 1e999}', '{"x": NaN}'])
def test_metadata_requires_an_object(project, name, payload):
    (project[0] / name).write_text(payload)
    report = checker.inspect_project(*project)
    assert not report["passed"]
    assert report["issues"][0]["code"] == "invalid_project_metadata"


@pytest.mark.parametrize("payload", ["", "invalid", "[]", "null"])
def test_malformed_probe_is_not_ready(project, monkeypatch, payload):
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=payload, stderr=""))
    report = checker.inspect_project(*project, probe=True)
    assert not report["passed"] and not report["runtime_ready"]
    assert report["issues"] == [{"code": "runtime_probe_failed"}]
    assert report["probe"]["outcome"]["code"] == "invalid_probe_report"


@pytest.mark.parametrize("timeout", [0, -1, float('inf'), float('nan'), True])
def test_probe_requires_a_real_finite_deadline(project, monkeypatch, timeout):
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: pytest.fail("Invalid deadline"))
    report = checker.inspect_project(*project, probe=True, timeout=timeout)
    assert not report["passed"] and not report["runtime_ready"]


def test_slow_host_can_choose_a_longer_probe_deadline(project, monkeypatch):
    def run(*args, **kwargs):
        assert kwargs['timeout'] == 450
        return SimpleNamespace(returncode=0, stdout='{"passed": true}', stderr='')
    monkeypatch.setattr(checker.subprocess, "run", run)
    assert checker.inspect_project(*project, probe=True, timeout=450)['runtime_ready']


def test_symlink_escape_is_refused_without_loading(project, monkeypatch):
    root, _ = project
    outside = root.parent / (root.name + '-outside')
    outside.write_text('not part of the frozen project')
    (root / 'escaping-link').symlink_to(outside)
    change_manifest(root, lambda m: m['files'].update({'escaping-link': '0' * 64}))
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: pytest.fail("Escaping input"))
    report = checker.inspect_project(*project, probe=True)
    assert not report['passed'] and not report['runtime_ready']


def test_existing_evidence_is_not_overwritten(project, monkeypatch, capsys):
    output = project[0] / 'existing-review.json'
    output.write_text('keep original evidence')
    monkeypatch.setattr(checker.sys, 'argv', ['check_project', str(project[0]),
        '--requirements', str(project[1]), '--output', str(output)])
    assert checker.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report['issues'][-1]['code'] == 'report_write_failed'
    assert output.read_text() == 'keep original evidence'
