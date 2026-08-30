from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from tools import promote_agent_trial


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wheel(path: Path, version: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"agentfem-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: agentfem\nVersion: {version}\n",
        )


def _acceptance(root: Path, commit: str, wheel: Path) -> Path:
    transcript = root / "transcript.md"
    explanation = root / "explanation.md"
    transcript.write_text("fresh agent transcript\n", encoding="utf-8")
    explanation.write_text("reviewed mechanics\n", encoding="utf-8")
    record = {
        "schema": "agentfem.agent-trial-acceptance",
        "status": "passed",
        "agent": "Codex",
        "agentfem_version": "0.2.6",
        "source_commit": commit,
        "installed_wheel": True,
        "fresh_context": True,
        "human_interventions": 0,
        "runtime": "passed",
        "capability_discovery": "passed",
        "project_check": "passed",
        "simulation_result": "passed",
        "verification": "passed",
        "scientific_explanation": "reviewed",
        "candidate_identity_verified": True,
        "wheel": str(wheel),
        "wheel_sha256": _sha256(wheel),
        "transcript": str(transcript),
        "transcript_sha256": _sha256(transcript),
        "explanation": str(explanation),
        "explanation_sha256": _sha256(explanation),
    }
    path = root / "acceptance.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_release_only_changes_can_promote_a_real_fresh_agent_trial(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    (repository / "src/agentfem/release").mkdir(parents=True)
    (repository / "skills/agentfem").mkdir(parents=True)
    (repository / "docs").mkdir()
    (repository / "src/agentfem/__init__.py").write_text(
        '__version__ = "0.2.6"\n', encoding="utf-8"
    )
    (repository / "src/agentfem/models.py").write_text("ANSWER = 42\n", encoding="utf-8")
    (repository / "skills/agentfem/SKILL.md").write_text("stable skill\n", encoding="utf-8")
    (repository / "AGENT_GUIDE.md").write_text("stable guide\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "agentfem"\nversion = "0.2.6"\n', encoding="utf-8"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "candidate")
    source_commit = _git(repository, "rev-parse", "HEAD")

    source_wheel = tmp_path / "agentfem-0.2.6.whl"
    target_wheel = tmp_path / "agentfem-0.3.0.whl"
    _write_wheel(source_wheel, "0.2.6")
    _write_wheel(target_wheel, "0.3.0")
    acceptance = _acceptance(tmp_path, source_commit, source_wheel)

    (repository / "src/agentfem/__init__.py").write_text(
        '__version__ = "0.3.0"\n', encoding="utf-8"
    )
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "agentfem"\nversion = "0.3.0"\n', encoding="utf-8"
    )
    (repository / "docs/release.md").write_text("0.3.0\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "release")

    report = promote_agent_trial.evaluate(
        repository=repository,
        source_acceptance=acceptance,
        target_wheel=target_wheel,
    )

    assert report["status"] == "passed"
    assert report["behavior_equivalent"] is True
    assert report["protected_path_count"] == 3


def test_runtime_change_blocks_agent_trial_promotion(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    (repository / "src/agentfem").mkdir(parents=True)
    (repository / "skills/agentfem").mkdir(parents=True)
    (repository / "src/agentfem/__init__.py").write_text(
        '__version__ = "0.2.6"\n', encoding="utf-8"
    )
    (repository / "src/agentfem/models.py").write_text("ANSWER = 42\n", encoding="utf-8")
    (repository / "skills/agentfem/SKILL.md").write_text("stable skill\n", encoding="utf-8")
    (repository / "AGENT_GUIDE.md").write_text("stable guide\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "agentfem"\nversion = "0.2.6"\n', encoding="utf-8"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "candidate")
    source_commit = _git(repository, "rev-parse", "HEAD")
    source_wheel = tmp_path / "agentfem-0.2.6.whl"
    target_wheel = tmp_path / "agentfem-0.3.0.whl"
    _write_wheel(source_wheel, "0.2.6")
    _write_wheel(target_wheel, "0.3.0")
    acceptance = _acceptance(tmp_path, source_commit, source_wheel)

    (repository / "src/agentfem/models.py").write_text("ANSWER = 43\n", encoding="utf-8")
    (repository / "src/agentfem/__init__.py").write_text(
        '__version__ = "0.3.0"\n', encoding="utf-8"
    )
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "agentfem"\nversion = "0.3.0"\n', encoding="utf-8"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "runtime change")

    report = promote_agent_trial.evaluate(
        repository=repository,
        source_acceptance=acceptance,
        target_wheel=target_wheel,
    )

    assert report["status"] == "failed"
    assert any("behavior-affecting" in gap for gap in report["gaps"])


def test_only_named_cross_platform_regression_repairs_are_nonruntime_changes():
    allowed = {
        "tests/periodic_void_fixture.py",
        "tests/test_periodic_void_realization.py",
    }

    assert all(promote_agent_trial._path_allowed(path) for path in allowed)
    assert not promote_agent_trial._path_allowed("tests/test_models.py")


def test_environment_bridge_only_ignores_python_gmsh_binding():
    source = b"dependencies:\n  - python=3.11\n  - gmsh\n"
    binding_added = (
        b"dependencies:\n  - python=3.11\n  - gmsh\n  - python-gmsh\n"
    )
    runtime_changed = b"dependencies:\n  - python=3.12\n  - gmsh\n"

    assert promote_agent_trial._path_allowed("environment.yml")
    assert promote_agent_trial._normalized_environment(source) == (
        promote_agent_trial._normalized_environment(binding_added)
    )
    assert promote_agent_trial._normalized_environment(source) != (
        promote_agent_trial._normalized_environment(runtime_changed)
    )
