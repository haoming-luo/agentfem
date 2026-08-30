"""Promote a fresh-agent trial across a behavior-equivalent release boundary.

This tool exists for the narrow case in which a deliberately tested release
candidate is promoted by changing only version, documentation and release
evidence.  It fails closed when the public agent guidance, templates, Skill,
or executable AgentFEM runtime changed.  It never rewrites the original trial.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib
import zipfile

import promotion_gate


_ALLOWED_EXACT = {
    "CHANGELOG.md",
    "CITATION.cff",
    "README.md",
    "mkdocs.yml",
    "pyproject.toml",
    "promotion_gate.py",
    "src/agentfem/__init__.py",
    "tests/test_agent_trial_promotion.py",
    "tests/test_promotion_gate.py",
    "tools/promote_agent_trial.py",
}
_ALLOWED_PREFIXES = ("docs/", "site/", "src/agentfem/release/")
_PROTECTED_PREFIXES = ("src/agentfem/", "skills/agentfem/")
_PROTECTED_EXACT = {"AGENT_GUIDE.md"}
_PROTECTED_EXCLUDED = {
    "src/agentfem/__init__.py",
}
_PROTECTED_EXCLUDED_PREFIXES = ("src/agentfem/release/",)


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata = [
            name for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise ValueError(f"Expected one wheel METADATA file, found {len(metadata)}.")
        content = archive.read(metadata[0]).decode("utf-8")
    for line in content.splitlines():
        if line.startswith("Version:"):
            return line.partition(":")[2].strip()
    raise ValueError("Wheel METADATA has no Version field.")


def _git_file(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _protected_runtime_tree(root: Path, commit: str) -> tuple[str, tuple[str, ...]]:
    listing = _run_git(root, "ls-tree", "-r", commit)
    entries = []
    paths = []
    for line in listing.splitlines():
        metadata, path = line.split("\t", 1)
        protected = path in _PROTECTED_EXACT or path.startswith(_PROTECTED_PREFIXES)
        excluded = path in _PROTECTED_EXCLUDED or path.startswith(
            _PROTECTED_EXCLUDED_PREFIXES
        )
        if protected and not excluded:
            entries.append(f"{metadata}\t{path}")
            paths.append(path)
    digest = hashlib.sha256(("\n".join(entries) + "\n").encode("utf-8")).hexdigest()
    return digest, tuple(paths)


def _normalized_pyproject(content: bytes) -> dict[str, object]:
    record = copy.deepcopy(tomllib.loads(content.decode("utf-8")))
    project = record.get("project")
    if isinstance(project, dict):
        project["version"] = "<release-version>"
    return record


def _normalized_init(content: bytes) -> str:
    return re.sub(
        r'(?m)^__version__\s*=\s*["\'][^"\']+["\']$',
        '__version__ = "<release-version>"',
        content.decode("utf-8"),
    )


def _path_allowed(path: str) -> bool:
    return path in _ALLOWED_EXACT or path.startswith(_ALLOWED_PREFIXES)


def _file_integrity(record: dict[str, object], field: str, hash_field: str) -> bool:
    value = record.get(field)
    expected = record.get(hash_field)
    if not value or not promotion_gate._is_sha256(expected):
        return False
    path = Path(str(value)).expanduser()
    return path.is_file() and _sha256(path) == expected


def evaluate(
    *,
    repository: Path,
    source_acceptance: Path,
    target_wheel: Path,
    target_commit: str | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    acceptance_path = Path(source_acceptance).resolve()
    wheel_path = Path(target_wheel).resolve()
    source = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Fresh-agent acceptance must be a JSON object.")
    source_commit = str(source.get("source_commit") or "")
    target = target_commit or _run_git(root, "rev-parse", "HEAD").strip()
    target_version = _wheel_version(wheel_path)
    changed = tuple(
        line for line in _run_git(
            root, "diff", "--name-only", f"{source_commit}..{target}"
        ).splitlines() if line
    )
    source_tree, source_paths = _protected_runtime_tree(root, source_commit)
    target_tree, target_paths = _protected_runtime_tree(root, target)
    gaps = []

    if not promotion_gate._valid_agent_trial(source):
        gaps.append("source record is not a complete fresh-agent acceptance")
    if not _file_integrity(source, "transcript", "transcript_sha256"):
        gaps.append("source transcript is missing or has changed")
    if not _file_integrity(source, "explanation", "explanation_sha256"):
        gaps.append("source explanation is missing or has changed")
    source_wheel = Path(str(source.get("wheel", ""))).expanduser()
    if not source_wheel.is_file() or _sha256(source_wheel) != source.get("wheel_sha256"):
        gaps.append("source wheel is missing or has changed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, target],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        gaps.append("source trial commit is not an ancestor of the target commit")
    disallowed = tuple(path for path in changed if not _path_allowed(path))
    if disallowed:
        gaps.append(f"behavior-affecting paths changed: {disallowed}")
    if source_tree != target_tree or source_paths != target_paths:
        gaps.append("protected runtime, agent guidance, templates, or Skill changed")
    if _normalized_pyproject(_git_file(root, source_commit, "pyproject.toml")) != (
        _normalized_pyproject(_git_file(root, target, "pyproject.toml"))
    ):
        gaps.append("pyproject.toml changed beyond the release version")
    if _normalized_init(_git_file(root, source_commit, "src/agentfem/__init__.py")) != (
        _normalized_init(_git_file(root, target, "src/agentfem/__init__.py"))
    ):
        gaps.append("agentfem.__init__ changed beyond the release version")
    target_pyproject = tomllib.loads(
        _git_file(root, target, "pyproject.toml").decode("utf-8")
    )
    if target_pyproject.get("project", {}).get("version") != target_version:
        gaps.append("target wheel version does not match target pyproject.toml")

    return {
        "schema": "agentfem.agent-trial-promotion",
        "schema_version": "0.1.0",
        "status": "passed" if not gaps else "failed",
        "allowed_changes_only": not disallowed,
        "behavior_equivalent": not gaps,
        "source_acceptance": str(acceptance_path),
        "source_acceptance_sha256": promotion_gate._record_sha256(source),
        "source_agentfem_version": source.get("agentfem_version"),
        "source_commit": source_commit,
        "source_wheel_sha256": source.get("wheel_sha256"),
        "target_agentfem_version": target_version,
        "target_commit": target,
        "target_wheel": str(wheel_path),
        "target_wheel_sha256": _sha256(wheel_path),
        "changed_paths": changed,
        "protected_path_count": len(target_paths),
        "protected_runtime_tree_source": source_tree,
        "protected_runtime_tree_target": target_tree,
        "gaps": gaps,
    }


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--source-acceptance", type=Path, required=True)
    parser.add_argument("--target-wheel", type=Path, required=True)
    parser.add_argument("--target-commit")
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args()
    report = evaluate(
        repository=options.repository,
        source_acceptance=options.source_acceptance,
        target_wheel=options.target_wheel,
        target_commit=options.target_commit,
    )
    _write(options.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
