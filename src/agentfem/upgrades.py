"""Version-aware project inspection for humans and AI agents.

The upgrade layer deliberately separates deterministic metadata migrations
from changes that may alter finite-element meaning.  It can identify legacy
workflow patterns and explain the preferred public API, but it never rewrites
loads, constraints, materials, weak forms, or solver choices automatically.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from . import __version__
from ._api_contract import COMPATIBILITY_MODEL_REPLACEMENTS
from .project import CURRENT_PROJECT_SCHEMA_VERSION, PROJECT_FILENAME, ProjectConfig


UPGRADE_SCHEMA = "agentfem.upgrade-report"
COHESIVE_CHECKPOINT_SCHEMA = "agentfem.dof-mapped-cohesive-force.v5"


@dataclass(frozen=True)
class UpgradeFinding:
    """One stable, addressable compatibility or migration finding."""

    code: str
    severity: str
    message: str
    path: Path
    line: int | None = None
    column: int | None = None
    replacement: str | None = None
    automatic: bool = False
    semantic_review: bool = False

    def as_dict(self, *, root: Path | None = None) -> dict[str, object]:
        selected_path = self.path
        if root is not None:
            try:
                selected_path = selected_path.relative_to(root)
            except ValueError:
                pass
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": str(selected_path),
            "line": self.line,
            "column": self.column,
            "replacement": self.replacement,
            "automatic": self.automatic,
            "semantic_review": self.semantic_review,
        }


@dataclass(frozen=True)
class UpgradeReport:
    """Dry-run migration plan for one installed-use project."""

    project: ProjectConfig
    findings: tuple[UpgradeFinding, ...]
    runtime_version: str = __version__

    @property
    def required(self) -> tuple[UpgradeFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "required")

    @property
    def errors(self) -> tuple[UpgradeFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def status(self) -> str:
        if self.errors:
            return "blocked"
        if self.required:
            return "migration_required"
        if self.findings:
            return "review_recommended"
        return "current"

    def summary(self) -> dict[str, object]:
        counts = {
            severity: sum(item.severity == severity for item in self.findings)
            for severity in ("error", "required", "advisory")
        }
        return {
            "schema": UPGRADE_SCHEMA,
            "schema_version": "0.1.0",
            "status": self.status,
            "agentfem_version": self.runtime_version,
            "project_schema": self.project.schema_version,
            "supported_project_schema": CURRENT_PROJECT_SCHEMA_VERSION,
            "project": self.project.summary(),
            "counts": counts,
            "findings": tuple(
                item.as_dict(root=self.project.root) for item in self.findings
            ),
            "policy": {
                "automatic_scope": "deterministic project metadata only",
                "semantic_scope": (
                    "physics, regions, loads, constraints, weak forms, materials, "
                    "and solver choices always require review"
                ),
            },
        }

    def format(self) -> str:
        lines = [
            f"AgentFEM project upgrade: {self.project.name}",
            f"  status: {self.status}",
            f"  runtime: {self.runtime_version}",
            (
                "  project schema: "
                f"{self.project.schema_version} -> {CURRENT_PROJECT_SCHEMA_VERSION}"
            ),
        ]
        if not self.findings:
            lines.append("  No migration findings.")
            return "\n".join(lines)
        for item in self.findings:
            location = str(item.path.relative_to(self.project.root))
            if item.line is not None:
                location += f":{item.line}"
            marker = "automatic" if item.automatic else "review"
            lines.append(
                f"  [{item.severity}] {item.code} {location} ({marker})\n"
                f"    {item.message}"
            )
            if item.replacement:
                lines.append(f"    preferred: {item.replacement}")
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def inspect_project(project: ProjectConfig) -> UpgradeReport:
    """Return a dry-run upgrade report without executing or changing the case."""

    findings = list(_metadata_findings(project))
    for path in _project_python_files(project):
        findings.extend(_python_findings(path))
    findings.sort(
        key=lambda item: (
            str(item.path),
            -1 if item.line is None else item.line,
            item.code,
        )
    )
    return UpgradeReport(project=project, findings=tuple(findings))


def _project_python_files(project: ProjectConfig) -> tuple[Path, ...]:
    excluded = {
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "outputs",
        "__pycache__",
    }
    paths = []
    for path in project.root.rglob("*.py"):
        relative = path.relative_to(project.root)
        if any(part in excluded or part.startswith(".") for part in relative.parts):
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(project.root)
        except ValueError:
            continue
        try:
            resolved.relative_to(project.output_directory)
        except ValueError:
            pass
        else:
            continue
        paths.append(resolved)
    if project.entrypoint.is_file() and project.entrypoint not in paths:
        paths.append(project.entrypoint)
    return tuple(sorted(set(paths)))


def apply_safe_metadata(project: ProjectConfig) -> tuple[Path, ...]:
    """Apply only deterministic project-metadata migrations, atomically.

    A timestamp-free adjacent ``.bak`` file preserves the original on the
    first migration. Existing backups are never overwritten.
    """

    config_path = project.root / PROJECT_FILENAME
    text = config_path.read_text(encoding="utf-8")
    updated = text
    schema_match = re.search(
        r'(?m)^(\s*schema_version\s*=\s*)["\']([^"\']+)["\']\s*$',
        updated,
    )
    if schema_match is None:
        updated = _insert_project_key(
            updated,
            "schema_version",
            CURRENT_PROJECT_SCHEMA_VERSION,
        )
    elif _compare_versions(
        schema_match.group(2), CURRENT_PROJECT_SCHEMA_VERSION
    ) == -1:
        updated = (
            updated[: schema_match.start()]
            + f'{schema_match.group(1)}"{CURRENT_PROJECT_SCHEMA_VERSION}"'
            + updated[schema_match.end() :]
        )
    if updated == text:
        return ()
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(config_path)
    return (config_path, backup)


def migrate_cohesive_checkpoint(
    snapshot: dict[str, object],
    *,
    tangential: str,
    tangential_stiffness: float | None = None,
    acknowledge_physics_change: bool = False,
) -> dict[str, object]:
    """Explicitly promote a physical-keyed scalar checkpoint to schema v5.

    Version 3 introduced physical facet identity and version 4 generalized the
    state field table, but neither recorded tangential interface behavior.
    Promotion to ``tie`` or ``degraded`` therefore requires an explicit
    acknowledgement. Mixed-mode state cannot be inferred from scalar opening
    history and is deliberately rejected.
    """

    if not isinstance(snapshot, dict):
        raise TypeError("A cohesive checkpoint migration requires a mapping.")
    source_schema = snapshot.get("schema")
    supported = {
        "agentfem.dof-mapped-cohesive-force.v3",
        "agentfem.dof-mapped-cohesive-force.v4",
    }
    if source_schema not in supported:
        raise ValueError(
            "Cohesive checkpoint migration supports physical-keyed schemas "
            "v3 and v4. Restore older state into a free-slip consumer and "
            "write a current checkpoint first."
        )
    mode = str(tangential).strip().lower().replace("-", "_")
    mode = {"none": "free", "normal_only": "free", "cohesive": "degraded"}.get(
        mode, mode
    )
    if mode not in {"free", "tie", "degraded"}:
        raise ValueError(
            "Scalar checkpoint migration supports free, tie, or degraded; "
            "mixed-mode initiation history cannot be reconstructed."
        )
    if mode != "free" and not acknowledge_physics_change:
        raise ValueError(
            "Changing a legacy free-slip checkpoint to a shear-carrying "
            "interface requires acknowledge_physics_change=True."
        )
    law = snapshot.get("law")
    if not isinstance(law, dict) or law.get("mode") != "normal":
        raise ValueError("Only scalar Mode-I cohesive checkpoints can migrate.")
    if mode == "free":
        stiffness = 0.0
        if tangential_stiffness is not None and float(tangential_stiffness) != 0.0:
            raise ValueError("Free-slip migration cannot add tangential stiffness.")
    else:
        stiffness = (
            float(law.get("initial_stiffness"))
            if tangential_stiffness is None
            else float(tangential_stiffness)
        )
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("Migrated tangential stiffness must be positive.")

    migrated = deepcopy(snapshot)
    if source_schema.endswith(".v3"):
        records = migrated.get("maximum_opening_by_key")
        if not isinstance(records, dict) or not records:
            raise ValueError("Version 3 cohesive checkpoint lacks keyed state.")
        migrated["state_by_field_and_key"] = {"maximum_opening": records}
    fields = migrated.get("state_by_field_and_key")
    if not isinstance(fields, dict) or set(fields) != {"maximum_opening"}:
        raise ValueError("Legacy scalar checkpoint state fields are incompatible.")
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    migrated.update(
        {
            "schema": COHESIVE_CHECKPOINT_SCHEMA,
            "interface_kinematics": mode,
            "tangential_stiffness": stiffness,
            "state_identity": "ordered_physical_facet_and_quadrature",
            "migration": {
                "schema": "agentfem.cohesive-checkpoint-migration.v1",
                "source_schema": source_schema,
                "source_sha256": sha256(encoded).hexdigest(),
                "target_schema": COHESIVE_CHECKPOINT_SCHEMA,
                "declared_legacy_kinematics": "free",
                "target_kinematics": mode,
                "physics_change_acknowledged": bool(acknowledge_physics_change),
            },
        }
    )
    return migrated


def _metadata_findings(project: ProjectConfig) -> Iterable[UpgradeFinding]:
    path = project.root / PROJECT_FILENAME
    text = path.read_text(encoding="utf-8")
    comparison = _compare_versions(
        project.schema_version,
        CURRENT_PROJECT_SCHEMA_VERSION,
    )
    if comparison is None:
        yield UpgradeFinding(
            "AFM-UPG-001",
            "error",
            f"Project schema {project.schema_version!r} is not a valid numeric version.",
            path,
        )
    elif comparison > 0:
        yield UpgradeFinding(
            "AFM-UPG-002",
            "error",
            (
                f"Project schema {project.schema_version} is newer than this AgentFEM "
                f"runtime supports ({CURRENT_PROJECT_SCHEMA_VERSION})."
            ),
            path,
            replacement="upgrade the AgentFEM runtime before running this project",
        )
    elif comparison < 0:
        yield UpgradeFinding(
            "AFM-UPG-003",
            "required",
            "The operational project schema requires a deterministic migration.",
            path,
            replacement=f'schema_version = "{CURRENT_PROJECT_SCHEMA_VERSION}"',
            automatic=True,
        )
    elif not re.search(r"(?m)^\s*schema_version\s*=", text):
        yield UpgradeFinding(
            "AFM-UPG-004",
            "advisory",
            (
                "The schema version is implicit. Recording it makes future upgrades "
                "deterministic for terminals, GUIs, and agents."
            ),
            path,
            replacement=f'schema_version = "{CURRENT_PROJECT_SCHEMA_VERSION}"',
            automatic=True,
        )


def _python_findings(path: Path) -> Iterable[UpgradeFinding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return ()
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name.endswith("io.XDMFTimeSeries") or name == "XDMFTimeSeries":
            findings.append(
                UpgradeFinding(
                    "AFM-UPG-101",
                    "advisory",
                    (
                        "Low-level XDMF writing remains supported, but ordinary static and "
                        "transient cases should use the Step result lifecycle so fields, "
                        "artifacts, and evidence stay in one contract."
                    ),
                    path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                    replacement='step.solve_result(output=run.artifact("fields.xdmf"))',
                    semantic_review=True,
                )
            )
        elif name.endswith("mesh.BoundaryRegion") or name == "BoundaryRegion":
            findings.append(
                UpgradeFinding(
                    "AFM-UPG-102",
                    "advisory",
                    (
                        "Direct BoundaryRegion construction can make imported tag identity "
                        "ambiguous. Prefer a named geometry or tagged-region factory."
                    ),
                    path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                    replacement=(
                        "mesh.boundary(...) for geometric selection, or "
                        "mesh.tagged_boundary_region(...) for imported physical tags"
                    ),
                    semantic_review=True,
                )
            )
        elif name.endswith("results.integral") and _uses_region_measure(node):
            findings.append(
                UpgradeFinding(
                    "AFM-UPG-103",
                    "advisory",
                    "A standard named-region measure no longer needs case-owned UFL plumbing.",
                    path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                    replacement="results.region_measure(on=region)",
                    semantic_review=True,
                )
            )
        method = name.rsplit(".", 1)[-1]
        replacement = COMPATIBILITY_MODEL_REPLACEMENTS.get(method)
        if replacement is not None and method.endswith("_step"):
            preferred = (
                f"model.{replacement}(...)"
                if " " not in replacement
                else replacement
            )
            findings.append(
                UpgradeFinding(
                    "AFM-UPG-104",
                    "advisory",
                    (
                        f"Model.{method}(...) remains executable as a compatibility "
                        "spelling, but new workflows use the unified public vocabulary."
                    ),
                    path,
                    line=node.lineno,
                    column=node.col_offset + 1,
                    replacement=preferred,
                    semantic_review=method.endswith("_step"),
                )
            )
    return tuple(findings)


def _uses_region_measure(node: ast.Call) -> bool:
    return any(
        keyword.arg == "measure"
        and isinstance(keyword.value, ast.Attribute)
        and keyword.value.attr == "measure"
        for keyword in node.keywords
    )


def _call_name(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _compare_versions(left: str, right: str) -> int | None:
    pattern = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
    left_match = pattern.match(str(left))
    right_match = pattern.match(str(right))
    if left_match is None or right_match is None:
        return None
    left_key = tuple(int(item) for item in left_match.groups())
    right_key = tuple(int(item) for item in right_match.groups())
    return (left_key > right_key) - (left_key < right_key)


def _insert_project_key(text: str, key: str, value: str) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == "[project]":
            lines.insert(index + 1, f'{key} = "{value}"\n')
            return "".join(lines)
    raise ValueError(f"{PROJECT_FILENAME} has no [project] table.")


__all__ = [
    "COHESIVE_CHECKPOINT_SCHEMA",
    "UPGRADE_SCHEMA",
    "UpgradeFinding",
    "UpgradeReport",
    "apply_safe_metadata",
    "inspect_project",
    "migrate_cohesive_checkpoint",
]
