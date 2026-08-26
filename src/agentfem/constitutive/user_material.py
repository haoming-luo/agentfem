"""Contracts for future Abaqus UMAT/UHYPER interoperability.

This module deliberately defines and validates the material-point boundary;
it does not claim that arbitrary Abaqus user subroutines can already be
executed by AgentFEM.  A real bridge additionally needs quadrature-state
storage, a global constitutive driver, compiler/ABI adapters, and reference
comparisons against Abaqus.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class MaterialPointInput:
    """Solver-neutral finite-strain input for one material-point update."""

    deformation_gradient_old: np.ndarray
    deformation_gradient_new: np.ndarray
    time: float
    time_increment: float
    properties: np.ndarray
    state_old: np.ndarray
    temperature: float | None = None
    temperature_increment: float | None = None
    field_variables: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("deformation_gradient_old", "deformation_gradient_new"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3x3 matrix.")
            if np.linalg.det(value) <= 0.0:
                raise ValueError(f"{name} must have a positive determinant.")
            object.__setattr__(self, name, value.copy())
        if not np.isfinite(self.time) or not np.isfinite(self.time_increment):
            raise ValueError("Material-point time values must be finite.")
        if self.time_increment <= 0.0:
            raise ValueError("Material-point time_increment must be positive.")
        for name in ("properties", "state_old"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values.")
            object.__setattr__(self, name, value.copy())
        if self.field_variables is not None:
            fields = np.asarray(self.field_variables, dtype=float).reshape(-1)
            if not np.all(np.isfinite(fields)):
                raise ValueError("field_variables must contain finite values.")
            object.__setattr__(self, "field_variables", fields.copy())


@dataclass(frozen=True)
class MaterialPointOutput:
    """Constitutive response returned to a nonlinear finite-element driver."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    strain_energy_density: float | None = None
    suggested_time_scale: float = 1.0

    def __post_init__(self) -> None:
        stress = np.asarray(self.cauchy_stress, dtype=float)
        tangent = np.asarray(self.consistent_tangent, dtype=float)
        state = np.asarray(self.state_new, dtype=float).reshape(-1)
        if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
            raise ValueError("cauchy_stress must be a finite 3x3 tensor.")
        if tangent.shape != (6, 6) or not np.all(np.isfinite(tangent)):
            raise ValueError("consistent_tangent must be a finite 6x6 matrix.")
        if not np.all(np.isfinite(state)):
            raise ValueError("state_new must contain finite values.")
        if (
            self.strain_energy_density is not None
            and not np.isfinite(self.strain_energy_density)
        ):
            raise ValueError("strain_energy_density must be finite when provided.")
        if (
            not np.isfinite(self.suggested_time_scale)
            or self.suggested_time_scale <= 0.0
        ):
            raise ValueError("suggested_time_scale must be finite and positive.")
        object.__setattr__(self, "cauchy_stress", stress.copy())
        object.__setattr__(self, "consistent_tangent", tangent.copy())
        object.__setattr__(self, "state_new", state.copy())


@runtime_checkable
class UserMaterial(Protocol):
    """Protocol implemented by native or adapted material-point models."""

    name: str

    def update(self, point: MaterialPointInput) -> MaterialPointOutput:
        """Advance one integration point and return stress, state, and tangent."""


@dataclass(frozen=True)
class AbaqusUserMaterialBridge:
    """Truthful capability description for an intended UMAT/UHYPER adapter."""

    kind: str
    source: str
    material_name: str
    property_count: int
    state_variable_count: int = 0
    tensor_order: str = "11,22,33,12,13,23"
    status: str = "adapter_specification"

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        if kind not in {"UMAT", "UHYPER"}:
            raise ValueError("Abaqus user-material kind must be UMAT or UHYPER.")
        if not self.source:
            raise ValueError("Abaqus user-material source must be named.")
        if self.property_count < 0 or self.state_variable_count < 0:
            raise ValueError("Property and state-variable counts cannot be negative.")
        object.__setattr__(self, "kind", kind)

    @property
    def executable(self) -> bool:
        """Return false until a compiled adapter and global driver exist."""

        return False

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_user_material_bridge",
            "interface": self.kind,
            "source": self.source,
            "material_name": self.material_name,
            "property_count": self.property_count,
            "state_variable_count": self.state_variable_count,
            "tensor_order": self.tensor_order,
            "status": self.status,
            "executable": self.executable,
            "required_runtime": (
                "quadrature state driver",
                "compiler/ABI adapter",
                "Abaqus-to-AgentFEM tensor and rotation semantics",
                "consistent-tangent verification",
            ),
        }


@dataclass(frozen=True)
class UserMaterialInspectionIssue:
    """One stable finding from source-only user-material inspection."""

    code: str
    severity: str
    message: str

    def summary(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class UserMaterialSourceFile:
    """One content-addressed source in a user-material source graph."""

    path: Path
    logical_path: str
    source_sha256: str
    include_files: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "logical_path": self.logical_path,
            "source_sha256": self.source_sha256,
            "include_files": list(self.include_files),
        }


@dataclass(frozen=True)
class UserMaterialIncludeEdge:
    """One Fortran INCLUDE relation and its resolution status."""

    source: str
    declaration: str
    target: str
    status: str

    def summary(self) -> dict[str, str]:
        return {
            "source": self.source,
            "declaration": self.declaration,
            "target": self.target,
            "status": self.status,
        }


@dataclass(frozen=True)
class UserMaterialSourceGraph:
    """Recursive identity of project-owned Fortran material sources."""

    root: Path
    files: tuple[UserMaterialSourceFile, ...]
    edges: tuple[UserMaterialIncludeEdge, ...]
    fingerprint: str
    runtime_includes: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.issues and all(
            item.status in {"resolved", "runtime_provided"} for item in self.edges
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.user-material-source-graph",
            "schema_version": "0.1.0",
            "root": str(self.root),
            "fingerprint": self.fingerprint,
            "complete": self.complete,
            "files": [item.summary() for item in self.files],
            "edges": [item.summary() for item in self.edges],
            "runtime_includes": list(self.runtime_includes),
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AbaqusUserMaterialInspection:
    """Source inventory that selects a migration route without executing code."""

    source: str
    source_sha256: str
    source_graph: UserMaterialSourceGraph
    interface: str | None
    entrypoints: tuple[str, ...]
    includes: tuple[str, ...]
    abaqus_utility_calls: tuple[str, ...]
    project_calls: tuple[str, ...]
    external_calls: tuple[str, ...]
    missing_contract_symbols: tuple[str, ...]
    route: str
    findings: tuple[UserMaterialInspectionIssue, ...]

    @property
    def status(self) -> str:
        return (
            "manual_adaptation_required"
            if any(item.severity == "error" for item in self.findings)
            else "adapter_candidate"
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.abaqus-user-material-inspection",
            "schema_version": "0.1.0",
            "status": self.status,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "source_graph": self.source_graph.summary(),
            "interface": self.interface,
            "entrypoints": list(self.entrypoints),
            "includes": list(self.includes),
            "abaqus_utility_calls": list(self.abaqus_utility_calls),
            "project_calls": list(self.project_calls),
            "external_calls": list(self.external_calls),
            "missing_contract_symbols": list(self.missing_contract_symbols),
            "recommended_route": self.route,
            "executable": False,
            "findings": [item.summary() for item in self.findings],
        }

    def format(self) -> str:
        lines = [
            f"Abaqus user-material inspection: {self.status}",
            f"  interface: {self.interface or '<not identified>'}",
            f"  route: {self.route}",
            f"  source SHA256: {self.source_sha256}",
            f"  source graph: {len(self.source_graph.files)} file(s), "
            f"{self.source_graph.fingerprint}",
        ]
        if self.abaqus_utility_calls:
            lines.append(
                "  Abaqus utilities: " + ", ".join(self.abaqus_utility_calls)
            )
        if self.project_calls:
            lines.append("  project calls: " + ", ".join(self.project_calls))
        if self.external_calls:
            lines.append("  unresolved calls: " + ", ".join(self.external_calls))
        if self.missing_contract_symbols:
            lines.append(
                "  missing contract symbols: "
                + ", ".join(self.missing_contract_symbols)
            )
        lines.extend(
            f"  [{item.severity}] {item.code}: {item.message}"
            for item in self.findings
        )
        return "\n".join(lines)


_ABAQUS_UTILITY_NAMES = {
    "GETENVVAR",
    "GETJOBNAME",
    "GETOUTDIR",
    "GETRANK",
    "GETVRM",
    "MATERIAL_LIB_MECH",
    "ROTSIG",
    "SINV",
    "SPRINC",
    "SPRIND",
    "STDB_ABQERR",
    "VGETVRM",
    "VSPRINC",
    "VSPRIND",
    "XIT",
}

_REQUIRED_CONTRACT_SYMBOLS = {
    "UMAT": {
        "STRESS",
        "STATEV",
        "DDSDDE",
        "STRAN",
        "DSTRAN",
        "TIME",
        "DTIME",
        "PROPS",
        "NPROPS",
    },
    "UHYPER": {
        "BI1",
        "BI2",
        "AJ",
        "U",
        "UI1",
        "UI2",
        "UI3",
        "STATEV",
        "PROPS",
    },
}

_ABAQUS_RUNTIME_INCLUDES = {
    "ABA_PARAM.INC",
    "VABA_PARAM.INC",
}


def read_user_material_source_graph(source: str | Path) -> UserMaterialSourceGraph:
    """Resolve local Fortran INCLUDE files and fingerprint the complete asset.

    Abaqus-provided parameter headers remain explicit runtime dependencies.
    Project-owned includes are resolved relative to the declaring file. Missing
    files and cycles are retained as addressable issues rather than ignored.
    """

    root = Path(source).expanduser().resolve()
    if not root.is_file():
        raise FileNotFoundError(root)
    files: dict[Path, UserMaterialSourceFile] = {}
    edges: list[UserMaterialIncludeEdge] = []
    runtime_includes: list[str] = []
    issues: list[str] = []

    def logical(selected: Path) -> str:
        return Path(os.path.relpath(selected, root.parent)).as_posix()

    def declarations(text: str) -> tuple[str, ...]:
        return tuple(
            match.strip()
            for match in re.findall(
                r"(?im)^\s*(?:#\s*)?INCLUDE\s+['\"]([^'\"]+)['\"]",
                text,
            )
        )

    def visit(selected: Path, ancestry: tuple[Path, ...]) -> None:
        resolved = selected.expanduser().resolve()
        if resolved in files:
            return
        raw = resolved.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        selected_includes = declarations(text)
        files[resolved] = UserMaterialSourceFile(
            path=resolved,
            logical_path=logical(resolved),
            source_sha256=sha256(raw).hexdigest(),
            include_files=selected_includes,
        )
        for declaration in selected_includes:
            if Path(declaration).name.upper() in _ABAQUS_RUNTIME_INCLUDES:
                normalized = Path(declaration).name.upper()
                runtime_includes.append(normalized)
                edges.append(
                    UserMaterialIncludeEdge(
                        logical(resolved),
                        declaration,
                        normalized,
                        "runtime_provided",
                    )
                )
                continue
            candidate = Path(declaration).expanduser()
            if not candidate.is_absolute():
                candidate = resolved.parent / candidate
            candidate = candidate.resolve()
            target = logical(candidate)
            if candidate in ancestry or candidate == resolved:
                edges.append(
                    UserMaterialIncludeEdge(
                        logical(resolved), declaration, target, "cycle"
                    )
                )
                issues.append(
                    "AFM-USERMAT-INCLUDE-002: recursive include cycle: "
                    + " -> ".join(
                        logical(item) for item in (*ancestry, resolved, candidate)
                    )
                    + "."
                )
                continue
            if not candidate.is_file():
                edges.append(
                    UserMaterialIncludeEdge(
                        logical(resolved), declaration, target, "missing"
                    )
                )
                issues.append(
                    f"AFM-USERMAT-INCLUDE-001: {logical(resolved)} references "
                    f"missing include {declaration!r}."
                )
                continue
            edges.append(
                UserMaterialIncludeEdge(
                    logical(resolved), declaration, target, "resolved"
                )
            )
            visit(candidate, (*ancestry, resolved))

    visit(root, ())
    payload = {
        "root": logical(root),
        "files": [
            {
                "logical_path": item.logical_path,
                "source_sha256": item.source_sha256,
                "include_files": list(item.include_files),
            }
            for item in files.values()
        ],
        "edges": [item.summary() for item in edges],
        "runtime_includes": sorted(set(runtime_includes)),
    }
    fingerprint = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return UserMaterialSourceGraph(
        root=root,
        files=tuple(files.values()),
        edges=tuple(edges),
        fingerprint=fingerprint,
        runtime_includes=tuple(sorted(set(runtime_includes))),
        issues=tuple(dict.fromkeys(issues)),
    )


def inspect_abaqus_user_material(
    source: str | Path,
    *,
    kind: str = "auto",
) -> AbaqusUserMaterialInspection:
    """Inventory a Fortran UMAT/UHYPER source and select a credible next route.

    Inspection is deliberately source-only. It neither compiles the routine nor
    claims Abaqus ABI compatibility; that distinction is part of the returned
    machine-readable contract.
    """

    path = Path(source).expanduser().resolve()
    if path.suffix.lower() not in {".f", ".for", ".f90", ".f95", ".f03", ".f08"}:
        raise ValueError("Abaqus user-material inspection requires a Fortran source file.")
    if not path.is_file():
        raise FileNotFoundError(path)
    graph = read_user_material_source_graph(path)
    raw = path.read_bytes()
    texts = [
        item.path.read_text(encoding="utf-8", errors="replace")
        for item in graph.files
    ]
    text = "\n".join(texts)
    upper = text.upper()
    entrypoints = tuple(
        dict.fromkeys(
            re.findall(
                r"(?im)^\s*(?:RECURSIVE\s+)?SUBROUTINE\s+([A-Z][A-Z0-9_]*)\s*\(",
                upper,
            )
        )
    )
    material_entrypoints = tuple(
        item for item in entrypoints if item in {"UMAT", "UHYPER"}
    )
    selected_kind = str(kind).strip().upper()
    if selected_kind == "AUTO":
        interface = material_entrypoints[0] if len(material_entrypoints) == 1 else None
    elif selected_kind in {"UMAT", "UHYPER"}:
        interface = selected_kind
    else:
        raise ValueError("kind must be 'auto', 'UMAT', or 'UHYPER'.")

    includes = tuple(
        dict.fromkeys(
            match.strip("'\"")
            for match in re.findall(
                r"(?im)^\s*INCLUDE\s+(['\"][^'\"]+['\"])", text
            )
        )
    )
    calls = tuple(
        dict.fromkeys(
            re.findall(r"(?im)\bCALL\s+([A-Z][A-Z0-9_]*)\s*\(", upper)
        )
    )
    utilities = tuple(item for item in calls if item in _ABAQUS_UTILITY_NAMES)
    project_calls = tuple(
        item
        for item in calls
        if item in entrypoints and item not in material_entrypoints
    )
    external = tuple(
        item
        for item in calls
        if item not in _ABAQUS_UTILITY_NAMES
        and item not in material_entrypoints
        and item not in project_calls
    )
    source_symbols = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", upper))
    missing_contract_symbols = tuple(
        sorted(_REQUIRED_CONTRACT_SYMBOLS.get(interface, set()) - source_symbols)
    )
    findings = []
    if len(material_entrypoints) != 1:
        findings.append(
            UserMaterialInspectionIssue(
                "AFM-USERMAT-SOURCE-001",
                "error",
                "Source inspection requires exactly one UMAT or UHYPER entry point.",
            )
        )
    elif interface not in material_entrypoints:
        findings.append(
            UserMaterialInspectionIssue(
                "AFM-USERMAT-SOURCE-002",
                "error",
                f"Requested {interface} entry point is not present in the source.",
            )
        )
    if utilities:
        findings.append(
            UserMaterialInspectionIssue(
                "AFM-USERMAT-SOURCE-003",
                "error",
                "The routine calls Abaqus utility functions that require explicit "
                "replacement or an Abaqus-compatible support library: "
                + ", ".join(utilities),
            )
        )
    if missing_contract_symbols:
        findings.append(
            UserMaterialInspectionIssue(
                "AFM-USERMAT-SOURCE-004",
                "error",
                "The detected entry point does not expose the minimum symbols "
                "needed by the restricted material-point contract: "
                + ", ".join(missing_contract_symbols),
            )
        )
    if external:
        findings.append(
            UserMaterialInspectionIssue(
                "AFM-USERMAT-SOURCE-101",
                "warning",
                "Unresolved subroutine calls require additional source or an "
                "explicit replacement before compilation: "
                + ", ".join(external),
            )
        )
    for issue in graph.issues:
        code, _, message = issue.partition(":")
        findings.append(
            UserMaterialInspectionIssue(
                code or "AFM-USERMAT-INCLUDE-000",
                "error",
                message.strip() or issue,
            )
        )
    if interface == "UHYPER":
        route = "restricted_uhyper_energy_adapter"
    elif interface == "UMAT":
        route = "restricted_umat_material_point_adapter"
    else:
        route = "manual_interface_identification"
    if interface is not None and not findings:
        findings.append(
            UserMaterialInspectionIssue(
                "AFM-USERMAT-SOURCE-100",
                "info",
                "The source is a candidate for adapter development. Compilation, "
                "material-point path comparison, tangent checks, and global FEM "
                "integration remain required before execution.",
            )
        )
    return AbaqusUserMaterialInspection(
        source=str(path),
        source_sha256=sha256(raw).hexdigest(),
        source_graph=graph,
        interface=interface,
        entrypoints=entrypoints,
        includes=includes,
        abaqus_utility_calls=utilities,
        project_calls=project_calls,
        external_calls=external,
        missing_contract_symbols=missing_contract_symbols,
        route=route,
        findings=tuple(findings),
    )
