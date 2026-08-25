"""Scope-aware, fail-closed planning for migration from Abaqus input decks.

This module does not execute an Abaqus deck and does not claim numerical
equivalence.  It resolves the engineering relationships that must survive a
migration: parts, instances, scoped sets, section assignments, and material
cards.  Numerical lowering remains a later, explicitly reviewed decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Mapping

from . import abaqus


@dataclass(frozen=True)
class AbaqusSourceLocation:
    """Stable source location for one migrated engineering declaration."""

    file: str
    line: int

    def summary(self) -> dict[str, object]:
        return {"file": self.file, "line": self.line}


@dataclass(frozen=True)
class AbaqusScope:
    """One Abaqus naming scope retained independently of numeric labels."""

    kind: str
    name: str
    parent: str | None
    key: str
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "parent": self.parent,
            "key": self.key,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusPart:
    """One reusable Abaqus Part declaration."""

    name: str
    scope: str
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusInstance:
    """One Part instance and its source-level positioning data."""

    name: str
    part: str
    assembly: str
    scope: str
    positioning: tuple[tuple[str, ...], ...]
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "part": self.part,
            "assembly": self.assembly,
            "scope": self.scope,
            "positioning": [list(row) for row in self.positioning],
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusScopedRegion:
    """Named set declaration qualified by its Abaqus scope."""

    name: str
    kind: str
    scope: str
    instance: str | None
    source: str
    location: AbaqusSourceLocation

    @property
    def key(self) -> str:
        return f"{self.scope}/{self.kind}:{self.name.upper()}"

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "scope": self.scope,
            "instance": self.instance,
            "source": self.source,
            "key": self.key,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusElementBlock:
    """One scoped element declaration with its numerical source identity."""

    scope: str
    region: str | None
    definition: abaqus.AbaqusElementDefinition
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "region": self.region,
            **self.definition.summary(),
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusPendingAsset:
    """Source engineering input retained until reviewed native lowering."""

    category: str
    keyword: str
    scope: str
    step: str | None
    options: Mapping[str, str]
    flags: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "category": self.category,
            "keyword": self.keyword,
            "scope": self.scope,
            "step": self.step,
            "options": dict(self.options),
            "flags": list(self.flags),
            "rows": [list(row) for row in self.rows],
            "status": "preserved_not_lowered",
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusMaterialBlock:
    """One keyword block belonging to an Abaqus material card."""

    keyword: str
    options: Mapping[str, str]
    flags: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "keyword": self.keyword,
            "options": dict(self.options),
            "flags": list(self.flags),
            "rows": [list(row) for row in self.rows],
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusMaterialCard:
    """Source material identity and behavior blocks, without eager execution."""

    name: str
    blocks: tuple[AbaqusMaterialBlock, ...]
    location: AbaqusSourceLocation
    translation_status: str
    native_candidate: Mapping[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "behavior_keywords": [item.keyword for item in self.blocks],
            "blocks": [item.summary() for item in self.blocks],
            "translation_status": self.translation_status,
            "native_candidate": dict(self.native_candidate),
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusSectionAssignment:
    """Material-to-region assignment retained at its Abaqus declaration level."""

    section_type: str
    scope: str
    region: str | None
    material: str | None
    orientation: str | None
    options: Mapping[str, str]
    flags: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    region_resolved: bool
    material_resolved: bool
    status: str
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "section_type": self.section_type,
            "scope": self.scope,
            "region": self.region,
            "material": self.material,
            "orientation": self.orientation,
            "options": dict(self.options),
            "flags": list(self.flags),
            "rows": [list(row) for row in self.rows],
            "region_resolved": self.region_resolved,
            "material_resolved": self.material_resolved,
            "status": self.status,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusEffectiveMaterialAssignment:
    """One section assignment projected onto an effective model/instance scope."""

    target_scope: str
    source_scope: str
    region: str
    material: str
    inherited_from_part: bool
    status: str
    location: AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "target_scope": self.target_scope,
            "source_scope": self.source_scope,
            "region": self.region,
            "material": self.material,
            "inherited_from_part": self.inherited_from_part,
            "status": self.status,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusMigrationIssue:
    """Addressable migration finding that cannot be silently inferred."""

    code: str
    severity: str
    message: str
    location: AbaqusSourceLocation | None = None

    def summary(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "location": None if self.location is None else self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusMigrationPlan:
    """Reviewable engineering plan between an Abaqus deck and AgentFEM model."""

    source_graph: abaqus.AbaqusSourceGraph
    scopes: tuple[AbaqusScope, ...]
    parts: tuple[AbaqusPart, ...]
    instances: tuple[AbaqusInstance, ...]
    regions: tuple[AbaqusScopedRegion, ...]
    element_blocks: tuple[AbaqusElementBlock, ...]
    materials: tuple[AbaqusMaterialCard, ...]
    sections: tuple[AbaqusSectionAssignment, ...]
    effective_assignments: tuple[AbaqusEffectiveMaterialAssignment, ...]
    pending_assets: tuple[AbaqusPendingAsset, ...]
    issues: tuple[AbaqusMigrationIssue, ...]

    @property
    def blocked(self) -> bool:
        return any(item.severity == "error" for item in self.issues)

    @property
    def review_required(self) -> bool:
        return self.blocked or any(
            item.translation_status != "native_candidate" for item in self.materials
        )

    @property
    def ready_to_solve(self) -> bool:
        return False

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.abaqus-migration-plan",
            "schema_version": "0.1.0",
            "status": "blocked" if self.blocked else "review_required",
            "ready_to_solve": self.ready_to_solve,
            "source_graph": self.source_graph.summary(),
            "scopes": [item.summary() for item in self.scopes],
            "parts": [item.summary() for item in self.parts],
            "instances": [item.summary() for item in self.instances],
            "regions": [item.summary() for item in self.regions],
            "element_blocks": [item.summary() for item in self.element_blocks],
            "materials": [item.summary() for item in self.materials],
            "sections": [item.summary() for item in self.sections],
            "effective_assignments": [
                item.summary() for item in self.effective_assignments
            ],
            "pending_assets": [item.summary() for item in self.pending_assets],
            "issues": [item.summary() for item in self.issues],
        }

    def text(self) -> str:
        return "\n".join(
            (
                "Abaqus migration plan",
                f"  source files: {len(self.source_graph.files)}",
                f"  parts/instances: {len(self.parts)}/{len(self.instances)}",
                f"  materials/sections: {len(self.materials)}/{len(self.sections)}",
                f"  element declarations: {len(self.element_blocks)}",
                f"  scoped regions: {len(self.regions)}",
                f"  preserved pending assets: {len(self.pending_assets)}",
                f"  status: {'blocked' if self.blocked else 'review required'}",
                "  native execution: deliberately disabled until reviewed lowering",
            )
        )


@dataclass
class _MaterialBuilder:
    name: str
    location: AbaqusSourceLocation
    blocks: list[
        tuple[
            str,
            dict[str, str],
            tuple[str, ...],
            list[tuple[str, ...]],
            AbaqusSourceLocation,
        ]
    ]


@dataclass
class _InstanceBuilder:
    name: str
    part: str
    assembly: str
    scope: str
    location: AbaqusSourceLocation
    positioning: list[tuple[str, ...]]


_PENDING_CATEGORIES = {
    "*STEP": "step",
    "*STATIC": "procedure",
    "*DYNAMIC": "procedure",
    "*HEAT TRANSFER": "procedure",
    "*COUPLED TEMPERATURE-DISPLACEMENT": "procedure",
    "*STEADY STATE DYNAMICS": "procedure",
    "*BOUNDARY": "boundary_condition",
    "*EQUATION": "constraint",
    "*MPC": "constraint",
    "*COUPLING": "constraint",
    "*KINEMATIC": "constraint",
    "*DISTRIBUTING": "constraint",
    "*CLOAD": "load",
    "*DLOAD": "load",
    "*DSLOAD": "load",
    "*BASE MOTION": "load",
    "*CONNECTOR LOAD": "load",
    "*TEMPERATURE": "predefined_field",
    "*FIELD": "predefined_field",
    "*PREDEFINED FIELD": "predefined_field",
    "*INITIAL CONDITIONS": "initial_condition",
    "*AMPLITUDE": "amplitude",
    "*FILM": "thermal_load",
    "*CFILM": "thermal_load",
    "*SFILM": "thermal_load",
    "*RADIATE": "thermal_load",
    "*CRADIATE": "thermal_load",
    "*SRADIATE": "thermal_load",
    "*BODY HEAT FLUX": "thermal_load",
    "*DFLUX": "thermal_load",
    "*SFLUX": "thermal_load",
    "*SURFACE": "region_definition",
    "*ORIENTATION": "orientation",
    "*TRANSFORM": "coordinate_system",
    "*TIE": "interaction",
    "*CONTACT": "interaction",
    "*CONTACT PAIR": "interaction",
    "*SURFACE INTERACTION": "interaction",
    "*SURFACE BEHAVIOR": "interaction_property",
    "*FRICTION": "interaction_property",
    "*MODEL CHANGE": "model_change",
    "*CONTROLS": "solver_control",
    "*RESTART": "restart",
    "*OUTPUT": "output_request",
    "*NODE OUTPUT": "output_request",
    "*ELEMENT OUTPUT": "output_request",
    "*HISTORY OUTPUT": "output_request",
    "*NODE PRINT": "output_request",
    "*EL PRINT": "output_request",
}


_MATERIAL_TERMINATORS = {
    "*AMPLITUDE",
    "*ASSEMBLY",
    "*BOUNDARY",
    "*CLOAD",
    "*DLOAD",
    "*END ASSEMBLY",
    "*END INSTANCE",
    "*END PART",
    "*END STEP",
    "*HEADING",
    "*INITIAL CONDITIONS",
    "*INSTANCE",
    "*MATERIAL",
    "*PART",
    "*STEP",
}


def _keyword(line: str) -> tuple[str, dict[str, str], set[str]]:
    fields = [item.strip() for item in line.split(",")]
    options: dict[str, str] = {}
    flags: set[str] = set()
    for item in fields[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            options[key.strip().upper()] = value.strip()
        elif item:
            flags.add(item.upper())
    return fields[0].upper(), options, flags


def _values(line: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in line.split(",") if item.strip())


def _logical(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root.parent)).as_posix()


def _expanded_lines(
    path: Path,
    *,
    root: Path,
    ancestry: tuple[Path, ...] = (),
) -> Iterable[tuple[AbaqusSourceLocation, str]]:
    """Yield an include-expanded view while retaining every source location."""

    selected = path.resolve()
    for line_number, raw in enumerate(
        selected.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if line.startswith("*") and not line.startswith("**"):
            keyword, options, _flags = _keyword(line)
            if keyword == "*INCLUDE":
                declaration = options.get("INPUT") or options.get("FILE")
                if declaration:
                    candidate = Path(declaration.strip().strip('"').strip("'"))
                    if not candidate.is_absolute():
                        candidate = selected.parent / candidate
                    candidate = candidate.resolve()
                    if candidate.is_file() and candidate not in (*ancestry, selected):
                        yield from _expanded_lines(
                            candidate,
                            root=root,
                            ancestry=(*ancestry, selected),
                        )
                continue
        yield AbaqusSourceLocation(_logical(selected, root), line_number), raw


def _native_material_candidate(
    name: str,
    blocks: tuple[AbaqusMaterialBlock, ...],
) -> tuple[str, dict[str, object]]:
    keywords = [item.keyword for item in blocks]
    if "*USER MATERIAL" in keywords or any(
        item.keyword == "*HYPERELASTIC" and "USER" in item.flags for item in blocks
    ):
        return "review_required_user_material", {}
    if not keywords or any(item not in {"*ELASTIC", "*DENSITY"} for item in keywords):
        return "review_required", {}
    elastic = next((item for item in blocks if item.keyword == "*ELASTIC"), None)
    if (
        elastic is None
        or elastic.options.get("TYPE", "ISOTROPIC").upper() != "ISOTROPIC"
    ):
        return "review_required", {}
    if not elastic.rows or len(elastic.rows[0]) < 2:
        return "review_required", {}
    try:
        young, poisson = float(elastic.rows[0][0]), float(elastic.rows[0][1])
        density_block = next(
            (item for item in blocks if item.keyword == "*DENSITY"), None
        )
        if density_block is None or not density_block.rows:
            return "review_required_missing_density", {}
        density = float(density_block.rows[0][0])
    except (TypeError, ValueError):
        return "review_required", {}
    candidate: dict[str, object] = {
        "constructor": "constitutive.isotropic_elastic",
        "name": name,
        "young": young,
        "poisson": poisson,
    }
    candidate["density"] = density
    return "native_candidate", candidate


def plan(path: str | Path) -> AbaqusMigrationPlan:
    """Build a scope-aware migration plan without converting or solving."""

    root = Path(path).expanduser().resolve()
    graph = abaqus.read_source_graph(root)
    issues = [
        AbaqusMigrationIssue(
            code=item.split(":", 1)[0],
            severity="error",
            message=item,
        )
        for item in graph.issues
    ]
    scopes: list[AbaqusScope] = [
        AbaqusScope(
            kind="model",
            name="MODEL",
            parent=None,
            key="model",
            location=AbaqusSourceLocation(_logical(root, root), 1),
        )
    ]
    parts: list[AbaqusPart] = []
    instance_builders: list[_InstanceBuilder] = []
    regions: list[AbaqusScopedRegion] = []
    element_blocks: list[AbaqusElementBlock] = []
    material_builders: list[_MaterialBuilder] = []
    raw_sections: list[
        tuple[
            str,
            str,
            dict[str, str],
            tuple[str, ...],
            list[tuple[str, ...]],
            AbaqusSourceLocation,
        ]
    ] = []
    raw_pending_assets: list[
        tuple[
            str,
            str,
            str,
            str | None,
            dict[str, str],
            tuple[str, ...],
            list[tuple[str, ...]],
            AbaqusSourceLocation,
        ]
    ] = []

    part_name: str | None = None
    assembly_name: str | None = None
    instance: _InstanceBuilder | None = None
    active_material: _MaterialBuilder | None = None
    active_material_block: (
        tuple[
            str,
            dict[str, str],
            tuple[str, ...],
            list[tuple[str, ...]],
            AbaqusSourceLocation,
        ]
        | None
    ) = None
    active_keyword = ""
    active_section_rows: list[tuple[str, ...]] | None = None
    active_pending_rows: list[tuple[str, ...]] | None = None
    step_name: str | None = None
    step_index = 0

    def scope_key() -> str:
        if instance is not None:
            return instance.scope
        if part_name is not None:
            return f"part:{part_name}"
        if assembly_name is not None:
            return f"assembly:{assembly_name}"
        return "model"

    for location, raw in _expanded_lines(root, root=root):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            active_keyword, options, flags = _keyword(line)
            active_material_block = None
            active_section_rows = None
            active_pending_rows = None
            if active_keyword == "*PART":
                name = options.get("NAME", "").strip()
                if not name:
                    issues.append(
                        AbaqusMigrationIssue(
                            "AFM-ABAQUS-SCOPE-001",
                            "error",
                            "*PART requires NAME=.",
                            location,
                        )
                    )
                    name = "<unnamed>"
                part_name = name
                active_material = None
                scope = f"part:{name}"
                parts.append(AbaqusPart(name, scope, location))
                scopes.append(AbaqusScope("part", name, "model", scope, location))
            elif active_keyword == "*END PART":
                part_name = None
                active_material = None
            elif active_keyword == "*ASSEMBLY":
                assembly_name = options.get("NAME", "ASSEMBLY").strip()
                active_material = None
                scope = f"assembly:{assembly_name}"
                scopes.append(
                    AbaqusScope("assembly", assembly_name, "model", scope, location)
                )
            elif active_keyword == "*END ASSEMBLY":
                assembly_name = None
                active_material = None
            elif active_keyword == "*INSTANCE":
                name = options.get("NAME", "").strip()
                part = options.get("PART", "").strip()
                assembly = assembly_name or "<missing-assembly>"
                scope = f"instance:{assembly}/{name or '<unnamed>'}"
                instance = _InstanceBuilder(
                    name or "<unnamed>",
                    part or "<unspecified>",
                    assembly,
                    scope,
                    location,
                    [],
                )
                instance_builders.append(instance)
                scopes.append(
                    AbaqusScope(
                        "instance",
                        instance.name,
                        f"assembly:{assembly}",
                        scope,
                        location,
                    )
                )
                active_material = None
            elif active_keyword == "*END INSTANCE":
                instance = None
                active_material = None
            elif active_keyword == "*MATERIAL":
                name = options.get("NAME", "").strip() or "<unnamed>"
                active_material = _MaterialBuilder(name, location, [])
                material_builders.append(active_material)
            elif active_keyword == "*STEP":
                step_index += 1
                step_name = options.get("NAME", f"Step-{step_index}").strip()
                active_pending_rows = []
                raw_pending_assets.append(
                    (
                        "step",
                        active_keyword,
                        scope_key(),
                        step_name,
                        options,
                        tuple(sorted(flags)),
                        active_pending_rows,
                        location,
                    )
                )
                active_material = None
            elif active_keyword == "*END STEP":
                step_name = None
                active_material = None
            elif active_keyword.endswith(" SECTION"):
                active_section_rows = []
                raw_sections.append(
                    (
                        active_keyword,
                        scope_key(),
                        options,
                        tuple(sorted(flags)),
                        active_section_rows,
                        location,
                    )
                )
                active_material = None
            elif active_keyword in {"*NSET", "*ELSET"}:
                active_material = None
                option = active_keyword[1:]
                name = options.get(option, "").strip()
                if name:
                    regions.append(
                        AbaqusScopedRegion(
                            name=name,
                            kind=option.lower(),
                            scope=scope_key(),
                            instance=options.get("INSTANCE"),
                            source="explicit_set",
                            location=location,
                        )
                    )
            elif active_keyword == "*ELEMENT" and options.get("ELSET"):
                active_material = None
                source_type = options.get("TYPE", "<unspecified>").strip().upper()
                element_blocks.append(
                    AbaqusElementBlock(
                        scope=scope_key(),
                        region=options["ELSET"].strip(),
                        definition=abaqus.describe_element_type(source_type),
                        location=location,
                    )
                )
                regions.append(
                    AbaqusScopedRegion(
                        name=options["ELSET"].strip(),
                        kind="elset",
                        scope=scope_key(),
                        instance=None,
                        source="element_header",
                        location=location,
                    )
                )
            elif active_keyword == "*ELEMENT":
                active_material = None
                source_type = options.get("TYPE", "<unspecified>").strip().upper()
                element_blocks.append(
                    AbaqusElementBlock(
                        scope=scope_key(),
                        region=None,
                        definition=abaqus.describe_element_type(source_type),
                        location=location,
                    )
                )
            elif active_keyword == "*NODE":
                active_material = None
            elif active_keyword in _PENDING_CATEGORIES:
                active_material = None
                active_pending_rows = []
                raw_pending_assets.append(
                    (
                        _PENDING_CATEGORIES[active_keyword],
                        active_keyword,
                        scope_key(),
                        step_name,
                        options,
                        tuple(sorted(flags)),
                        active_pending_rows,
                        location,
                    )
                )
            elif active_material is not None:
                if active_keyword in _MATERIAL_TERMINATORS:
                    active_material = None
                else:
                    block = (
                        active_keyword,
                        options,
                        tuple(sorted(flags)),
                        [],
                        location,
                    )
                    active_material.blocks.append(block)
                    active_material_block = block
            continue

        row = _values(line)
        if instance is not None and active_keyword == "*INSTANCE" and row:
            instance.positioning.append(row)
        elif active_material_block is not None and row:
            active_material_block[3].append(row)
        elif active_section_rows is not None and row:
            active_section_rows.append(row)
        elif active_pending_rows is not None and row:
            active_pending_rows.append(row)

    part_lookup = {item.name.upper(): item for item in parts}
    if len(part_lookup) != len(parts):
        issues.append(
            AbaqusMigrationIssue(
                "AFM-ABAQUS-SCOPE-002",
                "error",
                "Part names must be unique within one model.",
            )
        )
    instances = tuple(
        AbaqusInstance(
            item.name,
            item.part,
            item.assembly,
            item.scope,
            tuple(item.positioning),
            item.location,
        )
        for item in instance_builders
    )
    for item in instances:
        if item.part.upper() not in part_lookup:
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-SCOPE-003",
                    "error",
                    f"Instance {item.name!r} references unknown Part {item.part!r}.",
                    item.location,
                )
            )
        if item.assembly == "<missing-assembly>":
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-SCOPE-004",
                    "error",
                    f"Instance {item.name!r} is outside an *ASSEMBLY block.",
                    item.location,
                )
            )
        if item.positioning:
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-SCOPE-005",
                    "warning",
                    f"Instance {item.name!r} has positioning data that must be "
                    "explicitly lowered before native mesh conversion.",
                    item.location,
                )
            )

    instance_keys = {(item.assembly.upper(), item.name.upper()) for item in instances}
    if len(instance_keys) != len(instances):
        issues.append(
            AbaqusMigrationIssue(
                "AFM-ABAQUS-SCOPE-006",
                "error",
                "Instance names must be unique within each Assembly.",
            )
        )

    for item in element_blocks:
        definition = item.definition
        if definition.source_type == "<UNSPECIFIED>":
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-ELEMENT-001",
                    "error",
                    "*ELEMENT requires TYPE= before a numerical formulation can "
                    "be selected.",
                    item.location,
                )
            )
        elif definition.solver_capability in {"topology_only", "not_declared"}:
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-ELEMENT-002",
                    "warning",
                    f"Element {definition.source_type} can be preserved as "
                    "declared topology but has no verified native lowering in "
                    "the current capability catalog.",
                    item.location,
                )
            )

    materials: list[AbaqusMaterialCard] = []
    for item in material_builders:
        blocks = tuple(
            AbaqusMaterialBlock(keyword, options, flags, tuple(rows), location)
            for keyword, options, flags, rows, location in item.blocks
        )
        status, candidate = _native_material_candidate(item.name, blocks)
        materials.append(
            AbaqusMaterialCard(item.name, blocks, item.location, status, candidate)
        )
        if status == "review_required_user_material":
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-MATERIAL-004",
                    "warning",
                    f"Material {item.name!r} declares an Abaqus user material. "
                    "The deck preserves constants and state declarations but "
                    "does not contain the compiled source, solver ABI, or a "
                    "verified AgentFEM adapter.",
                    item.location,
                )
            )
        elif status == "review_required_missing_density":
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-MATERIAL-002",
                    "warning",
                    f"Material {item.name!r} declares isotropic elasticity but no "
                    "density. AgentFEM will not invent the missing property or "
                    "emit an executable material candidate.",
                    item.location,
                )
            )
        elif status != "native_candidate":
            keywords = ", ".join(block.keyword for block in blocks) or "none"
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-MATERIAL-003",
                    "warning",
                    f"Material {item.name!r} requires reviewed constitutive "
                    f"lowering; detected behavior blocks: {keywords}.",
                    item.location,
                )
            )
    material_lookup = {item.name.upper(): item for item in materials}
    if len(material_lookup) != len(materials):
        issues.append(
            AbaqusMigrationIssue(
                "AFM-ABAQUS-MATERIAL-001",
                "error",
                "Material names must be unique within one model.",
            )
        )

    region_keys = {item.key for item in regions}
    sections: list[AbaqusSectionAssignment] = []
    for section_type, scope, options, flags, rows, location in raw_sections:
        region = options.get("ELSET")
        material = options.get("MATERIAL")
        composite = "COMPOSITE" in flags
        region_resolved = (
            region is not None and f"{scope}/elset:{region.upper()}" in region_keys
        )
        material_resolved = composite or (
            material is not None and material.upper() in material_lookup
        )
        status = (
            "review_required_composite"
            if composite and region_resolved
            else "references_resolved"
            if region_resolved and material_resolved
            else "blocked_unresolved_reference"
        )
        sections.append(
            AbaqusSectionAssignment(
                section_type=section_type,
                scope=scope,
                region=region,
                material=material,
                orientation=options.get("ORIENTATION"),
                options=options,
                flags=flags,
                rows=tuple(rows),
                region_resolved=region_resolved,
                material_resolved=material_resolved,
                status=status,
                location=location,
            )
        )
        if not region_resolved:
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-SECTION-001",
                    "error",
                    f"{section_type} references ELSET {region!r} outside its "
                    f"resolved scope {scope!r}.",
                    location,
                )
            )
        if composite:
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-SECTION-003",
                    "warning",
                    f"Composite {section_type} layers require explicit material and "
                    "orientation lowering.",
                    location,
                )
            )
        elif not material_resolved:
            issues.append(
                AbaqusMigrationIssue(
                    "AFM-ABAQUS-SECTION-002",
                    "error",
                    f"{section_type} references unknown material {material!r}.",
                    location,
                )
            )

    effective_assignments: list[AbaqusEffectiveMaterialAssignment] = []
    for section in sections:
        if (
            section.status != "references_resolved"
            or section.region is None
            or section.material is None
        ):
            continue
        if section.scope.startswith("part:"):
            part = section.scope.split(":", 1)[1]
            selected_instances = [
                item for item in instances if item.part.upper() == part.upper()
            ]
            for selected in selected_instances:
                effective_assignments.append(
                    AbaqusEffectiveMaterialAssignment(
                        target_scope=selected.scope,
                        source_scope=section.scope,
                        region=f"{selected.name}.{section.region}",
                        material=section.material,
                        inherited_from_part=True,
                        status="references_resolved",
                        location=section.location,
                    )
                )
        else:
            effective_assignments.append(
                AbaqusEffectiveMaterialAssignment(
                    target_scope=section.scope,
                    source_scope=section.scope,
                    region=section.region,
                    material=section.material,
                    inherited_from_part=False,
                    status="references_resolved",
                    location=section.location,
                )
            )

    return AbaqusMigrationPlan(
        source_graph=graph,
        scopes=tuple(scopes),
        parts=tuple(parts),
        instances=instances,
        regions=tuple({item.key: item for item in regions}.values()),
        element_blocks=tuple(element_blocks),
        materials=tuple(materials),
        sections=tuple(sections),
        effective_assignments=tuple(effective_assignments),
        pending_assets=tuple(
            AbaqusPendingAsset(
                category,
                keyword,
                scope,
                step,
                options,
                flags,
                tuple(rows),
                location,
            )
            for (
                category,
                keyword,
                scope,
                step,
                options,
                flags,
                rows,
                location,
            ) in raw_pending_assets
        ),
        issues=tuple(issues),
    )


def _safe_project_name(value: str) -> str:
    selected = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    if not selected:
        raise ValueError("A migrated project name requires a letter or number.")
    return selected


def _case_source(root_entry: str) -> str:
    return f'''"""Reviewable AgentFEM scaffold generated from an Abaqus input deck.

This file deliberately does not solve until engineering semantics and native
formulations in migration.json have been reviewed.
"""

from pathlib import Path

from agentfem import mesh


PROJECT_ROOT = Path(__file__).resolve().parent
ABAQUS_SOURCE = PROJECT_ROOT / {root_entry!r}


def migration_plan():
    return mesh.plan_abaqus_migration(ABAQUS_SOURCE)


def main():
    plan = migration_plan()
    print(plan.text())
    raise RuntimeError(
        "This is a fail-closed migration scaffold. Review migration.json, "
        "select native AgentFEM formulations, then replace this guard with "
        "the visible Study -> Model -> assets -> step -> SimulationResult workflow."
    )


if __name__ == "__main__":
    main()
'''


def _material_candidates_source(plan_record: AbaqusMigrationPlan) -> str:
    lines = [
        '"""Reviewable material candidates extracted from the Abaqus source.',
        "",
        "Values remain in the source deck's consistent unit system. Nothing in",
        "this module is activated automatically by the generated case.",
        '"""',
        "",
        "from agentfem import constitutive, materials",
        "",
    ]
    candidate_names: list[tuple[str, str]] = []
    review_required: list[str] = []
    for index, material in enumerate(plan_record.materials, 1):
        if material.translation_status != "native_candidate":
            review_required.append(material.name)
            continue
        candidate = dict(material.native_candidate)
        symbol = f"material_{index}"
        arguments = [
            f"young={candidate['young']!r}",
            f"poisson={candidate['poisson']!r}",
            f"density={candidate['density']!r}",
        ]
        lines.extend(
            (
                f"{symbol} = materials.define(",
                f"    {material.name!r},",
                "    constitutive.isotropic_elastic(",
                *(f"        {item}," for item in arguments),
                "    ),",
                '    source="Abaqus migration candidate; unit review required",',
                "    metadata={",
                '        "migration_status": "candidate_not_activated",',
                f'        "source_file": {material.location.file!r},',
                f'        "source_line": {material.location.line!r},',
                "    },",
                ")",
                "",
            )
        )
        candidate_names.append((material.name, symbol))
    lines.append("MATERIAL_CANDIDATES = {")
    lines.extend(f"    {name!r}: {symbol}," for name, symbol in candidate_names)
    lines.extend(
        (
            "}",
            "",
            "MIGRATION_STATUS = {",
            *(
                f"    {material.name!r}: {material.translation_status!r},"
                for material in plan_record.materials
            ),
            "}",
            "",
            f"REVIEW_REQUIRED = {tuple(review_required)!r}",
            "",
        )
    )
    return "\n".join(lines)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _migration_report_markdown(plan_record: AbaqusMigrationPlan) -> str:
    lines = [
        "# Abaqus migration review",
        "",
        "> This report preserves source engineering intent. It does not authorize",
        "> execution or claim Abaqus numerical equivalence.",
        "",
        f"- Status: **{'blocked' if plan_record.blocked else 'review required'}**",
        f"- Source graph: `{plan_record.source_graph.fingerprint}`",
        f"- Files: {len(plan_record.source_graph.files)}",
        f"- Parts / Instances: {len(plan_record.parts)} / {len(plan_record.instances)}",
        f"- Pending assets: {len(plan_record.pending_assets)}",
        "",
        "## Element declarations",
        "",
        "| Scope | Region | Abaqus type | Topology | Native capability |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            _markdown_cell(value)
            for value in (
                item.scope,
                item.region or "",
                item.definition.source_type,
                item.definition.topology or "unknown",
                item.definition.solver_capability,
            )
        )
        + " |"
        for item in plan_record.element_blocks
    )
    if not plan_record.element_blocks:
        lines.append("| — | — | — | — | — |")
    lines.extend(
        (
            "",
            "## Materials",
            "",
            "| Name | Source behaviors | Migration status |",
            "|---|---|---|",
        )
    )
    lines.extend(
        "| "
        + " | ".join(
            _markdown_cell(value)
            for value in (
                item.name,
                ", ".join(block.keyword for block in item.blocks) or "none",
                item.translation_status,
            )
        )
        + " |"
        for item in plan_record.materials
    )
    if not plan_record.materials:
        lines.append("| — | — | — |")
    lines.extend(
        (
            "",
            "## Preserved assets awaiting lowering",
            "",
            "| Category | Keyword | Step | Source |",
            "|---|---|---|---|",
        )
    )
    lines.extend(
        "| "
        + " | ".join(
            _markdown_cell(value)
            for value in (
                item.category,
                item.keyword,
                item.step or "",
                f"{item.location.file}:{item.location.line}",
            )
        )
        + " |"
        for item in plan_record.pending_assets
    )
    if not plan_record.pending_assets:
        lines.append("| — | — | — | — |")
    lines.extend(("", "## Findings", ""))
    lines.extend(
        f"- **{item.severity.upper()} {item.code}** — {item.message}"
        for item in plan_record.issues
    )
    if not plan_record.issues:
        lines.append("- No structural-reference errors were detected.")
    lines.extend(
        (
            "",
            "## Required next decision",
            "",
            "Review `migration.json`, select native AgentFEM formulations, and",
            "replace the guard in `case.py` only after the complete scientific",
            "workflow and its verification evidence have been made explicit.",
            "",
        )
    )
    return "\n".join(lines)


def create_project(
    source: str | Path,
    destination: str | Path,
    *,
    name: str | None = None,
    created_with: str = "unknown",
) -> dict[str, object]:
    """Create an atomic, fail-closed migration project with copied sources."""

    source_path = Path(source).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if target.exists() and any(target.iterdir() if target.is_dir() else (target,)):
        raise FileExistsError(f"Migration target is not empty: {target}")
    plan_record = plan(source_path)
    if not plan_record.source_graph.complete:
        raise ValueError(
            "Cannot bundle an incomplete Abaqus source graph. Resolve missing or "
            "recursive *INCLUDE declarations first."
        )
    project_name = _safe_project_name(name or target.name)
    source_paths = [item.path for item in plan_record.source_graph.files]
    common_root = Path(os.path.commonpath([str(item) for item in source_paths]))
    if common_root.is_file():
        common_root = common_root.parent
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.migration-", dir=target.parent)
    )
    try:
        bundled: dict[str, str] = {}
        for item in plan_record.source_graph.files:
            relative = item.path.relative_to(common_root)
            destination_path = temporary / "source" / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.path, destination_path)
            bundled[item.logical_path] = destination_path.relative_to(
                temporary
            ).as_posix()
        root_entry = bundled[_logical(source_path, source_path)]
        (temporary / "case.py").write_text(_case_source(root_entry), encoding="utf-8")
        material_directory = temporary / "materials"
        material_directory.mkdir()
        (material_directory / "candidates.py").write_text(
            _material_candidates_source(plan_record), encoding="utf-8"
        )
        (material_directory / "README.md").write_text(
            "\n".join(
                (
                    "# Material migration candidates",
                    "",
                    "`candidates.py` contains source-valued candidates for the narrow",
                    "Abaqus material cards that AgentFEM can recognize unambiguously.",
                    "They are not imported by `case.py` and retain the source unit system.",
                    "Review units, formulation, calibration, and verification before use.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        mesh_directory = temporary / "mesh"
        mesh_directory.mkdir()
        (mesh_directory / "README.md").write_text(
            "\n".join(
                (
                    "# Derived mesh artifacts",
                    "",
                    "Keep the copied `source/` deck authoritative. Store only",
                    "fingerprinted conversion manifests and derived solver meshes here.",
                    "Never erase Abaqus formulation suffixes during conversion.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (temporary / "agentfem.toml").write_text(
            "\n".join(
                (
                    "[project]",
                    f'name = "{project_name}"',
                    'entrypoint = "case.py"',
                    'schema_version = "0.2.0"',
                    f'created_with = "{created_with}"',
                    "",
                    "[run]",
                    'output_directory = "outputs"',
                    "",
                )
            ),
            encoding="utf-8",
        )
        record = {
            **plan_record.summary(),
            "project": {
                "name": project_name,
                "entrypoint": "case.py",
                "source_entrypoint": root_entry,
                "bundled_sources": bundled,
            },
        }
        (temporary / "migration.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "migration.md").write_text(
            _migration_report_markdown(plan_record), encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "\n".join(
                (
                    f"# {project_name}",
                    "",
                    "This is a fail-closed AgentFEM migration scaffold.",
                    "",
                    "1. Run `agentfem check`.",
                    "2. Review `migration.json` and the copied `source/` deck.",
                    "   `migration.md` is the compact human review report.",
                    "3. Select and verify native AgentFEM formulations.",
                    "4. Replace the guard in `case.py` with the visible FEM workflow.",
                    "",
                    "The generated project does not claim Abaqus solver equivalence.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (temporary / "AGENTS.md").write_text(
            "\n".join(
                (
                    "# Abaqus Migration Review",
                    "",
                    "- Preserve every copied source file and migration fingerprint.",
                    "- Resolve every error in migration.json before solving.",
                    "- Do not discard Abaqus element suffixes or infer equivalence.",
                    "- Keep materials, regions, assignments, and steps distinct.",
                    "- Finish with a SimulationResult and explicit verification evidence.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (temporary / "outputs").mkdir()
        (temporary / ".gitignore").write_text(
            "outputs/\n__pycache__/\n", encoding="utf-8"
        )
        if target.exists():
            target.rmdir()
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema": "agentfem.abaqus-migration-project",
        "schema_version": "0.1.0",
        "status": "created_review_required",
        "project": str(target),
        "migration_plan": str(target / "migration.json"),
        "migration_report": str(target / "migration.md"),
        "source_entrypoint": str(target / root_entry),
    }
