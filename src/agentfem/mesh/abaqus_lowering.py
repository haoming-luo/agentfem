"""Reviewed native lowering for a deliberately narrow Abaqus project subset.

Migration preserves source intent first.  This module is the second gate: it
can emit an ordinary AgentFEM case only when every consumed source asset has a
declared native meaning.  Unsupported assets are findings, never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping

from . import abaqus_migration


class AbaqusNativeLoweringBlocked(ValueError):
    """Addressable refusal carrying the complete native-lowering assessment."""

    def __init__(self, assessment: "AbaqusNativeLoweringAssessment") -> None:
        self.assessment = assessment
        codes = ", ".join(
            item.code for item in assessment.findings if item.severity == "error"
        )
        super().__init__(f"Abaqus native lowering is blocked: {codes}")

    def details(self) -> dict[str, object]:
        return self.assessment.summary()


@dataclass(frozen=True)
class AbaqusNativeBoundary:
    """One source ``*BOUNDARY`` row accepted by the native draft."""

    region: str
    components: tuple[int, ...]
    value: float
    step: str | None
    location: abaqus_migration.AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "region": self.region,
            "components": list(self.components),
            "value": self.value,
            "step": self.step,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusNativePressure:
    """One source ``*DSLOAD`` pressure accepted by the native draft."""

    surface: str
    value: float
    step: str
    location: abaqus_migration.AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "value": self.value,
            "source_sign_convention": "positive_pressure_acts_inward",
            "native_configuration": "reference",
            "step": self.step,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusNativeGravity:
    """One whole-model Abaqus ``GRAV`` load accepted by the native draft."""

    acceleration: tuple[float, ...]
    step: str
    location: abaqus_migration.AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "acceleration": list(self.acceleration),
            "step": self.step,
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusNativeMaterialAssignment:
    """One reviewed isotropic material assigned to one Abaqus ``ELSET``."""

    region: str
    material_name: str
    material: Mapping[str, object]
    location: abaqus_migration.AbaqusSourceLocation

    def summary(self) -> dict[str, object]:
        return {
            "region": self.region,
            "material_name": self.material_name,
            "material": dict(self.material),
            "location": self.location.summary(),
        }


@dataclass(frozen=True)
class AbaqusNativeLoweringAssessment:
    """Eligibility and exact decisions for one native AgentFEM draft."""

    source_fingerprint: str
    dimension: int | None
    assumption: str | None
    topology: str | None
    degree: int | None
    material: Mapping[str, object]
    material_name: str | None
    material_assignments: tuple[AbaqusNativeMaterialAssignment, ...]
    step_name: str | None
    boundaries: tuple[AbaqusNativeBoundary, ...]
    pressures: tuple[AbaqusNativePressure, ...]
    gravities: tuple[AbaqusNativeGravity, ...]
    findings: tuple[abaqus_migration.AbaqusMigrationIssue, ...]

    @property
    def eligible(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.abaqus-native-lowering-assessment",
            "schema_version": "0.1.0",
            "status": "eligible" if self.eligible else "blocked",
            "source_fingerprint": self.source_fingerprint,
            "dimension": self.dimension,
            "assumption": self.assumption,
            "topology": self.topology,
            "degree": self.degree,
            "material_name": self.material_name,
            "material": dict(self.material),
            "material_assignments": [
                item.summary() for item in self.material_assignments
            ],
            "step_name": self.step_name,
            "boundaries": [item.summary() for item in self.boundaries],
            "pressures": [item.summary() for item in self.pressures],
            "gravities": [item.summary() for item in self.gravities],
            "findings": [item.summary() for item in self.findings],
        }


def _finding(code, severity, message, location=None):
    return abaqus_migration.AbaqusMigrationIssue(
        code=code,
        severity=severity,
        message=message,
        location=location,
    )


def _supported_instance(plan, findings):
    if not plan.parts and not plan.instances:
        return None
    if (
        len(plan.parts) != 1
        or len(plan.instances) != 1
        or plan.instances[0].part.upper() != plan.parts[0].name.upper()
    ):
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-SCOPE-001",
                "error",
                "Native draft generation supports either a flat orphan mesh or "
                "one Part instantiated once. Multi-instance assembly merging "
                "requires global label and interface decisions.",
            )
        )
        return None
    rows = plan.instances[0].positioning
    if (
        len(rows) > 2
        or any(len(row) not in {3, 7} for row in rows)
        or (len(rows) == 2 and (len(rows[0]) != 3 or len(rows[1]) != 7))
    ):
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-SCOPE-002",
                "error",
                "Instance positioning must be an optional translation row followed "
                "by an optional axis-angle rotation row.",
                plan.instances[0].location,
            )
        )
    else:
        try:
            tuple(float(value) for row in rows for value in row)
        except ValueError:
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-SCOPE-003",
                    "error",
                    "Instance positioning contains a non-numeric value.",
                    plan.instances[0].location,
                )
            )
    return plan.instances[0]


def _region_aliases(plan, *, kind, instance):
    aliases = {}
    part_scope = None if instance is None else f"part:{instance.part}"
    for item in plan.regions:
        if item.kind != kind:
            continue
        name = item.name.upper()
        if instance is None:
            if item.scope == "model":
                aliases[name] = name
            continue
        if item.scope == part_scope:
            aliases[name] = name
            aliases[f"{instance.name.upper()}.{name}"] = name
        elif item.scope in {instance.scope, f"assembly:{instance.assembly}"}:
            aliases[name] = name
    return aliases


def _kinematics(element_blocks, findings):
    definitions = [item.definition for item in element_blocks]
    if not definitions:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-ELEMENT-001",
                "error",
                "Native lowering requires at least one *ELEMENT declaration.",
            )
        )
        return None, None, None, None
    for block in element_blocks:
        definition = block.definition
        if (
            definition.family != "continuum_solid"
            or definition.physics != "solid_mechanics"
            or definition.solver_capability != "native_lagrange_analogue"
        ):
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-ELEMENT-002",
                    "error",
                    f"{definition.source_type} is not a verified displacement-based "
                    "native analogue for this lowering route.",
                    block.location,
                )
            )
    kinematics = {item.kinematics for item in definitions}
    topologies = {item.topology for item in definitions}
    interpolation = {item.interpolation for item in definitions}
    if len(kinematics) != 1 or len(topologies) != 1 or len(interpolation) != 1:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-ELEMENT-003",
                "error",
                "The first native route requires one kinematics, topology, and "
                "interpolation family across all element blocks.",
            )
        )
        return None, None, None, None
    selected = definitions[0]
    if selected.kinematics == "three_dimensional":
        dimension, assumption = 3, None
    elif selected.kinematics in {"plane_stress", "plane_strain", "axisymmetric"}:
        dimension, assumption = 2, selected.kinematics
    else:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-ELEMENT-004",
                "error",
                f"Kinematics {selected.kinematics!r} has no public static-solid "
                "lowering in this route.",
            )
        )
        dimension, assumption = None, None
    degree = 2 if selected.interpolation == "quadratic" else 1
    return dimension, assumption, selected.topology, degree


def _parse_boundaries(plan, dimension, region_aliases, findings):
    accepted = []
    for asset in plan.pending_assets:
        if asset.keyword != "*BOUNDARY":
            continue
        if asset.options.get("AMPLITUDE"):
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-BC-001",
                    "error",
                    "Amplitude-driven *BOUNDARY requires an explicit amplitude "
                    "lowering rather than a final-value substitution.",
                    asset.location,
                )
            )
            continue
        if asset.options.get("OP"):
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-BC-007",
                    "error",
                    "*BOUNDARY, OP=... changes inherited boundary state and needs "
                    "the multi-Step lifecycle rather than row-wise accumulation.",
                    asset.location,
                )
            )
            continue
        unsupported_flags = set(asset.flags) - {"USER"}
        if unsupported_flags or "USER" in asset.flags or asset.options.get("TYPE"):
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-BC-002",
                    "error",
                    "This route accepts only ordinary displacement *BOUNDARY rows.",
                    asset.location,
                )
            )
            continue
        for row in asset.rows:
            if len(row) < 2:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-BC-003",
                        "error",
                        f"Incomplete *BOUNDARY row: {row!r}.",
                        asset.location,
                    )
                )
                continue
            source_region = row[0].upper()
            region = region_aliases.get(source_region)
            if region is None:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-BC-004",
                        "error",
                        f"*BOUNDARY target {row[0]!r} is not a preserved NSET. "
                        "Direct node-label constraints are not inferred.",
                        asset.location,
                    )
                )
                continue
            try:
                first = int(row[1])
                last = int(row[2]) if len(row) >= 3 else first
                value = float(row[3]) if len(row) >= 4 else 0.0
            except ValueError:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-BC-005",
                        "error",
                        f"Non-numeric *BOUNDARY row: {row!r}.",
                        asset.location,
                    )
                )
                continue
            if dimension is None or first < 1 or last < first or last > dimension:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-BC-006",
                        "error",
                        f"Displacement dof range {first}:{last} is incompatible "
                        f"with model dimension {dimension}.",
                        asset.location,
                    )
                )
                continue
            accepted.append(
                AbaqusNativeBoundary(
                    region=region,
                    components=tuple(range(first - 1, last)),
                    value=value,
                    step=asset.step,
                    location=asset.location,
                )
            )
    # Abaqus can repeat inherited conditions across declarations.  Emit one
    # native condition per dof and refuse contradictory values instead of
    # relying on backend ordering of overlapping Dirichlet constraints.
    selected = []
    assigned: dict[tuple[str, int], float] = {}
    for item in accepted:
        remaining = []
        for component in item.components:
            key = (item.region, component)
            previous = assigned.get(key)
            if previous is None:
                assigned[key] = item.value
                remaining.append(component)
            elif not math.isclose(previous, item.value):
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-BC-008",
                        "error",
                        f"Conflicting prescribed values for {item.region} "
                        f"component {component + 1}: {previous} and {item.value}.",
                        item.location,
                    )
                )
        if remaining:
            selected.append(
                AbaqusNativeBoundary(
                    region=item.region,
                    components=tuple(remaining),
                    value=item.value,
                    step=item.step,
                    location=item.location,
                )
            )
    return tuple(selected)


def _parse_pressures(plan, surface_aliases, findings):
    accepted = []
    for asset in plan.pending_assets:
        if asset.category != "load":
            continue
        if asset.keyword == "*DLOAD" and all(
            len(row) >= 2 and row[1].upper() == "GRAV" for row in asset.rows
        ):
            continue
        if asset.keyword != "*DSLOAD":
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-LOAD-001",
                    "error",
                    f"{asset.keyword} is preserved but not lowered by the first "
                    "native route.",
                    asset.location,
                )
            )
            continue
        if asset.options.get("AMPLITUDE"):
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-LOAD-002",
                    "error",
                    "Amplitude-driven *DSLOAD requires an explicit amplitude lowering.",
                    asset.location,
                )
            )
            continue
        for row in asset.rows:
            if len(row) < 3 or row[1].upper() != "P":
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-LOAD-003",
                        "error",
                        f"Only *DSLOAD rows of the form surface, P, magnitude are "
                        f"accepted; received {row!r}.",
                        asset.location,
                    )
                )
                continue
            source_surface = row[0].upper()
            surface = surface_aliases.get(source_surface)
            if surface is None:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-LOAD-004",
                        "error",
                        f"*DSLOAD references unknown SURFACE {row[0]!r}.",
                        asset.location,
                    )
                )
                continue
            try:
                value = float(row[2])
            except ValueError:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-LOAD-005",
                        "error",
                        f"Pressure magnitude is not numeric: {row[2]!r}.",
                        asset.location,
                    )
                )
                continue
            accepted.append(
                AbaqusNativePressure(
                    surface=surface,
                    value=value,
                    step=asset.step or "<unspecified>",
                    location=asset.location,
                )
            )
    return tuple(accepted)


def _parse_gravities(plan, dimension, section_region, findings):
    accepted = []
    for asset in plan.pending_assets:
        if asset.keyword != "*DLOAD":
            continue
        if asset.options.get("AMPLITUDE"):
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-GRAV-001",
                    "error",
                    "Amplitude-driven GRAV requires explicit amplitude lowering.",
                    asset.location,
                )
            )
            continue
        for row in asset.rows:
            if len(row) < 6 or row[1].upper() != "GRAV":
                continue
            if dimension != 3:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-GRAV-002",
                        "error",
                        "The first GRAV lowering is restricted to three-dimensional "
                        "solids; axisymmetric component conventions require a "
                        "separate reviewed route.",
                        asset.location,
                    )
                )
                continue
            if section_region is None or row[0].upper() != section_region.upper():
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-GRAV-003",
                        "error",
                        "The first GRAV lowering requires its ELSET to equal the "
                        "single homogeneous Section region.",
                        asset.location,
                    )
                )
                continue
            try:
                magnitude = float(row[2])
                direction = tuple(float(value) for value in row[3:6])
            except ValueError:
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-GRAV-004",
                        "error",
                        f"Non-numeric GRAV row: {row!r}.",
                        asset.location,
                    )
                )
                continue
            norm = math.sqrt(sum(value * value for value in direction))
            if not math.isclose(norm, 1.0, rel_tol=1.0e-8, abs_tol=1.0e-10):
                findings.append(
                    _finding(
                        "AFM-ABAQUS-LOWER-GRAV-005",
                        "error",
                        "Abaqus GRAV direction cosines must form a unit vector.",
                        asset.location,
                    )
                )
                continue
            accepted.append(
                AbaqusNativeGravity(
                    acceleration=tuple(magnitude * value for value in direction),
                    step=asset.step or "<unspecified>",
                    location=asset.location,
                )
            )
    return tuple(accepted)


def assess(plan: abaqus_migration.AbaqusMigrationPlan) -> AbaqusNativeLoweringAssessment:
    """Assess the first reviewed linear-static Abaqus lowering route."""

    findings = [item for item in plan.issues if item.severity == "error"]
    instance = _supported_instance(plan, findings)
    dimension, assumption, topology, degree = _kinematics(
        plan.element_blocks, findings
    )

    candidates = {
        item.name.upper(): item
        for item in plan.materials
        if item.translation_status == "native_candidate"
    }
    if not plan.materials or len(candidates) != len(plan.materials):
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-MATERIAL-001",
                "error",
                "Native lowering requires every assigned material to be a complete "
                "constant isotropic elastic material with density.",
            )
        )
        material, material_name = {}, None
    elif len(candidates) == 1:
        selected_material = next(iter(candidates.values()))
        material = dict(selected_material.native_candidate)
        material_name = selected_material.name
    else:
        material, material_name = {}, None

    allowed_section_scopes = {"model"}
    if instance is not None:
        allowed_section_scopes.add(f"part:{instance.part}")
    valid_sections = [
        item
        for item in plan.sections
        if item.status == "references_resolved"
        and item.scope in allowed_section_scopes
        and item.section_type == "*SOLID SECTION"
    ]
    if not plan.sections or len(valid_sections) != len(plan.sections):
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-SECTION-001",
                "error",
                "Native lowering requires homogeneous *SOLID SECTION declarations "
                "with resolved ELSET and material references in the selected scope.",
            )
        )
    section_regions = [(item.region or "").upper() for item in valid_sections]
    if len(section_regions) != len(set(section_regions)):
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-SECTION-004",
                "error",
                "Each lowered ELSET must have exactly one SOLID SECTION assignment.",
            )
        )
    declared_regions = {(item.region or "").upper() for item in plan.element_blocks}
    assigned_regions = set(section_regions)
    if declared_regions != assigned_regions:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-SECTION-002",
                "error",
                "SOLID SECTION regions must exactly cover the ELSET declared by "
                "every *ELEMENT block; implicit subset inheritance is not guessed.",
            )
        )
    material_assignments = []
    for section in valid_sections:
        if dimension in {2} and assumption in {"plane_stress", "plane_strain"}:
            rows = section.rows
            if rows:
                try:
                    unit_thickness = (
                        len(rows) == 1
                        and len(rows[0]) == 1
                        and math.isclose(float(rows[0][0]), 1.0)
                    )
                except ValueError:
                    unit_thickness = False
                if not unit_thickness:
                    findings.append(
                        _finding(
                            "AFM-ABAQUS-LOWER-SECTION-003",
                            "error",
                            "A non-unit or non-scalar two-dimensional Section "
                            "thickness cannot be discarded by a per-unit-thickness "
                            "native formulation.",
                            section.location,
                        )
                    )
        selected = candidates.get((section.material or "").upper())
        if selected is not None and section.region is not None:
            material_assignments.append(
                AbaqusNativeMaterialAssignment(
                    region=section.region.upper(),
                    material_name=selected.name,
                    material=dict(selected.native_candidate),
                    location=section.location,
                )
            )

    steps = [item for item in plan.pending_assets if item.keyword == "*STEP"]
    procedures = [
        item for item in plan.pending_assets if item.category == "procedure"
    ]
    step_name = steps[0].step if len(steps) == 1 else None
    if len(steps) != 1 or len(procedures) != 1 or procedures[0].keyword != "*STATIC":
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-STEP-001",
                "error",
                "The first native route requires exactly one *STATIC Step.",
            )
        )
    elif steps[0].flags or set(steps[0].options) - {"NAME", "NLGEOM", "INC"}:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-STEP-003",
                "error",
                "The *STEP declaration contains unsupported procedure semantics.",
                steps[0].location,
            )
        )
    elif procedures[0].flags or procedures[0].options:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-STEP-004",
                "error",
                "The first route does not reinterpret *STATIC options such as "
                "stabilization or continuation controls.",
                procedures[0].location,
            )
        )
    elif steps[0].options.get("NLGEOM", "NO").upper() not in {"NO", "OFF"}:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-STEP-002",
                "error",
                "NLGEOM requires a reviewed finite-strain formulation and cannot "
                "enter the linear-static lowering.",
                steps[0].location,
            )
        )
    elif procedures[0].rows:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-STEP-101",
                "warning",
                "Abaqus static incrementation controls are retained in lowering.json; "
                "the linear native draft computes the same final load state in one solve.",
                procedures[0].location,
            )
        )

    allowed = {
        "step",
        "procedure",
        "boundary_condition",
        "load",
        "region_definition",
        "output_request",
        "solver_control",
        "restart",
    }
    for asset in plan.pending_assets:
        if asset.category not in allowed:
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-ASSET-001",
                    "error",
                    f"{asset.keyword} ({asset.category}) has no declared meaning in "
                    "the first native route.",
                    asset.location,
                )
            )
        elif asset.category in {"output_request", "solver_control", "restart"}:
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-ASSET-101",
                    "warning",
                    f"{asset.keyword} is preserved as source evidence; the native "
                    "draft uses AgentFEM's result lifecycle instead.",
                    asset.location,
                )
            )

    nsets = _region_aliases(plan, kind="nset", instance=instance)
    surfaces = {}
    for item in plan.pending_assets:
        if item.keyword != "*SURFACE":
            continue
        name = item.options.get("NAME", "").upper()
        if not name:
            continue
        if item.options.get("TYPE", "ELEMENT").upper() != "ELEMENT":
            findings.append(
                _finding(
                    "AFM-ABAQUS-LOWER-SURFACE-001",
                    "error",
                    f"SURFACE {name!r} is not element-based and cannot be "
                    "reconstructed as a native exterior facet set.",
                    item.location,
                )
            )
            continue
        if instance is None and item.scope == "model":
            surfaces[name] = name
        elif instance is not None:
            if item.scope == f"part:{instance.part}":
                surfaces[name] = name
                surfaces[f"{instance.name.upper()}.{name}"] = name
            elif item.scope in {instance.scope, f"assembly:{instance.assembly}"}:
                surfaces[name] = name
    boundaries = _parse_boundaries(plan, dimension, nsets, findings)
    pressures = _parse_pressures(plan, surfaces, findings)
    section_region = plan.sections[0].region if len(plan.sections) == 1 else None
    gravities = _parse_gravities(plan, dimension, section_region, findings)
    if not boundaries:
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-MODEL-001",
                "error",
                "A native static draft requires at least one accepted displacement "
                "boundary condition.",
            )
        )
    if (
        not pressures
        and not gravities
        and not any(item.value != 0.0 for item in boundaries)
    ):
        findings.append(
            _finding(
                "AFM-ABAQUS-LOWER-MODEL-101",
                "warning",
                "The accepted source assets contain no nonzero displacement or pressure.",
            )
        )

    return AbaqusNativeLoweringAssessment(
        source_fingerprint=plan.source_graph.fingerprint,
        dimension=dimension,
        assumption=assumption,
        topology=topology,
        degree=degree,
        material=material,
        material_name=material_name,
        material_assignments=tuple(material_assignments),
        step_name=step_name,
        boundaries=boundaries,
        pressures=pressures,
        gravities=gravities,
        findings=tuple(findings),
    )


def _transform_point(point, positioning):
    values = [float(value) for value in point]
    while len(values) < 3:
        values.append(0.0)
    translation = None
    rotation = None
    for row in positioning:
        if len(row) == 3:
            translation = tuple(float(value) for value in row)
        elif len(row) == 7:
            rotation = tuple(float(value) for value in row)
    if translation is not None:
        values = [values[index] + translation[index] for index in range(3)]
    if rotation is None:
        return tuple(values)
    axis_start = rotation[:3]
    axis_end = rotation[3:6]
    angle = math.radians(rotation[6])
    direction = [axis_end[index] - axis_start[index] for index in range(3)]
    norm = math.sqrt(sum(value * value for value in direction))
    if norm == 0.0:
        raise ValueError("Abaqus instance rotation axis has zero length.")
    unit = [value / norm for value in direction]
    vector = [values[index] - axis_start[index] for index in range(3)]
    cross = (
        unit[1] * vector[2] - unit[2] * vector[1],
        unit[2] * vector[0] - unit[0] * vector[2],
        unit[0] * vector[1] - unit[1] * vector[0],
    )
    dot = sum(unit[index] * vector[index] for index in range(3))
    cosine, sine = math.cos(angle), math.sin(angle)
    return tuple(
        axis_start[index]
        + vector[index] * cosine
        + cross[index] * sine
        + unit[index] * dot * (1.0 - cosine)
        for index in range(3)
    )


def _strip_instance_option(line: str) -> str:
    fields = [item.strip() for item in line.split(",")]
    return ", ".join(
        item for item in fields if not item.upper().startswith("INSTANCE=")
    )


def _strip_instance_reference(value: str, instance_name: str) -> str:
    prefix = f"{instance_name.upper()}."
    selected = value.strip()
    return selected[len(prefix) :] if selected.upper().startswith(prefix) else selected


def _single_instance_mesh_source(source: Path, plan, instance) -> str:
    """Create a derived orphan mesh for one positioned Part instance."""

    output = ["*Heading", "** AgentFEM derived one-instance orphan mesh"]
    part_name = None
    instance_name = None
    active_keyword = ""
    keep = False
    selected_scope = None
    mesh_keywords = {"*NODE", "*ELEMENT", "*NSET", "*ELSET", "*SURFACE"}
    for _location, raw in abaqus_migration.expanded_lines(source):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            keyword, options, _flags = abaqus_migration._keyword(line)
            active_keyword = keyword
            keep = False
            if keyword == "*PART":
                part_name = options.get("NAME")
                selected_scope = "part"
                continue
            if keyword == "*END PART":
                part_name = None
                selected_scope = None
                continue
            if keyword == "*INSTANCE":
                instance_name = options.get("NAME")
                selected_scope = "instance"
                continue
            if keyword == "*END INSTANCE":
                instance_name = None
                selected_scope = "assembly"
                continue
            if keyword == "*ASSEMBLY":
                selected_scope = "assembly"
                continue
            if keyword == "*END ASSEMBLY":
                selected_scope = None
                continue
            selected_part = (
                selected_scope == "part"
                and part_name is not None
                and part_name.upper() == instance.part.upper()
            )
            selected_instance = (
                selected_scope == "instance"
                and instance_name is not None
                and instance_name.upper() == instance.name.upper()
            )
            selected_assembly = selected_scope == "assembly"
            if keyword in mesh_keywords and (
                selected_part or selected_instance or selected_assembly
            ):
                keep = True
                output.append(_strip_instance_option(line))
            continue
        if not keep:
            continue
        values = [item.strip() for item in line.split(",")]
        if active_keyword == "*NODE":
            if len(values) < 3:
                raise ValueError(f"Invalid Abaqus node row in positioned Part: {line}")
            point = _transform_point(values[1:], instance.positioning)
            output.append(
                ", ".join((values[0], *(format(value, ".17g") for value in point)))
            )
        elif active_keyword in {"*NSET", "*ELSET"}:
            output.append(
                ", ".join(
                    _strip_instance_reference(value, instance.name) for value in values
                )
            )
        elif active_keyword == "*SURFACE":
            values[0] = _strip_instance_reference(values[0], instance.name)
            output.append(", ".join(values))
        else:
            output.append(line)
    return "\n".join(output) + "\n"


def _solver_mesh_source(source: Path, plan) -> str:
    if not plan.parts and not plan.instances:
        return "\n".join(
            raw for _location, raw in abaqus_migration.expanded_lines(source)
        ) + "\n"
    instance = plan.instances[0]
    return _single_instance_mesh_source(source, plan, instance)


def _native_case_source(assessment, *, source_entry):
    assumption = (
        ""
        if assessment.assumption is None
        else f",\n        assumption={assessment.assumption!r}"
    )
    lines = [
        '"""Reviewed native AgentFEM draft lowered from an Abaqus input deck.\n\n',
        'See lowering.json for reviewer, units, decisions, and limitations.\n"""',
        "",
        "from pathlib import Path",
        "",
        "from mpi4py import MPI",
        "",
        "from agentfem import fields, mesh, models, project, studies",
        "from agentfem.constitutive import elasticity",
        "",
        "",
        "PROJECT_ROOT = Path(__file__).resolve().parent",
        f"ABAQUS_SOURCE = PROJECT_ROOT / {source_entry!r}",
        "",
        "",
        "def main():",
        "    run = project.current_run(project_root=PROJECT_ROOT)",
        "    cell = mesh.read_abaqus_mesh(",
        "        ABAQUS_SOURCE,",
        '        PROJECT_ROOT / "mesh" / "native.xdmf",',
        "        comm=MPI.COMM_WORLD,",
        f"        cell_type={assessment.topology!r},",
        "    )",
        "    study = studies.static_solid(",
        f"        dimension={assessment.dimension}{assumption},",
        f"        name={assessment.step_name!r},",
        "    )",
        '    model = models.create(study=study, mesh=cell, name="migrated_model")',
        f"    displacement = model.field(fields.displacement(cell.domain, degree={assessment.degree}))",
    ]
    for assignment in assessment.material_assignments:
        material = assignment.material
        lines.extend(
            (
                "    model.material(",
                "        elasticity.isotropic_elastic(",
                f"            young={material['young']!r},",
                f"            poisson={material['poisson']!r},",
                f"            density={material['density']!r},",
                f"            name={assignment.material_name!r},",
                "        ),",
                f"        region=cell.element_set({assignment.region!r}),",
                "    )",
            )
        )
    for index, item in enumerate(assessment.boundaries, 1):
        lines.extend(
            (
                "    model.fix(",
                "        displacement,",
                f"        on=cell.node_set({item.region!r}),",
                f"        components={item.components!r},",
                f"        value={item.value!r},",
                f'        name="abaqus_boundary_{index}",',
                "    )",
            )
        )
    for index, item in enumerate(assessment.pressures, 1):
        lines.extend(
            (
                "    model.pressure(",
                f"        {item.value!r},",
                f"        on=cell.boundary({item.surface!r}, tag={100 + index}),",
                f'        name="abaqus_pressure_{index}",',
                "    )",
            )
        )
    for index, item in enumerate(assessment.gravities, 1):
        lines.extend(
            (
                "    model.gravity(",
                f"        {item.acceleration!r},",
                f'        name="abaqus_gravity_{index}",',
                "    )",
            )
        )
    lines.extend(
        (
            "    model.check()",
            "    simulation = model.step(",
            "        target=displacement,",
            f"        name={assessment.step_name!r},",
            '        output=run.artifact("fields.xdmf"),',
            "    ).solve_result()",
            '    simulation.add_dof_statistics(displacement, prefix="displacement")',
            "    if MPI.COMM_WORLD.rank == 0:",
            "        run.publish(simulation)",
            "        print(simulation.format())",
            "    MPI.COMM_WORLD.barrier()",
            "    return simulation",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        )
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _activate_entrypoint(configuration: Path, entrypoint: str) -> None:
    text = configuration.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^entrypoint\s*=\s*["\'][^"\']+["\']\s*$',
        f'entrypoint = "{entrypoint}"',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("agentfem.toml must contain exactly one project entrypoint.")
    _atomic_text(configuration, updated)


def lower_project(
    project: str | Path,
    *,
    reviewed_by: str,
    unit_system: str,
    activate: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Emit, and optionally activate, an eligible reviewed native draft."""

    reviewer = str(reviewed_by).strip()
    units = str(unit_system).strip()
    if not reviewer or not units:
        raise ValueError("Native lowering requires reviewed_by and unit_system.")
    root = Path(project).expanduser().resolve()
    migration_path = root / "migration.json"
    if not migration_path.is_file():
        raise FileNotFoundError(f"No migration.json found in {root}.")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    source_entry = root / migration["project"]["source_entrypoint"]
    plan = abaqus_migration.plan(source_entry)
    expected = migration["source_graph"]["fingerprint"]
    if plan.source_graph.fingerprint != expected:
        raise ValueError(
            "Copied Abaqus sources changed after migration planning. Recreate the "
            "migration project before lowering."
        )
    assessment = assess(plan)
    if not assessment.eligible:
        raise AbaqusNativeLoweringBlocked(assessment)

    case_path = root / "case.native.py"
    lowering_path = root / "lowering.json"
    if not force and (case_path.exists() or lowering_path.exists()):
        raise FileExistsError(
            "Native lowering artifacts already exist. Pass --force only after review."
        )
    mesh_directory = root / "mesh"
    mesh_directory.mkdir(exist_ok=True)
    expanded_path = mesh_directory / "abaqus-expanded.inp"
    expanded_text = _solver_mesh_source(source_entry, plan)
    _atomic_text(expanded_path, expanded_text)

    assessment_record = assessment.summary()
    review_identity = {
        "reviewed_by": reviewer,
        "unit_system": units,
        "source_values_reinterpreted": False,
        "claim": "native_analogue_not_abaqus_solver_equivalence",
    }
    review = {
        **review_identity,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    content = json.dumps(
        {"assessment": assessment_record, "review": review_identity},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record = {
        "schema": "agentfem.abaqus-native-lowering",
        "schema_version": "0.1.0",
        "status": "activated" if activate else "drafted",
        "source_fingerprint": assessment.source_fingerprint,
        "expanded_source": expanded_path.relative_to(root).as_posix(),
        "expanded_source_sha256": sha256(expanded_text.encode("utf-8")).hexdigest(),
        "review": review,
        "assessment": assessment_record,
        "decision_fingerprint": sha256(content).hexdigest(),
        "entrypoint": case_path.name,
    }
    _atomic_text(
        case_path,
        _native_case_source(
            assessment,
            source_entry=expanded_path.relative_to(root).as_posix(),
        ),
    )
    _atomic_text(
        lowering_path,
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    if activate:
        _activate_entrypoint(root / "agentfem.toml", case_path.name)
    return {
        "schema": "agentfem.abaqus-native-lowering-result",
        "schema_version": "0.1.0",
        "status": record["status"],
        "project": str(root),
        "entrypoint": str(case_path),
        "lowering_record": str(lowering_path),
        "decision_fingerprint": record["decision_fingerprint"],
    }
