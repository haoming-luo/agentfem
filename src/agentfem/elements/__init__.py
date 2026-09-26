# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite-element identity and discretization compatibility contracts.

Basix and DOLFINx continue to own basis construction, dof maps, quadrature,
and assembly. This module owns the smaller AgentFEM boundary between an
engineering field and those backend objects: what element is actually in use,
whether it belongs to the registered mesh, and whether its value shape is
compatible with the declared Study.

The contract deliberately does not translate source element names into
numerical formulations. A C3D8R, shell, beam, hybrid, cohesive, or other
specialized source element still needs an explicit Step provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agentfem._model_support import domain as model_domain
from agentfem.mesh.compatibility import TopologyCompatibility, describe_topology
from agentfem.validation import ValidationIssue, ValidationReport, issue


@dataclass(frozen=True)
class ElementIdentity:
    """Backend-readable identity of one scalar, vector, tensor, or mixed element."""

    family: str
    cell: str | None
    degree: int | tuple[int, ...] | None
    value_shape: tuple[int, ...]
    sobolev_space: str | None
    discontinuous: bool
    mixed: bool
    subelements: tuple["ElementIdentity", ...] = ()
    signature: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "family": self.family,
            "cell": self.cell,
            "degree": self.degree,
            "value_shape": self.value_shape,
            "sobolev_space": self.sobolev_space,
            "discontinuous": self.discontinuous,
            "mixed": self.mixed,
            "subelements": tuple(item.summary() for item in self.subelements),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class FieldDiscretization:
    """One model field bound to its actual finite-element space."""

    name: str
    kind: str
    element: ElementIdentity
    mesh_cell: str
    mesh_matches_model: bool

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "mesh_cell": self.mesh_cell,
            "mesh_matches_model": self.mesh_matches_model,
            "element": self.element.summary(),
        }


@dataclass(frozen=True)
class DiscretizationAudit:
    """Inspectable mesh/element/Study preflight with optional quality evidence."""

    topology: TopologyCompatibility | None
    fields: tuple[FieldDiscretization, ...]
    validation: ValidationReport
    quality: object | None = None

    @property
    def acceptable(self) -> bool:
        quality_ok = self.quality is None or bool(
            getattr(self.quality, "acceptable", False)
        )
        return self.validation.is_valid and quality_ok

    def summary(self) -> dict[str, object]:
        return {
            "kind": "discretization_audit",
            "acceptable": self.acceptable,
            "topology": None if self.topology is None else self.topology.summary(),
            "fields": tuple(item.summary() for item in self.fields),
            "validation": self.validation.as_dict(),
            "quality": (
                None
                if self.quality is None
                else getattr(self.quality, "summary", lambda: self.quality)()
            ),
        }


def describe_element(element_or_space) -> ElementIdentity:
    """Describe a UFL element or a function space without constructing forms."""

    element = _element(element_or_space)
    subelements = tuple(
        describe_element(item) for item in tuple(getattr(element, "sub_elements", ()))
    )
    family = str(
        _value(getattr(element, "family_name", None)) or type(element).__name__
    )
    degree = _normal_degree(_value(getattr(element, "degree", None)))
    value_shape = tuple(
        int(value)
        for value in (_value(getattr(element, "reference_value_shape", ())) or ())
    )
    sobolev = _value(getattr(element, "sobolev_space", None))
    sobolev_name = None if sobolev is None else str(sobolev)
    normalized_family = family.lower()
    discontinuous = (
        "discontinuous" in normalized_family
        or normalized_family in {"dg", "dpc"}
        or (sobolev_name is not None and "l2" in sobolev_name.lower())
    )
    mixed = "mixed" in normalized_family
    return ElementIdentity(
        family=family,
        cell=_element_cell(element),
        degree=degree,
        value_shape=value_shape,
        sobolev_space=sobolev_name,
        discontinuous=discontinuous,
        # UFL blocked vector/tensor elements also expose scalar sub-elements.
        # Only the explicit mixed family owns independent field blocks.
        mixed=mixed,
        subelements=subelements,
        signature=str(element),
    )


def describe_field(field, *, registered_mesh=None) -> FieldDiscretization:
    """Describe the runtime discretization of one AgentFEM or DOLFINx field."""

    space = _space(field)
    field_mesh = getattr(space, "mesh", None)
    if field_mesh is None:
        raise TypeError("A field discretization requires a function space with a mesh.")
    topology = getattr(field_mesh, "topology", None)
    mesh_cell = _mesh_cell(topology)
    return FieldDiscretization(
        name=str(
            getattr(
                field,
                "name",
                getattr(getattr(field, "value", None), "name", "field"),
            )
        ),
        kind=str(getattr(field, "kind", "field")),
        element=describe_element(space),
        mesh_cell=mesh_cell,
        mesh_matches_model=registered_mesh is None or field_mesh is registered_mesh,
    )


def audit(
    model,
    *,
    check_quality: bool = False,
    quality_threshold: float = 0.1,
    reject_poor_quality: bool = False,
) -> DiscretizationAudit:
    """Audit mesh topology, field elements, Study shapes, and mesh quality.

    The default path is backend-light and suitable for ``Model.validate``.
    ``check_quality=True`` performs the collective cell-level calculation and
    is intended for explicit preflight, release evidence, and long runs.
    """

    domain = model_domain(getattr(model, "mesh", None))
    issues: list[ValidationIssue] = []
    topology = None
    field_records: list[FieldDiscretization] = []
    quality_report = None

    if domain is not None:
        topology_object = getattr(domain, "topology", None)
        cell_name = _mesh_cell(topology_object)
        if cell_name:
            topology = describe_topology(cell_name)
            if not topology.inspectable:
                issues.append(
                    issue(
                        "AFM-DISCRETIZATION-001",
                        "model.mesh.topology",
                        f"Runtime cell topology {cell_name!r} is not supported.",
                        hint="Convert or split the mesh into a declared solver topology.",
                        topology=topology.summary(),
                    )
                )
            elif not topology.release_ready:
                issues.append(
                    issue(
                        "AFM-DISCRETIZATION-002",
                        "model.mesh.topology",
                        f"Runtime cell topology {cell_name!r} has conditional maturity.",
                        severity="warning",
                        hint=(
                            "Require an analysis-specific provider and benchmark before "
                            "treating this route as release evidence."
                        ),
                        topology=topology.summary(),
                    )
                )

        for index, field in enumerate(getattr(model, "fields", ())):
            if not _has_element(field):
                continue
            try:
                record = describe_field(field, registered_mesh=domain)
            except (AttributeError, TypeError, ValueError) as exc:
                issues.append(
                    issue(
                        "AFM-DISCRETIZATION-003",
                        f"model.fields[{index}]",
                        f"The field element cannot be inspected: {exc}",
                        hint="Register a DOLFINx function space or AgentFEM unknown.",
                    )
                )
                continue
            field_records.append(record)
            issues.extend(
                _field_issues(
                    record,
                    index=index,
                    study=getattr(model, "study", None),
                )
            )

        if check_quality and topology is not None and topology.inspectable:
            from agentfem.mesh.quality import audit as audit_quality

            try:
                quality_report = audit_quality(
                    domain,
                    threshold=float(quality_threshold),
                    strict=False,
                )
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                issues.append(
                    issue(
                        "AFM-MESH-QUALITY-001",
                        "model.mesh.quality",
                        f"Mesh quality could not be established: {exc}",
                        hint="Repair or convert the mesh before a trusted long run.",
                    )
                )
            else:
                if not quality_report.valid:
                    issues.append(
                        issue(
                            "AFM-MESH-QUALITY-002",
                            "model.mesh.quality",
                            (
                                "The mesh contains "
                                f"{quality_report.invalid_cells} invalid or folded cells."
                            ),
                            hint="Repair the invalid cells; do not start the solver.",
                            quality=quality_report.summary(),
                        )
                    )
                elif quality_report.poor_cells:
                    issues.append(
                        issue(
                            "AFM-MESH-QUALITY-003",
                            "model.mesh.quality",
                            (
                                f"{quality_report.poor_cells} cells fall below the "
                                f"declared quality threshold {quality_report.threshold:g}."
                            ),
                            severity="error" if reject_poor_quality else "warning",
                            hint=(
                                "Refine or improve the mesh, or justify a "
                                "task-specific threshold."
                            ),
                            quality=quality_report.summary(),
                        )
                    )

    report = ValidationReport.from_issues(
        issues,
        scope=f"discretization:{getattr(model, 'name', 'model')}",
    )
    return DiscretizationAudit(
        topology=topology,
        fields=tuple(field_records),
        validation=report,
        quality=quality_report,
    )


def _lightweight_issues(model) -> tuple[ValidationIssue, ...]:
    """Return only metadata-level issues for the common Model validator."""

    # Model validation already owns the generic cross-mesh field diagnostic.
    # Keep the standalone audit richer without duplicating that issue in the
    # common report.
    return tuple(
        item
        for item in audit(model, check_quality=False).validation.issues
        if item.code != "AFM-DISCRETIZATION-004"
    )


def _field_issues(
    record: FieldDiscretization,
    *,
    index: int,
    study,
) -> Iterable[ValidationIssue]:
    path = f"model.fields[{index}]"
    if not record.mesh_matches_model:
        yield issue(
            "AFM-DISCRETIZATION-004",
            f"{path}.space.mesh",
            "The field element belongs to a different mesh from the model.",
            hint="Recreate the field on model.mesh.",
        )
    element_cells = tuple(sorted(set(_leaf_cells(record.element))))
    if element_cells and element_cells != (record.mesh_cell,):
        yield issue(
            "AFM-DISCRETIZATION-005",
            f"{path}.space.element",
            (
                f"Element cell identity {element_cells!r} does not match "
                f"runtime mesh cell {record.mesh_cell!r}."
            ),
            hint="Rebuild the function space on the registered mesh topology.",
            element=record.element.summary(),
        )
    if study is None:
        return
    dimension = int(getattr(study, "dimension", 0) or 0)
    physics = str(getattr(study, "physics", ""))
    kind = record.kind.lower()
    if physics == "solid_mechanics" and kind in {
        "displacement",
        "displacement_subfield",
    }:
        shape = record.element.value_shape
        if shape != (dimension,):
            yield issue(
                "AFM-DISCRETIZATION-006",
                f"{path}.space.element.value_shape",
                (
                    f"A {dimension}D solid displacement requires value shape "
                    f"({dimension},), received {shape!r}."
                ),
                hint="Create the displacement field with the Study dimension.",
            )
    if (
        physics == "heat_transfer"
        and kind in {"temperature", "scalar_unknown"}
        and record.element.value_shape
    ):
        yield issue(
            "AFM-DISCRETIZATION-007",
            f"{path}.space.element.value_shape",
            "A temperature primary field must be scalar.",
            hint="Use fields.temperature(...) or fields.scalar_unknown(...).",
        )


def _element(element_or_space):
    candidate = getattr(element_or_space, "ufl_element", None)
    return candidate() if callable(candidate) else element_or_space


def _space(field):
    space = getattr(field, "space", None)
    if space is not None:
        return space
    value = getattr(field, "value", field)
    space = getattr(value, "function_space", None)
    if space is None:
        raise TypeError("The field does not expose a function space.")
    return space


def _has_element(field) -> bool:
    try:
        space = _space(field)
    except TypeError:
        return False
    return callable(getattr(space, "ufl_element", None))


def _mesh_cell(topology) -> str:
    if topology is None:
        return ""
    name = getattr(topology, "cell_name", None)
    if callable(name):
        return str(name()).lower()
    cell_type = getattr(topology, "cell_type", None)
    return str(cell_type).split(".")[-1].lower() if cell_type is not None else ""


def _element_cell(element) -> str | None:
    cell = _value(getattr(element, "cell", None))
    if cell is None:
        return None
    name = getattr(cell, "cellname", None)
    if callable(name):
        return str(name()).lower()
    return str(cell).lower()


def _leaf_cells(element: ElementIdentity) -> Iterable[str]:
    if element.subelements:
        for subelement in element.subelements:
            yield from _leaf_cells(subelement)
    elif element.cell:
        yield element.cell


def _normal_degree(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _value(value):
    return value() if callable(value) else value


__all__ = [
    "DiscretizationAudit",
    "ElementIdentity",
    "FieldDiscretization",
    "audit",
    "describe_element",
    "describe_field",
]
