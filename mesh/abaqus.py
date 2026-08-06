"""Focused Abaqus keyword readers used by AgentFEM interoperability.

The mesh topology is delegated to :mod:`meshio`.  This module retains the
information that a generic mesh converter normally discards but scientific
constraints still need: Abaqus node labels and ``*EQUATION`` terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AbaqusNamedSet:
    """A source-labelled Abaqus node or element set."""

    name: str
    labels: tuple[int, ...]
    kind: str

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "count": len(self.labels),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class AbaqusSurfaceEntry:
    """One source set/element and face identifier in an Abaqus surface."""

    reference: str
    face: str | None = None


@dataclass(frozen=True)
class AbaqusSurface:
    """A node- or element-based Abaqus surface definition."""

    name: str
    surface_type: str
    entries: tuple[AbaqusSurfaceEntry, ...]

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "abaqus_surface",
            "surface_type": self.surface_type,
            "entries": [
                {"reference": item.reference, "face": item.face}
                for item in self.entries
            ],
        }


@dataclass(frozen=True)
class AbaqusModelSemantics:
    """Engineering names retained independently of neutral mesh topology."""

    node_sets: tuple[AbaqusNamedSet, ...] = ()
    element_sets: tuple[AbaqusNamedSet, ...] = ()
    surfaces: tuple[AbaqusSurface, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "node_sets": {item.name: item.summary() for item in self.node_sets},
            "element_sets": {
                item.name: item.summary() for item in self.element_sets
            },
            "surfaces": {item.name: item.summary() for item in self.surfaces},
        }


def read_model_semantics(path: str | Path) -> AbaqusModelSemantics:
    """Read named node/element sets and explicit surface entries."""

    source = Path(path)
    nsets: dict[str, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    surfaces: dict[str, tuple[str, list[AbaqusSurfaceEntry]]] = {}
    keyword, options, flags = "", {}, set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            parts = [item.strip() for item in line.split(",")]
            keyword, options, flags = parts[0].upper(), {}, set()
            for item in parts[1:]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    options[key.strip().upper()] = value.strip().upper()
                elif item:
                    flags.add(item.upper())
            if keyword == "*SURFACE":
                name = options.get("NAME")
                if not name:
                    raise ValueError(f"*SURFACE at {source}:{line_number} requires NAME=.")
                surfaces.setdefault(name, (options.get("TYPE", "ELEMENT"), []))
            continue
        values = _csv_values(line)
        if keyword == "*NODE" and options.get("NSET"):
            nsets.setdefault(options["NSET"], []).append(int(values[0]))
        elif keyword == "*ELEMENT" and options.get("ELSET"):
            elsets.setdefault(options["ELSET"], []).append(int(values[0]))
        elif keyword in {"*NSET", "*ELSET"}:
            option = keyword[1:]
            name = options.get(option)
            if not name:
                raise ValueError(f"{keyword} at {source}:{line_number} requires {option}=.")
            selected = (nsets if keyword == "*NSET" else elsets).setdefault(name, [])
            if "GENERATE" in flags:
                if len(values) not in {2, 3}:
                    raise ValueError(f"Invalid {keyword}, GENERATE at {source}:{line_number}.")
                start, end = int(values[0]), int(values[1])
                increment = 1 if len(values) == 2 else int(values[2])
                if increment <= 0 or end < start:
                    raise ValueError("Abaqus set generation requires a positive range.")
                selected.extend(range(start, end + 1, increment))
            else:
                selected.extend(int(value) for value in values)
        elif keyword == "*SURFACE":
            reference = values[0].upper()
            face = values[1].upper() if len(values) > 1 and values[1] else None
            surfaces[options["NAME"]][1].append(AbaqusSurfaceEntry(reference, face))

    def named(values, kind):
        return tuple(AbaqusNamedSet(name, tuple(dict.fromkeys(labels)), kind) for name, labels in sorted(values.items()))

    return AbaqusModelSemantics(
        named(nsets, "abaqus_node_set"), named(elsets, "abaqus_element_set"),
        tuple(AbaqusSurface(name, value[0], tuple(value[1])) for name, value in sorted(surfaces.items())),
    )


@dataclass(frozen=True)
class AbaqusElementDefinition:
    """Scientific identity attached to one Abaqus ``*ELEMENT`` type.

    Neutral mesh formats preserve topology and geometry, but normally discard
    formulation suffixes such as ``H``.  Keeping that distinction beside the
    converted mesh prevents a readable ``tetra10`` grid from being mistaken
    for a verified Abaqus hybrid formulation.
    """

    source_type: str
    topology: str | None = None
    interpolation: str | None = None
    formulation: str = "source_defined"
    pressure_interpolation: str | None = None
    additional_pressure_variables: int = 0
    additional_displacement_variables: int = 0

    @property
    def is_hybrid(self) -> bool:
        return self.pressure_interpolation is not None

    def summary(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "topology": self.topology,
            "interpolation": self.interpolation,
            "formulation": self.formulation,
            "pressure_interpolation": self.pressure_interpolation,
            "additional_pressure_variables": self.additional_pressure_variables,
            "additional_displacement_variables": self.additional_displacement_variables,
        }


_C3D10_FAMILY = {
    "C3D10": {
        "formulation": "displacement",
    },
    "C3D10H": {
        "formulation": "hybrid",
        "pressure_interpolation": "constant",
        "additional_pressure_variables": 1,
    },
    "C3D10HS": {
        "formulation": "hybrid_improved_surface_stress",
        "pressure_interpolation": "linear",
        "additional_pressure_variables": 4,
    },
    "C3D10M": {
        "formulation": "modified_hourglass_control",
        "additional_displacement_variables": 3,
    },
    "C3D10MH": {
        "formulation": "modified_hybrid_hourglass_control",
        "pressure_interpolation": "linear",
        "additional_pressure_variables": 4,
        "additional_displacement_variables": 3,
    },
}


def describe_element_type(element_type: str) -> AbaqusElementDefinition:
    """Describe formulation information that is lost in neutral mesh I/O."""

    selected = str(element_type).strip().upper()
    details = _C3D10_FAMILY.get(selected)
    if details is None:
        return AbaqusElementDefinition(source_type=selected)
    return AbaqusElementDefinition(
        source_type=selected,
        topology="tetra10",
        interpolation="quadratic",
        **details,
    )


def read_element_definitions(path: str | Path) -> tuple[AbaqusElementDefinition, ...]:
    """Read distinct Abaqus element declarations in source order.

    This intentionally reads keyword headers rather than inferring element
    semantics from the topology produced by meshio.
    """

    path = Path(path)
    selected: list[AbaqusElementDefinition] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("*") or line.startswith("**"):
            continue
        fields = [item.strip() for item in line.split(",")]
        if fields[0].upper() != "*ELEMENT":
            continue
        options = {}
        for item in fields[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            options[key.strip().upper()] = value.strip()
        element_type = options.get("TYPE")
        if element_type is None:
            raise ValueError(f"Abaqus *ELEMENT declaration in {path} has no TYPE= option.")
        normalized = element_type.upper()
        if normalized not in seen:
            selected.append(describe_element_type(normalized))
            seen.add(normalized)
    return tuple(selected)


@dataclass(frozen=True)
class AbaqusNodeTable:
    """Abaqus node labels and coordinates in source-file order."""

    labels: np.ndarray
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] != labels.size:
            raise ValueError("Abaqus node labels and coordinates must have equal length.")
        if coordinates.shape[1] not in {1, 2, 3}:
            raise ValueError("Abaqus node coordinates must have dimension 1, 2, or 3.")
        if np.unique(labels).size != labels.size:
            raise ValueError("Abaqus node labels must be unique.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Abaqus node coordinates must be finite.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "coordinates", coordinates)

    def index(self, label: int) -> int:
        """Return the source-order index of one node label."""

        matches = np.flatnonzero(self.labels == int(label))
        if matches.size != 1:
            raise KeyError(f"Abaqus node label {label} is not present.")
        return int(matches[0])

    def coordinate(self, label: int) -> np.ndarray:
        """Return a copy of one node coordinate."""

        return self.coordinates[self.index(label)].copy()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_node_table",
            "node_count": int(self.labels.size),
            "geometric_dimension": int(self.coordinates.shape[1]),
            "minimum": np.min(self.coordinates, axis=0).tolist(),
            "maximum": np.max(self.coordinates, axis=0).tolist(),
        }


@dataclass(frozen=True)
class EquationTerm:
    """One ``coefficient * nodal_dof`` term in an Abaqus equation."""

    node: int
    dof: int
    coefficient: float

    def __post_init__(self) -> None:
        if int(self.node) <= 0:
            raise ValueError("Abaqus equation node labels must be positive.")
        if int(self.dof) <= 0:
            raise ValueError("Abaqus equation dof numbers must be positive.")
        if not np.isfinite(float(self.coefficient)):
            raise ValueError("Abaqus equation coefficients must be finite.")


@dataclass(frozen=True)
class LinearEquation:
    """One homogeneous Abaqus ``*EQUATION`` constraint."""

    terms: tuple[EquationTerm, ...]

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise ValueError("An Abaqus equation requires at least two terms.")
        if self.terms[0].coefficient == 0.0:
            raise ValueError("The first Abaqus equation coefficient cannot be zero.")

    @property
    def slave(self) -> tuple[int, int]:
        """Return the node/dof eliminated by Abaqus/Standard."""

        return self.terms[0].node, self.terms[0].dof


@dataclass(frozen=True)
class AbaqusEquationSet:
    """A parsed collection of Abaqus linear constraint equations."""

    equations: tuple[LinearEquation, ...]
    source: Path | None = None

    def __post_init__(self) -> None:
        slaves = [equation.slave for equation in self.equations]
        if len(set(slaves)) != len(slaves):
            raise ValueError("An Abaqus dof is eliminated by more than one equation.")

    def summary(self) -> dict[str, object]:
        term_counts: dict[int, int] = {}
        component_counts: dict[int, int] = {}
        for equation in self.equations:
            term_counts[len(equation.terms)] = term_counts.get(len(equation.terms), 0) + 1
            component = equation.terms[0].dof
            component_counts[component] = component_counts.get(component, 0) + 1
        return {
            "kind": "abaqus_equation_set",
            "source": None if self.source is None else str(self.source),
            "equation_count": len(self.equations),
            "term_counts": term_counts,
            "slave_dofs_by_component": component_counts,
        }


@dataclass(frozen=True)
class AbaqusMeshImport:
    """DOLFINx mesh together with Abaqus labels and conversion evidence."""

    fem_mesh: object
    nodes: AbaqusNodeTable
    conversion: object

    @property
    def domain(self):
        return self.fem_mesh.domain

    @property
    def cell_tags(self):
        return self.fem_mesh.cell_tags

    @property
    def facet_tags(self):
        return self.fem_mesh.facet_tags

    @property
    def element_definitions(self) -> tuple[dict[str, object], ...]:
        values = self.conversion.source_metadata.get(
            "abaqus_element_definitions",
            (),
        )
        return tuple(dict(value) for value in values)

    @property
    def model_semantics(self) -> dict[str, object]:
        """Return preserved NSET/ELSET/SURFACE source semantics."""

        return dict(
            self.conversion.source_metadata.get("abaqus_model_semantics", {})
        )

    @property
    def node_sets(self) -> dict[str, object]:
        return dict(self.model_semantics.get("node_sets", {}))

    @property
    def element_sets(self) -> dict[str, object]:
        return dict(self.model_semantics.get("element_sets", {}))

    @property
    def surfaces(self) -> dict[str, object]:
        return dict(self.model_semantics.get("surfaces", {}))

    def surface_faces(self, name: str) -> tuple[tuple[int, str | None], ...]:
        """Expand an element-based surface to source element-label/face pairs."""

        selected = self.surfaces.get(str(name).upper())
        if selected is None:
            raise KeyError(f"Abaqus surface {name!r} is not present.")
        if selected.get("surface_type") != "ELEMENT":
            raise TypeError(f"Abaqus surface {name!r} is not element based.")
        expanded = []
        for entry in selected.get("entries", ()):
            reference = str(entry["reference"]).upper()
            labels = self.element_sets.get(reference, {}).get("labels")
            if labels is None:
                try:
                    labels = (int(reference),)
                except ValueError as exc:
                    raise KeyError(
                        f"Surface {name!r} references unknown ELSET {reference!r}."
                    ) from exc
            expanded.extend((int(label), entry.get("face")) for label in labels)
        return tuple(expanded)

    def require_formulation(
        self,
        *supported: str,
        operation: str = "selected solution procedure",
    ) -> None:
        """Reject known source formulations not consumed by a solver path."""

        accepted = {str(value) for value in supported}
        incompatible = [
            value
            for value in self.element_definitions
            if value.get("formulation") not in accepted
            and value.get("formulation") != "source_defined"
        ]
        if incompatible:
            labels = ", ".join(
                f"{value.get('source_type')} ({value.get('formulation')})"
                for value in incompatible
            )
            raise NotImplementedError(
                f"{operation} does not consume the imported Abaqus formulation: "
                f"{labels}. Neutral-mesh conversion preserves geometry/topology "
                "but cannot replace the element's pressure or enhanced variables."
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_mesh_import",
            "source": str(self.conversion.source_path),
            "cell_type": self.conversion.cell_type,
            "nodes": self.nodes.summary(),
            "mesh": self.fem_mesh.summary().as_dict(),
            "region_tags": self.conversion.region_tags,
            "element_definitions": self.element_definitions,
            "node_sets": self.node_sets,
            "element_sets": self.element_sets,
            "surfaces": self.surfaces,
            "source_metadata": self.conversion.source_metadata,
            "warnings": self.conversion.warnings,
        }


def read_node_table(path: str | Path) -> AbaqusNodeTable:
    """Read all ``*NODE`` sections while preserving Abaqus node labels."""

    path = Path(path)
    labels: list[int] = []
    coordinates: list[tuple[float, ...]] = []
    in_nodes = False
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            in_nodes = line.split(",", 1)[0].strip().upper() == "*NODE"
            continue
        if not in_nodes:
            continue
        values = _csv_values(line)
        if len(values) < 2:
            raise ValueError(f"Invalid Abaqus node at {path}:{line_number}.")
        labels.append(int(values[0]))
        coordinates.append(tuple(float(value) for value in values[1:]))
    if not labels:
        raise ValueError(f"No *NODE data were found in {path}.")
    dimensions = {len(value) for value in coordinates}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent Abaqus node dimensions in {path}.")
    return AbaqusNodeTable(np.asarray(labels), np.asarray(coordinates))


def read_equations(path: str | Path) -> AbaqusEquationSet:
    """Read Abaqus ``*EQUATION`` data or a keyword-free included equation file.

    Abaqus allows the ``3*N`` term values to continue across multiple lines.
    The include used by the example contains only equation data, which is also
    accepted here.
    """

    path = Path(path)
    records = _data_records(path)
    equations: list[LinearEquation] = []
    index = 0
    while index < len(records):
        line_number, values = records[index]
        index += 1
        if len(values) != 1:
            raise ValueError(
                f"Expected an equation term count at {path}:{line_number}, got {values!r}."
            )
        term_count = int(values[0])
        if term_count < 2:
            raise ValueError(
                f"Abaqus equation at {path}:{line_number} has fewer than two terms."
            )
        flat: list[str] = []
        while len(flat) < 3 * term_count and index < len(records):
            _, continuation = records[index]
            index += 1
            flat.extend(continuation)
        if len(flat) != 3 * term_count:
            raise ValueError(
                f"Abaqus equation at {path}:{line_number} expected "
                f"{3 * term_count} term values, got {len(flat)}."
            )
        terms = tuple(
            EquationTerm(
                node=int(flat[offset]),
                dof=int(flat[offset + 1]),
                coefficient=float(flat[offset + 2]),
            )
            for offset in range(0, len(flat), 3)
        )
        equations.append(LinearEquation(terms))
    if not equations:
        raise ValueError(f"No equation data were found in {path}.")
    return AbaqusEquationSet(tuple(equations), source=path)


def displacement_in_source_order(
    displacement,
    nodes: AbaqusNodeTable,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Return a vector field ordered by Abaqus source node labels.

    Under MPI each rank contributes its owned dofs and receives the complete
    source-ordered array. This keeps history extraction deterministic without
    treating ghost values as independent data.
    """

    function = getattr(displacement, "value", displacement)
    space = function.function_space
    comm = space.mesh.comm
    block_size = int(space.dofmap.index_map_bs)
    coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    values = np.asarray(function.x.array).reshape(-1, block_size)
    dimension = coordinates.shape[1]
    source_coordinates = np.zeros((nodes.labels.size, dimension), dtype=float)
    copied_dimension = min(dimension, nodes.coordinates.shape[1])
    source_coordinates[:, :copied_dimension] = nodes.coordinates[:, :copied_dimension]
    source_buckets: dict[tuple[int, ...], list[int]] = {}
    for source_index, coordinate in enumerate(source_coordinates):
        key = tuple(np.rint(coordinate / tolerance).astype(np.int64))
        source_buckets.setdefault(key, []).append(source_index)
    local: dict[int, np.ndarray] = {}
    owned_count = space.dofmap.index_map.size_local
    for block, coordinate in enumerate(coordinates[:owned_count]):
        key = tuple(np.rint(coordinate / tolerance).astype(np.int64))
        candidates: list[int] = []
        for shift in product((-1, 0, 1), repeat=dimension):
            candidates.extend(
                source_buckets.get(
                    tuple(key[i] + shift[i] for i in range(dimension)),
                    (),
                )
            )
        if not candidates:
            continue
        distances = np.linalg.norm(
            source_coordinates[candidates] - coordinate,
            axis=1,
        )
        closest = int(np.argmin(distances))
        if distances[closest] > tolerance:
            continue
        source_index = int(candidates[closest])
        if source_index in local:
            raise ValueError(
                f"Abaqus node {int(nodes.labels[source_index])} matched "
                "multiple owned field dofs."
            )
        local[source_index] = values[block].copy()

    gathered: dict[int, np.ndarray] = {}
    for rank_values in comm.allgather(local):
        overlap = set(gathered) & set(rank_values)
        if overlap:
            raise ValueError(
                "Abaqus source nodes have duplicate distributed owners: "
                f"{sorted(overlap)[:8]}."
            )
        gathered.update(rank_values)
    missing = set(range(nodes.labels.size)) - set(gathered)
    if missing:
        labels = [int(nodes.labels[index]) for index in sorted(missing)[:8]]
        raise ValueError(
            f"Abaqus nodes have no matching distributed field dofs: {labels}."
        )
    ordered = np.empty((nodes.labels.size, block_size), dtype=float)
    for source_index, value in gathered.items():
        ordered[source_index] = value
    return ordered


def write_deformation_vtu_pair(
    source_path: str | Path,
    nodes: AbaqusNodeTable,
    displacement,
    output_directory: str | Path,
    *,
    deformation_scale: float = 1.0,
    basename: str = "periodic_cell",
) -> tuple[Path, Path]:
    """Write ParaView-ready undeformed and deformed Abaqus meshes."""

    from .formats import require_meshio

    if not np.isfinite(deformation_scale):
        raise ValueError("deformation_scale must be finite.")
    meshio = require_meshio()
    source = meshio.read(Path(source_path), file_format="abaqus")
    values = displacement_in_source_order(displacement, nodes)
    points = np.asarray(source.points, dtype=float)
    if values.shape != points.shape:
        raise ValueError(
            "Abaqus point and displacement shapes differ: "
            f"{points.shape} versus {values.shape}."
        )
    magnitude = np.linalg.norm(values, axis=1)
    point_data = {
        "Displacement": values,
        "DisplacementMagnitude": magnitude,
        "AbaqusNodeLabel": nodes.labels,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    undeformed_path = output / f"{basename}_undeformed.vtu"
    deformed_path = output / f"{basename}_deformed.vtu"
    common = {
        "cells": source.cells,
        "cell_data": dict(source.cell_data),
        "field_data": dict(source.field_data),
    }
    meshio.write(
        undeformed_path,
        meshio.Mesh(points=points, point_data=point_data, **common),
    )
    meshio.write(
        deformed_path,
        meshio.Mesh(
            points=points + float(deformation_scale) * values,
            point_data=point_data,
            **common,
        ),
    )
    return undeformed_path, deformed_path


def periodic_cell_volume(
    nodes: AbaqusNodeTable,
    *,
    anchor_node: int,
    reference_nodes,
) -> float:
    """Return the reference parallelepiped volume from four control nodes."""

    references = tuple(int(node) for node in reference_nodes)
    if len(references) != 3:
        raise ValueError("periodic_cell_volume requires three reference nodes.")
    origin = nodes.coordinate(int(anchor_node))
    lattice = np.column_stack(
        [nodes.coordinate(node) - origin for node in references]
    )
    volume = abs(float(np.linalg.det(lattice)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Periodic-cell control nodes define a degenerate volume.")
    return volume


def _data_records(path: Path) -> list[tuple[int, list[str]]]:
    records: list[tuple[int, list[str]]] = []
    in_equations = True
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            in_equations = line.split(",", 1)[0].strip().upper() == "*EQUATION"
            continue
        if in_equations:
            records.append((line_number, _csv_values(line)))
    return records


def _csv_values(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]
