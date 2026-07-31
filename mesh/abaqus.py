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

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_mesh_import",
            "source": str(self.conversion.source_path),
            "cell_type": self.conversion.cell_type,
            "nodes": self.nodes.summary(),
            "mesh": self.fem_mesh.summary().as_dict(),
            "region_tags": self.conversion.region_tags,
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
