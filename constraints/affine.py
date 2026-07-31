"""Inspectable affine elimination for Abaqus-style nodal equations.

For a displacement vector ``u`` the reduction stores

``u = T q + u_bar``

where ``q`` contains independent degrees of freedom.  Nonlinear residuals and
tangents are reduced by ``T.T @ R`` and ``T.T @ K @ T``.  The first release is
deliberately serial: the algebra is exact, while distributed ownership of
chained Abaqus equations remains a separate backend capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np
from petsc4py import PETSc

from .. import fields as field_api
from ..mesh.abaqus import AbaqusEquationSet, AbaqusNodeTable


@dataclass(frozen=True)
class AffineReduction:
    """Sparse serial representation of ``u = T q + offset``."""

    row_offsets: np.ndarray
    column_indices: np.ndarray
    coefficients: np.ndarray
    offset: np.ndarray
    independent_full_dofs: np.ndarray

    def __post_init__(self) -> None:
        row_offsets = np.asarray(self.row_offsets, dtype=PETSc.IntType)
        columns = np.asarray(self.column_indices, dtype=PETSc.IntType)
        coefficients = np.asarray(self.coefficients, dtype=PETSc.ScalarType)
        offset = np.asarray(self.offset, dtype=PETSc.ScalarType)
        independent = np.asarray(self.independent_full_dofs, dtype=PETSc.IntType)
        if row_offsets.ndim != 1 or row_offsets.size != offset.size + 1:
            raise ValueError("AffineReduction row_offsets must have n_full + 1 entries.")
        if row_offsets[-1] != columns.size or columns.size != coefficients.size:
            raise ValueError("AffineReduction CSR arrays are inconsistent.")
        object.__setattr__(self, "row_offsets", row_offsets)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "independent_full_dofs", independent)

    @property
    def full_size(self) -> int:
        return int(self.offset.size)

    @property
    def reduced_size(self) -> int:
        return int(self.independent_full_dofs.size)

    @property
    def eliminated_count(self) -> int:
        return self.full_size - self.reduced_size

    def matrix(self, comm=PETSc.COMM_SELF):
        """Create the PETSc sparse transformation matrix ``T``."""

        return PETSc.Mat().createAIJ(
            size=(self.full_size, self.reduced_size),
            csr=(self.row_offsets, self.column_indices, self.coefficients),
            comm=comm,
        )

    def reconstruct(self, reduced_values) -> np.ndarray:
        """Return full values from a NumPy reduced vector."""

        reduced = np.asarray(reduced_values, dtype=PETSc.ScalarType).reshape(-1)
        if reduced.size != self.reduced_size:
            raise ValueError(
                f"Expected {self.reduced_size} reduced values, got {reduced.size}."
            )
        full = self.offset.copy()
        for row in range(self.full_size):
            start, end = self.row_offsets[row : row + 2]
            full[row] += np.dot(
                self.coefficients[start:end],
                reduced[self.column_indices[start:end]],
            )
        return full

    def initial_reduced_values(
        self,
        dof_coordinates: np.ndarray,
        deformation_gradient,
        *,
        block_size: int,
    ) -> np.ndarray:
        """Sample an affine displacement field at independent dofs."""

        F = _deformation_gradient(deformation_gradient, block_size)
        coordinates = np.asarray(dof_coordinates, dtype=float)
        values = np.empty(self.reduced_size, dtype=PETSc.ScalarType)
        for index, full_dof in enumerate(self.independent_full_dofs):
            block, component = divmod(int(full_dof), int(block_size))
            values[index] = ((F - np.eye(block_size)) @ coordinates[block])[component]
        return values

    def summary(self) -> dict[str, object]:
        return {
            "kind": "affine_dof_reduction",
            "full_dofs": self.full_size,
            "independent_dofs": self.reduced_size,
            "eliminated_or_prescribed_dofs": self.eliminated_count,
            "transformation_nonzeros": int(self.coefficients.size),
        }


@dataclass(frozen=True)
class AbaqusPeriodicConstraint:
    """Periodic cell equations controlled by a macroscopic deformation gradient."""

    target: object
    nodes: AbaqusNodeTable
    equations: AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int
    reference_nodes: tuple[int, ...]
    tolerance: float = 1.0e-9
    name: str = "abaqus_periodic_cell"

    def __post_init__(self) -> None:
        function = field_api.unwrap(self.target)
        space = function.function_space
        domain = space.mesh
        if domain.comm.size != 1:
            raise NotImplementedError(
                "Abaqus equation elimination is serial in this release. "
                "The constraint representation is backend-independent; distributed "
                "ownership and ghost propagation are the remaining parallel step."
            )
        block_size = int(space.dofmap.index_map_bs)
        F = _deformation_gradient(self.deformation_gradient, block_size)
        if len(self.reference_nodes) != block_size:
            raise ValueError(
                "reference_nodes must contain one control node per spatial direction."
            )
        control = (int(self.anchor_node), *(int(node) for node in self.reference_nodes))
        for node in control:
            self.nodes.index(node)
        if self.tolerance <= 0.0:
            raise ValueError("AbaqusPeriodicConstraint.tolerance must be positive.")
        object.__setattr__(self, "deformation_gradient", F)
        object.__setattr__(self, "reference_nodes", tuple(int(node) for node in self.reference_nodes))

    @property
    def control_nodes(self) -> tuple[int, ...]:
        return (self.anchor_node, *self.reference_nodes)

    @property
    def reference_cell_volume(self) -> float:
        """Return the lattice-cell volume implied by the control nodes."""

        dimension = self.deformation_gradient.shape[0]
        origin = self.nodes.coordinate(self.anchor_node)[:dimension]
        lattice = np.column_stack(
            [
                self.nodes.coordinate(node)[:dimension] - origin
                for node in self.reference_nodes
            ]
        )
        volume = abs(float(np.linalg.det(lattice)))
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("Periodic control nodes define a degenerate lattice cell.")
        return volume

    def reduction(self, load_factor: float = 1.0) -> AffineReduction:
        """Build the exact affine reduction for one load factor."""

        if not np.isfinite(load_factor) or load_factor < 0.0:
            raise ValueError("load_factor must be finite and non-negative.")
        function = field_api.unwrap(self.target)
        space = function.function_space
        block_size = int(space.dofmap.index_map_bs)
        dof_coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
        full_size = int(space.dofmap.index_map.size_local * block_size)
        node_to_block = _match_nodes_to_dof_blocks(
            self.nodes,
            dof_coordinates,
            labels=_equation_node_labels(self.equations) | set(self.control_nodes),
            tolerance=self.tolerance,
        )

        relations: dict[int, dict[int, float]] = {}
        for equation in self.equations.equations:
            slave_term = equation.terms[0]
            slave = _scalar_dof(slave_term.node, slave_term.dof, node_to_block, block_size)
            dependency: dict[int, float] = {}
            for term in equation.terms[1:]:
                dof = _scalar_dof(term.node, term.dof, node_to_block, block_size)
                dependency[dof] = dependency.get(dof, 0.0) - (
                    term.coefficient / slave_term.coefficient
                )
            relations[slave] = dependency

        F = np.eye(block_size) + float(load_factor) * (
            self.deformation_gradient - np.eye(block_size)
        )
        prescribed: dict[int, float] = {}
        for label in self.control_nodes:
            coordinate = self.nodes.coordinate(label)[:block_size]
            displacement = (F - np.eye(block_size)) @ coordinate
            block = node_to_block[label]
            for component in range(block_size):
                prescribed[block * block_size + component] = float(displacement[component])
        conflicts = set(relations) & set(prescribed)
        if conflicts:
            raise ValueError(
                "Abaqus equation slave dofs cannot also be prescribed control dofs: "
                f"{sorted(conflicts)[:8]}."
            )
        return _build_reduction(full_size, relations, prescribed)

    def mismatch(self, load_factor: float = 1.0) -> float:
        """Return the maximum absolute original ``*EQUATION`` residual."""

        function = field_api.unwrap(self.target)
        space = function.function_space
        block_size = int(space.dofmap.index_map_bs)
        labels = _equation_node_labels(self.equations)
        node_to_block = _match_nodes_to_dof_blocks(
            self.nodes,
            np.asarray(space.tabulate_dof_coordinates(), dtype=float),
            labels=labels,
            tolerance=self.tolerance,
        )
        values = function.x.array
        maximum = 0.0
        for equation in self.equations.equations:
            value = 0.0
            for term in equation.terms:
                dof = _scalar_dof(term.node, term.dof, node_to_block, block_size)
                value += term.coefficient * values[dof]
            maximum = max(maximum, abs(float(value)))
        return maximum

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "abaqus_periodic_constraint",
            "enforcement": "exact_affine_elimination",
            "equations": self.equations.summary(),
            "anchor_node": self.anchor_node,
            "reference_nodes": self.reference_nodes,
            "target_deformation_gradient": self.deformation_gradient.tolist(),
            "reference_cell_volume": self.reference_cell_volume,
            "supports_parallel": False,
        }


def abaqus_periodic_cell(
    target,
    *,
    nodes: AbaqusNodeTable,
    equations: AbaqusEquationSet,
    deformation_gradient,
    anchor_node: int,
    reference_nodes,
    tolerance: float = 1.0e-9,
    name: str = "abaqus_periodic_cell",
) -> AbaqusPeriodicConstraint:
    """Create exact periodic-cell constraints from Abaqus equation data."""

    return AbaqusPeriodicConstraint(
        target=target,
        nodes=nodes,
        equations=equations,
        deformation_gradient=np.asarray(deformation_gradient, dtype=float),
        anchor_node=int(anchor_node),
        reference_nodes=tuple(int(node) for node in reference_nodes),
        tolerance=float(tolerance),
        name=name,
    )


def _build_reduction(
    full_size: int,
    relations: Mapping[int, Mapping[int, float]],
    prescribed: Mapping[int, float],
) -> AffineReduction:
    slaves = set(relations)
    fixed = set(prescribed)
    invalid = (slaves | fixed) - set(range(full_size))
    if invalid:
        raise ValueError(f"Affine constraints reference invalid full dofs: {sorted(invalid)[:8]}.")
    independent = np.asarray(
        sorted(set(range(full_size)) - slaves - fixed),
        dtype=PETSc.IntType,
    )
    reduced_index = {int(dof): index for index, dof in enumerate(independent)}
    cache: dict[int, tuple[dict[int, float], float]] = {}
    visiting: set[int] = set()

    def expand(dof: int) -> tuple[dict[int, float], float]:
        if dof in cache:
            coefficients, value = cache[dof]
            return dict(coefficients), value
        if dof in prescribed:
            result = ({}, float(prescribed[dof]))
            cache[dof] = result
            return ({}, result[1])
        if dof not in relations:
            result = ({reduced_index[dof]: 1.0}, 0.0)
            cache[dof] = result
            return ({reduced_index[dof]: 1.0}, 0.0)
        if dof in visiting:
            raise ValueError(f"Cyclic Abaqus equation dependency detected at full dof {dof}.")
        visiting.add(dof)
        coefficients: dict[int, float] = {}
        offset = 0.0
        for dependency, multiplier in relations[dof].items():
            child_coefficients, child_offset = expand(int(dependency))
            for column, coefficient in child_coefficients.items():
                coefficients[column] = coefficients.get(column, 0.0) + (
                    float(multiplier) * coefficient
                )
            offset += float(multiplier) * child_offset
        visiting.remove(dof)
        coefficients = {
            column: coefficient
            for column, coefficient in coefficients.items()
            if abs(coefficient) > 1.0e-15
        }
        cache[dof] = (dict(coefficients), offset)
        return coefficients, offset

    row_offsets = [0]
    columns: list[int] = []
    coefficients: list[float] = []
    offsets = np.empty(full_size, dtype=PETSc.ScalarType)
    for dof in range(full_size):
        row, offsets[dof] = expand(dof)
        for column in sorted(row):
            columns.append(column)
            coefficients.append(row[column])
        row_offsets.append(len(columns))
    return AffineReduction(
        np.asarray(row_offsets),
        np.asarray(columns),
        np.asarray(coefficients),
        offsets,
        independent,
    )


def _match_nodes_to_dof_blocks(
    nodes: AbaqusNodeTable,
    dof_coordinates: np.ndarray,
    *,
    labels: set[int],
    tolerance: float,
) -> dict[int, int]:
    dimension = dof_coordinates.shape[1]
    buckets: dict[tuple[int, ...], list[int]] = {}
    for block, coordinate in enumerate(dof_coordinates):
        key = _coordinate_key(coordinate, tolerance)
        buckets.setdefault(key, []).append(block)
    mapping: dict[int, int] = {}
    for label in labels:
        coordinate = nodes.coordinate(label)[:dimension]
        key = _coordinate_key(coordinate, tolerance)
        candidates: list[int] = []
        for shift in product((-1, 0, 1), repeat=dimension):
            candidates.extend(
                buckets.get(tuple(key[i] + shift[i] for i in range(dimension)), ())
            )
        if not candidates:
            raise ValueError(
                f"Abaqus node {label} at {coordinate.tolist()} has no matching field dof."
            )
        distances = np.linalg.norm(dof_coordinates[candidates] - coordinate, axis=1)
        closest = int(np.argmin(distances))
        if distances[closest] > tolerance:
            raise ValueError(
                f"Abaqus node {label} is {distances[closest]:.3e} from its closest field dof."
            )
        mapping[label] = int(candidates[closest])
    return mapping


def _coordinate_key(coordinate, tolerance: float) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(coordinate) / tolerance).astype(np.int64))


def _scalar_dof(
    node: int,
    dof: int,
    node_to_block: Mapping[int, int],
    block_size: int,
) -> int:
    component = int(dof) - 1
    if component < 0 or component >= block_size:
        raise ValueError(
            f"Abaqus dof {dof} is outside a {block_size}-component displacement space."
        )
    try:
        block = node_to_block[int(node)]
    except KeyError as exc:
        raise KeyError(f"Abaqus node {node} has no matching displacement dof.") from exc
    return int(block * block_size + component)


def _equation_node_labels(equations: AbaqusEquationSet) -> set[int]:
    return {
        int(term.node)
        for equation in equations.equations
        for term in equation.terms
    }


def _deformation_gradient(value, dimension: int) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (dimension, dimension):
        raise ValueError(
            f"deformation_gradient must have shape {(dimension, dimension)}, "
            f"got {selected.shape}."
        )
    if not np.all(np.isfinite(selected)):
        raise ValueError("deformation_gradient must contain only finite values.")
    if np.linalg.det(selected) <= 0.0:
        raise ValueError("deformation_gradient must have positive determinant.")
    return selected
