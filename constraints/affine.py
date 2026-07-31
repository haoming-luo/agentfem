"""Inspectable affine elimination for Abaqus-style nodal equations.

For a displacement vector ``u`` the reduction stores

``u = T q + u_bar``

where ``q`` contains independent degrees of freedom. Nonlinear residuals and
tangents are reduced by ``T.T @ R`` and ``T.T @ K @ T``. Serial problems use
an explicit transformation matrix. Distributed problems flatten the same
constraint graph and delegate ownership-aware assembly to ``dolfinx_mpc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np
from dolfinx import fem
from mpi4py import MPI
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


@dataclass
class DistributedAffineReduction:
    """Homogeneous correction space for a distributed affine constraint."""

    mpc: object
    bcs: tuple[object, ...]
    original_space: object
    full_size: int
    reduced_size: int
    slave_count: int
    control_dof_count: int

    @property
    def eliminated_count(self) -> int:
        return self.full_size - self.reduced_size

    def correction(self):
        """Create a correction function on the augmented MPC space."""

        return fem.Function(self.mpc.function_space, name="AffineCorrection")

    def validate_prefix_layout(self, correction) -> None:
        """Check that augmented local dofs retain the original ordering."""

        original = np.asarray(
            self.original_space.tabulate_dof_coordinates(),
            dtype=float,
        )
        augmented = np.asarray(
            correction.function_space.tabulate_dof_coordinates(),
            dtype=float,
        )
        if augmented.shape[0] < original.shape[0] or not np.allclose(
            augmented[: original.shape[0]],
            original,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "MPC function-space layout does not preserve original local dof order."
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "distributed_affine_dof_reduction",
            "backend": "dolfinx_mpc",
            "full_dofs": self.full_size,
            "independent_dofs": self.reduced_size,
            "eliminated_or_prescribed_dofs": self.eliminated_count,
            "equation_slave_dofs": self.slave_count,
            "prescribed_control_dofs": self.control_dof_count,
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

    def deformation_gradient_at(self, load_factor: float) -> np.ndarray:
        """Return the macroscopic deformation gradient at a step load factor."""

        if not np.isfinite(load_factor) or load_factor < 0.0:
            raise ValueError("load_factor must be finite and non-negative.")
        identity = np.eye(self.deformation_gradient.shape[0])
        return identity + float(load_factor) * (
            self.deformation_gradient - identity
        )

    def reduction(self, load_factor: float = 1.0) -> AffineReduction:
        """Build the exact affine reduction for one load factor."""

        if not np.isfinite(load_factor) or load_factor < 0.0:
            raise ValueError("load_factor must be finite and non-negative.")
        function = field_api.unwrap(self.target)
        space = function.function_space
        if space.mesh.comm.size != 1:
            raise RuntimeError(
                "Explicit AffineReduction is a serial backend. "
                "Use distributed_reduction() under MPI."
            )
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

        F = self.deformation_gradient_at(load_factor)
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

    def distributed_reduction(self) -> DistributedAffineReduction:
        """Build an ownership-aware correction MPC from ``*EQUATION`` data."""

        try:
            import dolfinx_mpc
        except ImportError as exc:
            raise ImportError(
                "Distributed Abaqus *EQUATION constraints require dolfinx_mpc "
                "matching the installed DOLFINx version."
            ) from exc

        function = field_api.unwrap(self.target)
        space = function.function_space
        domain = space.mesh
        block_size = int(space.dofmap.index_map_bs)
        node_map = _distributed_node_map(
            space,
            self.nodes,
            labels=_equation_node_labels(self.equations) | set(self.control_nodes),
            tolerance=self.tolerance,
        )
        relations, prescribed = _semantic_relations(
            self.equations,
            self.control_nodes,
            block_size,
        )
        expanded = _expand_semantic_relations(relations, prescribed)

        index_map = space.dofmap.index_map
        slaves: list[int] = []
        masters: list[int] = []
        coefficients: list[float] = []
        owners: list[int] = []
        offsets = [0]
        dummy_master = next(
            (
                master
                for relation in expanded.values()
                for master in relation
            ),
            None,
        )
        for slave in sorted(expanded):
            label, component = slave
            global_block, _ = node_map[label]
            local_block = int(
                index_map.global_to_local(
                    np.asarray([global_block], dtype=np.int64)
                )[0]
            )
            if local_block < 0:
                continue
            slaves.append(local_block * block_size + component)
            relation = expanded[slave]
            encoded_relation = relation
            if not relation:
                if dummy_master is None:
                    raise ValueError(
                        "All Abaqus equation slaves are fully prescribed; "
                        "a distributed MPC requires at least one independent dof."
                    )
                # dolfinx_mpc 0.11 only clears a slave element-vector entry
                # while iterating over its masters. A zero coefficient keeps
                # the exact relation delta_u_slave = 0 while exercising that
                # assembly path.
                encoded_relation = {dummy_master: 0.0}
            for master, coefficient in sorted(encoded_relation.items()):
                master_label, master_component = master
                master_block, master_owner = node_map[master_label]
                masters.append(int(master_block * block_size + master_component))
                coefficients.append(float(coefficient))
                owners.append(int(master_owner))
            offsets.append(len(masters))

        mpc = dolfinx_mpc.MultiPointConstraint(space)
        mpc.add_constraint(
            space,
            np.asarray(slaves, dtype=np.int32),
            np.asarray(masters, dtype=np.int64),
            np.asarray(coefficients, dtype=PETSc.ScalarType),
            np.asarray(owners, dtype=np.int32),
            np.asarray(offsets, dtype=np.int32),
        )
        mpc.finalize()

        control_globals = np.asarray(
            [node_map[label][0] for label in self.control_nodes],
            dtype=np.int64,
        )
        control_local = index_map.global_to_local(control_globals)
        control_blocks = [
            int(block)
            for block in control_local
            if int(block) >= 0
        ]
        control_bc = fem.dirichletbc(
            np.zeros(block_size, dtype=PETSc.ScalarType),
            np.asarray(control_blocks, dtype=np.int32),
            space,
        )
        full_size = int(index_map.size_global * block_size)
        slave_count = len(expanded)
        control_dof_count = len(prescribed)
        return DistributedAffineReduction(
            mpc=mpc,
            bcs=(control_bc,),
            original_space=space,
            full_size=full_size,
            reduced_size=full_size - slave_count - control_dof_count,
            slave_count=slave_count,
            control_dof_count=control_dof_count,
        )

    def apply_affine_increment(
        self,
        start_factor: float,
        target_factor: float,
    ) -> None:
        """Add the macroscopic affine predictor between two load factors."""

        function = field_api.unwrap(self.target)
        space = function.function_space
        block_size = int(space.dofmap.index_map_bs)
        coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
        values = function.x.array.reshape(-1, block_size)
        if values.shape[0] != coordinates.shape[0]:
            raise RuntimeError("Displacement values and dof coordinates do not align.")
        delta = float(target_factor) - float(start_factor)
        values[:] += delta * (
            coordinates[:, :block_size]
            @ (self.deformation_gradient - np.eye(block_size)).T
        )
        function.x.scatter_forward()

    def mismatch(self, load_factor: float = 1.0) -> float:
        """Return the maximum absolute original ``*EQUATION`` residual."""

        function = field_api.unwrap(self.target)
        space = function.function_space
        block_size = int(space.dofmap.index_map_bs)
        labels = _equation_node_labels(self.equations)
        if space.mesh.comm.size == 1:
            node_to_block = _match_nodes_to_dof_blocks(
                self.nodes,
                np.asarray(space.tabulate_dof_coordinates(), dtype=float),
                labels=labels,
                tolerance=self.tolerance,
            )
            nodal_values = {
                label: function.x.array[
                    node_to_block[label] * block_size : (node_to_block[label] + 1)
                    * block_size
                ]
                for label in labels
            }
        else:
            nodal_values = _distributed_nodal_values(
                function,
                self.nodes,
                labels=labels,
                tolerance=self.tolerance,
            )
        maximum = 0.0
        for equation in self.equations.equations:
            value = 0.0
            for term in equation.terms:
                value += (
                    term.coefficient
                    * nodal_values[int(term.node)][int(term.dof) - 1]
                )
            maximum = max(maximum, abs(float(value)))
        return float(space.mesh.comm.allreduce(maximum, op=MPI.MAX))

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
            "supports_parallel": _dolfinx_mpc_available(),
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


SemanticDof = tuple[int, int]


def _semantic_relations(
    equations: AbaqusEquationSet,
    control_nodes,
    block_size: int,
) -> tuple[
    dict[SemanticDof, dict[SemanticDof, float]],
    set[SemanticDof],
]:
    """Return equation relations in source-label/component coordinates."""

    relations: dict[SemanticDof, dict[SemanticDof, float]] = {}
    for equation in equations.equations:
        slave_term = equation.terms[0]
        slave_component = int(slave_term.dof) - 1
        if slave_component < 0 or slave_component >= block_size:
            raise ValueError(
                f"Abaqus dof {slave_term.dof} is outside a "
                f"{block_size}-component displacement space."
            )
        slave = (int(slave_term.node), slave_component)
        dependencies: dict[SemanticDof, float] = {}
        for term in equation.terms[1:]:
            component = int(term.dof) - 1
            if component < 0 or component >= block_size:
                raise ValueError(
                    f"Abaqus dof {term.dof} is outside a "
                    f"{block_size}-component displacement space."
                )
            master = (int(term.node), component)
            dependencies[master] = dependencies.get(master, 0.0) - (
                float(term.coefficient) / float(slave_term.coefficient)
            )
        relations[slave] = dependencies
    prescribed = {
        (int(label), component)
        for label in control_nodes
        for component in range(block_size)
    }
    conflicts = set(relations) & prescribed
    if conflicts:
        raise ValueError(
            "Abaqus equation slave dofs cannot also be prescribed controls: "
            f"{sorted(conflicts)[:8]}."
        )
    return relations, prescribed


def _expand_semantic_relations(
    relations: Mapping[SemanticDof, Mapping[SemanticDof, float]],
    prescribed: set[SemanticDof],
) -> dict[SemanticDof, dict[SemanticDof, float]]:
    """Flatten chained slaves to independent masters for MPC assembly."""

    cache: dict[SemanticDof, dict[SemanticDof, float]] = {}
    visiting: set[SemanticDof] = set()

    def expand(dof: SemanticDof) -> dict[SemanticDof, float]:
        if dof in cache:
            return dict(cache[dof])
        if dof in prescribed:
            cache[dof] = {}
            return {}
        if dof not in relations:
            return {dof: 1.0}
        if dof in visiting:
            raise ValueError(
                "Cyclic Abaqus equation dependency detected at "
                f"node {dof[0]}, component {dof[1] + 1}."
            )
        visiting.add(dof)
        coefficients: dict[SemanticDof, float] = {}
        for dependency, multiplier in relations[dof].items():
            for master, coefficient in expand(dependency).items():
                coefficients[master] = coefficients.get(master, 0.0) + (
                    float(multiplier) * coefficient
                )
        visiting.remove(dof)
        coefficients = {
            master: coefficient
            for master, coefficient in coefficients.items()
            if abs(coefficient) > 1.0e-15
        }
        cache[dof] = dict(coefficients)
        return coefficients

    return {slave: expand(slave) for slave in relations}


def _distributed_node_map(
    space,
    nodes: AbaqusNodeTable,
    *,
    labels: set[int],
    tolerance: float,
) -> dict[int, tuple[int, int]]:
    """Map Abaqus labels to global block dofs and owning MPI ranks."""

    domain = space.mesh
    comm = domain.comm
    index_map = space.dofmap.index_map
    coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    dimension = coordinates.shape[1]
    selected_labels = {int(label) for label in labels}
    source_buckets: dict[tuple[int, ...], list[int]] = {}
    for label in selected_labels:
        coordinate = _coordinate_in_dimension(nodes.coordinate(label), dimension)
        source_buckets.setdefault(_coordinate_key(coordinate, tolerance), []).append(label)

    owned_coordinates = coordinates[: index_map.size_local]
    local_globals = index_map.local_to_global(
        np.arange(index_map.size_local, dtype=np.int32)
    )
    local: dict[int, tuple[int, int]] = {}
    for local_block, coordinate in enumerate(owned_coordinates):
        key = _coordinate_key(coordinate, tolerance)
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
        distances = np.asarray(
            [
                np.linalg.norm(
                    _coordinate_in_dimension(nodes.coordinate(label), dimension)
                    - coordinate
                )
                for label in candidates
            ]
        )
        closest = int(np.argmin(distances))
        if distances[closest] <= tolerance:
            label = int(candidates[closest])
            if label in local:
                raise ValueError(
                    f"Abaqus node {label} matched multiple owned field dofs."
                )
            local[label] = (int(local_globals[local_block]), int(comm.rank))

    mapping: dict[int, tuple[int, int]] = {}
    for rank_mapping in comm.allgather(local):
        for label, entry in rank_mapping.items():
            if label in mapping and mapping[label] != entry:
                raise ValueError(
                    f"Abaqus node {label} has inconsistent distributed ownership."
                )
            mapping[label] = entry
    missing = selected_labels - set(mapping)
    if missing:
        raise ValueError(
            "Abaqus nodes have no matching distributed field dofs: "
            f"{sorted(missing)[:8]}."
        )
    return mapping


def _distributed_nodal_values(
    function,
    nodes: AbaqusNodeTable,
    *,
    labels: set[int],
    tolerance: float,
) -> dict[int, np.ndarray]:
    """Collect selected owned nodal values on every rank."""

    space = function.function_space
    comm = space.mesh.comm
    block_size = int(space.dofmap.index_map_bs)
    node_map = _distributed_node_map(
        space,
        nodes,
        labels=labels,
        tolerance=tolerance,
    )
    local_start, _ = space.dofmap.index_map.local_range
    values = function.x.array.reshape(-1, block_size)
    local = {
        label: values[int(global_block - local_start)].copy()
        for label, (global_block, owner) in node_map.items()
        if owner == comm.rank
    }
    gathered: dict[int, np.ndarray] = {}
    for rank_values in comm.allgather(local):
        gathered.update(rank_values)
    if set(gathered) != set(labels):
        raise RuntimeError("Distributed nodal value collection is incomplete.")
    return gathered


def _dolfinx_mpc_available() -> bool:
    try:
        import dolfinx_mpc  # noqa: F401
    except ImportError:
        return False
    return True


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
        coordinate = _coordinate_in_dimension(nodes.coordinate(label), dimension)
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


def _coordinate_in_dimension(coordinate, dimension: int) -> np.ndarray:
    selected = np.asarray(coordinate, dtype=float).reshape(-1)
    if selected.size > dimension:
        return selected[:dimension]
    output = np.zeros(dimension, dtype=float)
    output[: selected.size] = selected
    return output


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
