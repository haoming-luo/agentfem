"""Inspectable affine elimination for Abaqus-style nodal equations.

For a displacement vector ``u`` the reduction stores

``u = T q + u_bar``

where ``q`` contains independent degrees of freedom. Nonlinear residuals and
tangents are reduced by ``T.T @ R`` and ``T.T @ K @ T``. Serial problems use
an explicit transformation matrix. Distributed problems flatten the same
constraint graph and delegate ownership-aware assembly to ``dolfinx_mpc``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Mapping

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from .. import fields as field_api
from ..mesh.abaqus import AbaqusEquationSet, AbaqusNodeTable


@dataclass(frozen=True)
class DeformationGradientPath:
    """Piecewise-linear macroscopic deformation-gradient history.

    The path coordinate remains monotone on ``[0, 1]`` while the physical
    deformation is free to unload, reload, or change direction.  Keeping the
    complete matrix history in one immutable object makes non-proportional RVE
    loading inspectable, fingerprintable, and checkpoint-safe.
    """

    coordinates: tuple[float, ...]
    gradients: tuple[tuple[tuple[float, ...], ...], ...]
    name: str = "deformation_gradient_path"

    def __post_init__(self) -> None:
        coordinates = tuple(float(value) for value in self.coordinates)
        gradients = np.asarray(self.gradients, dtype=float)
        if len(coordinates) < 3:
            raise ValueError(
                "A deformation-gradient path requires identity, at least one "
                "internal knot, and a final state. Use deformation_gradient "
                "for a two-state proportional path."
            )
        if any(not np.isfinite(value) for value in coordinates):
            raise ValueError("Path coordinates must be finite.")
        if abs(coordinates[0]) > 1.0e-12 or abs(coordinates[-1] - 1.0) > 1.0e-12:
            raise ValueError("Path coordinates must start at 0 and end at 1.")
        coordinates = (0.0, *coordinates[1:-1], 1.0)
        if any(
            right <= left
            for left, right in zip(coordinates, coordinates[1:])
        ):
            raise ValueError("Path coordinates must be strictly increasing.")
        if (
            gradients.ndim != 3
            or gradients.shape[0] != len(coordinates)
            or gradients.shape[1] != gradients.shape[2]
            or gradients.shape[1] not in (2, 3)
        ):
            raise ValueError(
                "Path gradients must have shape (states, dimension, dimension) "
                "with dimension 2 or 3."
            )
        if not np.isfinite(gradients).all():
            raise ValueError("Path gradients must contain only finite values.")
        dimension = int(gradients.shape[1])
        if not np.allclose(
            gradients[0],
            np.eye(dimension),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "A deformation-gradient path must begin at the identity."
            )
        gradients = gradients.copy()
        gradients[0] = np.eye(dimension)
        for index, (left, right) in enumerate(
            zip(gradients[:-1], gradients[1:])
        ):
            minimum = _minimum_linear_path_determinant(left, right)
            if not np.isfinite(minimum) or minimum <= 0.0:
                raise ValueError(
                    "A deformation-gradient path must preserve positive "
                    f"determinant throughout segment {index}; minimum J={minimum:.6g}."
                )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(
            self,
            "gradients",
            tuple(
                tuple(tuple(float(value) for value in row) for row in gradient)
                for gradient in gradients
            ),
        )

    @property
    def dimension(self) -> int:
        return len(self.gradients[0])

    @property
    def final(self) -> np.ndarray:
        return np.asarray(self.gradients[-1], dtype=float)

    def at(self, coordinate: float) -> np.ndarray:
        """Interpolate the macroscopic deformation gradient at one coordinate."""

        selected = float(coordinate)
        if not np.isfinite(selected) or not -1.0e-12 <= selected <= 1.0 + 1.0e-12:
            raise ValueError("Path coordinate must be finite and lie in [0, 1].")
        selected = min(1.0, max(0.0, selected))
        if selected <= self.coordinates[0]:
            return np.asarray(self.gradients[0], dtype=float)
        if selected >= self.coordinates[-1]:
            return np.asarray(self.gradients[-1], dtype=float)
        right = int(np.searchsorted(self.coordinates, selected, side="right"))
        left = right - 1
        interval = self.coordinates[right] - self.coordinates[left]
        fraction = (selected - self.coordinates[left]) / interval
        first = np.asarray(self.gradients[left], dtype=float)
        second = np.asarray(self.gradients[right], dtype=float)
        gradient = first + fraction * (second - first)
        if float(np.linalg.det(gradient)) <= 0.0:
            raise ValueError("Interpolated deformation gradient has non-positive J.")
        return gradient

    def summary(self) -> dict[str, object]:
        payload = {
            "schema": "agentfem.deformation-gradient-path",
            "schema_version": 1,
            "kind": "piecewise_linear_deformation_gradient_path",
            "name": self.name,
            "coordinate_name": "normalized_step_coordinate",
            "coordinates": list(self.coordinates),
            "gradients": [
                [list(row) for row in gradient] for gradient in self.gradients
            ],
            "interpolation": "piecewise_linear",
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {
            **payload,
            "fingerprint": sha256(canonical.encode("utf-8")).hexdigest(),
        }


def deformation_gradient_path(
    coordinates: Iterable[float],
    gradients: Iterable[Iterable[Iterable[float]]],
    *,
    name: str = "deformation_gradient_path",
) -> DeformationGradientPath:
    """Create an inspectable unload/reload or non-proportional macro path."""

    return DeformationGradientPath(
        coordinates=tuple(coordinates),
        gradients=tuple(gradients),
        name=str(name),
    )


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
    """Periodic equations controlled by prescribed or free reference dofs.

    A complete ``deformation_gradient`` preserves the original affine-control
    API for a proportional path.  ``deformation_gradient_path`` supplies an
    identity-starting, piecewise-linear matrix history for unloading, reloading,
    or non-proportional loading. ``control_displacements`` instead mirrors
    Abaqus reference-node boundary conditions: one row per ``reference_nodes``
    entry and one value per spatial component, with ``None`` marking a free
    macroscopic degree of freedom whose conjugate global reaction is zero.
    Exactly one of these three macro-control forms must be supplied.
    """

    target: object
    nodes: AbaqusNodeTable
    equations: AbaqusEquationSet
    anchor_node: int
    reference_nodes: tuple[int, ...]
    deformation_gradient: np.ndarray | None = None
    control_displacements: tuple[tuple[float | None, ...], ...] | None = None
    tolerance: float = 1.0e-9
    name: str = "abaqus_periodic_cell"
    deformation_gradient_path: DeformationGradientPath | None = None

    def __post_init__(self) -> None:
        _, space, _, block_size = self._displacement_layout()
        if len(self.reference_nodes) != block_size:
            raise ValueError(
                "reference_nodes must contain one control node per spatial direction."
            )
        control = (int(self.anchor_node), *(int(node) for node in self.reference_nodes))
        for node in control:
            self.nodes.index(node)
        if self.tolerance <= 0.0:
            raise ValueError("AbaqusPeriodicConstraint.tolerance must be positive.")
        references = tuple(int(node) for node in self.reference_nodes)
        object.__setattr__(self, "reference_nodes", references)
        supplied = sum(
            value is not None
            for value in (
                self.deformation_gradient,
                self.control_displacements,
                self.deformation_gradient_path,
            )
        )
        if supplied != 1:
            raise ValueError(
                "Pass exactly one of deformation_gradient, "
                "deformation_gradient_path, or control_displacements."
            )
        lattice = self._reference_lattice()
        if self.deformation_gradient_path is not None:
            if not isinstance(
                self.deformation_gradient_path, DeformationGradientPath
            ):
                raise TypeError(
                    "deformation_gradient_path must be created by "
                    "constraints.deformation_gradient_path(...)."
                )
            if self.deformation_gradient_path.dimension != block_size:
                raise ValueError(
                    "Deformation-gradient path dimension does not match the "
                    "periodic displacement field."
                )
            F = self.deformation_gradient_path.final
            prescribed = ((F - np.eye(block_size)) @ lattice).T
            controls = tuple(
                tuple(float(value) for value in row) for row in prescribed
            )
        elif self.deformation_gradient is not None:
            F = _deformation_gradient(self.deformation_gradient, block_size)
            prescribed = ((F - np.eye(block_size)) @ lattice).T
            controls = tuple(
                tuple(float(value) for value in row) for row in prescribed
            )
        else:
            selected = tuple(tuple(row) for row in self.control_displacements)
            if len(selected) != block_size or any(
                len(row) != block_size for row in selected
            ):
                raise ValueError(
                    "control_displacements must contain one vector per "
                    "reference node and one entry per spatial component."
                )
            controls = tuple(
                tuple(
                    None
                    if value is None
                    else _finite_control_value(value)
                    for value in row
                )
                for row in selected
            )
            nominal = np.asarray(
                [
                    [0.0 if value is None else float(value) for value in row]
                    for row in controls
                ],
                dtype=float,
            )
            F = np.eye(block_size) + nominal.T @ np.linalg.inv(lattice)
        object.__setattr__(self, "deformation_gradient", F)
        object.__setattr__(self, "control_displacements", controls)

    @property
    def is_mixed(self) -> bool:
        """Whether pressure dofs share the constrained solution vector."""

        return getattr(self.target, "kind", None) == "displacement_pressure"

    def _displacement_layout(self):
        """Return full field, collapsed displacement space, and parent map."""

        if self.is_mixed:
            function = self.target.value
            collapsed, maps = self.target.space.sub(0).collapse()
            parent_map = np.asarray(
                maps[0] if isinstance(maps, (list, tuple)) else maps,
                dtype=PETSc.IntType,
            )
            block_size = int(collapsed.dofmap.index_map_bs)
            return function, collapsed, parent_map, block_size
        function = field_api.unwrap(self.target)
        space = function.function_space
        block_size = int(space.dofmap.index_map_bs)
        # The local coefficient vector contains owned and ghost dofs.  The
        # affine predictor must update both before scatter_forward(); using
        # only index_map.size_local breaks the coordinate/value alignment in
        # distributed displacement spaces.
        local_size = int(function.x.array.size)
        parent_map = np.arange(local_size, dtype=PETSc.IntType)
        return function, space, parent_map, block_size

    @property
    def control_nodes(self) -> tuple[int, ...]:
        return (self.anchor_node, *self.reference_nodes)

    @property
    def prescribed_control_dofs(self) -> frozenset[tuple[int, int]]:
        prescribed = {
            (int(self.anchor_node), component)
            for component in range(len(self.reference_nodes))
        }
        for label, values in zip(self.reference_nodes, self.control_displacements):
            prescribed.update(
                (int(label), component)
                for component, value in enumerate(values)
                if value is not None
            )
        return frozenset(prescribed)

    @property
    def has_free_macro_dofs(self) -> bool:
        return any(
            value is None
            for row in self.control_displacements
            for value in row
        )

    def _reference_lattice(self) -> np.ndarray:
        dimension = len(self.reference_nodes)
        origin = self.nodes.coordinate(int(self.anchor_node))[:dimension]
        lattice = np.column_stack(
            [
                self.nodes.coordinate(int(node))[:dimension] - origin
                for node in self.reference_nodes
            ]
        )
        determinant = float(np.linalg.det(lattice))
        if not np.isfinite(determinant) or abs(determinant) <= self.tolerance:
            raise ValueError(
                "Periodic reference nodes must define an independent lattice basis."
            )
        return lattice

    def prescribed_values_at(self, load_factor: float) -> dict[tuple[int, int], float]:
        if not np.isfinite(load_factor) or load_factor < 0.0:
            raise ValueError("load_factor must be finite and non-negative.")
        values = {
            (int(self.anchor_node), component): 0.0
            for component in range(len(self.reference_nodes))
        }
        if self.deformation_gradient_path is not None:
            dimension = len(self.reference_nodes)
            prescribed = (
                (self.deformation_gradient_at(load_factor) - np.eye(dimension))
                @ self._reference_lattice()
            ).T
            for label, row in zip(self.reference_nodes, prescribed):
                values.update(
                    {
                        (int(label), component): float(value)
                        for component, value in enumerate(row)
                    }
                )
            return values
        for label, row in zip(self.reference_nodes, self.control_displacements):
            for component, value in enumerate(row):
                if value is not None:
                    values[(int(label), component)] = float(load_factor) * float(value)
        return values

    def scientific_identity(self) -> dict[str, object]:
        """Return a backend- and partition-independent constraint fingerprint."""

        equations = []
        labels = set(self.control_nodes)
        for equation in self.equations.equations:
            first, *dependencies = equation.terms
            labels.update(int(term.node) for term in equation.terms)
            equations.append(
                {
                    "slave": {
                        "node": int(first.node),
                        "dof": int(first.dof),
                        "coefficient": float(first.coefficient),
                    },
                    "dependencies": sorted(
                        (
                            {
                                "node": int(term.node),
                                "dof": int(term.dof),
                                "coefficient": float(term.coefficient),
                            }
                            for term in dependencies
                        ),
                        key=lambda item: (
                            item["node"], item["dof"], item["coefficient"]
                        ),
                    ),
                }
            )
        equations.sort(
            key=lambda item: (
                item["slave"]["node"],
                item["slave"]["dof"],
                item["slave"]["coefficient"],
            )
        )
        payload = {
            "kind": "abaqus_periodic_constraint",
            "equations": equations,
            "nodes": [
                {
                    "label": int(label),
                    "coordinate": self.nodes.coordinate(int(label)).tolist(),
                }
                for label in sorted(labels)
            ],
            "anchor_node": int(self.anchor_node),
            "reference_nodes": [int(value) for value in self.reference_nodes],
            "control_displacements": [
                [None if value is None else float(value) for value in row]
                for row in self.control_displacements
            ],
            "nominal_deformation_gradient": self.deformation_gradient.tolist(),
            "tolerance": float(self.tolerance),
            "reference_cell_volume": float(self.reference_cell_volume),
        }
        if self.deformation_gradient_path is not None:
            payload["deformation_gradient_path"] = (
                self.deformation_gradient_path.summary()
            )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        identity = {
            "kind": payload["kind"],
            "fingerprint": sha256(canonical.encode("utf-8")).hexdigest(),
            "equation_count": len(equations),
            "referenced_node_count": len(labels),
            "anchor_node": payload["anchor_node"],
            "reference_nodes": payload["reference_nodes"],
            "control_displacements": payload["control_displacements"],
            "nominal_deformation_gradient": payload[
                "nominal_deformation_gradient"
            ],
            "reference_cell_volume": payload["reference_cell_volume"],
        }
        if self.deformation_gradient_path is not None:
            identity["deformation_gradient_path"] = payload[
                "deformation_gradient_path"
            ]
        return identity

    @property
    def reference_cell_volume(self) -> float:
        """Return the lattice-cell volume implied by the control nodes."""

        lattice = self._reference_lattice()
        volume = abs(float(np.linalg.det(lattice)))
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("Periodic control nodes define a degenerate lattice cell.")
        return volume

    def deformation_gradient_at(self, load_factor: float) -> np.ndarray:
        """Return the nominal prescribed-gradient predictor at one factor.

        Free macroscopic components use zero displacement in this predictor.
        Use :meth:`measured_deformation_gradient` after a solve to recover the
        actual macroscopic gradient from the reference-node solution.
        """

        if not np.isfinite(load_factor) or load_factor < 0.0:
            raise ValueError("load_factor must be finite and non-negative.")
        if self.deformation_gradient_path is not None:
            return self.deformation_gradient_path.at(load_factor)
        identity = np.eye(self.deformation_gradient.shape[0])
        return identity + float(load_factor) * (
            self.deformation_gradient - identity
        )

    def required_load_factors(self) -> tuple[float, ...]:
        """Return physical path knots that no accepted solve may skip."""

        if self.deformation_gradient_path is None:
            return ()
        return tuple(
            float(value)
            for value in self.deformation_gradient_path.coordinates[1:]
        )

    def measured_deformation_gradient(self, displacement) -> np.ndarray:
        """Recover the actual macroscopic gradient from control-node motion."""

        from ..mesh.abaqus import displacement_in_source_order

        values = displacement_in_source_order(displacement, self.nodes)
        anchor = values[self.nodes.index(int(self.anchor_node))]
        displacement_lattice = np.column_stack(
            [
                values[self.nodes.index(int(node))] - anchor
                for node in self.reference_nodes
            ]
        )
        return np.eye(len(self.reference_nodes)) + (
            displacement_lattice @ np.linalg.inv(self._reference_lattice())
        )

    def reduction(self, load_factor: float = 1.0) -> AffineReduction:
        """Build the exact affine reduction for one load factor."""

        if not np.isfinite(load_factor) or load_factor < 0.0:
            raise ValueError("load_factor must be finite and non-negative.")
        function, displacement_space, parent_map, block_size = (
            self._displacement_layout()
        )
        space = function.function_space
        if space.mesh.comm.size != 1:
            raise RuntimeError(
                "Explicit AffineReduction is a serial backend. "
                "Use distributed_reduction() under MPI."
            )
        dof_coordinates = np.asarray(
            displacement_space.tabulate_dof_coordinates(), dtype=float
        )
        full_size = int(function.x.array.size)
        node_to_block = _match_nodes_to_dof_blocks(
            self.nodes,
            dof_coordinates,
            labels=_equation_node_labels(self.equations) | set(self.control_nodes),
            tolerance=self.tolerance,
        )

        relations: dict[int, dict[int, float]] = {}
        for equation in self.equations.equations:
            slave_term = equation.terms[0]
            collapsed_slave = _scalar_dof(
                slave_term.node,
                slave_term.dof,
                node_to_block,
                block_size,
            )
            slave = int(parent_map[collapsed_slave])
            dependency: dict[int, float] = {}
            for term in equation.terms[1:]:
                collapsed_dof = _scalar_dof(
                    term.node,
                    term.dof,
                    node_to_block,
                    block_size,
                )
                dof = int(parent_map[collapsed_dof])
                dependency[dof] = dependency.get(dof, 0.0) - (
                    term.coefficient / slave_term.coefficient
                )
            relations[slave] = dependency

        prescribed: dict[int, float] = {}
        for (label, component), value in self.prescribed_values_at(load_factor).items():
            block = node_to_block[label]
            parent_dof = int(parent_map[block * block_size + component])
            prescribed[parent_dof] = float(value)
        conflicts = set(relations) & set(prescribed)
        if conflicts:
            raise ValueError(
                "Abaqus equation slave dofs cannot also be prescribed control dofs: "
                f"{sorted(conflicts)[:8]}."
            )
        return _build_reduction(full_size, relations, prescribed)

    def initial_reduced_values(
        self,
        reduction: AffineReduction,
        deformation_gradient,
    ) -> np.ndarray:
        """Sample affine displacement while initializing pressure to zero."""

        _, displacement_space, parent_map, block_size = self._displacement_layout()
        F = _deformation_gradient(deformation_gradient, block_size)
        coordinates = np.asarray(
            displacement_space.tabulate_dof_coordinates(), dtype=float
        )
        origin = self.nodes.coordinate(int(self.anchor_node))[:block_size]
        parent_to_displacement = {
            int(parent): local for local, parent in enumerate(parent_map)
        }
        values = np.zeros(reduction.reduced_size, dtype=PETSc.ScalarType)
        for index, full_dof in enumerate(reduction.independent_full_dofs):
            local = parent_to_displacement.get(int(full_dof))
            if local is None:
                continue
            block, component = divmod(local, block_size)
            values[index] = (
                (F - np.eye(block_size)) @ (coordinates[block] - origin)
            )[component]
        return values

    def distributed_reduction(self) -> DistributedAffineReduction:
        """Build an ownership-aware correction MPC from ``*EQUATION`` data."""

        if self.is_mixed:
            raise NotImplementedError(
                "Distributed affine MPC for a mixed displacement-pressure "
                "space is not yet implemented. Run this C3D10H periodic route "
                "in serial; ordinary C3D10 displacement MPC remains parallel."
            )
        if self.has_free_macro_dofs:
            raise NotImplementedError(
                "Distributed affine MPC with free macroscopic control dofs is "
                "not yet implemented; run this mixed-control route in serial."
            )

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
            self.prescribed_control_dofs,
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
        # DOLFINx records the owned/ghost split of a Dirichlet dof array as a
        # single prefix position.  Preserve that contract explicitly: local
        # owned blocks are numbered before ghosts, whereas control-node order
        # is unrelated to partition ownership.  Passing an unsorted sequence
        # such as ``[ghost, ghost, owned]`` makes the owned prefix appear empty
        # and leaves the corresponding global matrix rows unconstrained.
        control_blocks = np.unique(
            control_local[control_local >= 0]
        ).astype(np.int32, copy=False)
        control_bc = fem.dirichletbc(
            np.zeros(block_size, dtype=PETSc.ScalarType),
            control_blocks,
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

        function, displacement_space, parent_map, block_size = (
            self._displacement_layout()
        )
        coordinates = np.asarray(
            displacement_space.tabulate_dof_coordinates(), dtype=float
        )
        origin = self.nodes.coordinate(int(self.anchor_node))[:block_size]
        delta_gradient = self.deformation_gradient_at(
            target_factor
        ) - self.deformation_gradient_at(start_factor)
        increment = (
            (coordinates[:, :block_size] - origin) @ delta_gradient.T
        )
        flattened = increment.reshape(-1)
        if flattened.size != parent_map.size:
            raise RuntimeError("Displacement values and dof coordinates do not align.")
        function.x.array[parent_map] += flattened
        function.x.scatter_forward()

    def mismatch(self, load_factor: float = 1.0) -> float:
        """Return the maximum absolute original ``*EQUATION`` residual."""

        function = (
            self.target.collapsed_displacement()
            if self.is_mixed
            else field_api.unwrap(self.target)
        )
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
        values = {
            "name": self.name,
            "kind": "abaqus_periodic_constraint",
            "enforcement": "exact_affine_elimination",
            "equations": self.equations.summary(),
            "anchor_node": self.anchor_node,
            "reference_nodes": self.reference_nodes,
            "macro_control_kind": (
                "mixed_prescribed_and_free"
                if self.has_free_macro_dofs
                else "prescribed_deformation_gradient"
            ),
            "nominal_deformation_gradient": self.deformation_gradient.tolist(),
            "control_displacements": [
                list(row) for row in self.control_displacements
            ],
            "free_macro_dofs": [
                {"node": int(label), "component": component + 1}
                for label, row in zip(self.reference_nodes, self.control_displacements)
                for component, value in enumerate(row)
                if value is None
            ],
            "reference_cell_volume": self.reference_cell_volume,
            "unknown_layout": (
                "mixed_displacement_pressure" if self.is_mixed else "displacement"
            ),
            "supports_parallel": (
                _dolfinx_mpc_available()
                and not self.is_mixed
                and not self.has_free_macro_dofs
            ),
        }
        if not self.has_free_macro_dofs:
            values["target_deformation_gradient"] = self.deformation_gradient.tolist()
        if self.deformation_gradient_path is not None:
            values["macro_control_kind"] = "prescribed_deformation_gradient_path"
            values["deformation_gradient_path"] = (
                self.deformation_gradient_path.summary()
            )
        return values


def abaqus_periodic_cell(
    target,
    *,
    nodes: AbaqusNodeTable,
    equations: AbaqusEquationSet,
    anchor_node: int,
    reference_nodes,
    deformation_gradient=None,
    deformation_gradient_path: DeformationGradientPath | None = None,
    control_displacements=None,
    tolerance: float = 1.0e-9,
    name: str = "abaqus_periodic_cell",
) -> AbaqusPeriodicConstraint:
    """Create exact periodic equations and explicit macro-control semantics."""

    return AbaqusPeriodicConstraint(
        target=target,
        nodes=nodes,
        equations=equations,
        anchor_node=int(anchor_node),
        reference_nodes=tuple(int(node) for node in reference_nodes),
        deformation_gradient=(
            None
            if deformation_gradient is None
            else np.asarray(deformation_gradient, dtype=float)
        ),
        control_displacements=control_displacements,
        deformation_gradient_path=deformation_gradient_path,
        tolerance=float(tolerance),
        name=name,
    )


def _minimum_linear_path_determinant(left, right) -> float:
    """Return the minimum determinant along ``left + s * (right-left)``."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    dimension = int(first.shape[0])
    samples = np.linspace(0.0, 1.0, dimension + 1)
    determinants = np.asarray(
        [np.linalg.det(first + value * (second - first)) for value in samples],
        dtype=float,
    )
    coefficients = np.polynomial.polynomial.polyfit(
        samples,
        determinants,
        deg=dimension,
    )
    derivative = np.polynomial.polynomial.polyder(coefficients)
    candidates = [0.0, 1.0]
    for root in np.polynomial.polynomial.polyroots(derivative):
        if abs(float(np.imag(root))) <= 1.0e-10:
            value = float(np.real(root))
            if 0.0 < value < 1.0:
                candidates.append(value)
    return min(
        float(np.linalg.det(first + value * (second - first)))
        for value in candidates
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


def _finite_control_value(value) -> float:
    selected = float(value)
    if not np.isfinite(selected):
        raise ValueError("Periodic control displacements must be finite or None.")
    return selected


def _semantic_relations(
    equations: AbaqusEquationSet,
    prescribed_controls,
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
        (int(label), int(component))
        for label, component in prescribed_controls
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
