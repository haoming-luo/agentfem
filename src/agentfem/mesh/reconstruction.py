# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Auditable cell-neighbour gradient reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import mesh as dolfinx_mesh

from .neighborhood import cell_stencil_neighborhood


@dataclass(frozen=True)
class CellGradientReconstruction:
    """Owned-cell gradients and conditioning evidence for one reconstruction."""

    gradients: np.ndarray
    tangent_bases: np.ndarray
    neighbor_counts: np.ndarray
    ranks: np.ndarray
    condition_numbers: np.ndarray
    rings: int
    weight_power: float

    def __post_init__(self) -> None:
        gradients = np.asarray(self.gradients, dtype=float)
        bases = np.asarray(self.tangent_bases, dtype=float)
        counts = np.asarray(self.neighbor_counts, dtype=np.int32)
        ranks = np.asarray(self.ranks, dtype=np.int32)
        conditions = np.asarray(self.condition_numbers, dtype=float)
        owned = gradients.shape[0]
        if bases.ndim != 3 or bases.shape[0] != owned:
            raise ValueError("tangent_bases require one matrix per owned cell.")
        if counts.shape != (owned,) or ranks.shape != (owned,):
            raise ValueError("neighbor_counts and ranks require one value per cell.")
        if conditions.shape != (owned,):
            raise ValueError("condition_numbers require one value per cell.")
        if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(bases)):
            raise ValueError("Reconstructed gradients and bases must be finite.")
        if not np.all(np.isfinite(conditions)) or np.any(conditions < 1.0):
            raise ValueError("Reconstruction condition numbers must be finite and >= 1.")
        object.__setattr__(self, "gradients", gradients.copy())
        object.__setattr__(self, "tangent_bases", bases.copy())
        object.__setattr__(self, "neighbor_counts", counts.copy())
        object.__setattr__(self, "ranks", ranks.copy())
        object.__setattr__(self, "condition_numbers", conditions.copy())

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "cell_gradient_reconstruction",
            "owned_cells": int(self.gradients.shape[0]),
            "gradient_shape": list(self.gradients.shape),
            "rings": self.rings,
            "weight_power": self.weight_power,
            "minimum_neighbors": int(np.min(self.neighbor_counts)),
            "maximum_neighbors": int(np.max(self.neighbor_counts)),
            "minimum_rank": int(np.min(self.ranks)),
            "maximum_condition_number": float(np.max(self.condition_numbers)),
            "method": "weighted_least_squares_in_local_tangent_basis",
            "identity_scope": "runtime_partition",
        }


@dataclass(frozen=True)
class CellGradientStencil:
    """One owned cell's compact linear gradient stencil."""

    cell: int
    neighbors: tuple[int, ...]
    neighbor_weights: np.ndarray
    center_weight: np.ndarray
    tangent_basis: np.ndarray
    rank: int
    condition_number: float

    def __post_init__(self) -> None:
        neighbors = tuple(int(value) for value in self.neighbors)
        weights = np.asarray(self.neighbor_weights, dtype=float)
        center = np.asarray(self.center_weight, dtype=float)
        basis = np.asarray(self.tangent_basis, dtype=float)
        if weights.ndim != 2 or weights.shape[0] != len(neighbors):
            raise ValueError("neighbor_weights require one row per neighbor.")
        if center.shape != (weights.shape[1],):
            raise ValueError("center_weight shape must match gradient dimension.")
        if basis.ndim != 2 or basis.shape[0] != weights.shape[1]:
            raise ValueError("tangent_basis has incompatible geometric dimension.")
        if not all(np.all(np.isfinite(item)) for item in (weights, center, basis)):
            raise ValueError("Cell gradient stencil coefficients must be finite.")
        if not np.allclose(center + np.sum(weights, axis=0), 0.0, atol=1.0e-13):
            raise ValueError("Cell gradient stencil must annihilate constant fields.")
        object.__setattr__(self, "cell", int(self.cell))
        object.__setattr__(self, "neighbors", neighbors)
        object.__setattr__(self, "neighbor_weights", weights.copy())
        object.__setattr__(self, "center_weight", center.copy())
        object.__setattr__(self, "tangent_basis", basis.copy())
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "condition_number", float(self.condition_number))


@dataclass(frozen=True)
class CellGradientOperator:
    """Reusable local sparse operator from cell values to owned-cell gradients."""

    stencils: tuple[CellGradientStencil, ...]
    total_cells: int
    geometric_dimension: int
    topological_dimension: int
    rings: int
    weight_power: float
    condition_limit: float

    @property
    def owned_cells(self) -> int:
        return len(self.stencils)

    @property
    def nonzero_blocks(self) -> int:
        return sum(len(item.neighbors) + 1 for item in self.stencils)

    def apply(self, cell_values) -> CellGradientReconstruction:
        """Apply cached geometry weights to scalar or vector cell data."""

        values = np.asarray(cell_values, dtype=float)
        if (
            values.ndim < 1
            or values.shape[0] != self.total_cells
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "cell_values must be finite with one leading entry for every "
                "local and ghost cell."
            )
        flattened = values.reshape((self.total_cells, -1))
        gradients = np.empty(
            (self.owned_cells, flattened.shape[1], self.geometric_dimension),
            dtype=float,
        )
        for index, stencil in enumerate(self.stencils):
            neighbors = np.asarray(stencil.neighbors, dtype=np.int32)
            gradients[index] = (
                flattened[neighbors].T @ stencil.neighbor_weights
                + flattened[stencil.cell][:, None] * stencil.center_weight
            )
        return CellGradientReconstruction(
            gradients=gradients.reshape(
                (self.owned_cells, *values.shape[1:], self.geometric_dimension)
            ),
            tangent_bases=np.asarray(
                [item.tangent_basis for item in self.stencils],
                dtype=float,
            ),
            neighbor_counts=np.asarray(
                [len(item.neighbors) for item in self.stencils],
                dtype=np.int32,
            ),
            ranks=np.asarray([item.rank for item in self.stencils], dtype=np.int32),
            condition_numbers=np.asarray(
                [item.condition_number for item in self.stencils],
                dtype=float,
            ),
            rings=self.rings,
            weight_power=self.weight_power,
        )

    def apply_adjoint(self, gradient_duals) -> np.ndarray:
        """Apply the exact transpose to owned-cell gradient dual values.

        The returned leading axis contains local and ghost cells.  On MPI,
        ghost contributions are intentionally *not* communicated here; a
        backend assembly adapter must reverse-scatter them to cell owners.
        Keeping that distinction visible prevents a rank-local transpose from
        being mistaken for a globally assembled residual.
        """

        duals = np.asarray(gradient_duals, dtype=float)
        if (
            duals.ndim < 2
            or duals.shape[0] != self.owned_cells
            or duals.shape[-1] != self.geometric_dimension
            or not np.all(np.isfinite(duals))
        ):
            raise ValueError(
                "gradient_duals must be finite with shape "
                "(owned_cells, ..., geometric_dimension)."
            )
        value_shape = duals.shape[1:-1]
        flattened = duals.reshape(
            (self.owned_cells, -1, self.geometric_dimension)
        )
        result = np.zeros((self.total_cells, flattened.shape[1]), dtype=float)
        for index, stencil in enumerate(self.stencils):
            neighbor_contributions = flattened[index] @ stencil.neighbor_weights.T
            for neighbor_index, neighbor in enumerate(stencil.neighbors):
                result[neighbor] += neighbor_contributions[:, neighbor_index]
            result[stencil.cell] += flattened[index] @ stencil.center_weight
        return result.reshape((self.total_cells, *value_shape))

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "cell_gradient_operator",
            "owned_cells": self.owned_cells,
            "total_local_and_ghost_cells": self.total_cells,
            "geometric_dimension": self.geometric_dimension,
            "topological_dimension": self.topological_dimension,
            "rings": self.rings,
            "weight_power": self.weight_power,
            "condition_limit": self.condition_limit,
            "nonzero_blocks": self.nonzero_blocks,
            "maximum_condition_number": max(
                item.condition_number for item in self.stencils
            ),
            "linearity": "geometry_cached_linear_action_on_cell_values",
            "adjoint": "local_and_ghost_contributions_require_mpi_reverse_scatter",
            "identity_scope": "runtime_partition",
        }


def _neighbor_graph(pairs, size: int) -> tuple[set[int], ...]:
    graph = tuple(set() for _ in range(size))
    for pair in pairs:
        left, right = pair.cell_locals
        graph[left].add(right)
        graph[right].add(left)
    return graph


def _ring_neighbors(graph, cell: int, rings: int) -> tuple[int, ...]:
    visited = {cell}
    frontier = {cell}
    for _ in range(rings):
        following = set()
        for current in frontier:
            following.update(graph[current])
        following.difference_update(visited)
        visited.update(following)
        frontier = following
        if not frontier:
            break
    visited.remove(cell)
    return tuple(sorted(visited))


def cell_gradient_operator(
    domain,
    *,
    rings: int = 2,
    weight_power: float = 1.0,
    condition_limit: float = 1.0e10,
) -> CellGradientOperator:
    """Precompute a reusable, compact cell-gradient operator from geometry.

    A local SVD supplies the best-fit tangent basis, so surfaces embedded in a
    higher-dimensional coordinate space do not require an arbitrary global
    projection.  The least-squares system must span the mesh topological
    dimension and remain below the caller-declared condition limit.  Failure
    is explicit; AgentFEM does not return zero curvature at an under-resolved
    boundary cell.
    """

    domain = getattr(domain, "domain", domain)
    rings = int(rings)
    weight_power = float(weight_power)
    condition_limit = float(condition_limit)
    if rings < 1:
        raise ValueError("rings must be at least one.")
    if not np.isfinite(weight_power) or weight_power < 0.0:
        raise ValueError("weight_power must be finite and nonnegative.")
    if not np.isfinite(condition_limit) or condition_limit < 1.0:
        raise ValueError("condition_limit must be finite and at least one.")

    topology = domain.topology
    tdim = int(topology.dim)
    gdim = int(domain.geometry.dim)
    cell_map = topology.index_map(tdim)
    owned = int(cell_map.size_local)
    total = owned + int(cell_map.num_ghosts)
    cells = np.arange(total, dtype=np.int32)
    centroids = np.asarray(
        dolfinx_mesh.compute_midpoints(domain, tdim, cells),
        dtype=float,
    )[:, :gdim]
    stencil = cell_stencil_neighborhood(domain)
    graph = _neighbor_graph(stencil.pairs, total)

    stencils = []
    for cell in range(owned):
        neighbors = _ring_neighbors(graph, cell, rings)
        if len(neighbors) < tdim:
            raise RuntimeError(
                f"Cell {cell} has only {len(neighbors)} reconstruction neighbors; "
                f"need at least {tdim}. Increase rings or refine the mesh."
            )
        offsets = centroids[np.asarray(neighbors)] - centroids[cell]
        _, singular, right = np.linalg.svd(offsets, full_matrices=False)
        tolerance = max(offsets.shape) * np.finfo(float).eps * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        if rank < tdim:
            raise RuntimeError(
                f"Cell {cell} reconstruction stencil has rank {rank}; need {tdim}."
            )
        basis = right[:tdim].T
        projected = offsets @ basis
        distances = np.linalg.norm(projected, axis=1)
        if np.any(distances <= np.finfo(float).eps):
            raise RuntimeError(f"Cell {cell} has coincident reconstruction centroids.")
        weights = distances ** (-weight_power)
        weighted_coordinates = projected * np.sqrt(weights)[:, None]
        pseudo_inverse = np.linalg.pinv(weighted_coordinates)
        local_singular = np.linalg.svd(weighted_coordinates, compute_uv=False)
        local_tolerance = (
            max(weighted_coordinates.shape)
            * np.finfo(float).eps
            * local_singular[0]
        )
        local_rank = int(np.count_nonzero(local_singular > local_tolerance))
        if int(local_rank) < tdim:
            raise RuntimeError(
                f"Cell {cell} weighted reconstruction has rank {local_rank}; need {tdim}."
            )
        condition = float(local_singular[0] / local_singular[tdim - 1])
        if condition > condition_limit:
            raise RuntimeError(
                f"Cell {cell} reconstruction condition number {condition:g} exceeds "
                f"the declared limit {condition_limit:g}."
            )
        neighbor_weights = (
            basis @ pseudo_inverse @ np.diag(np.sqrt(weights))
        ).T
        stencils.append(
            CellGradientStencil(
                cell=cell,
                neighbors=neighbors,
                neighbor_weights=neighbor_weights,
                center_weight=-np.sum(neighbor_weights, axis=0),
                tangent_basis=basis,
                rank=local_rank,
                condition_number=condition,
            )
        )

    return CellGradientOperator(
        stencils=tuple(stencils),
        total_cells=total,
        geometric_dimension=gdim,
        topological_dimension=tdim,
        rings=rings,
        weight_power=weight_power,
        condition_limit=condition_limit,
    )


def reconstruct_cell_gradient(
    domain,
    cell_values,
    *,
    rings: int = 2,
    weight_power: float = 1.0,
    condition_limit: float = 1.0e10,
) -> CellGradientReconstruction:
    """Build and apply a cell-gradient operator in one convenience call."""

    return cell_gradient_operator(
        domain,
        rings=rings,
        weight_power=weight_power,
        condition_limit=condition_limit,
    ).apply(cell_values)


__all__ = [
    "CellGradientOperator",
    "CellGradientReconstruction",
    "CellGradientStencil",
    "cell_gradient_operator",
    "reconstruct_cell_gradient",
]
