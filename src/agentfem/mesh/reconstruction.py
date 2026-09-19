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


def reconstruct_cell_gradient(
    domain,
    cell_values,
    *,
    rings: int = 2,
    weight_power: float = 1.0,
    condition_limit: float = 1.0e10,
) -> CellGradientReconstruction:
    """Reconstruct owned-cell gradients from local/ghost cell-center values.

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
    values = np.asarray(cell_values, dtype=float)
    if values.ndim < 1 or values.shape[0] != total or not np.all(np.isfinite(values)):
        raise ValueError(
            "cell_values must be finite with one leading entry for every local "
            "and ghost cell."
        )
    flattened = values.reshape((total, -1))
    cells = np.arange(total, dtype=np.int32)
    centroids = np.asarray(
        dolfinx_mesh.compute_midpoints(domain, tdim, cells),
        dtype=float,
    )[:, :gdim]
    stencil = cell_stencil_neighborhood(domain)
    graph = _neighbor_graph(stencil.pairs, total)

    gradients = np.empty((owned, flattened.shape[1], gdim), dtype=float)
    bases = np.empty((owned, gdim, tdim), dtype=float)
    counts = np.empty(owned, dtype=np.int32)
    ranks = np.empty(owned, dtype=np.int32)
    conditions = np.empty(owned, dtype=float)
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
        delta = flattened[np.asarray(neighbors)] - flattened[cell]
        weighted_delta = delta * np.sqrt(weights)[:, None]
        coefficients, _, local_rank, local_singular = np.linalg.lstsq(
            weighted_coordinates,
            weighted_delta,
            rcond=None,
        )
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
        gradients[cell] = coefficients.T @ basis.T
        bases[cell] = basis
        counts[cell] = len(neighbors)
        ranks[cell] = int(local_rank)
        conditions[cell] = condition

    return CellGradientReconstruction(
        gradients=gradients.reshape((owned, *values.shape[1:], gdim)),
        tangent_bases=bases,
        neighbor_counts=counts,
        ranks=ranks,
        condition_numbers=conditions,
        rings=rings,
        weight_power=weight_power,
    )


__all__ = ["CellGradientReconstruction", "reconstruct_cell_gradient"]
