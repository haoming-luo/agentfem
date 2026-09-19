# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Partition-aware cell neighborhoods built from mesh topology.

Neighbouring-element shell methods, discontinuous Galerkin operators, local
error estimators, and crack-front algorithms all need the same primitive:
which two cells meet at an owned interior facet, and which local facet belongs
to each cell.  This module exposes that runtime topology without pretending
that partition-dependent global numbering is scientific model identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InteriorFacetPair:
    """Two cells adjacent to one owned interior facet on the current partition."""

    facet_local: int
    facet_global: int
    cell_locals: tuple[int, int]
    cell_globals: tuple[int, int]
    cell_local_facets: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "interior_facet_pair",
            "facet_local": self.facet_local,
            "facet_global": self.facet_global,
            "cell_locals": list(self.cell_locals),
            "cell_globals": list(self.cell_globals),
            "cell_local_facets": list(self.cell_local_facets),
            "identity_scope": "runtime_partition",
        }


@dataclass(frozen=True)
class CellNeighborhood:
    """Owned interior-facet adjacency plus explicit partition evidence."""

    topological_dimension: int
    pairs: tuple[InteriorFacetPair, ...]
    owned_facets: int
    owned_exterior_facets: int
    ghost_cells: int

    @property
    def owned_interior_facets(self) -> int:
        return len(self.pairs)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "cell_neighborhood",
            "topological_dimension": self.topological_dimension,
            "owned_facets": self.owned_facets,
            "owned_interior_facets": self.owned_interior_facets,
            "owned_exterior_facets": self.owned_exterior_facets,
            "ghost_cells": self.ghost_cells,
            "pairs": [item.as_dict() for item in self.pairs],
            "identity_scope": "runtime_partition",
        }


@dataclass(frozen=True)
class InteriorFacetGeometry:
    """Geometric scale and direction for one interior-facet cell pair."""

    pair: InteriorFacetPair
    facet_midpoint: tuple[float, ...]
    cell_centroids: tuple[tuple[float, ...], tuple[float, ...]]
    center_vector: tuple[float, ...]
    center_distance: float
    center_direction: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "interior_facet_geometry",
            "pair": self.pair.as_dict(),
            "facet_midpoint": list(self.facet_midpoint),
            "cell_centroids": [list(value) for value in self.cell_centroids],
            "center_vector": list(self.center_vector),
            "center_distance": self.center_distance,
            "center_direction": list(self.center_direction),
            "identity_scope": "runtime_partition",
        }


@dataclass(frozen=True)
class CellNeighborhoodGeometry:
    """Geometric evidence for every pair in a :class:`CellNeighborhood`."""

    geometric_dimension: int
    facets: tuple[InteriorFacetGeometry, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "cell_neighborhood_geometry",
            "geometric_dimension": self.geometric_dimension,
            "interior_facet_count": len(self.facets),
            "facets": [item.as_dict() for item in self.facets],
            "identity_scope": "runtime_partition",
        }


@dataclass(frozen=True)
class CellPairDifference:
    """Directional cell-value difference on every interior-facet pair."""

    values: np.ndarray
    directions: np.ndarray
    distances: np.ndarray
    facet_globals: tuple[int, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        directions = np.asarray(self.directions, dtype=float)
        distances = np.asarray(self.distances, dtype=float)
        facet_globals = tuple(int(value) for value in self.facet_globals)
        count = len(facet_globals)
        if values.ndim < 1 or values.shape[0] != count:
            raise ValueError("CellPairDifference values require one row per facet.")
        if directions.ndim != 2 or directions.shape[0] != count:
            raise ValueError("CellPairDifference directions require one row per facet.")
        if distances.shape != (count,) or np.any(distances <= 0.0):
            raise ValueError("CellPairDifference distances must be positive.")
        if not all(
            np.all(np.isfinite(item)) for item in (values, directions, distances)
        ):
            raise ValueError("CellPairDifference arrays must be finite.")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "directions", directions.copy())
        object.__setattr__(self, "distances", distances.copy())
        object.__setattr__(self, "facet_globals", facet_globals)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "cell_pair_directional_difference",
            "values": self.values.tolist(),
            "directions": self.directions.tolist(),
            "distances": self.distances.tolist(),
            "facet_globals": list(self.facet_globals),
            "definition": "(right_cell_value-left_cell_value)/center_distance",
            "identity_scope": "runtime_partition",
        }


def _local_facet_number(cell_to_facets, cell: int, facet: int) -> int:
    facets = np.asarray(cell_to_facets.links(cell), dtype=np.int64)
    locations = np.flatnonzero(facets == facet)
    if locations.size != 1:
        raise RuntimeError(
            "Mesh topology does not expose exactly one local-facet position "
            f"for cell {cell} and facet {facet}."
        )
    return int(locations[0])


def cell_neighborhood(domain) -> CellNeighborhood:
    """Return every owned interior facet and its two adjacent cells.

    The mesh must provide ghost cells across partition boundaries.  AgentFEM
    fails closed when an owned facet has only one locally visible cell but is
    not marked exterior; silently treating that facet as a boundary would make
    neighbouring-element and DG operators partition dependent.
    """

    domain = getattr(domain, "domain", domain)
    topology = domain.topology
    tdim = int(topology.dim)
    if tdim < 1:
        raise ValueError("cell_neighborhood requires a mesh of dimension at least one.")
    fdim = tdim - 1
    topology.create_entities(fdim)
    topology.create_connectivity(fdim, tdim)
    topology.create_connectivity(tdim, fdim)
    facet_to_cells = topology.connectivity(fdim, tdim)
    cell_to_facets = topology.connectivity(tdim, fdim)
    if facet_to_cells is None or cell_to_facets is None:
        raise RuntimeError("Required cell/facet mesh connectivity is unavailable.")

    facet_map = topology.index_map(fdim)
    cell_map = topology.index_map(tdim)
    owned_facets = int(facet_map.size_local)
    owned_cells = int(cell_map.size_local)
    ghost_cells = int(cell_map.num_ghosts)
    facet_globals = facet_map.local_to_global(
        np.arange(owned_facets, dtype=np.int32)
    )
    cell_globals = cell_map.local_to_global(
        np.arange(owned_cells + ghost_cells, dtype=np.int32)
    )

    pairs = []
    exterior = 0
    for facet in range(owned_facets):
        cells = np.asarray(facet_to_cells.links(facet), dtype=np.int32)
        if cells.size == 1:
            exterior += 1
            continue
        if cells.size != 2:
            raise RuntimeError(
                "Neighbour operators require a manifold mesh with one or two "
                f"cells per facet; owned facet {facet} has {cells.size}."
            )
        global_ids = tuple(int(cell_globals[int(cell)]) for cell in cells)
        order = np.argsort(global_ids)
        ordered_cells = tuple(int(cells[index]) for index in order)
        ordered_globals = tuple(global_ids[index] for index in order)
        local_facets = tuple(
            _local_facet_number(cell_to_facets, cell, facet)
            for cell in ordered_cells
        )
        pairs.append(
            InteriorFacetPair(
                facet_local=facet,
                facet_global=int(facet_globals[facet]),
                cell_locals=ordered_cells,
                cell_globals=ordered_globals,
                cell_local_facets=local_facets,
            )
        )
    pairs.sort(key=lambda item: item.facet_global)
    return CellNeighborhood(
        topological_dimension=tdim,
        pairs=tuple(pairs),
        owned_facets=owned_facets,
        owned_exterior_facets=exterior,
        ghost_cells=ghost_cells,
    )


def cell_neighborhood_geometry(
    domain,
    neighborhood: CellNeighborhood | None = None,
) -> CellNeighborhoodGeometry:
    """Attach centroids, facet midpoints, and pair distances to a neighborhood.

    Distances are computed in the mesh embedding coordinates, so the same
    contract works for planar meshes and two-dimensional surfaces embedded in
    three dimensions.  No curvature or finite-difference rule is inferred.
    """

    from dolfinx import mesh as dolfinx_mesh

    domain = getattr(domain, "domain", domain)
    selected = cell_neighborhood(domain) if neighborhood is None else neighborhood
    if not isinstance(selected, CellNeighborhood):
        raise TypeError("neighborhood must be a CellNeighborhood.")
    if selected.topological_dimension != int(domain.topology.dim):
        raise ValueError("Neighborhood topological dimension does not match domain.")
    gdim = int(domain.geometry.dim)
    facets = []
    for pair in selected.pairs:
        cells = np.asarray(pair.cell_locals, dtype=np.int32)
        facet = np.asarray((pair.facet_local,), dtype=np.int32)
        centroids = np.asarray(
            dolfinx_mesh.compute_midpoints(domain, domain.topology.dim, cells),
            dtype=float,
        )[:, :gdim]
        midpoint = np.asarray(
            dolfinx_mesh.compute_midpoints(
                domain,
                domain.topology.dim - 1,
                facet,
            ),
            dtype=float,
        )[0, :gdim]
        vector = centroids[1] - centroids[0]
        distance = float(np.linalg.norm(vector))
        if not np.isfinite(distance) or distance <= np.finfo(float).eps:
            raise RuntimeError(
                "Adjacent cells must have distinct finite centroids for "
                f"facet {pair.facet_local}."
            )
        facets.append(
            InteriorFacetGeometry(
                pair=pair,
                facet_midpoint=tuple(float(value) for value in midpoint),
                cell_centroids=tuple(
                    tuple(float(value) for value in centroid)
                    for centroid in centroids
                ),
                center_vector=tuple(float(value) for value in vector),
                center_distance=distance,
                center_direction=tuple(float(value / distance) for value in vector),
            )
        )
    return CellNeighborhoodGeometry(
        geometric_dimension=gdim,
        facets=tuple(facets),
    )


def cell_pair_directional_difference(
    geometry: CellNeighborhoodGeometry,
    cell_values,
) -> CellPairDifference:
    """Difference local/ghost cell values along each center-to-center line.

    ``cell_values`` follows current local cell numbering and must include ghost
    entries referenced by ``geometry``.  The caller owns ghost synchronization.
    The result is a directional difference, not a full gradient or a shell
    curvature; combining several directions is a separate reconstruction.
    """

    if not isinstance(geometry, CellNeighborhoodGeometry):
        raise TypeError("geometry must be CellNeighborhoodGeometry.")
    values = np.asarray(cell_values, dtype=float)
    if values.ndim < 1 or not np.all(np.isfinite(values)):
        raise ValueError("cell_values must be a finite array with a cell axis.")
    if not geometry.facets:
        return CellPairDifference(
            values=np.empty((0, *values.shape[1:]), dtype=float),
            directions=np.empty((0, geometry.geometric_dimension), dtype=float),
            distances=np.empty(0, dtype=float),
            facet_globals=(),
        )
    largest_cell = max(
        max(item.pair.cell_locals) for item in geometry.facets
    )
    if values.shape[0] <= largest_cell:
        raise ValueError(
            "cell_values does not include every local or ghost cell referenced "
            f"by the neighborhood; need index {largest_cell}."
        )
    differences = []
    directions = []
    distances = []
    facet_globals = []
    for item in geometry.facets:
        left, right = item.pair.cell_locals
        differences.append((values[right] - values[left]) / item.center_distance)
        directions.append(item.center_direction)
        distances.append(item.center_distance)
        facet_globals.append(item.pair.facet_global)
    return CellPairDifference(
        values=np.asarray(differences, dtype=float),
        directions=np.asarray(directions, dtype=float),
        distances=np.asarray(distances, dtype=float),
        facet_globals=tuple(facet_globals),
    )


__all__ = [
    "CellPairDifference",
    "CellNeighborhood",
    "CellNeighborhoodGeometry",
    "InteriorFacetGeometry",
    "InteriorFacetPair",
    "cell_neighborhood",
    "cell_neighborhood_geometry",
    "cell_pair_directional_difference",
]
