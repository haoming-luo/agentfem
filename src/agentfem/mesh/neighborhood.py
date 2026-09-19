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


__all__ = ["CellNeighborhood", "InteriorFacetPair", "cell_neighborhood"]
