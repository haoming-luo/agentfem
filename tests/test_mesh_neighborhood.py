# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import mesh


def test_triangle_cell_neighborhood_preserves_both_local_facet_positions():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 2, 1)
    evidence = mesh.cell_neighborhood(domain)

    assert evidence.topological_dimension == 2
    assert evidence.owned_interior_facets == 3
    assert evidence.owned_exterior_facets == 6
    assert evidence.owned_facets == 9
    assert evidence.ghost_cells == 0
    assert len({item.facet_global for item in evidence.pairs}) == 3
    assert all(left < right for left, right in (item.cell_globals for item in evidence.pairs))
    assert all(
        0 <= local_facet <= 2
        for item in evidence.pairs
        for local_facet in item.cell_local_facets
    )
    assert evidence.as_dict()["identity_scope"] == "runtime_partition"


def test_quadrilateral_cell_neighborhood_has_one_shared_edge():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        2,
        1,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    evidence = mesh.cell_neighborhood(domain)

    assert evidence.owned_interior_facets == 1
    assert evidence.owned_exterior_facets == 6
    pair = evidence.pairs[0]
    assert pair.cell_globals == (0, 1)
    assert all(0 <= value <= 3 for value in pair.cell_local_facets)
    assert pair.as_dict()["identity_scope"] == "runtime_partition"


def test_fem_mesh_facade_is_accepted():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 3)
    evidence = mesh.cell_neighborhood(mesh.FEMMesh(domain))

    assert evidence.topological_dimension == 1
    assert evidence.owned_interior_facets == 2
    assert evidence.owned_exterior_facets == 2
