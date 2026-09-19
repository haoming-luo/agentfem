# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
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

    geometry = mesh.cell_neighborhood_geometry(domain, evidence)
    assert geometry.geometric_dimension == 2
    assert len(geometry.facets) == 1
    shared = geometry.facets[0]
    assert shared.facet_midpoint == (0.5, 0.5)
    assert shared.cell_centroids == ((0.25, 0.5), (0.75, 0.5))
    assert shared.center_vector == (0.5, 0.0)
    assert shared.center_distance == 0.5
    assert shared.center_direction == (1.0, 0.0)


def test_neighborhood_geometry_is_translation_invariant():
    first = dolfinx_mesh.create_rectangle(
        MPI.COMM_SELF,
        ((0.0, 0.0), (2.0, 1.0)),
        (2, 1),
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    shifted = dolfinx_mesh.create_rectangle(
        MPI.COMM_SELF,
        ((3.0, -4.0), (5.0, -3.0)),
        (2, 1),
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    original = mesh.cell_neighborhood_geometry(first).facets[0]
    translated = mesh.cell_neighborhood_geometry(shifted).facets[0]

    assert translated.center_vector == original.center_vector
    assert translated.center_distance == original.center_distance
    assert translated.center_direction == original.center_direction


def test_pair_difference_is_exact_for_affine_scalar_and_vector_cell_fields():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        3,
        2,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    geometry = mesh.cell_neighborhood_geometry(domain)
    cell_count = domain.topology.index_map(domain.topology.dim).size_local
    cells = np.arange(cell_count, dtype=np.int32)
    centroids = dolfinx_mesh.compute_midpoints(
        domain,
        domain.topology.dim,
        cells,
    )[:, :2]
    scalar_gradient = np.array((2.0, -3.0))
    scalar = centroids @ scalar_gradient + 4.0
    scalar_result = mesh.cell_pair_directional_difference(geometry, scalar)
    np.testing.assert_allclose(
        scalar_result.values,
        scalar_result.directions @ scalar_gradient,
        atol=1.0e-14,
    )

    vector_gradient = np.array(((2.0, -3.0), (0.5, 1.25), (-1.0, 4.0)))
    vector = centroids @ vector_gradient.T + np.array((4.0, -2.0, 1.0))
    vector_result = mesh.cell_pair_directional_difference(geometry, vector)
    np.testing.assert_allclose(
        vector_result.values,
        vector_result.directions @ vector_gradient.T,
        atol=1.0e-14,
    )
    assert vector_result.as_dict()["definition"].startswith("(right_cell_value")


def test_pair_difference_rejects_missing_ghost_or_local_values():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 2, 1)
    geometry = mesh.cell_neighborhood_geometry(domain)
    with pytest.raises(ValueError, match="does not include"):
        mesh.cell_pair_directional_difference(geometry, np.zeros(1))


def test_fem_mesh_facade_is_accepted():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 3)
    evidence = mesh.cell_neighborhood(mesh.FEMMesh(domain))

    assert evidence.topological_dimension == 1
    assert evidence.owned_interior_facets == 2
    assert evidence.owned_exterior_facets == 2


def test_serial_stencil_matches_unique_owned_interior_pairs():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 3, 2)
    unique = mesh.cell_neighborhood(domain)
    stencil = mesh.cell_stencil_neighborhood(domain)

    assert stencil.ghost_cells == 0
    assert stencil.owned_cells == 12
    assert tuple(item.facet_global for item in stencil.pairs) == tuple(
        item.facet_global for item in unique.pairs
    )
    assert all(item.facet_owned for item in stencil.pairs)
    assert all(all(item.cell_owned) for item in stencil.pairs)
    assert stencil.as_dict()["identity_scope"] == "local_computation_stencil"


@pytest.mark.parametrize("cell_type", ["triangle", "quadrilateral"])
def test_cell_gradient_reconstruction_is_affine_exact(cell_type):
    kind = getattr(dolfinx_mesh.CellType, cell_type)
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        3,
        cell_type=kind,
    )
    cell_map = domain.topology.index_map(domain.topology.dim)
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    centroids = dolfinx_mesh.compute_midpoints(
        domain,
        domain.topology.dim,
        cells,
    )[:, :2]
    gradient = np.array(((2.0, -3.0), (0.5, 1.25), (-1.0, 4.0)))
    values = centroids @ gradient.T + np.array((4.0, -2.0, 1.0))

    reconstructed = mesh.reconstruct_cell_gradient(domain, values, rings=2)

    np.testing.assert_allclose(
        reconstructed.gradients,
        np.broadcast_to(gradient, reconstructed.gradients.shape),
        atol=5.0e-14,
    )
    assert np.all(reconstructed.ranks == 2)
    assert reconstructed.as_dict()["method"].startswith("weighted_least_squares")

    operator = mesh.cell_gradient_operator(domain, rings=2)
    shifted_values = values + np.array((10.0, -3.0, 7.0))
    reused = operator.apply(shifted_values)
    np.testing.assert_allclose(reused.gradients, reconstructed.gradients, atol=5.0e-14)
    assert operator.nonzero_blocks > operator.owned_cells
    assert operator.as_dict()["linearity"].startswith("geometry_cached")


def test_cell_gradient_reconstruction_fails_closed_on_insufficient_stencil():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 1)
    values = np.array((0.5,))
    with pytest.raises(RuntimeError, match="only 0 reconstruction neighbors"):
        mesh.reconstruct_cell_gradient(domain, values, rings=1)


@pytest.mark.parametrize("components", [(), (3,)])
def test_cell_gradient_operator_adjoint_satisfies_inner_product_identity(components):
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        3,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    operator = mesh.cell_gradient_operator(domain, rings=2)
    rng = np.random.default_rng(20260920)
    values = rng.normal(size=(operator.total_cells, *components))
    duals = rng.normal(
        size=(operator.owned_cells, *components, operator.geometric_dimension)
    )

    gradients = operator.apply(values).gradients
    transpose = operator.apply_adjoint(duals)

    assert np.vdot(gradients, duals) == pytest.approx(
        np.vdot(values, transpose),
        rel=1.0e-13,
        abs=1.0e-13,
    )


def test_cell_gradient_operator_adjoint_validates_dual_shape():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    operator = mesh.cell_gradient_operator(domain)
    with pytest.raises(ValueError, match="gradient_duals"):
        operator.apply_adjoint(np.zeros((operator.owned_cells, 3)))
