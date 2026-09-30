from __future__ import annotations

from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models


def _tetrahedral_cube(comm):
    return mesh.create_unit_cube(
        comm,
        1,
        1,
        1,
        cell_type=mesh.CellType.tetrahedron,
    )


def test_dolfinx_exterior_adapter_builds_outward_serial_cube_surface():
    domain = _tetrahedral_cube(MPI.COMM_SELF)

    partition = boundary_models.dolfinx_exterior_triangle_partition(domain)

    assert partition.global_facet_count == 12
    assert partition.local_facet_count == 12
    assert partition.summary()["ownership_method"].startswith("dolfinx_owned")
    assert partition.local_surface is not None
    assert partition.local_surface.summary()["closed"] is True
    query = np.asarray(
        (
            (-0.1, 0.3, 0.4),
            (1.1, 0.3, 0.4),
            (0.3, -0.1, 0.4),
            (0.3, 1.1, 0.4),
            (0.3, 0.4, -0.1),
            (0.3, 0.4, 1.1),
        )
    )
    projection = partition.local_surface.project(query)
    assert projection.valid.tolist() == [True] * 6
    np.testing.assert_allclose(projection.signed_gaps, 0.1, atol=1.0e-14)


def test_dolfinx_exterior_adapter_preserves_identity_when_input_order_changes():
    domain = _tetrahedral_cube(MPI.COMM_SELF)
    domain.topology.create_entities(domain.topology.dim - 1)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    exterior = mesh.exterior_facet_indices(domain.topology)

    forward = boundary_models.dolfinx_exterior_triangle_partition(
        domain, facets=exterior
    )
    reversed_partition = boundary_models.dolfinx_exterior_triangle_partition(
        domain, facets=exterior[::-1]
    )

    assert (
        forward.global_geometry_fingerprint
        == reversed_partition.global_geometry_fingerprint
    )
    assert forward.global_facet_identity_fingerprint == (
        reversed_partition.global_facet_identity_fingerprint
    )
    assert set(forward.local_facet_ids) == set(reversed_partition.local_facet_ids)


def test_dolfinx_exterior_adapter_rejects_internal_facet_selection():
    domain = _tetrahedral_cube(MPI.COMM_SELF)
    facet_dimension = domain.topology.dim - 1
    domain.topology.create_entities(facet_dimension)
    domain.topology.create_connectivity(facet_dimension, domain.topology.dim)
    facet_map = domain.topology.index_map(facet_dimension)
    exterior = set(int(value) for value in mesh.exterior_facet_indices(domain.topology))
    internal = next(
        index for index in range(facet_map.size_local) if index not in exterior
    )

    with pytest.raises(ValueError, match="owned exterior facets"):
        boundary_models.dolfinx_exterior_triangle_partition(
            domain, facets=np.asarray((internal,), dtype=np.int32)
        )


def test_dolfinx_exterior_adapter_rejects_unsupported_hexahedra():
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF,
        1,
        1,
        1,
        cell_type=mesh.CellType.hexahedron,
    )

    with pytest.raises(ValueError, match="only tetrahedral volume topology"):
        boundary_models.dolfinx_exterior_triangle_partition(domain)


def test_dolfinx_exterior_partition_drives_routed_search_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("DOLFINx boundary partition is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(
        comm,
        2,
        1,
        1,
        cell_type=mesh.CellType.tetrahedron,
    )
    partition = boundary_models.dolfinx_exterior_triangle_partition(domain)
    search = boundary_models.routed_distributed_triangle_surface_bvh(partition, comm)
    query = (
        np.asarray(((-0.1, 0.3, 0.4),))
        if comm.rank == 0
        else np.asarray(((1.1, 0.3, 0.4),))
    )

    routed = search.project_with_diagnostics(query)
    oracle = search.correctness_oracle.project(query)

    np.testing.assert_array_equal(routed.projection.valid, oracle.valid)
    np.testing.assert_array_equal(routed.projection.entity_ids, oracle.entity_ids)
    np.testing.assert_allclose(
        routed.projection.closest_points,
        oracle.closest_points,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(routed.projection.signed_gaps, 0.1, atol=1.0e-14)
    assert routed.diagnostics.summary()["packed_numeric_transport"] is True
