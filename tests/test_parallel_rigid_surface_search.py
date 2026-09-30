from __future__ import annotations

from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models


def _require_two_ranks():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Distributed triangle-search contract is reviewed on two ranks.")


def _assert_projection_equal(distributed, reference):
    np.testing.assert_array_equal(distributed.valid, reference.valid)
    np.testing.assert_array_equal(distributed.status_codes, reference.status_codes)
    np.testing.assert_array_equal(distributed.entity_ids, reference.entity_ids)
    np.testing.assert_allclose(
        distributed.closest_points,
        reference.closest_points,
        atol=1.0e-14,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        distributed.normals,
        reference.normals,
        atol=1.0e-14,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        distributed.signed_gaps,
        reference.signed_gaps,
        atol=1.0e-14,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        distributed.local_coordinates,
        reference.local_coordinates,
        atol=1.0e-13,
        equal_nan=True,
    )
    assert distributed.geometry_fingerprint == reference.geometry_fingerprint


def _square_surface():
    return boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        triangles=((0, 1, 2), (0, 2, 3)),
        facet_ids=(20, 10),
        name="square_tool",
    )


def test_distributed_triangle_search_matches_global_reference_for_local_queries():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _square_surface()
    partition = boundary_models.partition_triangle_surface(surface, comm)
    search = boundary_models.distributed_triangle_surface_bvh(partition, comm)
    points = (
        np.asarray(((0.75, 0.25, 0.2), (1.2, 0.5, -0.1)))
        if comm.rank == 0
        else np.asarray(((0.25, 0.75, -0.3),))
    )

    outcome = search.project_with_diagnostics(points)
    reference = surface.project(points)

    _assert_projection_equal(outcome.projection, reference)
    summary = outcome.diagnostics.summary()
    assert summary["distributed_ownership"] is True
    assert summary["scalable_neighbor_routing"] is False
    assert summary["global_query_count"] == 3
    assert summary["local_facet_count"] == 1
    assert search.summary()["retains_replicated_global_geometry"] is False


def test_distributed_triangle_search_preserves_cross_rank_ambiguity():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
            (1.0, 0.0, 2.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
        facet_ids=(3, 4),
        name="parallel_sheets",
    )
    search = boundary_models.distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    points = np.asarray(((0.2, 0.2, 1.0),)) if comm.rank == 0 else np.empty((0, 3))

    distributed = search.project(points)
    reference = surface.project(points)

    _assert_projection_equal(distributed, reference)
    if comm.rank == 0:
        assert distributed.status_codes.tolist() == ["ambiguous_projection"]


def test_distributed_triangle_search_uses_global_stable_id_on_shared_edge():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _square_surface()
    search = boundary_models.distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    points = np.asarray(((0.5, 0.5, 0.1),))

    distributed = search.project(points)
    reference = surface.project(points)

    _assert_projection_equal(distributed, reference)
    assert distributed.entity_ids.tolist() == [10]


def test_far_local_ambiguity_does_not_hide_a_closer_remote_candidate():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.0, 0.0, 0.9),
            (1.0, 0.0, 0.9),
            (0.0, 1.0, 0.9),
        ),
        triangles=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        facet_ids=(1, 3, 2),
        name="three_parallel_sheets",
    )
    search = boundary_models.distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    points = np.asarray(((0.2, 0.2, 1.0),))

    distributed = search.project(points)
    reference = surface.project(points)

    _assert_projection_equal(distributed, reference)
    assert distributed.status_codes.tolist() == ["ok"]
    assert distributed.entity_ids.tolist() == [2]


def test_distributed_triangle_search_matches_reference_under_rigid_motion():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _square_surface()
    search = boundary_models.distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.0, 0.5),
        rotation=(0.0, 0.0, np.pi / 2.0),
        reference_point=(0.0, 0.0, 0.0),
    )
    points = (
        np.asarray(((-0.25, 0.75, 0.7),))
        if comm.rank == 0
        else np.asarray(((-0.75, 0.25, 0.4),))
    )

    distributed = search.project(points, motion=motion, factor=1.0)
    reference = surface.project(points, motion=motion, factor=1.0)

    _assert_projection_equal(distributed, reference)


def test_distributed_triangle_partition_rejects_different_global_geometry():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _square_surface()
    if comm.rank == 1:
        surface = boundary_models.triangulated_rigid_surface(
            vertices=np.asarray(surface.vertices) * 2.0,
            triangles=surface.triangles,
            facet_ids=surface.facet_ids,
            name=surface.name,
        )

    with pytest.raises(ValueError, match="same reviewed triangle geometry"):
        boundary_models.partition_triangle_surface(surface, comm)
