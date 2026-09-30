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


def _strip_surface(cell_count: int = 8):
    vertices = [
        (float(index), float(side), 0.0)
        for index in range(cell_count + 1)
        for side in (0, 1)
    ]
    triangles = []
    for index in range(cell_count):
        lower_left = 2 * index
        upper_left = lower_left + 1
        lower_right = lower_left + 2
        upper_right = lower_left + 3
        triangles.extend(
            (
                (lower_left, lower_right, upper_right),
                (lower_left, upper_right, upper_left),
            )
        )
    return boundary_models.triangulated_rigid_surface(
        vertices=vertices,
        triangles=triangles,
        facet_ids=tuple(range(100, 100 + len(triangles))),
        name="strip_tool",
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
        boundary_models.partition_triangle_surface(
            surface,
            comm,
            ownership_method="stable_facet_id_round_robin",
        ),
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
        boundary_models.partition_triangle_surface(
            surface,
            comm,
            ownership_method="stable_facet_id_round_robin",
        ),
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


def test_routed_triangle_search_matches_oracle_and_avoids_remote_queries():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _strip_surface()
    partition = boundary_models.partition_triangle_surface(surface, comm)
    search = boundary_models.routed_distributed_triangle_surface_bvh(
        partition,
        comm,
    )
    points = (
        np.asarray(((1.25, 0.4, 0.1), (1.75, 0.6, -0.2)))
        if comm.rank == 0
        else np.asarray(((6.25, 0.4, 0.15), (6.75, 0.6, -0.05)))
    )

    routed = search.project_with_diagnostics(points)
    oracle = search.correctness_oracle.project(points)
    reference = surface.project(points)

    _assert_projection_equal(routed.projection, oracle)
    _assert_projection_equal(routed.projection, reference)
    diagnostics = routed.diagnostics.summary()
    assert diagnostics["phase_two_query_messages"] == 0
    assert diagnostics["avoided_query_messages"] == points.shape[0]
    assert diagnostics["maximum_queried_ranks"] == 1
    assert diagnostics["packed_numeric_transport"] is False


def test_routed_triangle_search_queries_both_ranks_at_partition_seam():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _strip_surface()
    search = boundary_models.routed_distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    points = np.asarray(((4.0, 0.5, 0.1),))

    routed = search.project_with_diagnostics(points)
    oracle = search.correctness_oracle.project(points)

    _assert_projection_equal(routed.projection, oracle)
    assert routed.diagnostics.queried_rank_counts == (2,)
    assert routed.diagnostics.phase_two_query_messages == 1


def test_routed_triangle_search_preserves_cross_rank_ambiguity_and_distance_limit():
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
    partition = boundary_models.partition_triangle_surface(
        surface,
        comm,
        ownership_method="stable_facet_id_round_robin",
    )
    search = boundary_models.routed_distributed_triangle_surface_bvh(
        partition,
        comm,
    )
    points = (
        np.asarray(((0.2, 0.2, 1.0), (0.2, 0.2, 4.0)))
        if comm.rank == 0
        else np.empty((0, 3))
    )

    routed = search.project(points, maximum_distance=1.5)
    oracle = search.correctness_oracle.project(points, maximum_distance=1.5)

    _assert_projection_equal(routed, oracle)
    if comm.rank == 0:
        assert routed.status_codes.tolist() == [
            "ambiguous_projection",
            "no_candidate",
        ]


def test_routed_triangle_search_matches_oracle_under_rigid_motion():
    _require_two_ranks()
    comm = MPI.COMM_WORLD
    surface = _strip_surface()
    search = boundary_models.routed_distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.0, 0.5),
        rotation=(0.0, 0.0, np.pi / 6.0),
        reference_point=(0.0, 0.0, 0.0),
    )
    points = (
        np.asarray(((0.9, 1.0, 0.7),))
        if comm.rank == 0
        else np.asarray(((4.8, 3.8, 0.4),))
    )

    routed = search.project(points, motion=motion, factor=1.0)
    oracle = search.correctness_oracle.project(
        points,
        motion=motion,
        factor=1.0,
    )

    _assert_projection_equal(routed, oracle)
