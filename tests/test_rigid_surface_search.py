from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models


def _grid_surface(count: int = 8):
    vertices = [
        (float(i) / count, float(j) / count, 0.0)
        for j in range(count + 1)
        for i in range(count + 1)
    ]
    triangles = []
    for j in range(count):
        for i in range(count):
            lower_left = j * (count + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + count + 1
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    facet_ids = tuple(range(1000, 1000 + len(triangles)))
    return boundary_models.triangulated_rigid_surface(
        vertices=vertices,
        triangles=triangles,
        facet_ids=facet_ids,
        name="grid_tool",
    )


def _assert_projection_equal(accelerated, reference):
    np.testing.assert_array_equal(accelerated.valid, reference.valid)
    np.testing.assert_array_equal(accelerated.status_codes, reference.status_codes)
    np.testing.assert_array_equal(accelerated.entity_ids, reference.entity_ids)
    np.testing.assert_allclose(
        accelerated.closest_points,
        reference.closest_points,
        equal_nan=True,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        accelerated.normals,
        reference.normals,
        equal_nan=True,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        accelerated.signed_gaps,
        reference.signed_gaps,
        equal_nan=True,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        accelerated.local_coordinates,
        reference.local_coordinates,
        equal_nan=True,
        atol=1.0e-13,
    )
    assert accelerated.local_coordinate_system == reference.local_coordinate_system
    assert accelerated.geometry_fingerprint == reference.geometry_fingerprint


def test_triangle_bvh_matches_exhaustive_reference_for_faces_edges_and_vertices():
    surface = _grid_surface()
    search = boundary_models.triangle_surface_bvh(surface)
    rng = np.random.default_rng(20260929)
    random_points = np.column_stack(
        (
            rng.uniform(-0.2, 1.2, 40),
            rng.uniform(-0.2, 1.2, 40),
            rng.uniform(-0.4, 0.4, 40),
        )
    )
    reviewed_points = np.asarray(
        (
            (0.5, 0.5, 0.2),
            (0.0, 0.0, -0.1),
            (1.0, 1.0, 0.3),
            (0.25, 0.125, 0.05),
        )
    )
    points = np.vstack((random_points, reviewed_points))

    outcome = search.project_with_diagnostics(points)
    reference = surface.project(points)

    _assert_projection_equal(outcome.projection, reference)
    assert outcome.projection.method == "aabb_bvh_exact_triangle_projection"
    assert outcome.diagnostics.query_count == points.shape[0]
    assert outcome.diagnostics.facet_count == surface.triangles.shape[0]


def test_triangle_bvh_matches_reference_on_closed_nonplanar_surface():
    surface = boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        facet_ids=(41, 17, 29, 5),
        name="tetrahedral_tool",
    )
    search = boundary_models.triangle_surface_bvh(surface)
    rng = np.random.default_rng(37)
    points = rng.uniform(-0.5, 1.5, size=(100, 3))

    accelerated = search.project(points)
    reference = surface.project(points)

    _assert_projection_equal(accelerated, reference)
    assert surface.summary()["closed"] is True


def test_triangle_bvh_preserves_ambiguous_and_no_candidate_failures():
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
    )
    search = boundary_models.triangle_surface_bvh(surface)
    points = ((0.2, 0.2, 1.0), (0.2, 0.2, 4.0))

    accelerated = search.project(points, maximum_distance=1.5)
    reference = surface.project(points, maximum_distance=1.5)

    _assert_projection_equal(accelerated, reference)
    assert accelerated.status_codes.tolist() == [
        "ambiguous_projection",
        "no_candidate",
    ]


def test_triangle_bvh_matches_reference_under_prescribed_motion():
    surface = _grid_surface(4)
    search = boundary_models.triangle_surface_bvh(surface)
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.0, 1.0),
        rotation=(0.0, 0.0, np.pi / 3.0),
        reference_point=(0.0, 0.0, 0.0),
    )
    points = ((-0.1, 0.4, 1.2), (0.2, 0.1, 0.8), (2.0, 2.0, 1.0))

    accelerated = search.project(points, motion=motion, factor=1.0)
    reference = surface.project(points, motion=motion, factor=1.0)

    _assert_projection_equal(accelerated, reference)


def test_triangle_bvh_prunes_exact_facet_evaluations_and_reports_scope():
    surface = _grid_surface(20)
    search = boundary_models.triangle_surface_bvh(surface)

    outcome = search.project_with_diagnostics(((0.01, 0.01, 0.1),))
    summary = outcome.diagnostics.summary()

    assert outcome.projection.all_valid
    assert summary["facet_count"] == 800
    assert summary["maximum_evaluated_facets"] < 80
    assert summary["maximum_visited_nodes"] < search.node_count
    assert summary["distributed_ownership"] is False
    assert search.summary()["execution_scope"] == "replicated_process_local"
    assert search.summary()["ambiguity_policy"] == (
        "preserve_all_equidistant_candidates"
    )


def test_triangle_bvh_supports_empty_query_batches():
    search = boundary_models.triangle_surface_bvh(_grid_surface(2))

    outcome = search.project_with_diagnostics(np.empty((0, 3)))

    assert outcome.projection.point_count == 0
    assert outcome.projection.all_valid
    assert outcome.diagnostics.query_count == 0
    assert outcome.diagnostics.summary()["mean_evaluated_facets"] == 0.0


def test_triangle_local_coordinates_reconstruct_reviewed_closest_points():
    surface = _grid_surface(4)
    projection = boundary_models.triangle_surface_bvh(surface).project(
        ((0.18, 0.37, 0.2), (0.75, 0.75, -0.1), (1.2, 0.5, 0.3))
    )
    id_to_local = {
        int(facet_id): local
        for local, facet_id in enumerate(surface.facet_ids.tolist())
    }

    for point_index, facet_id in enumerate(projection.entity_ids.tolist()):
        local = id_to_local[facet_id]
        triangle = surface.vertices[surface.triangles[local]]
        weights = projection.local_coordinates[point_index]
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1.0e-13)
        np.testing.assert_allclose(
            weights @ triangle,
            projection.closest_points[point_index],
            atol=1.0e-13,
        )
        assert np.min(weights) >= -1.0e-12
        assert np.max(weights) <= 1.0 + 1.0e-12


def test_triangle_bvh_rejects_unreviewed_geometry_type():
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )

    with pytest.raises(TypeError, match="TriangulatedRigidSurface"):
        boundary_models.triangle_surface_bvh(surface)


def test_triangle_search_evidence_rejects_mismatched_geometry():
    search = boundary_models.triangle_surface_bvh(_grid_surface(2))
    outcome = search.project_with_diagnostics(((0.2, 0.2, 0.1),))
    mismatched = boundary_models.TriangleSearchDiagnostics(
        geometry_fingerprint="0" * 64,
        tree_node_count=outcome.diagnostics.tree_node_count,
        facet_count=outcome.diagnostics.facet_count,
        visited_node_counts=outcome.diagnostics.visited_node_counts,
        evaluated_facet_counts=outcome.diagnostics.evaluated_facet_counts,
    )

    with pytest.raises(ValueError, match="geometry differ"):
        boundary_models.TriangleSearchOutcome(
            projection=outcome.projection,
            diagnostics=mismatched,
        )

    with pytest.raises(ValueError, match="aligned"):
        boundary_models.TriangleSearchDiagnostics(
            geometry_fingerprint="0" * 64,
            tree_node_count=1,
            facet_count=1,
            visited_node_counts=(1,),
            evaluated_facet_counts=(),
        )
