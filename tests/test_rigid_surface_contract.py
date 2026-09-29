from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models


def test_plane_projection_preserves_signed_gap_and_closest_point_semantics():
    surface = boundary_models.rigid_plane(
        point=(1.0, 0.5),
        normal=(-1.0, 0.0),
        name="forming_tool",
    )

    projection = surface.project(
        (
            (0.8, 0.2),
            (1.0, 0.7),
            (1.2, 0.9),
        )
    )

    np.testing.assert_allclose(projection.signed_gaps, (0.2, 0.0, -0.2))
    np.testing.assert_allclose(
        projection.closest_points,
        ((1.0, 0.2), (1.0, 0.7), (1.0, 0.9)),
    )
    np.testing.assert_allclose(
        projection.normals,
        ((-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)),
    )
    assert projection.all_valid is True
    assert projection.entity_ids is None
    assert projection.summary() == {
        "surface_name": "forming_tool",
        "surface_kind": "rigid_plane_surface",
        "dimension": 2,
        "point_count": 3,
        "valid_count": 3,
        "invalid_count": 0,
        "all_valid": True,
        "status_counts": {"ok": 3},
        "method": "exact_orthogonal_projection",
        "geometry_fingerprint": surface.geometry_fingerprint,
        "signed_gap_convention": "positive_admissible_negative_penetration",
        "entity_identity": "not_applicable",
    }


def test_plane_projection_follows_prescribed_translation_and_rotation():
    surface = boundary_models.rigid_plane(
        point=(1.0, 0.5),
        normal=(-1.0, 0.0),
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.1, -0.2),
        rotation=np.pi / 2.0,
        reference_point=(1.0, 0.5),
    )

    projection = surface.project((1.4, 0.2), motion=motion, factor=1.0)

    np.testing.assert_allclose(projection.signed_gaps, (0.1,), atol=1.0e-15)
    np.testing.assert_allclose(
        projection.closest_points,
        ((1.4, 0.3),),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(projection.normals, ((0.0, -1.0),), atol=1.0e-15)


def test_plane_projection_supports_empty_batches_and_three_dimensions():
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0, 1.0),
        normal=(0.0, 0.0, 1.0),
    )

    projection = surface.project(np.empty((0, 3)))

    assert projection.query_points.shape == (0, 3)
    assert projection.closest_points.shape == (0, 3)
    assert projection.signed_gaps.shape == (0,)
    assert projection.all_valid is True


@pytest.mark.parametrize(
    "points, message",
    [
        (((0.0, 0.0, 0.0),), "shape"),
        (((0.0, np.nan),), "finite"),
    ],
)
def test_plane_projection_rejects_invalid_queries(points, message):
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match=message):
        surface.project(points)


def test_surface_projection_is_immutable_and_validates_backend_evidence():
    projection = boundary_models.SurfaceProjection(
        surface_name="mesh_tool",
        surface_kind="triangulated_rigid_surface",
        query_points=((0.0, 0.0, 0.0),),
        closest_points=((0.0, 0.0, 1.0),),
        normals=((0.0, 0.0, 1.0),),
        signed_gaps=(-1.0,),
        valid=(True,),
        status_codes=("ok",),
        method="reviewed_test_projection",
        entity_ids=(7,),
        geometry_fingerprint="0" * 64,
    )

    assert projection.summary()["entity_identity"] == "provided"
    assert projection.entity_ids.tolist() == [7]
    with pytest.raises(ValueError):
        projection.signed_gaps[0] = 2.0

    with pytest.raises(ValueError, match="unit vectors"):
        boundary_models.SurfaceProjection(
            surface_name="broken",
            surface_kind="test",
            query_points=((0.0, 0.0),),
            closest_points=((0.0, 0.0),),
            normals=((2.0, 0.0),),
            signed_gaps=(0.0,),
            valid=(True,),
            status_codes=("ok",),
            method="invalid",
        )


def test_surface_projection_represents_failed_search_without_fake_geometry():
    projection = boundary_models.SurfaceProjection(
        surface_name="mesh_tool",
        surface_kind="triangulated_rigid_surface",
        query_points=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        closest_points=((0.0, 0.0, 1.0), (np.nan, np.nan, np.nan)),
        normals=((0.0, 0.0, 1.0), (np.nan, np.nan, np.nan)),
        signed_gaps=(-1.0, np.nan),
        valid=(True, False),
        status_codes=("ok", "no_candidate"),
        method="reviewed_test_projection",
        entity_ids=(7, -1),
        geometry_fingerprint="0" * 64,
    )

    assert projection.all_valid is False
    assert projection.summary()["valid_count"] == 1
    assert projection.summary()["invalid_count"] == 1
    assert projection.summary()["status_counts"] == {"no_candidate": 1, "ok": 1}

    with pytest.raises(ValueError, match="entity ID -1"):
        boundary_models.SurfaceProjection(
            surface_name="broken",
            surface_kind="triangulated_rigid_surface",
            query_points=((0.0, 0.0, 0.0),),
            closest_points=((np.nan, np.nan, np.nan),),
            normals=((np.nan, np.nan, np.nan),),
            signed_gaps=(np.nan,),
            valid=(False,),
            status_codes=("no_candidate",),
            method="invalid",
            entity_ids=(7,),
            geometry_fingerprint="0" * 64,
        )

    with pytest.raises(ValueError, match="require a geometry fingerprint"):
        boundary_models.SurfaceProjection(
            surface_name="broken",
            surface_kind="triangulated_rigid_surface",
            query_points=((0.0, 0.0, 0.0),),
            closest_points=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, 1.0),),
            signed_gaps=(0.0,),
            valid=(True,),
            status_codes=("ok",),
            method="invalid",
            entity_ids=(7,),
        )


def test_surface_summary_declares_projection_and_gap_conventions():
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )

    assert isinstance(surface, boundary_models.RigidSurface)
    assert surface.summary() == {
        "name": "rigid_plane",
        "kind": "rigid_plane_surface",
        "dimension": 2,
        "point": (0.0, 0.0),
        "normal": (0.0, 1.0),
        "geometry_fingerprint": surface.geometry_fingerprint,
        "representation": "analytical",
        "projection": "exact_orthogonal",
        "signed_gap_convention": "positive_admissible_negative_penetration",
        "entity_identity": "not_applicable",
    }


def test_sphere_projection_supports_circle_and_explicit_admissible_side():
    exterior = boundary_models.rigid_sphere(
        center=(1.0, -1.0),
        radius=2.0,
        name="round_tool",
    )
    interior = boundary_models.rigid_sphere(
        center=(1.0, -1.0),
        radius=2.0,
        admissible_side="interior",
    )

    outside = exterior.project(((4.0, -1.0), (2.0, -1.0)))
    inside = interior.project(((4.0, -1.0), (2.0, -1.0)))

    np.testing.assert_allclose(outside.closest_points, ((3.0, -1.0),) * 2)
    np.testing.assert_allclose(outside.normals, ((1.0, 0.0),) * 2)
    np.testing.assert_allclose(outside.signed_gaps, (1.0, -1.0))
    np.testing.assert_allclose(inside.normals, ((-1.0, 0.0),) * 2)
    np.testing.assert_allclose(inside.signed_gaps, (-1.0, 1.0))
    assert outside.summary()["method"] == "exact_radial_projection"
    assert exterior.summary()["geometric_shape"] == "circle"
    assert exterior.summary()["admissible_side"] == "exterior"


def test_sphere_projection_tracks_prescribed_rigid_translation():
    surface = boundary_models.rigid_sphere(center=(1.0, 0.0, 0.0), radius=0.5)
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 1.0, 0.0),
        rotation=(0.0, 0.0, np.pi / 2.0),
        reference_point=(0.0, 0.0, 0.0),
    )

    projection = surface.project(
        ((0.0, 3.0, 0.0),),
        motion=motion,
        factor=1.0,
    )

    np.testing.assert_allclose(
        projection.closest_points,
        ((0.0, 2.5, 0.0),),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        projection.normals,
        ((0.0, 1.0, 0.0),),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(projection.signed_gaps, (0.5,))


def test_sphere_projection_fails_closed_at_center_and_outside_search_radius():
    surface = boundary_models.rigid_sphere(center=(0.0, 0.0, 0.0), radius=1.0)

    projection = surface.project(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        maximum_distance=0.5,
    )

    assert projection.valid.tolist() == [False, False]
    assert projection.status_codes.tolist() == [
        "singular_projection",
        "no_candidate",
    ]
    assert np.isnan(projection.closest_points).all()


def test_infinite_cylinder_projection_and_motion_are_exact():
    surface = boundary_models.rigid_cylinder(
        axis_point=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        radius=2.0,
        name="roller",
    )
    projection = surface.project(((3.0, 0.0, 4.0), (1.0, 0.0, -2.0)))

    np.testing.assert_allclose(
        projection.closest_points,
        ((2.0, 0.0, 4.0), (2.0, 0.0, -2.0)),
    )
    np.testing.assert_allclose(projection.normals, ((1.0, 0.0, 0.0),) * 2)
    np.testing.assert_allclose(projection.signed_gaps, (1.0, -1.0))

    motion = boundary_models.prescribed_rigid_motion(
        translation=(1.0, 0.0, 0.0),
        rotation=(0.0, np.pi / 2.0, 0.0),
    )
    moved = surface.project(
        ((7.0, 3.0, 0.0),),
        motion=motion,
        factor=1.0,
    )
    np.testing.assert_allclose(
        moved.closest_points,
        ((7.0, 2.0, 0.0),),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        moved.normals,
        ((0.0, 1.0, 0.0),),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(moved.signed_gaps, (1.0,))
    assert surface.summary()["geometric_shape"] == "infinite_circular_cylinder"


def test_infinite_cylinder_fails_closed_on_axis():
    surface = boundary_models.rigid_cylinder(
        axis_point=(1.0, 2.0, 3.0),
        axis_direction=(0.0, 1.0, 0.0),
        radius=0.5,
    )

    projection = surface.project(((1.0, 7.0, 3.0),))

    assert projection.all_valid is False
    assert projection.status_codes.tolist() == ["singular_projection"]
    assert np.isnan(projection.signed_gaps).all()


@pytest.mark.parametrize(
    "factory, kwargs, message",
    [
        (boundary_models.rigid_sphere, {"center": (0.0, 0.0), "radius": 0.0}, "positive"),
        (
            boundary_models.rigid_sphere,
            {"center": (0.0, 0.0), "radius": 1.0, "admissible_side": "both"},
            "exterior.*interior",
        ),
        (
            boundary_models.rigid_cylinder,
            {
                "axis_point": (0.0, 0.0, 0.0),
                "axis_direction": (0.0, 0.0, 2.0),
                "radius": 1.0,
            },
            "unit vector",
        ),
    ],
)
def test_analytical_curved_surfaces_reject_ambiguous_geometry(
    factory,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_analytical_curved_surface_fingerprints_include_gap_orientation():
    exterior = boundary_models.rigid_sphere(center=(0.0, 0.0), radius=1.0)
    repeated = boundary_models.rigid_sphere(center=(0.0, 0.0), radius=1.0)
    interior = boundary_models.rigid_sphere(
        center=(0.0, 0.0),
        radius=1.0,
        admissible_side="interior",
    )

    assert exterior.geometry_fingerprint == repeated.geometry_fingerprint
    assert exterior.geometry_fingerprint != interior.geometry_fingerprint

    forward = boundary_models.rigid_cylinder(
        axis_point=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        radius=1.0,
    )
    reverse = boundary_models.rigid_cylinder(
        axis_point=(0.0, 0.0, 7.0),
        axis_direction=(0.0, 0.0, -1.0),
        radius=1.0,
    )
    assert forward.geometry_fingerprint == reverse.geometry_fingerprint
    assert forward.axis_direction == reverse.axis_direction
    assert forward.axis_point == reverse.axis_point


def _square_triangle_surface(**overrides):
    arguments = {
        "vertices": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        "triangles": ((0, 1, 2), (0, 2, 3)),
        "facet_ids": (20, 10),
        "name": "square_tool",
    }
    arguments.update(overrides)
    return boundary_models.triangulated_rigid_surface(**arguments)


def test_triangulated_surface_projects_faces_and_shared_coplanar_edge():
    surface = _square_triangle_surface()

    projection = surface.project(
        (
            (0.75, 0.25, 0.2),
            (0.25, 0.75, -0.3),
            (0.5, 0.5, 0.1),
        )
    )

    np.testing.assert_allclose(
        projection.closest_points,
        ((0.75, 0.25, 0.0), (0.25, 0.75, 0.0), (0.5, 0.5, 0.0)),
    )
    np.testing.assert_allclose(projection.signed_gaps, (0.2, -0.3, 0.1))
    np.testing.assert_allclose(
        projection.normals,
        ((0.0, 0.0, 1.0),) * 3,
    )
    np.testing.assert_array_equal(projection.entity_ids, (20, 10, 10))
    np.testing.assert_array_equal(projection.status_codes, ("ok", "ok", "ok"))
    assert projection.all_valid


def test_triangulated_surface_motion_and_search_radius_preserve_contract():
    surface = _square_triangle_surface()
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.0, 1.0),
        rotation=(0.0, 0.0, np.pi / 2.0),
        reference_point=(0.0, 0.0, 0.0),
    )

    moved = surface.project(
        ((-0.25, 0.75, 1.2),),
        motion=motion,
        factor=1.0,
    )
    np.testing.assert_allclose(moved.closest_points, ((-0.25, 0.75, 1.0),))
    np.testing.assert_allclose(moved.normals, ((0.0, 0.0, 1.0),))
    np.testing.assert_allclose(moved.signed_gaps, (0.2,))

    absent = surface.project(((0.5, 0.5, 2.0),), maximum_distance=0.5)
    assert absent.all_valid is False
    assert absent.status_codes.tolist() == ["no_candidate"]
    assert absent.entity_ids.tolist() == [-1]
    assert np.isnan(absent.closest_points).all()


def test_triangulated_surface_reports_ambiguous_equidistant_projection():
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

    projection = surface.project(((0.2, 0.2, 1.0),))

    assert projection.all_valid is False
    assert projection.status_codes.tolist() == ["ambiguous_projection"]
    assert projection.entity_ids.tolist() == [-1]
    assert np.isnan(projection.signed_gaps).all()


def test_triangulated_surface_summary_and_fingerprint_are_stable():
    surface = _square_triangle_surface()
    repeated = _square_triangle_surface()
    changed = _square_triangle_surface(facet_ids=(21, 10))

    assert surface.geometry_fingerprint == repeated.geometry_fingerprint
    assert surface.geometry_fingerprint != changed.geometry_fingerprint
    summary = surface.summary()
    assert summary["vertex_count"] == 4
    assert summary["facet_count"] == 2
    assert summary["edge_count"] == 5
    assert summary["boundary_edge_count"] == 4
    assert summary["closed"] is False
    assert summary["manifold"] is True
    assert summary["orientation_consistent"] is True
    assert summary["facet_identity"] == "user_provided"
    assert summary["projection"] == "exhaustive_reference"


@pytest.mark.parametrize(
    "vertices, triangles, message",
    [
        (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ((0, 1, 2),),
            "degenerate",
        ),
        (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0, 1, 2), (2, 1, 0)),
            "duplicate",
        ),
        (
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
            ),
            ((0, 1, 2), (0, 1, 3)),
            "inconsistent orientation",
        ),
        (
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            ((0, 1, 2), (1, 0, 3), (0, 1, 4)),
            "non-manifold",
        ),
        (
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            ((0, 1, 2), (0, 1, 3)),
            "duplicate vertices",
        ),
        (
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 2.0, 2.0),
            ),
            ((0, 1, 2),),
            "Every surface vertex",
        ),
    ],
)
def test_triangulated_surface_rejects_invalid_geometry(vertices, triangles, message):
    with pytest.raises(ValueError, match=message):
        boundary_models.triangulated_rigid_surface(
            vertices=vertices,
            triangles=triangles,
        )


def test_triangulated_surface_rejects_invalid_facet_identity():
    with pytest.raises(ValueError, match="unique non-negative"):
        _square_triangle_surface(facet_ids=(7, 7))

    with pytest.raises(ValueError, match="must be integers"):
        _square_triangle_surface(facet_ids=(7.5, 8.0))
