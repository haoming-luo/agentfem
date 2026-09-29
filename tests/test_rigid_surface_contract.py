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
        "representation": "analytical",
        "projection": "exact_orthogonal",
        "signed_gap_convention": "positive_admissible_negative_penetration",
        "entity_identity": "not_applicable",
    }
