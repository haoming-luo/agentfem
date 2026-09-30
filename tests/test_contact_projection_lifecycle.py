from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models


def _surface():
    return boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        triangles=((0, 1, 2), (0, 2, 3)),
        facet_ids=(101, 202),
        name="forming_tool",
    )


def test_contact_projection_lifecycle_updates_every_evaluation_and_commits():
    search = boundary_models.triangle_surface_bvh(_surface())
    lifecycle = boundary_models.ContactProjectionLifecycle(search, (8, 4))

    first = lifecycle.evaluate(((0.75, 0.25, -0.1), (0.25, 0.75, -0.2)))
    second = lifecycle.evaluate(((0.25, 0.75, -0.3), (0.75, 0.25, -0.4)))

    assert first.attempt == 1
    assert second.attempt == 2
    assert second.diagnostics is not None
    assert second.record.projection.entity_ids.tolist() == [101, 202]
    assert lifecycle.state.accepted is None
    accepted = lifecycle.commit_increment()
    assert accepted is second.record
    assert lifecycle.summary()["projection_update"] == "every_evaluation"
    assert lifecycle.summary()["successful_projections"] == 2
    assert lifecycle.summary()["accepted_increments"] == 1


def test_contact_projection_lifecycle_rolls_back_failed_search_without_stale_trial():
    search = boundary_models.triangle_surface_bvh(_surface())
    lifecycle = boundary_models.ContactProjectionLifecycle(search, (4, 8))
    lifecycle.evaluate(((0.25, 0.75, -0.1), (0.75, 0.25, -0.2)))
    lifecycle.commit_increment()
    accepted = lifecycle.state.accepted

    lifecycle.evaluate(((0.2, 0.8, -0.3), (0.8, 0.2, -0.4)))
    with pytest.raises(ValueError, match="invalid points"):
        lifecycle.evaluate(
            ((0.2, 0.8, -0.3), (100.0, 100.0, 100.0)),
            maximum_distance=0.1,
        )

    assert lifecycle.state.accepted is accepted
    assert lifecycle.state.trial is None
    assert lifecycle.summary()["rejected_projections"] == 1


def test_contact_projection_lifecycle_can_explicitly_retain_inactive_points():
    search = boundary_models.triangle_surface_bvh(_surface())
    lifecycle = boundary_models.ContactProjectionLifecycle(
        search,
        (4, 8),
        require_all_valid=False,
    )

    result = lifecycle.evaluate(
        ((0.2, 0.8, -0.1), (100.0, 100.0, 100.0)),
        maximum_distance=0.1,
    )

    assert result.record.projection.valid.tolist() == [True, False]
    lifecycle.rollback_increment()
    assert lifecycle.state.trial is None
    assert lifecycle.summary()["rejected_increments"] == 1


def test_contact_projection_lifecycle_rejects_identity_and_projector_errors():
    search = boundary_models.triangle_surface_bvh(_surface())
    with pytest.raises(TypeError, match="project"):
        boundary_models.ContactProjectionLifecycle(object(), (4, 8))
    with pytest.raises(TypeError, match="explicit integers"):
        boundary_models.ContactProjectionLifecycle(search, (4.5, 8.0))

    lifecycle = boundary_models.ContactProjectionLifecycle(search, (4, 8))
    with pytest.raises(ValueError, match="point count"):
        lifecycle.evaluate(((0.2, 0.8, -0.1),))
    assert lifecycle.state.trial is None


def test_contact_projection_lifecycle_supports_analytical_surfaces():
    plane = boundary_models.rigid_plane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    lifecycle = boundary_models.ContactProjectionLifecycle(plane, (1, 2))

    result = lifecycle.evaluate(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))

    np.testing.assert_allclose(result.record.projection.signed_gaps, (1.0, -1.0))
    assert result.diagnostics is None
