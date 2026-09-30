from __future__ import annotations

import pytest

from agentfem import boundary_models


def _surface(offset: float = 0.0):
    return boundary_models.rigid_plane(
        point=(offset, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
        name=f"plane_{offset}",
    )


def test_fixed_rigid_body_has_stable_scientific_identity():
    first = boundary_models.rigid_body(
        _surface(),
        reference_point=(0.0, 0.5, 0.5),
        name="fixed_tool",
    )
    second = boundary_models.rigid_body(
        _surface(),
        reference_point=(0.0, 0.5, 0.5),
        name="fixed_tool",
    )

    assert first.kinematic_mode == "fixed"
    assert first.scientific_identity == second.scientific_identity
    assert first.summary()["free_body_dynamics"] is False
    assert first.summary()["surface"]["kind"] == "rigid_plane_surface"


def test_prescribed_rigid_body_owns_schedule_and_reference_point():
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.02, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.1),
            reference_point=(0.0, 0.5, 0.5),
        ),
        start_time=0.1,
        end_time=0.2,
    )
    body = boundary_models.rigid_body(
        _surface(),
        motion_schedule=schedule,
        name="moving_tool",
    )

    assert body.kinematic_mode == "prescribed"
    assert body.reference_point == (0.0, 0.5, 0.5)
    assert body.summary()["motion_schedule"] == schedule.summary()


def test_rigid_body_rejects_conflicting_reference_point():
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.02, 0.0, 0.0),
            reference_point=(0.0, 0.5, 0.5),
        ),
        end_time=1.0,
    )

    with pytest.raises(ValueError, match="must match"):
        boundary_models.rigid_body(
            _surface(),
            motion_schedule=schedule,
            reference_point=(0.0, 0.0, 0.0),
        )


def test_rigid_body_identity_changes_with_geometry_or_schedule():
    fixed = boundary_models.rigid_body(_surface(), name="tool")
    moved_geometry = boundary_models.rigid_body(_surface(0.1), name="tool")
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.02, 0.0, 0.0),
        ),
        end_time=1.0,
    )
    moving = boundary_models.rigid_body(
        _surface(),
        motion_schedule=schedule,
        name="tool",
    )

    assert fixed.scientific_identity != moved_geometry.scientific_identity
    assert fixed.scientific_identity != moving.scientific_identity
