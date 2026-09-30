from __future__ import annotations

import copy

import pytest

from agentfem import boundary_models


def _station(
    time,
    force,
    coordinate,
    *,
    factor=0.0,
    resultant=(0.0, 0.0),
):
    return boundary_models.PrescribedContactWorkStation(
        time=time,
        factor=factor,
        generalized_force=force,
        generalized_coordinate=coordinate,
        contact_resultant=resultant,
        contact_potential_energy=0.0,
        active_point_count=1,
        invalid_point_count=0,
    )


def test_contact_work_is_transactional_and_uses_accepted_path_only():
    state = boundary_models.prescribed_contact_work_state(identity="tool")
    state.initialize(_station(0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    state.begin(_station(1.0, (4.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    state.rollback()
    assert state.path_work == pytest.approx(0.0)

    state.begin(_station(1.0, (4.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    state.commit()

    assert state.path_work == pytest.approx(9.0)
    assert state.latest_interval_power == pytest.approx(9.0)


def test_contact_work_includes_rotational_moment():
    state = boundary_models.prescribed_contact_work_state(identity="rotating_tool")
    state.initialize(_station(0.0, (0.0, 0.0, 2.0), (0.0, 0.0, 0.0)))
    state.begin(_station(0.5, (0.0, 0.0, 4.0), (0.0, 0.0, 0.25)))
    state.commit()

    assert state.path_work == pytest.approx(0.75)
    assert state.latest_interval_power == pytest.approx(1.5)


def test_contact_work_snapshot_round_trip_and_time_corruption_rejected():
    state = boundary_models.prescribed_contact_work_state(identity="tool")
    state.initialize(_station(0.0, (1.0, 0.0), (0.0, 0.0)))
    state.begin(_station(1.0, (1.0, 0.0), (1.0, 0.0)))
    state.commit()
    snapshot = state.snapshot()

    restored = boundary_models.prescribed_contact_work_state(identity="tool")
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert restored.path_work == pytest.approx(1.0)

    corrupted = copy.deepcopy(snapshot)
    corrupted["accepted"][1]["time"] = 0.0
    with pytest.raises(ValueError, match="strictly increase"):
        restored.restore(corrupted)


def test_contact_work_rejects_layout_changes_and_trial_checkpoint():
    state = boundary_models.prescribed_contact_work_state(identity="tool")
    state.initialize(_station(0.0, (1.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError, match="layout changed"):
        state.begin(_station(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    state.begin(_station(1.0, (1.0, 0.0), (1.0, 0.0)))
    with pytest.raises(RuntimeError, match="accepted boundary"):
        state.snapshot()


def test_rigid_motion_schedule_has_explicit_physical_time_semantics():
    motion = boundary_models.prescribed_rigid_motion(
        translation=(2.0, -1.0),
        rotation=0.5,
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion,
        start_time=1.0,
        end_time=3.0,
    )

    assert schedule.factor_at(0.0) == pytest.approx(0.0)
    assert schedule.factor_at(2.0) == pytest.approx(0.5)
    assert schedule.factor_at(4.0) == pytest.approx(1.0)
    assert schedule.generalized_coordinate_at(2.0).tolist() == pytest.approx(
        [1.0, -0.5, 0.25]
    )
    assert schedule.summary()["interpolation"] == "linear_factor_clamped"
