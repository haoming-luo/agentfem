from __future__ import annotations

import copy

import numpy as np
import pytest

from agentfem import boundary_models, state


def _surface(*, offset=0.0):
    return boundary_models.triangulated_rigid_surface(
        vertices=(
            (offset + 0.0, 0.0, 0.0),
            (offset + 1.0, 0.0, 0.0),
            (offset + 1.0, 1.0, 0.0),
            (offset + 0.0, 1.0, 0.0),
        ),
        triangles=((0, 1, 2), (0, 2, 3)),
        facet_ids=(2**54 + 7, 2**54 + 11),
        name="forming_tool",
    )


def _projection(points=((0.75, 0.25, -0.1), (0.25, 0.75, -0.2))):
    return _surface().project(points)


def test_contact_projection_state_is_atomic_and_inspectable():
    selected = boundary_models.ContactProjectionState()

    trial = selected.begin((9, 3), _projection())
    assert trial.point_ids.tolist() == [3, 9]
    assert trial.projection.entity_ids.tolist() == [2**54 + 11, 2**54 + 7]
    assert selected.accepted is None
    assert state.capabilities(selected).summary() == {
        "restartable": True,
        "replaceable": True,
        "begins_trial": True,
        "increment_transaction": False,
        "cycle_transaction": False,
    }

    selected.commit()
    assert selected.accepted is trial
    assert selected.trial is None

    changed = _projection(((0.8, 0.2, -0.3), (0.2, 0.8, -0.4)))
    selected.begin((9, 3), changed)
    selected.rollback()
    np.testing.assert_allclose(selected.accepted.projection.signed_gaps, (-0.2, -0.1))


def test_contact_projection_allows_facet_change_but_not_identity_change():
    selected = boundary_models.ContactProjectionState()
    selected.begin((3, 9), _projection())
    selected.commit()

    crossing = _projection(((0.25, 0.75, -0.1), (0.75, 0.25, -0.2)))
    trial = selected.begin((3, 9), crossing)
    assert (
        trial.projection.entity_ids.tolist()
        != selected.accepted.projection.entity_ids.tolist()
    )

    selected.rollback()
    with pytest.raises(ValueError, match="point identity changed"):
        selected.begin((3, 10), crossing)
    with pytest.raises(ValueError, match="surface contract changed"):
        selected.begin(
            (3, 9), _surface(offset=2.0).project(((2.2, 0.2, -0.1), (2.8, 0.8, -0.2)))
        )


def test_contact_projection_checkpoint_round_trip_preserves_int64_identity():
    selected = boundary_models.ContactProjectionState()
    selected.begin((2**55 + 1, 2**55 + 3), _projection())
    selected.commit()
    snapshot = selected.snapshot()

    restored = boundary_models.ContactProjectionState()
    restored.restore(snapshot)

    assert restored.accepted.point_ids.tolist() == [2**55 + 1, 2**55 + 3]
    assert restored.accepted.projection.entity_ids.tolist() == [2**54 + 7, 2**54 + 11]
    np.testing.assert_allclose(
        restored.accepted.projection.local_coordinates,
        selected.accepted.projection.local_coordinates,
    )
    assert restored.summary()["checkpoint_boundary"] == "accepted_only"


def test_contact_projection_checkpoint_rejects_trial_and_corruption():
    selected = boundary_models.ContactProjectionState()
    selected.begin((3, 9), _projection())
    with pytest.raises(RuntimeError, match="accepted boundary"):
        selected.snapshot()
    selected.commit()

    corrupted = copy.deepcopy(selected.snapshot())
    corrupted["accepted"]["entity_ids"][0] = -1
    with pytest.raises(ValueError, match="non-negative entity IDs"):
        boundary_models.ContactProjectionState().restore(corrupted)

    selected.begin((3, 9), _projection())
    with pytest.raises(RuntimeError, match="Rollback"):
        selected.restore(selected.accepted.snapshot())


def test_contact_projection_restore_cannot_clear_initialized_identity():
    selected = boundary_models.ContactProjectionState()
    selected.begin((3, 9), _projection())
    selected.commit()

    with pytest.raises(ValueError, match="unexpectedly empty"):
        selected.restore(
            {
                "schema": "agentfem.contact-projection-state.v1",
                "accepted": None,
            }
        )


def test_contact_projection_supports_empty_local_partition():
    projection = _surface().project(np.empty((0, 3)))
    selected = boundary_models.ContactProjectionState()
    selected.begin((), projection)
    selected.commit()

    assert selected.accepted.point_count == 0
    assert selected.accepted.summary()["point_ids_sorted"] is True


@pytest.mark.parametrize(
    ("point_ids", "message"),
    [
        ((1.5, 2.0), "explicit integers"),
        ((1, 1), "unique"),
        (((1,), (2,)), "one-dimensional"),
        (np.asarray([2**63], dtype=np.uint64), "signed 64-bit"),
    ],
)
def test_contact_projection_rejects_ambiguous_point_identity(point_ids, message):
    with pytest.raises((TypeError, ValueError), match=message):
        boundary_models.ContactProjectionState().begin(point_ids, _projection())
