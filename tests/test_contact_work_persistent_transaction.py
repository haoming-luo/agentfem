"""Rollback tokens share only immutable history and do not scan accepted paths."""

import pytest

from agentfem.boundary_models.contact_work import PrescribedContactWorkState
from test_contact_work_state import _station


def append(state, i, force=2.0):
    sample = _station(float(i), (force, 0.0), (i * 0.01, 0.0))
    if state.current is None:
        state.initialize(sample)
    else:
        state.begin(sample)
        state.commit()


@pytest.mark.parametrize("count", [0, 1, 63, 64, 65, 128, 129, 1000])
def test_persistent_token_survives_commit_restore_and_alternative_branch(count):
    state = PrescribedContactWorkState(identity="tool")
    for i in range(count):
        append(state, i)
    before = state.snapshot()
    before_summary = state.summary()
    token = state.transaction_snapshot()
    assert len(token.tail) < 64
    for i in range(count, count + 70):
        append(state, i)
    future = state.transaction_snapshot()
    future_snapshot = state.snapshot()
    state.restore(token)
    assert state.snapshot() == before
    assert state.summary() == before_summary
    for i in range(count, count + 70):
        append(state, i, force=7.0)
    assert state.snapshot() != future_snapshot
    state.restore(future)
    assert state.snapshot() == future_snapshot
    # Tokens can restore compatible independently owned State, not just undo
    # list truncation on the original object.
    other = PrescribedContactWorkState(identity="tool")
    other.restore(token)
    assert other.summary() == before_summary


def test_transaction_snapshot_does_not_iterate_or_copy_observational_history():
    state = PrescribedContactWorkState(identity="tool")
    for i in range(1000):
        append(state, i)

    class NoScan(list):
        def __iter__(self):
            raise AssertionError("Transaction creation scanned full history")

    state.accepted = NoScan(state.accepted)
    token = state.transaction_snapshot()
    assert len(token.accepted) == 1000
    assert token.blocks is state._history_blocks
    assert token.tail is state._history_tail


@pytest.mark.parametrize("numeric", [False, True])
def test_durable_restore_rebuilds_persistent_history_before_next_commit(numeric):
    source = PrescribedContactWorkState(identity="tool")
    for i in range(130):
        append(source, i)
    snapshot = source.checkpoint_snapshot() if numeric else source.snapshot()
    restored = PrescribedContactWorkState(identity="tool")
    restored.restore(snapshot)
    token = restored.transaction_snapshot()
    append(restored, 130)
    restored.restore(token)
    assert restored.summary() == source.summary()
    assert restored.snapshot() == source.snapshot()


def test_deep_history_token_comparison_does_not_recurse():
    from agentfem.boundary_models.contact_work import _WorkTransaction

    # Same immutable station references, independently allocated block chains.
    # This checks a long-token operation without constructing a large FE model.
    sample = _station(0.0, (1.0, 0.0), (0.0, 0.0))
    stations = [sample] * (64 * 2000)
    left = PrescribedContactWorkState._blocked_history(stations)
    right = PrescribedContactWorkState._blocked_history(stations)
    assert _WorkTransaction("tool", *left) == _WorkTransaction("tool", *right)
