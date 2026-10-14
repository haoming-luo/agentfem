# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Immutable resident records must retain external writes and rollback truth."""
from types import SimpleNamespace
import numpy as np
import pytest
from agentfem._resident_field_records import ResidentFieldRecords, _freeze


def field(values):
    return SimpleNamespace(x=SimpleNamespace(array=values))


def owner(value):
    while isinstance(value, np.ndarray):
        value = value.base
    return value


def test_immutable_response_reuse_and_independent_metadata():
    values = _freeze(np.arange(24.).reshape(4, 6))
    records = ResidentFieldRecords()
    records.remember('state', values.reshape(-1))
    live = field(values.ravel().copy())
    a = records.capture({'state': live})['state']
    b = records.capture({'state': live})['state']
    assert owner(a) is owner(b) is owner(values)
    assert not np.shares_memory(a, live.x.array)
    with pytest.raises(ValueError):
        a.setflags(write=True)
    a.shape = (4, 6)
    assert b.shape == (24,)
    assert records.capture({'state': live})['state'].shape == (24,)


@pytest.mark.parametrize('replacement', [-0., 3., np.nan, np.inf])
def test_external_write_and_old_snapshot_survive(replacement):
    records = ResidentFieldRecords()
    live = field(np.zeros(4))
    old = records.capture({'a': live})['a']
    live.x.array[0] = replacement
    new = records.capture({'a': live})['a']
    assert new.tobytes() == live.x.array.tobytes()
    assert old.tobytes() == np.zeros(4).tobytes()
    assert owner(old) is not owner(new)


def test_cache_replacement_and_slice_offsets():
    records = ResidentFieldRecords()
    original = _freeze(np.arange(30.))
    records.remember('a', original[3:13])
    live = field(original[3:13].copy())
    old = records.capture({'a': live})['a']
    assert owner(old) is owner(original)
    live.x.array[:] = -2
    records.capture({'a': live})
    records.capture({})
    np.testing.assert_array_equal(old, np.arange(3., 13.))


def test_restart_and_failed_commit_keep_old_payload(monkeypatch, tmp_path):
    from test_response_hex_explicit import make_step

    step = make_step(bond=True, viscosity=True)
    step.residual._resident_records_enabled = True
    step.run(until_step=2)
    before = step.residual.snapshot()
    saved = step.residual.transaction_snapshot()
    original = step.residual.commit

    def fail():
        original()
        raise RuntimeError('after cache publication')

    monkeypatch.setattr(step.residual, 'commit', fail)
    with pytest.raises(RuntimeError, match='cache publication'):
        step.run(until_step=3)
    assert step.residual.snapshot() == before
    monkeypatch.setattr(step.residual, 'commit', original)
    step.run(until_step=3)
    step.residual.restore(saved)
    assert step.residual.snapshot() == before


def test_borrowed_material_history_stays_immutable_and_observes_callback():
    from test_explicit_material_quadrature import fixture
    from agentfem._field_rollback_scope import _accepted_field_scope
    from agentfem.constitutive._explicit_material_driver import _immutable_committed_history

    _, state, _ = fixture()
    selected = next(iter(state.committed.values())).function
    records = ResidentFieldRecords()
    saved = records.capture({'state': selected})
    with _accepted_field_scope({'state': selected}, saved):
        first = _immutable_committed_history(state)
        assert owner(first) is owner(saved['state'])
        selected.x.array[0] = 2
        second = _immutable_committed_history(state)
        assert second.ravel()[0] == 2
        assert first.ravel()[0] == 0
