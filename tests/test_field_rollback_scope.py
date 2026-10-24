# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Accepted increment scopes must never turn caught failures into acceptance."""
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem._field_rollback_scope import _accepted_field_scope, _borrowed_field_snapshot
from agentfem.state import field_transaction


def fixture():
    domain = mesh.create_unit_interval(MPI.COMM_SELF, 2)
    f = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
    f.x.array[:] = 1
    return f, {"f": f.x.array.copy()}


def test_existing_snapshot_is_borrowed_not_recopied():
    f, snapshots = fixture()
    with _accepted_field_scope({"f": f}, snapshots):
        assert _borrowed_field_snapshot(f) is snapshots["f"]
        with field_transaction(f=f):
            f.x.array[:] = 2
    np.testing.assert_array_equal(f.x.array, 2)
    assert _borrowed_field_snapshot(f) is None
    np.testing.assert_array_equal(snapshots["f"], 1)


def test_copied_context_cannot_revive_an_ended_increment_snapshot():
    from contextvars import copy_context

    f, snapshots = fixture()
    with _accepted_field_scope({"f": f}, snapshots):
        inherited = copy_context()
        f.x.array[:] = 2

    def later_transaction():
        assert _borrowed_field_snapshot(f) is None
        with pytest.raises(ValueError, match="later rejected"):
            with field_transaction(f=f):
                f.x.array[:] = 3
                raise ValueError("later rejected")
        np.testing.assert_array_equal(f.x.array, 2)
        with _accepted_field_scope({"f": f}, {"f": f.x.array.copy()}):
            assert _borrowed_field_snapshot(f) is not None

    inherited.run(later_transaction)


def test_ended_scope_releases_snapshot_even_when_context_is_retained():
    from contextvars import copy_context
    import weakref

    f, snapshots = fixture()
    observed = weakref.ref(snapshots["f"])
    with pytest.raises(ValueError, match="owner rejected"):
        with _accepted_field_scope({"f": f}, snapshots):
            inherited = copy_context()
            raise ValueError("owner rejected")
    del snapshots
    assert observed() is None
    assert inherited.run(_borrowed_field_snapshot, f) is None


def test_caught_child_failure_poison_rejects_scope():
    f, snapshots = fixture()
    with pytest.raises(RuntimeError, match="joined field transaction failed"):
        with _accepted_field_scope({"f": f}, snapshots):
            with field_transaction(f=f):
                f.x.array[:] = 2
            try:
                with field_transaction(f=f):
                    f.x.array[:] = 3
                    raise ValueError("child rejected")
            except ValueError:
                pass
            np.testing.assert_array_equal(f.x.array, 1)
    assert _borrowed_field_snapshot(f) is None


def test_uncovered_fields_and_nonfinite_response_restore():
    f, snapshots = fixture()
    g = fem.Function(f.function_space)
    g.x.array[:] = 4
    with pytest.raises(ValueError, match="finite"):
        with _accepted_field_scope({"f": f}, snapshots):
            with field_transaction(f=f, g=g):
                f.x.array[:] = np.nan
                g.x.array[:] = 9
    np.testing.assert_array_equal(f.x.array, 1)
    np.testing.assert_array_equal(g.x.array, 4)


def test_default_nested_transaction_retains_savepoint():
    f, _ = fixture()
    with field_transaction(f=f):
        f.x.array[:] = 2
        with pytest.raises(ValueError):
            with field_transaction(f=f):
                f.x.array[:] = 3
                raise ValueError("restore inner only")
        np.testing.assert_array_equal(f.x.array, 2)


def test_reject_borrowed_live_storage():
    f, _ = fixture()
    with pytest.raises(ValueError, match="detached"):
        with _accepted_field_scope({"f": f}, {"f": f.x.array}):
            pytest.fail("Mutable live history is not a rollback snapshot")


def test_borrowed_snapshot_does_not_hide_invalid_live_entry():
    f, snapshots = fixture()
    with pytest.raises(ValueError, match="finite"):
        with _accepted_field_scope({"f": f}, snapshots):
            f.x.array[:] = np.nan
            with field_transaction(f=f):
                pytest.fail("Borrowing must retain the live-field entry check")


@pytest.mark.parametrize("contact", [False, "cylinder"])
def test_response_commit_failure_restores_complete_increment(monkeypatch, contact):
    from test_response_hex_explicit import make_step

    step = make_step(contact=contact, bond=True, viscosity=True)
    step.run(until_step=1)
    residual = step.residual
    while hasattr(residual, "base"):
        residual = residual.base
    before = step.residual.snapshot()
    motion = step.state.snapshot()
    original = residual.internal.response.commit

    def fail():
        original()
        raise RuntimeError("failed after material publication")

    monkeypatch.setattr(residual.internal.response, "commit", fail)
    with pytest.raises(RuntimeError, match="after material publication"):
        step.run()
    assert step.residual.snapshot() == before
    for name, value in motion["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], value)
    monkeypatch.setattr(residual.internal.response, "commit", original)
    step.run()
    assert step.completed_steps == 10


def test_joined_scope_matches_independent_savepoints(monkeypatch):
    from contextlib import nullcontext
    import agentfem._field_rollback_scope as ownership
    from test_response_hex_explicit import make_step

    joined = make_step(contact="cylinder", bond=True, viscosity=True)
    joined.run()
    monkeypatch.setattr(ownership, "_increment_field_scope", lambda *args: nullcontext())
    reference = make_step(contact="cylinder", bond=True, viscosity=True)
    reference.run()
    assert joined.residual.snapshot() == reference.residual.snapshot()
    for name, value in joined.state.snapshot()["fields"].items():
        np.testing.assert_array_equal(reference.state.snapshot()["fields"][name], value)


@pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="two-rank collective gate")
def test_distributed_default_rollback_and_borrowing_refusal():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 4)
    f = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
    f.x.array[:] = 2
    saved = f.x.array.copy()
    with pytest.raises(ValueError, match="serial-only"):
        with _accepted_field_scope({"f": f}, {"f": saved}):
            pytest.fail("Distributed borrowing has not been admitted")
    with pytest.raises((ValueError, RuntimeError)):
        with field_transaction(f=f):
            f.x.array[:] = 4
            if MPI.COMM_WORLD.rank == 0:
                raise ValueError("one-rank failure")
    np.testing.assert_array_equal(f.x.array, saved)
