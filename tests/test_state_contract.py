from __future__ import annotations

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import state


def _space():
    domain = mesh.create_unit_interval(MPI.COMM_SELF, 2)
    return fem.functionspace(domain, ("Lagrange", 1))


def test_first_order_state_has_atomic_accept_reject_and_restart():
    selected = state.TransientState.create(_space())
    selected.current.x.array[:] = 1.0
    selected.next.x.array[:] = 2.0

    assert state.capabilities(selected).summary() == {
        "restartable": True,
        "replaceable": True,
        "begins_trial": False,
        "increment_transaction": False,
        "cycle_transaction": False,
    }
    snapshot = selected.snapshot()
    selected.commit()
    assert np.allclose(selected.current.x.array, 2.0)
    selected.next.x.array[:] = 7.0
    selected.rollback()
    assert np.allclose(selected.next.x.array, 2.0)
    selected.restore(snapshot)
    assert np.allclose(selected.current.x.array, 1.0)
    assert np.allclose(selected.next.x.array, 2.0)


def test_second_order_state_rolls_back_every_trial_time_level():
    selected = state.SecondOrderDynamicsState.create(_space())
    selected.u.value.x.array[:] = 1.0
    selected.v.value.x.array[:] = 2.0
    selected.a.value.x.array[:] = 3.0
    selected.u_next.value.x.array[:] = 4.0
    selected.v_next.value.x.array[:] = 5.0
    selected.a_next.value.x.array[:] = 6.0
    selected.v_mid.value.x.array[:] = 7.0

    snapshot = selected.snapshot()
    selected.rollback()
    assert np.allclose(selected.u_next.value.x.array, 1.0)
    assert np.allclose(selected.v_next.value.x.array, 2.0)
    assert np.allclose(selected.a_next.value.x.array, 3.0)
    assert np.allclose(selected.v_mid.value.x.array, 2.0)
    selected.restore(snapshot)
    assert np.allclose(selected.u_next.value.x.array, 4.0)
    assert np.allclose(selected.v_mid.value.x.array, 7.0)


def test_state_requirements_fail_with_ownership_language():
    class NotAState:
        pass

    try:
        state.require_replaceable(NotAState(), name="material history")
    except TypeError as exc:
        assert "commit(), rollback(), snapshot(), and restore()" in str(exc)
    else:
        raise AssertionError("A non-transactional state was accepted.")


def test_state_restore_rejects_unstructured_snapshots():
    selected = state.TransientState.create(_space())

    try:
        selected.restore([1.0, 2.0])
    except TypeError as exc:
        assert "snapshot must be a mapping" in str(exc)
    else:
        raise AssertionError("An unstructured state snapshot was accepted.")
