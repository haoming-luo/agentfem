# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded compatibility seam for reentrant communicator finalization.

mpi4py 4.1.2 creates object-collective locks under a non-reentrant global
lock. An allocation there can run Python cyclic GC, release an old DOLFINx
communicator, and reenter that same lock through its deletion callback.
Defer automatic GC only until the object call has released the MPI locks.
Do not collect explicitly here, change thresholds, or change MPI payloads.
"""
from contextlib import contextmanager
import gc
from threading import Lock

_state_lock = Lock()
_active_calls = 0
_restore_gc = False


@contextmanager
def _defer_automatic_gc():
    global _active_calls, _restore_gc
    with _state_lock:
        if _active_calls == 0:
            _restore_gc = gc.isenabled()
            if _restore_gc:
                gc.disable()
        _active_calls += 1
    try:
        yield
    finally:
        with _state_lock:
            _active_calls -= 1
            if _active_calls == 0 and _restore_gc:
                gc.enable()
                _restore_gc = False


def object_allgather(comm, value):
    """Gather unchanged objects with exception-safe, overlap-safe GC deferral.

    Explicit external ``gc.collect()`` or concurrent GC-policy changes are
    outside this guard; this only controls automatic cyclic collection.
    """
    with _defer_automatic_gc():
        return comm.allgather(value)
