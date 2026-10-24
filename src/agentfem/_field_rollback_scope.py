# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial increment ownership; ordinary transactions stay savepoints.

A Procedure may lend its already detached accepted snapshot to child field
transactions. Children then join that increment, rather than introducing a new
savepoint. A caught child failure poisons the scope: it cannot be accepted.
The Procedure still owns restoration of non-field state and time inputs.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import numpy as np

_active = ContextVar("agentfem_field_rollback_owner", default=None)


def _increment_field_scope(residual, snapshot):
    factory = getattr(residual, "_increment_field_scope", None)
    return nullcontext() if factory is None else factory(snapshot)


@contextmanager
def _accepted_field_scope(functions, snapshots):
    previous = _active.get()
    if previous is not None and previous["active"]:
        raise RuntimeError("Accepted field rollback owners cannot be nested.")
    if set(functions) != set(snapshots):
        raise ValueError("Accepted rollback fields and snapshots differ.")
    entries = {}
    for name, function in functions.items():
        current = function.x.array
        saved = np.asarray(snapshots[name])
        if function.function_space.mesh.comm.size != 1:
            raise ValueError("Borrowed field rollback is serial-only.")
        if (saved.shape != current.shape or saved.dtype != current.dtype
                or np.shares_memory(saved, current) or not np.isfinite(saved).all()):
            raise ValueError("Accepted rollback requires detached finite matching arrays.")
        if id(function) in entries:
            raise ValueError("Accepted rollback contains aliased field identities.")
        entries[id(function)] = (function, saved, current.ctypes.data, current.shape)
    owner = {"entries": entries, "failed": False, "active": True}
    token = _active.set(owner)
    try:
        yield
        if owner["failed"]:
            raise RuntimeError("A joined field transaction failed; reject this increment.")
    finally:
        # ContextVar.reset only updates this context. A copied context must
        # neither revive the ended increment nor keep its large snapshots.
        owner["active"] = False
        entries.clear()
        _active.reset(token)


def _borrowed_field_snapshot(function):
    owner = _active.get()
    if owner is None or not owner["active"]:
        return None
    # Cleanup/rollback may itself use field transactions. Keep lending the
    # accepted snapshot; scope exit rejects acceptance without masking the
    # original scientific failure with a secondary cleanup error.
    entry = owner["entries"].get(id(function))
    if entry is None:
        return None
    original, saved, pointer, shape = entry
    current = function.x.array
    if original is not function or current.ctypes.data != pointer or current.shape != shape:
        raise RuntimeError("Field storage changed within its accepted rollback scope.")
    return saved


def _reject_field_scope():
    owner = _active.get()
    if owner is not None and owner["active"]:
        owner["failed"] = True
