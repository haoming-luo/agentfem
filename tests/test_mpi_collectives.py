# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Automatic GC cannot reenter MPI lock creation; caller state is preserved."""
import gc
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from agentfem._mpi_collectives import object_allgather


@pytest.mark.parametrize("initially_enabled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_object_gather_preserves_payload_and_gc_on_success_or_failure(initially_enabled, fails):
    before = gc.isenabled()
    try:
        (gc.enable if initially_enabled else gc.disable)()
        payload = {"total": {"seconds": 1.5, "calls": 1}}

        def gather(value):
            assert not gc.isenabled() and value is payload
            if fails:
                raise RuntimeError("collective failed")
            # Nested calls must not restore GC before the outer MPI call ends.
            inner = SimpleNamespace(allgather=lambda value: [value])
            assert object_allgather(inner, value) == [payload]
            assert not gc.isenabled()
            return [value]

        comm = SimpleNamespace(allgather=gather)
        if fails:
            with pytest.raises(RuntimeError, match="collective failed"):
                object_allgather(comm, payload)
        else:
            assert object_allgather(comm, payload)[0] is payload
        assert gc.isenabled() is initially_enabled
    finally:
        (gc.enable if before else gc.disable)()


@pytest.mark.parametrize('fail_first', [False, True])
def test_overlapping_gathers_restore_gc_only_after_last_call(fail_first):
    before = gc.isenabled()
    entered = [Event(), Event()]
    release = [Event(), Event()]
    errors = []
    failures = []

    class ExpectedFailure(RuntimeError):
        pass

    def worker(index):
        def gather(value):
            assert not gc.isenabled()
            entered[index].set()
            assert release[index].wait(5)
            assert not gc.isenabled()
            if fail_first and index == 0:
                raise ExpectedFailure('first call failed')
            return [value]
        try:
            assert object_allgather(SimpleNamespace(allgather=gather), index) == [index]
        except ExpectedFailure:
            failures.append(index)
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=worker, args=(index,)) for index in range(2)]
    gc.enable()
    try:
        threads[0].start()
        assert entered[0].wait(5)
        threads[1].start()
        assert entered[1].wait(5)
        release[0].set()
        threads[0].join(5)
        assert not threads[0].is_alive() and not gc.isenabled()
        release[1].set()
        threads[1].join(5)
        assert not threads[1].is_alive() and gc.isenabled() and not errors
        assert failures == ([0] if fail_first else [])
    finally:
        for event in release:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
        (gc.enable if before else gc.disable)()


def test_caught_nested_failure_does_not_restore_gc_inside_outer_call():
    before = gc.isenabled()
    gc.enable()
    def outer(value):
        def fail(_):
            raise RuntimeError('nested failure')
        with pytest.raises(RuntimeError, match='nested failure'):
            object_allgather(SimpleNamespace(allgather=fail), value)
        assert not gc.isenabled()
        return [value]
    try:
        assert object_allgather(SimpleNamespace(allgather=outer), 3) == [3]
        assert gc.isenabled()
    finally:
        (gc.enable if before else gc.disable)()


def test_deferred_cycles_are_collectable_and_thresholds_unchanged():
    import weakref
    before, thresholds = gc.isenabled(), gc.get_threshold()
    refs = []
    class Cycle:
        pass
    def gather(value):
        item = Cycle()
        item.self = item
        refs.append(weakref.ref(item))
        return [value]
    try:
        gc.enable()
        object_allgather(SimpleNamespace(allgather=gather), None)
        gc.collect()  # Outside the MPI call, as ordinary application cleanup.
        assert refs[0]() is None
        assert gc.isenabled() and gc.get_threshold() == thresholds
    finally:
        (gc.enable if before else gc.disable)()
