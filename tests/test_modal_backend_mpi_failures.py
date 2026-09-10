from __future__ import annotations

from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest

from agentfem.backends._modal import _collect_reduced_mode_values


def test_rank_local_reduced_vector_mismatch_fails_collectively():
    """A malformed rank must not strand its peers in modal collectives."""

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("collective modal failure requires at least two MPI ranks")

    value_count = 2 if comm.rank == 1 else 1
    vector = SimpleNamespace(array_r=np.ones(value_count, dtype=float))
    free_local = np.asarray([0], dtype=np.int32)
    free_global = np.asarray([comm.rank], dtype=np.int64)

    with pytest.raises(
        RuntimeError,
        match=(
            "Modal backend candidate 0 reduced-vector layout failed on rank 1: "
            "RuntimeError: distributed modal subspace layout does not match the "
            "free-dof map"
        ),
    ):
        _collect_reduced_mode_values(
            comm,
            vector,
            free_local=free_local,
            free_global=free_global,
            candidate=0,
        )

    # Reaching a later collective on every rank proves that failure delivery
    # did not leave a peer behind in a PETSc/SLEPc operation.
    comm.barrier()

