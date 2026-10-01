from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import assembly


def test_inverse_diagonal_requires_positive_mass_entries():
    with pytest.raises(
        ValueError,
        match="non-positive entries.*not a valid explicit mass-lumped element",
    ):
        assembly.inverse_diagonal(np.array([1.0, 0.0, -0.25]))


def test_inverse_diagonal_preserves_positive_mass_entries():
    np.testing.assert_allclose(
        assembly.inverse_diagonal(np.array([0.25, 2.0])),
        np.array([4.0, 0.5]),
    )


def test_inverse_diagonal_accepts_an_empty_mpi_shard():
    result = assembly.inverse_diagonal(np.empty((0,), dtype=float))

    assert result.shape == (0,)


def test_inverse_diagonal_rejects_a_non_positive_entry_collectively():
    values = (
        np.array([-0.25], dtype=float)
        if MPI.COMM_WORLD.rank == 0
        else np.empty((0,), dtype=float)
    )

    with pytest.raises(ValueError, match=r"count=1, minimum=-0.25"):
        assembly.inverse_diagonal(values, comm=MPI.COMM_WORLD)
