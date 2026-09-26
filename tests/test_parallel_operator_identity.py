from __future__ import annotations

from mpi4py import MPI
import pytest

from agentfem import mesh
from agentfem.operators.identity import mesh_executable_identity


def test_mesh_identity_is_shared_across_two_rank_partition():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed executable identity requires at least two ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )

    identity = mesh_executable_identity(domain)
    gathered = MPI.COMM_WORLD.allgather(identity)

    assert all(item == gathered[0] for item in gathered)
    assert identity["schema"] == "agentfem.mesh-executable-identity.v2"
    assert identity["global_cells"] == 8
    assert identity["coordinate_element"]["cell_type"] == "quadrilateral"
    assert identity["coordinate_element"]["degree"] == 1
    assert identity["mesh_sha256"] == identity["connectivity_sha256"]
