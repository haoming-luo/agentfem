from __future__ import annotations

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import mesh
from agentfem.operators.identity import (
    mesh_executable_identity,
    meshtags_executable_identity,
)


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


def test_meshtags_identity_is_shared_across_two_rank_partition():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed executable identity requires at least two ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    tdim = domain.topology.dim
    cell_map = domain.topology.index_map(tdim)
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    midpoints = dolfinx_mesh.compute_midpoints(domain, tdim, cells)
    values = np.where(midpoints[:, 0] < 1.0, 1, 2).astype(np.int32)
    tags = dolfinx_mesh.meshtags(domain, tdim, cells, values)

    identity = meshtags_executable_identity(domain, tags)
    gathered = MPI.COMM_WORLD.allgather(identity)

    assert all(item == gathered[0] for item in gathered)
    assert identity["global_tagged_entities"] == 8
    assert identity["mesh_sha256"] == mesh_executable_identity(domain)["mesh_sha256"]
