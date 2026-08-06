from __future__ import annotations

from pathlib import Path
import tempfile
import uuid

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constraints, fields, loads, mesh, results


def test_distributed_abaqus_regions_quality_and_remote_resultant():
    pytest.importorskip("meshio")
    comm = MPI.COMM_WORLD
    token = comm.bcast(uuid.uuid4().hex if comm.rank == 0 else None, root=0)
    directory = Path(tempfile.gettempdir()) / f"agentfem-parallel-mesh-{token}"
    source = directory / "two_tetra.inp"
    converted = directory / "two_tetra.xdmf"
    if comm.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "\n".join(
                (
                    "*Heading",
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "5, 1., 1., 1.",
                    "*Nset, nset=FIXED",
                    "1, 4",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1, 2, 3, 4",
                    "2, 2, 3, 4, 5",
                    "*Surface, name=LOADED, type=ELEMENT",
                    "1, S1",
                )
            ),
            encoding="utf-8",
        )
    comm.barrier()

    imported = mesh.read_abaqus_mesh(
        source,
        converted,
        comm=comm,
        cell_type="tetra",
        reuse_conversion=False,
    )
    fixed_nodes = imported.node_set("FIXED")
    loaded = imported.boundary("LOADED", tag=17)
    fixed = constraints.fixed(fields.displacement(imported.domain), on=fixed_nodes)
    quality = mesh.audit_quality(imported.domain, threshold=0.1, strict=True)
    remote = loads.remote_force(
        (3.0, -4.0, 5.0),
        reference_point=(1.0 / 3.0, 1.0 / 3.0, 0.0),
        on=loaded,
    )
    resultant = results.boundary_resultant(remote.traction, on=loaded)

    assert fixed_nodes.summary()["global_nodes"] == 2
    assert loaded.audit(strict=True)["global_tagged_facets"] == 1
    assert len(fixed.bcs) == 3
    assert quality.global_cells == 2
    np.testing.assert_allclose(resultant, (3.0, -4.0, 5.0), atol=1.0e-11)
