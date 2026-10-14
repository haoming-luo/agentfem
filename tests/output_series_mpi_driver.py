from pathlib import Path
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from agentfem import mesh, fields, results

comm = MPI.COMM_WORLD
out = Path(sys.argv[1])
for dim in (2, 3):
    d = (
        mesh.rectangle((0.0, 0.0), (1.0, 1.0), (3, 3), comm=comm)
        if dim == 2
        else mesh.cuboid((0.0,) * 3, (1.0,) * 3, (2, 2, 2), comm=comm)
    )
    u = fields.displacement(d, degree=2).value
    u.name = "Displacement"
    q = fem.Function(fem.functionspace(d, ("DG", 0)), name="Energy")
    r = results.SimulationResult(name="mpi_output")
    with results.output_series(
        r,
        out / f"dim{dim}.xdmf",
        paraview=True,
        primary_name="U",
        primary_semantic_name="Displacement",
    ) as writer:
        for t in (0.0, 1.0):
            u.interpolate(lambda x, t=t, dim=dim: np.vstack([x[0] * t] * dim))
            q.x.array[:] = t + 1
            writer.write_accepted(coordinate=t, primary=u, fields=(q,))
    assert results.inspect_output(out / f"dim{dim}.xdmf")["valid"]
    # Rank-local invalid input must fail collectively before backend I/O.
    try:
        with results.output_series(r, out / f"invalid{dim}.xdmf") as writer:
            if comm.rank == 1:
                u.x.array[0] = np.nan
            writer.write_accepted(coordinate=0, primary=u)
    except ValueError:
        pass
    else:
        raise AssertionError("rank-local invalid input was accepted")
    if comm.rank == 0:
        print(f"{dim}D: two-rank series and collective rejection passed", flush=True)
