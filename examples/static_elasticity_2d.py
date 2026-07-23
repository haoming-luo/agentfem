"""Small static linear-elasticity example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import fields
from agentfem import io as fem_io
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import studies
from agentfem.constitutive import elasticity
from agentfem.solvers import LinearSolverOptions


def main() -> None:
    comm = MPI.COMM_WORLD
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
        name="cantilever_plane_strain_static",
    )
    mesh_cells = (40, 8)
    element_degree = 2
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.2),
        cells=mesh_cells,
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="cantilever_model")

    displacement = model.field(fields.displacement(domain, degree=element_degree))

    model.material(
        elasticity.isotropic_elastic(
            young=210.0e9,
            poisson=0.30,
            density=7800.0,
            name="steel-like plane strain",
        )
    )

    def left(x):
        return np.isclose(x[0], 0.0)

    left_boundary = fem_mesh.boundary(domain, left, name="left", tag=1)
    model.fix(displacement, on=left_boundary, value=0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    right_boundary = fem_mesh.boundary(domain, right, name="right", tag=2)
    model.traction(value=(1.0e8, -1.0e6), on=right_boundary)
    model.check()

    step = model.linear_static_step(
        target=displacement,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
        name="cantilever_Ku_eq_F",
    )
    step.solve()

    out = Path(__file__).resolve().parents[1] / "examples_output" / "static_elasticity_2d.xdmf"
    output_displacement = fem_io.interpolate_for_xdmf(displacement.value, degree=1)
    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, output_displacement)

    if comm.rank == 0:
        print(model.tree())
        print(f"Static elasticity result: {out}")


if __name__ == "__main__":
    main()
