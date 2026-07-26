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
from agentfem.diagnostics import print_on_root
from agentfem.solvers import LinearSolverOptions


def main() -> None:
    comm = MPI.COMM_WORLD

    # 1. Study: define the analysis type and mechanical assumption.
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
        name="cantilever_plane_strain_static",
    )

    # 2. Mesh and model: create the geometry mesh and attach it to a model.
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

    # 3. Field: create the unknown displacement field.
    displacement = model.field(fields.displacement(domain, degree=element_degree))

    # 4. Material: assign an isotropic linear-elastic material.
    model.material(
        elasticity.isotropic_elastic(
            young=210.0e9,
            poisson=0.30,
            density=7800.0,
            name="steel-like plane strain",
        )
    )

    # 5. Constraints: clamp the left boundary.
    def left(x):
        return np.isclose(x[0], 0.0)

    left_boundary = fem_mesh.boundary(domain, left, name="left", tag=1)
    model.fix(displacement, on=left_boundary, value=0.0)

    # 6. Load: apply a traction on the right boundary.
    def right(x):
        return np.isclose(x[0], 1.0)

    right_boundary = fem_mesh.boundary(domain, right, name="right", tag=2)
    model.traction(value=(1.0e8, -1.0e6), on=right_boundary)
    model.check()

    # 7. Step and solve: assemble K u = F and solve the linear system.
    step = model.linear_static_step(
        target=displacement,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
        name="cantilever_Ku_eq_F",
    )
    step.solve()

    # 8. Output: write displacement to XDMF for ParaView.
    out = Path(__file__).resolve().parents[1] / "examples_output" / "static_elasticity_2d.xdmf"
    output_displacement = fem_io.interpolate_for_xdmf(displacement.value, degree=1)
    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, output_displacement)

    print_on_root(comm, model.tree())
    print_on_root(comm, f"Static elasticity result: {out}")


if __name__ == "__main__":
    main()
