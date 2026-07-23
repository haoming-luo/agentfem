"""Small static linear-elasticity example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import constraints
from agentfem import fields
from agentfem import io as fem_io
from agentfem import loads
from agentfem import mesh as fem_mesh
from agentfem import operators
from agentfem.constitutive import elasticity
from agentfem.problems import LinearSystemProblem
from agentfem.solvers import LinearSolverOptions


def main() -> None:
    comm = MPI.COMM_WORLD
    mesh_cells = (40, 8)
    element_degree = 2
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.2),
        cells=mesh_cells,
        comm=comm,
        cell_type="quadrilateral",
    )

    displacement = fields.displacement(domain, degree=element_degree)

    material = elasticity.isotropic_elastic(
        young=210.0e9,
        poisson=0.30,
        density=7800.0,
        name="steel-like plane strain",
    )

    def left(x):
        return np.isclose(x[0], 0.0)

    left_boundary = fem_mesh.boundary(domain, left, name="left", tag=1)
    fixed_left = constraints.fixed(
        displacement,
        location=left_boundary,
        value=0.0,
    )

    def right(x):
        return np.isclose(x[0], 1.0)

    right_boundary = fem_mesh.boundary(domain, right, name="right", tag=2)
    right_traction = loads.traction(
        value=(0.0, -1.0e6),
        location=right_boundary,
    )
    K = operators.stiffness_operator(displacement, material)
    F = operators.force_vector(
        target=displacement,
        loads=[right_traction],
    )
    system = operators.LinearSystem(stiffness=K, force=F, name="cantilever_Kx_eq_F")

    problem = LinearSystemProblem(
        system=system,
        unknown=displacement,
        bcs=fixed_left.bcs,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
    )
    problem.solve()

    out = Path(__file__).resolve().parents[1] / "examples_output" / "static_elasticity_2d.xdmf"
    output_displacement = fem_io.interpolate_for_xdmf(displacement.value, degree=1)
    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, output_displacement)

    if comm.rank == 0:
        print(f"Static elasticity result: {out}")


if __name__ == "__main__":
    main()
