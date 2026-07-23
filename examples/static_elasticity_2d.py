"""Small static linear-elasticity example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import boundary
from agentfem import forms
from agentfem import io as fem_io
from agentfem import mesh as fem_mesh
from agentfem import spaces
from agentfem.constitutive import elasticity
from agentfem.problems import LinearVariationalProblem
from agentfem.solvers import LinearSolverOptions


def main() -> None:
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 0.2])],
        [40, 8],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = spaces.vector_lagrange_space(domain, degree=1)
    u = spaces.named_function(V, "Displacement")
    du = spaces.trial_function(V)
    v = spaces.test_function(V)

    material = elasticity.isotropic_elastic(
        young=210.0e9,
        poisson=0.30,
        density=7800.0,
        name="steel-like plane strain",
    )
    sigma = material.sigma(du)

    def left(x):
        return np.isclose(x[0], 0.0)

    _, bc_x = boundary.component_dirichlet_bc(V, 0, left, value=0.0)
    _, bc_y = boundary.component_dirichlet_bc(V, 1, left, value=0.0)

    traction = fem.Constant(domain, np.array((0.0, -1.0e6), dtype=PETSc.ScalarType))

    def right(x):
        return np.isclose(x[0], 1.0)

    ds_right, _ = fem_mesh.tagged_boundary_measure(domain, right, tag=1)
    a = fem.form(forms.stiffness_form(sigma, elasticity.strain(v)))
    L = fem.form(forms.boundary_load_form(traction, v, ds_right(1)))

    problem = LinearVariationalProblem(
        bilinear_form=a,
        linear_form=L,
        solution=u,
        bcs=[bc_x, bc_y],
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
    )
    problem.solve()

    out = Path(__file__).resolve().parents[1] / "examples_output" / "static_elasticity_2d.xdmf"
    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, u)

    if comm.rank == 0:
        print(f"Static elasticity result: {out}")


if __name__ == "__main__":
    main()
