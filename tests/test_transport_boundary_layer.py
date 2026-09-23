"""Streamwise metric on an anisotropically graded physical outflow layer."""

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from agentfem import operators, solvers


def test_directional_supg_resolves_graded_outflow_layer():
    domain = mesh.create_unit_cube(
        MPI.COMM_WORLD, 8, 4, 4, cell_type=mesh.CellType.hexahedron
    )
    domain.geometry.x[:, 0] = 1 - (1 - domain.geometry.x[:, 0]) ** 3
    V = fem.functionspace(domain, ("Lagrange", 2))
    u, g = fem.Function(V), fem.Function(V)
    eps = 0.01
    g.interpolate(
        lambda X: (np.exp((X[0] - 1) / eps) - np.exp(-1 / eps)) / (1 - np.exp(-1 / eps))
    )
    facets = mesh.locate_entities_boundary(
        domain, 2, lambda X: np.full(X.shape[1], True)
    )
    bc = fem.dirichletbc(g, fem.locate_dofs_topological(V, 2, facets))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    tau = operators.intrinsic_time_scale(
        domain, (1.0, 0.0, 0.0), diffusivity=eps, degree=2, directional=True
    )
    a = (
        eps * ufl.inner(ufl.grad(trial), ufl.grad(test))
        + trial.dx(0) * test
        + tau * (trial.dx(0) - eps * ufl.div(ufl.grad(trial))) * test.dx(0)
    ) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * test * ufl.dx
    solvers.solve_linear_problem(
        fem.form(a), fem.form(L), u, bcs=[bc], bc_assembly="matrix_elimination"
    )
    X = ufl.SpatialCoordinate(domain)
    ref = (ufl.exp((X[0] - 1) / eps) - np.exp(-1 / eps)) / (1 - np.exp(-1 / eps))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 14})
    error = domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u - ref) ** 2 * dx)), op=MPI.SUM
    )
    norm = domain.comm.allreduce(fem.assemble_scalar(fem.form(ref**2 * dx)), op=MPI.SUM)
    assert np.sqrt(error / norm) < 0.06
    assert domain.comm.allreduce(float(u.x.array.min()), op=MPI.MIN) > -1e-6
