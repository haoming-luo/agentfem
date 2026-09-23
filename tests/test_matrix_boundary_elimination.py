"""Independent equivalence checks for assembled-matrix Dirichlet elimination.
The same tests may be run with mpiexec -n 2 to exercise distributed rows.
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from agentfem import solvers


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("nonzero_boundary", [False, True])
def test_matrix_elimination_matches_element_lifting(dimension, nonzero_boundary):
    if dimension == 2:
        domain = mesh.create_unit_square(
            MPI.COMM_WORLD, 3, 3, cell_type=mesh.CellType.triangle
        )
    else:
        domain = mesh.create_unit_cube(
            MPI.COMM_WORLD, 2, 2, 2, cell_type=mesh.CellType.hexahedron
        )
    V = fem.functionspace(domain, ("Lagrange", 2, (dimension,)))
    uh = fem.Function(V)
    vh = fem.Function(V)
    g = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    exact = ufl.as_vector(
        [
            (i + 1) * sum(x[j] ** 2 for j in range(dimension)) + x[i]
            for i in range(dimension)
        ]
    )
    if nonzero_boundary:
        g.interpolate(
            lambda X: np.vstack(
                [
                    (i + 1) * sum(X[j] ** 2 for j in range(dimension)) + X[i]
                    for i in range(dimension)
                ]
            )
        )
    facets = mesh.locate_entities_boundary(
        domain, dimension - 1, lambda X: np.full(X.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(V, dimension - 1, facets)
    bcs = [fem.dirichletbc(g, dofs)]
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)

    def stress(u):
        return 6 * ufl.sym(ufl.grad(u)) + 2 * ufl.div(u) * ufl.Identity(dimension)

    force = (
        -ufl.div(stress(exact))
        if nonzero_boundary
        else fem.Constant(domain, np.arange(1, dimension + 1, dtype=PETSc.ScalarType))
    )
    a = fem.form(ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx)
    L = fem.form(ufl.inner(force, test) * ufl.dx)
    options = solvers.LinearSolverOptions(
        ksp_type="cg", pc_type="jacobi", rtol=1e-12, atol=1e-13, max_it=1000
    )
    solvers.solve_linear_problem(a, L, uh, bcs=bcs, options=options)
    solvers.solve_linear_problem(
        a, L, vh, bcs=bcs, options=options, bc_assembly="matrix_elimination"
    )
    count = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    local = np.linalg.norm(uh.x.array[:count] - vh.x.array[:count]) ** 2
    diff = np.sqrt(domain.comm.allreduce(local, op=MPI.SUM))
    assert diff < 1e-9
    if nonzero_boundary:
        local = np.linalg.norm(vh.x.array[:count] - g.x.array[:count]) ** 2
        assert np.sqrt(domain.comm.allreduce(local, op=MPI.SUM)) < 1e-9
