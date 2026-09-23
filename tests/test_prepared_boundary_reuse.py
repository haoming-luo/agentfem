"""Repeated, changing Dirichlet data; runnable on one or multiple MPI ranks."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from agentfem import solvers


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("vector", [False, True])
def test_prepared_boundary_matrix_matches_lifting(dim, vector):
    domain = (
        mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
        if dim == 2
        else mesh.create_unit_cube(
            MPI.COMM_WORLD, 2, 2, 2, cell_type=mesh.CellType.hexahedron
        )
    )
    element = ("Lagrange", 2, (dim,)) if vector else ("Lagrange", 2)
    V = fem.functionspace(domain, element)
    lhs, rhs = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    q = sum(x[i] ** 2 for i in range(dim)) + x[0] + 1
    exact = ufl.as_vector([(i + 1) * q for i in range(dim)]) if vector else q
    g = fem.Function(V)
    scale = fem.Constant(domain, PETSc.ScalarType(0.0))
    force = scale * (exact - ufl.div(ufl.grad(exact)))
    a = (ufl.inner(lhs, rhs) + ufl.inner(ufl.grad(lhs), ufl.grad(rhs))) * ufl.dx
    L = ufl.inner(force, rhs) * ufl.dx
    boundary = mesh.locate_entities_boundary(
        domain, dim - 1, lambda X: np.full(X.shape[1], True)
    )
    bcs = [fem.dirichletbc(g, fem.locate_dofs_topological(V, dim - 1, boundary))]
    new, old = fem.Function(V), fem.Function(V)
    options = solvers.LinearSolverOptions(
        ksp_type="cg", pc_type="jacobi", rtol=1e-12, atol=1e-13, max_it=2000
    )
    with solvers.prepare_linear_problem(
        a, L, new, bcs=bcs, bc_assembly="matrix_elimination", options=options
    ) as cached:
        for amplitude in [0.0, 1.0, -0.4, 0.0]:
            scale.value = amplitude

            def values(X, amplitude=amplitude):
                scalar = amplitude * (sum(X[j] ** 2 for j in range(dim)) + X[0] + 1)
                return (
                    np.vstack([(i + 1) * scalar for i in range(dim)])
                    if vector
                    else scalar
                )

            g.interpolate(values)
            g.x.scatter_forward()
            cached.solve()
            solvers.solve_linear_problem(
                fem.form(a), fem.form(L), old, bcs=bcs, options=options
            )
            owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            for ref in [old, g]:
                error = np.linalg.norm(new.x.array[:owned] - ref.x.array[:owned]) ** 2
                assert np.sqrt(domain.comm.allreduce(error, op=MPI.SUM)) < 1e-8
        assert cached.solve_count == 4
    assert cached.closed
    with pytest.raises(RuntimeError, match="closed"):
        cached.solve()
