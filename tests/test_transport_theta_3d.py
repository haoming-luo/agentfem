"""Time convergence of public transport operators, independent of benchmark adapters."""
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from agentfem import operators, solvers, expressions, results


def solve_transport(dt, theta, *, varying_boundary=False):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2,
                                  cell_type=mesh.CellType.hexahedron)
    V = fem.functionspace(domain, ("Lagrange", 3))
    previous, current, boundary = (fem.Function(V) for _ in range(3))
    q = "x+2*y+3*z+1" if varying_boundary else "x*(1-x)*y*(1-y)*z*(1-z)"
    expressions.interpolate(previous, q)
    x = ufl.SpatialCoordinate(domain)
    shape = x[0]+2*x[1]+3*x[2]+1 if varying_boundary else x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])
    old_time = fem.Constant(domain, PETSc.ScalarType(0.))
    new_time = fem.Constant(domain, PETSc.ScalarType(0.))
    beta = ufl.as_vector((0.8, 0.3, 0.0))
    power = 2 if varying_boundary else 3
    def forcing(t):
        exact = (1+t)**power*shape
        return power*(1+t)**(power-1)*shape - 0.2*ufl.div(ufl.grad(exact)) + ufl.dot(beta,ufl.grad(exact))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    tau = operators.intrinsic_time_scale(domain, beta, diffusivity=0.2, degree=3)
    a, L = operators.transient_transport_forms(trial, test, previous,
        forcing(new_time), forcing(old_time), beta, 0.2, dt=dt, theta=theta, tau=tau)
    facets = mesh.locate_entities_boundary(domain,2,lambda x:np.ones(x.shape[1],dtype=bool))
    bc = fem.dirichletbc(boundary,fem.locate_dofs_topological(V,2,facets))
    with solvers.prepare_linear_problem(a,L,current,bcs=[bc]) as prepared:
        for step in range(round(0.2/dt)):
            old_time.value = step*dt
            new_time.value = (step+1)*dt
            expressions.interpolate(boundary,f"(1+t)**{power}*({q})",parameters={"t":float(new_time.value)})
            prepared.solve()
            previous.x.array[:] = current.x.array
            previous.x.scatter_forward()
    z,y,x = np.meshgrid(*([np.linspace(0,1,9)]*3),indexing="ij")
    points = np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    value = results.sample_points(current,points).reshape(x.shape)
    exact = 1.2**power*(x+2*y+3*z+1 if varying_boundary else x*(1-x)*y*(1-y)*z*(1-z))
    return np.linalg.norm(value-exact)/np.linalg.norm(exact)


@pytest.mark.parametrize("theta,minimum_ratio",[(0.5,3.4),(1.,1.7)])
def test_transport_supg_time_order(theta,minimum_ratio):
    errors=[solve_transport(dt,theta) for dt in [0.05,0.025]]
    assert errors[0]/errors[1]>minimum_ratio,errors


def test_transport_time_varying_boundary():
    assert max(solve_transport(dt,0.5,varying_boundary=True) for dt in [0.05,0.025])<1e-8
