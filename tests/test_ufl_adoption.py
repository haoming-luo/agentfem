"""Numerical equivalence of incremental adoption, not benchmark-specific dispatch."""
import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI
from agentfem import mesh, operators


def vector(form):
    return fem.assemble_vector(fem.form(form)).array.copy()


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('direction', [None, 'explicit'])
def test_burgers_matches_direct_weak_form(dim, direction):
    domain = (dmesh.create_unit_square(MPI.COMM_SELF, 3, 3) if dim == 2
              else dmesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2))
    V = fem.functionspace(domain, ('Lagrange', 1))
    state = fem.Function(V)
    state.interpolate(lambda x: 1 + x[0] + 2*x[1])
    test = ufl.TestFunction(V)
    chosen = tuple(range(1, dim+1)) if direction else None
    op = operators.burgers_convection_operator(state, state, test, direction=chosen)
    weights = chosen or (1,)*dim
    reference = state * sum(weights[j]*state.dx(j) for j in range(dim)) * test * ufl.dx
    np.testing.assert_allclose(vector(op.expression), vector(reference), rtol=1e-13, atol=1e-13)


def test_adoption_preserves_transient_operator_and_nonlinear_tangent():
    domain = dmesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    V = fem.functionspace(domain, ('Lagrange', 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    state = fem.Function(V); old = fem.Function(V)
    state.interpolate(lambda x: 1+x[0]); old.interpolate(lambda x: x[1])
    mass = operators.from_ufl(u*v*ufl.dx, name='M')
    diffusion = operators.diffusion_operator(u, v, conductivity=0.7)
    combined = (mass/0.2 + diffusion).expression
    direct = (u*v/0.2 + 0.7*ufl.inner(ufl.grad(u),ufl.grad(v)))*ufl.dx
    np.testing.assert_allclose(vector(ufl.action(combined,state)), vector(ufl.action(direct,state)),atol=1e-13)
    raw = ((state-old)/0.2*v + state**3*v)*ufl.dx
    residual = operators.from_ufl(raw,name='R',role='residual')
    assert residual.expression is raw
    tangent = operators.linearize(residual,state)
    np.testing.assert_allclose(vector(ufl.action(tangent.expression,state)),
                               vector(ufl.action(ufl.derivative(raw,state,u),state)),atol=1e-13)
    assert mass.role == 'matrix'
    assert operators.from_ufl(old*v*ufl.dx,name='history').role == 'vector'
    with pytest.raises(TypeError):operators.from_ufl(u*v,name='integrand')
    with pytest.raises(ValueError):operators.from_ufl(old*v*ufl.dx,name='wrong',role='matrix')


def test_keyword_mesh_adoption_preserves_geometry():
    e = ufl.Mesh(element('Lagrange','triangle',1,shape=(2,)))
    domain = mesh.from_arrays(cells=np.array([[0,1,2],[1,3,2]]),
        coordinates=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),
        coordinate_element=e,comm=MPI.COMM_SELF)
    area = fem.assemble_scalar(fem.form(1*ufl.dx(domain=domain)))
    assert area == pytest.approx(1.0)
    assert domain.topology.index_map(2).size_local == 2


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_mesh_adoption_coordinate_types_and_volume(dim, dtype):
    cell = "triangle" if dim == 2 else "tetrahedron"
    coordinates = np.vstack([np.zeros(dim), np.eye(dim)]).astype(dtype)
    expected_dtype = np.float64 if dtype == np.int64 else dtype
    coordinate_element = ufl.Mesh(element("Lagrange", cell, 1,
                                          shape=(dim,), dtype=expected_dtype))
    domain = mesh.from_arrays(cells=[list(range(dim + 1))],
                              coordinates=coordinates,
                              coordinate_element=coordinate_element,
                              comm=MPI.COMM_SELF)
    assert domain.geometry.x.dtype == expected_dtype
    volume = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain), dtype=expected_dtype))
    assert volume == pytest.approx(0.5 if dim == 2 else 1 / 6, rel=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_transient_residual_action_and_integrand_agree(dim):
    domain = (dmesh.create_unit_square(MPI.COMM_SELF, 3, 3) if dim == 2
              else dmesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2))
    V = fem.functionspace(domain, ("Lagrange", 1))
    current, previous = fem.Function(V), fem.Function(V)
    current.interpolate(lambda x: 1 + x[0]**2)
    previous.interpolate(lambda x: 0.5 + x[1])
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    mass = operators.mass_operator(u, v)
    dt = 0.03
    raw = ((current - previous) / dt * v + current**3 * v) * ufl.dx
    reused = (1.0 / dt) * (ufl.action(mass.expression, current)
                           - ufl.action(mass.expression, previous)) + current**3*v*ufl.dx
    np.testing.assert_allclose(vector(raw), vector(reused), rtol=1e-12, atol=1e-12)
    R = operators.from_ufl(reused, name="transient", role="residual")
    tangent = operators.linearize(R, current).expression
    np.testing.assert_allclose(vector(ufl.action(tangent, current)),
        vector(ufl.action(ufl.derivative(raw, current, u), current)), rtol=1e-12, atol=1e-12)
