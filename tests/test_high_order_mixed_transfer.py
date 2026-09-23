"""Physical field transfer must preserve high-order mixed-component meaning."""
import numpy as np
import pytest
import ufl
from dolfinx import mesh, fem
from mpi4py import MPI
from agentfem import fields, spaces

@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('degree',[2,3,4])
def test_high_order_mixed_polynomial_transfer(dimension,degree):
    d=(mesh.create_unit_square(MPI.COMM_WORLD,3,3) if dimension==2
       else mesh.create_unit_cube(MPI.COMM_WORLD,2,2,2))
    state=fields.velocity_pressure(d,velocity_degree=degree,pressure_degree=degree-1)
    V=spaces.independent_subspace(state.space,0)
    g=fem.Function(V)
    g.interpolate(lambda x:np.vstack([(j+1)*x[0]**2+x[j] for j in range(dimension)]))
    state.value.sub(0).interpolate(g)
    Q=spaces.independent_subspace(state.space,1);p=fem.Function(Q)
    p.interpolate(lambda x:1.+x[0]);state.value.sub(1).interpolate(p)
    actual=state.collapsed_velocity();pressure=state.collapsed_pressure()
    x=ufl.SpatialCoordinate(d);exact=ufl.as_vector([(j+1)*x[0]**2+x[j] for j in range(dimension)])
    local=fem.assemble_scalar(fem.form(ufl.inner(actual-exact,actual-exact)*ufl.dx))
    error=d.comm.allreduce(local,op=MPI.SUM)
    assert error<1e-22
    local=fem.assemble_scalar(fem.form((pressure-1.-x[0])**2*ufl.dx))
    assert d.comm.allreduce(local,op=MPI.SUM)<1e-22
