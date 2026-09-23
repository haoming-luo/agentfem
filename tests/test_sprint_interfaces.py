import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import fem
from mpi4py import MPI
from agentfem import mesh, fields, expressions, solvers

@pytest.mark.parametrize('cells,coords,message', [
 ([[0.,1.,2.]],[[0.,0.],[1.,0.],[0.,1.]],'integer'),
 ([[0,1,-2]],[[0.,0.],[1.,0.],[0.,1.]],'negative'),
 ([[0,1,2]],[[0.,0.],[np.nan,0.],[0.,1.]],'finite'),
 ([[0,1,2]],[0.,1.,2.],'shape')])
def test_mesh_input_reports_parameter(cells,coords,message):
    e=ufl.Mesh(element('Lagrange','triangle',1,shape=(2,)))
    with pytest.raises(ValueError,match=message):
        mesh.from_arrays(cells=cells,coordinates=coords,coordinate_element=e,comm=MPI.COMM_SELF)

def test_nonfinite_expression_preserves_field():
    d=mesh.rectangle((0.,0.),(1.,1.),(2,2),comm=MPI.COMM_SELF)
    f=fields.scalar_unknown(d,value=7.)
    with np.errstate(invalid='ignore'):
        with pytest.raises(expressions.ExpressionError,match='non-finite'):
            expressions.interpolate(f,'sqrt(x-2)')
    np.testing.assert_array_equal(f.value.x.array,7.)

@pytest.mark.parametrize('mode',['lifting','matrix_elimination'])
def test_prepared_coefficient_refresh_and_rhs_reuse(mode):
    d=mesh.rectangle((0.,0.),(1.,1.),(3,3),comm=MPI.COMM_SELF)
    V=fem.functionspace(d,('Lagrange',1));u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
    k=fem.Constant(d,1.);f=fem.Constant(d,2.);solution=fem.Function(V)
    a=k*u*v*ufl.dx;L=f*v*ufl.dx
    p=solvers.PreparedLinearProblem(a,L,solution,bc_assembly=mode,
        options=solvers.LinearSolverOptions(ksp_type='preonly',pc_type='lu'))
    try:
        p.solve();np.testing.assert_allclose(solution.x.array,2.,atol=1e-12)
        f.value=4.;p.solve();np.testing.assert_allclose(solution.x.array,4.,atol=1e-12)
        assert p.summary()['matrix_assembly_count']==1
        k.value=2.;p.refresh_matrix();p.solve()
        np.testing.assert_allclose(solution.x.array,2.,atol=1e-12)
        assert p.summary()['matrix_assembly_count']==2
        assert p.summary()['rhs_assembly_count']==3
    finally:p.close()


def test_high_order_geometry_is_not_linearized():
    e=ufl.Mesh(element('Lagrange','triangle',2,shape=(2,)))
    coords=np.array([[0.,0.],[1.,0.],[0.,1.],[0.5,0.5],[0.,0.5],[0.5,0.]])
    d=mesh.from_arrays(cells=[[0,1,2,3,4,5]],coordinates=coords,
                       coordinate_element=e,comm=MPI.COMM_SELF)
    assert (d.geometry.cmaps[0] if hasattr(d.geometry, "cmaps") else d.geometry.cmap).degree==2
    assert fem.assemble_scalar(fem.form(1*ufl.dx(domain=d)))==pytest.approx(0.5)


@pytest.mark.parametrize('mode',['lifting','matrix_elimination'])
def test_matrix_refresh_with_nonzero_boundary_matches_fresh_solve(mode):
    d=mesh.rectangle((0.,0.),(1.,1.),(3,3),comm=MPI.COMM_SELF)
    V=fem.functionspace(d,('Lagrange',1));u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
    k=fem.Constant(d,1.);g=fem.Function(V);g.x.array[:]=3.
    dofs=fem.locate_dofs_geometrical(V,lambda x:np.isclose(x[0],0.))
    bc=fem.dirichletbc(g,dofs)
    a=(k*ufl.inner(ufl.grad(u),ufl.grad(v))+u*v)*ufl.dx;L=2*v*ufl.dx
    out=fem.Function(V);ref=fem.Function(V)
    opt=solvers.LinearSolverOptions(ksp_type='preonly',pc_type='lu')
    p=solvers.PreparedLinearProblem(a,L,out,bcs=[bc],options=opt,bc_assembly=mode)
    try:
        p.solve();k.value=4.;g.x.array[:]=5.;p.refresh_matrix();p.solve()
        solvers.solve_linear_problem(fem.form(a),fem.form(L),ref,bcs=[bc],options=opt)
        np.testing.assert_allclose(out.x.array,ref.x.array,rtol=1e-11,atol=1e-11)
    finally:p.close()
