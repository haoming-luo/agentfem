import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from agentfem import mesh, operators, solvers, time


def test_error_step_factor_boundaries():
    assert time.error_step_factor(0.,1.)==2.
    assert time.error_step_factor(1.,1.)==pytest.approx(0.9)
    assert time.error_step_factor(100.,1.)==0.5
    with pytest.raises(ValueError):time.error_step_factor(float('nan'),1.)


def test_supg_stagnation_diffusion_and_temporal_bound():
    d=mesh.rectangle((0.,0.),(1.,1.),(2,2),comm=MPI.COMM_SELF)
    tau=operators.intrinsic_time_scale(d,(0.,0.),diffusivity=0.1,directional=True)
    val=fem.assemble_scalar(fem.form(tau*ufl.dx(domain=d)))
    assert np.isfinite(val) and val>0
    limited=operators.intrinsic_time_scale(d,(0.,0.),diffusivity=0.1,time_step=0.01)
    v=fem.assemble_scalar(fem.form(limited*ufl.dx(domain=d)))
    assert 0<v<=0.005 and v<val
    with pytest.raises(ValueError,match='time_step'):
        operators.intrinsic_time_scale(d,(1.,0.),time_step=0.)


def test_pressure_nullspace_gauge_preserves_velocity():
    # A velocity block coupled to pressure difference; constant pressure is free.
    A=PETSc.Mat().createDense([3,3],comm=MPI.COMM_SELF)
    A.setUp();A.setValues(range(3),range(3),[[2.,1.,-1.],[1.,0.,0.],[-1.,0.,0.]]);A.assemble()
    b=A.createVecRight();b.setValues(range(3),[2.,1.,-1.]);b.assemble()
    mode=b.duplicate();mode.setValues(range(3),[0.,1.,1.]);mode.assemble()
    ns=solvers.attach_nullspace(A,[mode],rhs=b)
    ksp=PETSc.KSP().create(MPI.COMM_SELF);ksp.setOperators(A);ksp.setType('gmres');ksp.getPC().setType('none');ksp.setTolerances(rtol=1e-12)
    x=b.duplicate();x.set(0);ksp.solve(b,x);assert ksp.getConvergedReason()>0
    first=x.getArray().copy();x.axpy(7.,mode)
    ksp.setInitialGuessNonzero(True);ksp.solve(b,x)
    assert x.getArray()[0]==pytest.approx(first[0],abs=1e-10)
    assert x.getArray()[1]-x.getArray()[2]==pytest.approx(first[1]-first[2],abs=1e-10)
    invalid=b.copy();invalid.setValue(1,3.);invalid.assemble()
    with pytest.raises(ValueError,match='incompatible'):solvers.attach_nullspace(A,[mode],rhs=invalid)
    invalid.destroy();x.destroy();ksp.destroy();ns.destroy();mode.destroy();b.destroy();A.destroy()


def test_stokes_constant_pressure_mode_and_manufactured_velocity():
    from basix.ufl import element, mixed_element
    from dolfinx import mesh as dm
    from dolfinx.fem import petsc as fp
    d=dm.create_unit_square(MPI.COMM_SELF,2,2)
    W=fem.functionspace(d,mixed_element([
        element('Lagrange','triangle',2,shape=(2,)),element('Lagrange','triangle',1)]))
    V,_=W.sub(0).collapse();Q,pmap=W.sub(1).collapse()
    g=fem.Function(V);g.interpolate(lambda x:np.vstack((x[0]**2,-2*x[0]*x[1])))
    facets=dm.locate_entities_boundary(d,1,lambda x:np.ones(x.shape[1],dtype=bool))
    bc=fem.dirichletbc(g,fem.locate_dofs_topological((W.sub(0),V),1,facets),W.sub(0))
    u,p=ufl.TrialFunctions(W);v,q=ufl.TestFunctions(W)
    a=fem.form((ufl.inner(ufl.grad(u),ufl.grad(v))-p*ufl.div(v)-q*ufl.div(u))*ufl.dx)
    L=fem.form(ufl.inner(ufl.as_vector((-1.,1.)),v)*ufl.dx)
    A=fp.assemble_matrix(a,bcs=[bc]);A.assemble();b=fp.assemble_vector(L)
    fp.apply_lifting(b,[a],[[bc]]);b.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE);fp.set_bc(b,[bc])
    mode=fem.Function(W);mode.x.array[:]=0.;mode.x.array[pmap]=1.
    ns=solvers.attach_nullspace(A,[mode.x.petsc_vec],rhs=b)
    out=fem.Function(W);ksp=PETSc.KSP().create(MPI.COMM_SELF);ksp.setOperators(A)
    ksp.setType('gmres');ksp.setGMRESRestart(100);ksp.getPC().setType('none');ksp.setTolerances(rtol=1e-11,atol=1e-12,max_it=300)
    ksp.solve(b,out.x.petsc_vec);assert ksp.getConvergedReason()>0
    velocity=out.sub(0).collapse();np.testing.assert_allclose(velocity.x.array,g.x.array,atol=1e-8)
    before=velocity.x.array.copy();out.x.petsc_vec.axpy(5.,mode.x.petsc_vec)
    ksp.setInitialGuessNonzero(True);ksp.solve(b,out.x.petsc_vec)
    np.testing.assert_allclose(out.sub(0).collapse().x.array,before,atol=1e-8)
    ksp.destroy();ns.destroy();b.destroy();A.destroy()
