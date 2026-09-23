import json
import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from agentfem import mesh, fields, results, solvers, diagnostics

def test_point_sampling_reports_missing_and_nonfinite_separately():
    d=mesh.rectangle((0.,0.),(1.,1.),(2,2),comm=MPI.COMM_SELF)
    f=fields.scalar_unknown(d,value=2.)
    s=results.sample_points(f,[[0.25,0.25],[2.,2.]],missing='nan',return_info=True)
    assert isinstance(s,results.PointSample)
    assert s.summary()['mesh_coverage']==0.5
    assert s.summary()['valid_coverage']==0.5
    assert s.summary()['missing_indices']==[1]
    f.value.x.array[:]=np.nan
    s=results.sample_points(f,[[0.25,0.25],[2.,2.]],missing='nan',return_info=True)
    assert s.summary()['nonfinite_inside_indices']==[0]
    assert s.summary()['valid_coverage']==0.
    assert results.sample_points(f,np.empty((0,2)),return_info=True).summary()['mesh_coverage'] is None
    with pytest.raises(ValueError,match='finite'):
        results.sample_points(f,[[np.nan,0.]])


def test_failed_linear_solve_has_machine_readable_next_checks():
    d=mesh.rectangle((0.,0.),(1.,1.),(5,5),comm=MPI.COMM_SELF)
    V=fem.functionspace(d,('Lagrange',1));u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
    x=ufl.SpatialCoordinate(d);out=fem.Function(V)
    a=(u*v+ufl.inner(ufl.grad(u),ufl.grad(v)))*ufl.dx
    L=ufl.exp(x[0]+2*x[1])*v*ufl.dx
    with pytest.raises(diagnostics.ComputationalFailure) as caught:
        solvers.solve_linear_problem(fem.form(a),fem.form(L),out,options=solvers.LinearSolverOptions(
            ksp_type='cg',pc_type='none',rtol=1e-14,max_it=1))
    d=caught.value.as_dict()
    assert d['code']=='AF-LINEAR-ITERATION-LIMIT'
    assert d['stage']=='linear_solve' and d['iterations']==1
    assert d['suggestions']
    json.dumps(d)



def test_nonfinite_failure_report_is_strict_json():
    from petsc4py import PETSc
    record=diagnostics.linear_failure_diagnostic(PETSc.KSP.ConvergedReason.DIVERGED_NANORINF,2,float('inf'))
    assert record['residual_norm'] is None
    assert record['code']=='AF-LINEAR-NONFINITE'
    json.dumps(record,allow_nan=False)
