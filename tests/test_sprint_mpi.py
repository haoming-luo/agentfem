import numpy as np
import pytest
import ufl
from basix.ufl import element
from mpi4py import MPI
from agentfem import mesh, expressions, fields, results

pytestmark=pytest.mark.skipif(MPI.COMM_WORLD.size<2,reason='requires two MPI ranks')

def test_one_rank_bad_mesh_fails_collectively():
    cells=[[0.,1.,2.]] if MPI.COMM_WORLD.rank==0 else [[0,1,2]]
    with pytest.raises(ValueError,match='rank 0'):
        mesh.from_arrays(cells=cells,coordinates=[[0.,0.],[1.,0.],[0.,1.]],
            coordinate_element=ufl.Mesh(element('Lagrange','triangle',1,shape=(2,))),comm=MPI.COMM_WORLD)

def test_one_rank_nonfinite_interpolation_rolls_back_everywhere():
    d=mesh.rectangle((0.,0.),(1.,1.),(3,3),comm=MPI.COMM_WORLD)
    f=fields.scalar_unknown(d,value=7.)
    source='sqrt(x-2)' if MPI.COMM_WORLD.rank==0 else '3.0'
    with np.errstate(invalid='ignore'):
        with pytest.raises(expressions.ExpressionError):expressions.interpolate(f,source)
    np.testing.assert_array_equal(f.value.x.array,7.)
    points=[[np.nan,0.]] if MPI.COMM_WORLD.rank==0 else [[0.,0.]]
    with pytest.raises(ValueError,match='rank 0'):results.sample_points(f,points)


def test_one_rank_invalid_expression_fails_before_interpolation():
    d=mesh.rectangle((0.,0.),(1.,1.),(3,3),comm=MPI.COMM_WORLD)
    f=fields.scalar_unknown(d,value=7.)
    source='sin(' if MPI.COMM_WORLD.rank==0 else '3.0'
    with pytest.raises(expressions.ExpressionError):
        expressions.interpolate(f,source)
    np.testing.assert_array_equal(f.value.x.array,7.)


def test_one_rank_invalid_sampling_option_fails_collectively():
    d=mesh.rectangle((0.,0.),(1.,1.),(3,3),comm=MPI.COMM_WORLD)
    f=fields.scalar_unknown(d,value=7.)
    with pytest.raises(ValueError,match='rank 0'):
        results.sample_points(f,[[0.5,0.5]],padding=-1 if MPI.COMM_WORLD.rank==0 else 1e-10)
