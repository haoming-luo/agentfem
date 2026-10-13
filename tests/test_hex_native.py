# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional kernels retain arithmetic order, rejection and reference fallbacks."""
import numpy as np
import pytest

native=pytest.importorskip('agentfem._hex_native')


@pytest.mark.parametrize('dtype',[np.int32,np.int64])
@pytest.mark.parametrize('additional',[False,True])
def test_scatter_matches_bincount_bitwise(dtype,additional):
    rng=np.random.default_rng(90)
    nodes=rng.integers(0,99,size=(421,8),dtype=dtype)
    force=rng.normal(size=(421,8,3))
    extra=rng.normal(size=force.shape) if additional else None
    expected=np.empty((99,3))
    for i in range(3):
        weights=force.reshape(-1,3)[:,i]
        if extra is not None:weights=weights+extra.reshape(-1,3)[:,i]
        expected[:,i]=np.bincount(nodes.ravel(),weights=weights,minlength=99)
    result=np.frombuffer(native.scatter(force,nodes,extra,99),dtype=float).reshape(-1,3)
    np.testing.assert_array_equal(result,expected)
    assert result.tobytes()==expected.tobytes()


def test_scatter_refuses_indices_nonfinite_and_overflow():
    force=np.ones((2,3));ids=np.array([0,1],dtype=np.int32)
    for bad in (np.array([-1,1],dtype=np.int32),np.array([0,2],dtype=np.int32)):
        with pytest.raises(ValueError):native.scatter(force,bad,None,2)
    for bad in (np.full((2,3),np.nan),np.full((2,3),np.inf)):
        with pytest.raises(ValueError):native.scatter(bad,ids,None,2)
    with pytest.raises(ValueError):native.scatter(np.full((2,3),1e308),np.zeros(2,dtype=np.int32),None,1)


def test_orientation_matches_reference(monkeypatch):
    from agentfem import _orientation as module
    rng=np.random.default_rng(905)
    f=np.eye(3)+rng.normal(0,.25,(1701,3,3))
    f[::17]*=-1
    f[0,0,0]=np.nan;f[1,0,0]=np.inf;f[2]=0
    result=module.positive_finite_determinants(f)
    monkeypatch.setattr(module,'_native_orientation',None)
    expected=module.positive_finite_determinants(f)
    np.testing.assert_array_equal(result,expected)


def test_finite_bits_include_signed_nan_subnormal_and_unaligned():
    values=np.array([0.,-0.,1.,-1.,np.nextafter(0.,1.),np.finfo(float).max])
    assert native.all_finite(values)
    for value in (np.nan,-np.nan,np.inf,-np.inf):
        for at in (0,3,5):
            bad=values.copy();bad[at]=value
            assert not native.all_finite(bad)
    storage=b'x'+values.tobytes()
    assert native.all_finite(np.frombuffer(storage,offset=1,dtype=float))
    assert native.all_finite(np.empty(0))


@pytest.mark.parametrize('chunk',[1,31,4096])
def test_cell_work_retains_force_motion_conjugacy(monkeypatch,chunk):
    from agentfem.elements import _bulk_viscosity as module
    rng=np.random.default_rng(810)
    force=rng.normal(size=(1201,8,3))
    nodes=rng.integers(0,105,size=(1201,8),dtype=np.int32)
    motion=rng.normal(size=(105,3))
    result=module._cell_increment_work(force,nodes,motion,chunk_size=chunk)
    monkeypatch.setattr(module,'_native_cell_work',None)
    reference=module._cell_increment_work(force,nodes,motion,chunk_size=chunk)
    assert abs(result[0]-reference[0]) <= 8*np.finfo(float).eps*reference[1]
    np.testing.assert_allclose(result[1],reference[1],rtol=8*np.finfo(float).eps)


def test_cell_work_refuses_invalid_indices_and_products():
    force=np.ones((1,8,3));nodes=np.zeros((1,8),dtype=np.int32);motion=np.ones((1,3))
    bad=nodes.copy();bad[0,0]=1
    with pytest.raises(ValueError):native.cell_work(force,bad,motion,32)
    badforce=force.copy();badforce[0,0,0]=np.nan
    with pytest.raises(ValueError):native.cell_work(badforce,nodes,motion,32)


def test_determinants_keep_reference_arithmetic_and_lu(monkeypatch):
    from agentfem import _small_matrix as module
    rng=np.random.default_rng(900)
    a=np.eye(3)+rng.normal(0,.15,(502,3,3))
    a[-2]=np.diag([1e-200,1e100,1e100])
    a[-1]=np.ones((3,3))
    actual=module.determinants_3x3(a)
    monkeypatch.setattr(module,'_native_determinants',None)
    expected=module.determinants_3x3(a)
    np.testing.assert_array_equal(actual,expected)
@pytest.mark.parametrize("scale", [0., 1.e-310, 1.e-200, 1., 1.e100, 1.e300])
def test_native_symmetric_stress_matches_reference_at_threshold(scale):
    native = pytest.importorskip("agentfem._hex_native")
    random = np.random.default_rng(4921).normal(size=(100, 3, 3))
    stress = scale * (random + random.transpose(0, 2, 1))
    def reference(values):
        tolerance = 1e-12*np.maximum(np.max(np.abs(values), axis=(1, 2)), np.finfo(float).tiny)
        return all(not np.any(np.abs(values[:, i,j]-values[:, j,i]) > tolerance)
                   for i,j in ((0,1),(0,2),(1,2)))
    for multiplier in (0., .999, 1., 1.001, 2.):
        sample = stress.copy()
        sample[0, 0, 1] += multiplier * 1e-12 * max(np.max(np.abs(stress[0])), np.finfo(float).tiny)
        assert native.symmetric_stress(sample) == reference(sample)
        unaligned = np.frombuffer(b"x" + sample.tobytes(), dtype=float, offset=1).reshape(sample.shape)
        assert native.symmetric_stress(unaligned) == reference(sample)
    assert native.symmetric_stress(np.empty((0,3,3)))
    for invalid in (np.nan, np.inf, -np.inf):
        sample = stress.copy()
        sample[0,0,0] = invalid
        assert not native.symmetric_stress(sample)
