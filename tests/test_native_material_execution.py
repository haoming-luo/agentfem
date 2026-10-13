# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import ctypes
import numpy as np
import pytest
from agentfem import elements, constitutive
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


@pytest.mark.parametrize('width', [1, 7, 31])
@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('reject', [False, True])
def test_generic_native_state_width_atomic_repair(width, workers, reject):
    pytest.importorskip('agentfem._body_pipeline')
    real = ctypes.POINTER(ctypes.c_double)
    abi = ctypes.CFUNCTYPE(None, ctypes.c_int, real, real, real, real,
        ctypes.c_double, ctypes.c_double, real, real, real, ctypes.POINTER(ctypes.c_int))
    schema = MaterialStateSchema('native-test', (MaterialStateVariable('values', shape=(width,)),))
    n = 19
    f = np.tile(np.eye(3), (n,1,1))
    state = np.arange(n*width, dtype=float).reshape(n,width)
    request = MaterialPointArrayBatchInput(f, f, state, schema, .1, .001)
    repaired = []
    def reference(r):
        return constitutive.ExplicitMaterialBatchOutput(np.tile(np.eye(3)*2, (r.point_count,1,1)),
            r.state_old+1, schema, np.full(r.point_count, .1))
    material = constitutive.explicit_user_material(state_schema=schema, update=reference,
        identity={'test': 'native-law', 'width': width}, density=1.)
    def fallback(r):
        repaired.extend(r.state_old[:,0].tolist())
        return reference(r)
    def implementation(n, f0, f1, old, properties, dt, density, stress, new, work, status):
        old = np.ctypeslib.as_array(old, shape=(n*width,)).reshape(n,width)
        np.ctypeslib.as_array(new, shape=(n*width,))[:] = (old+1).ravel()
        np.ctypeslib.as_array(stress, shape=(n*9,))[:] = np.tile(np.eye(3)*2, (n,1,1)).ravel()
        np.ctypeslib.as_array(work, shape=(n,))[:] = .1
        np.ctypeslib.as_array(status, shape=(n,))[:] = (old[:,0] % (2*width) == 0) if reject else 0
    function = abi(implementation)
    kernel = elements.native_explicit_hex_kernel(material, function, properties=[], density=1.,
        identity={'fixture': width}, fallback=fallback, workers=workers)
    rng = np.random.default_rng(21)
    geometry = elements.ExplicitHexGeometry(rng.normal(size=(n,8,3)),
        rng.normal(size=(n,8,4)), np.ones(n), np.full(n,.01))
    u = rng.normal(size=(n,8,3))
    actual = kernel.evaluate(material, request, geometry, u)
    expected = elements.explicit_hex_kernel(backend='native').evaluate(material, request, geometry, u)
    for name in ('first_piola_stress', 'internal_force', 'hourglass_energy'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    np.testing.assert_array_equal(actual.material_response.state_new, state+1)
    assert sorted(repaired) == (state[::2,0].tolist() if reject else [])
    np.testing.assert_array_equal(request.state_old, state)
    if reject:
        def fail(request):
            raise RuntimeError('injected repair failure')
        bad = elements.native_explicit_hex_kernel(material, function, properties=[], density=1.,
            identity={'fixture': width}, fallback=fail, workers=workers)
        with pytest.raises(RuntimeError, match='repair failure'):
            bad.evaluate(material, request, geometry, u)
        np.testing.assert_array_equal(request.state_old, state)
