# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import ctypes
import numpy as np
import pytest
from agentfem import elements, constitutive
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


@pytest.mark.parametrize('tile_size', [True, 0, -1, 1.5, 2147483648])
def test_invalid_tile_size_rejected_before_binding(tile_size):
    pytest.importorskip('agentfem._body_pipeline')
    with pytest.raises(ValueError, match='tile_size'):
        elements.native_explicit_hex_kernel(None, None, properties=[], density=1.,
            identity={}, fallback=lambda _: None, tile_size=tile_size)


@pytest.mark.parametrize('fault', ['stress', 'state', 'work', 'asymmetric', 'untouched', 'force_overflow'])
@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('tile_size', [None, 7])
def test_checked_worker_rejects_invalid_output_before_publication(fault, workers, tile_size):
    native = pytest.importorskip('agentfem._body_pipeline')
    if not hasattr(native, 'run_checked'):
        pytest.skip('requires newly built checked dispatcher')
    real = ctypes.POINTER(ctypes.c_double)
    abi = ctypes.CFUNCTYPE(None, ctypes.c_int, real, real, real, real,
        ctypes.c_double, ctypes.c_double, real, real, real, ctypes.POINTER(ctypes.c_int))
    schema = MaterialStateSchema('one', (MaterialStateVariable('q'),))
    n = 37
    f = np.tile(np.eye(3), (n, 1, 1))
    request = MaterialPointArrayBatchInput(f, f, np.zeros((n, 1)), schema, 0., .001)
    def reference(r):
        return constitutive.ExplicitMaterialBatchOutput(np.zeros((r.point_count,3,3)),
            np.zeros((r.point_count,1)), schema, np.zeros(r.point_count))
    material = constitutive.explicit_user_material(state_schema=schema, update=reference,
        identity={'test': fault}, density=1.)
    def callback(count, f0, f1, old, props, dt, rho, stress, state, work, status):
        if fault == 'untouched':
            return
        np.ctypeslib.as_array(stress, shape=(count*9,))[:] = 0.
        np.ctypeslib.as_array(state, shape=(count,))[:] = 0.
        np.ctypeslib.as_array(work, shape=(count,))[:] = 0.
        if fault == 'stress': stress[0] = float('nan')
        if fault == 'state': state[0] = float('nan')
        if fault == 'work': work[0] = float('inf')
        if fault == 'asymmetric': stress[1] = 1.
        if fault == 'force_overflow': stress[0] = 1.e308
    geometry = elements.ExplicitHexGeometry(np.full((n,8,3), 10.),
        np.zeros((n,8,4)), np.ones(n), np.zeros(n))
    with elements.native_explicit_hex_kernel(material, abi(callback), properties=[],
            density=1., identity={'test': fault}, fallback=reference, workers=workers,
            tile_size=tile_size) as kernel:
        with pytest.raises(ValueError, match='native material/body output'):
            kernel.evaluate(material, request, geometry, np.zeros((n,8,3)))
    np.testing.assert_array_equal(request.state_old, 0.)


@pytest.mark.parametrize('width,n', [(1,19), (7,19), (31,19), (7,17003)])
@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('reject', [False, True, 'single', 'all'])
@pytest.mark.parametrize('tile_size', [None, 257])
def test_generic_native_state_width_atomic_repair(width, n, workers, reject, tile_size):
    pytest.importorskip('agentfem._body_pipeline')
    real = ctypes.POINTER(ctypes.c_double)
    abi = ctypes.CFUNCTYPE(None, ctypes.c_int, real, real, real, real,
        ctypes.c_double, ctypes.c_double, real, real, real, ctypes.POINTER(ctypes.c_int))
    schema = MaterialStateSchema('native-test', (MaterialStateVariable('values', shape=(width,)),))
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
        rejected = ((old[:,0] % (2*width) == 0) if reject is True else
                    old[:,0] == 0 if reject == 'single' else
                    np.ones(n, dtype=bool) if reject == 'all' else np.zeros(n, dtype=bool))
        np.ctypeslib.as_array(status, shape=(n,))[:] = rejected
        # Rejected bytes are deliberately unusable until exact repair.
        np.ctypeslib.as_array(stress, shape=(n*9,)).reshape(n, 9)[rejected] = np.nan
        np.ctypeslib.as_array(new, shape=(n*width,)).reshape(n, width)[rejected] = np.nan
        np.ctypeslib.as_array(work, shape=(n,))[rejected] = np.nan
    function = abi(implementation)
    kernel = elements.native_explicit_hex_kernel(material, function, properties=[], density=1.,
        identity={'fixture': width}, fallback=fallback, workers=workers, tile_size=tile_size)
    rng = np.random.default_rng(21)
    geometry = elements.ExplicitHexGeometry(rng.normal(size=(n,8,3)),
        rng.normal(size=(n,8,4)), np.ones(n), np.full(n,.01))
    u = rng.normal(size=(n,8,3))
    actual = kernel.evaluate(material, request, geometry, u)
    expected = elements.explicit_hex_kernel(backend='native').evaluate(material, request, geometry, u)
    for name in ('first_piola_stress', 'internal_force', 'hourglass_energy'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    np.testing.assert_array_equal(actual.material_response.state_new, state+1)
    expected_repaired = (state[::2,0].tolist() if reject is True else
                         state[:1,0].tolist() if reject == 'single' else
                         state[:,0].tolist() if reject == 'all' else [])
    assert sorted(repaired) == expected_repaired
    np.testing.assert_array_equal(request.state_old, state)
    if reject:
        def fail(request):
            raise RuntimeError('injected repair failure')
        bad = elements.native_explicit_hex_kernel(material, function, properties=[], density=1.,
            identity={'fixture': width}, fallback=fail, workers=workers, tile_size=tile_size)
        with pytest.raises(RuntimeError, match='repair failure'):
            bad.evaluate(material, request, geometry, u)
        np.testing.assert_array_equal(request.state_old, state)
        bad.close()
    again = kernel.evaluate(material, request, geometry, u)
    np.testing.assert_array_equal(again.internal_force, actual.internal_force)
    kernel.close()
    kernel.close()
    np.testing.assert_array_equal(actual.material_response.state_new, state+1)
    if workers > 1:
        with pytest.raises(RuntimeError, match='closed'):
            kernel.evaluate(material, request, geometry, u)


@pytest.mark.parametrize('failure', ['cutback', 'energy', 'nonfinite'])
def test_native_repair_preserves_scientific_rejection(failure):
    pytest.importorskip('agentfem._body_pipeline')
    from agentfem.constitutive._explicit_material import ExplicitMaterialCutbackRequest
    real = ctypes.POINTER(ctypes.c_double)
    abi = ctypes.CFUNCTYPE(None, ctypes.c_int, real, real, real, real,
        ctypes.c_double, ctypes.c_double, real, real, real, ctypes.POINTER(ctypes.c_int))
    schema = MaterialStateSchema('one', (MaterialStateVariable('q'),))
    f = np.eye(3)[None]
    request = MaterialPointArrayBatchInput(f, f, np.zeros((1, 1)), schema, 0., .001)
    def reference(r):
        options = {'requested_time_scale': np.full(1, .5)} if failure == 'cutback' else (
            {'stored_energy_density': np.zeros(1)} if failure == 'energy' else {})
        stress = np.full((1, 3, 3), np.nan) if failure == 'nonfinite' else np.zeros((1, 3, 3))
        return constitutive.ExplicitMaterialBatchOutput(stress, np.ones((1, 1)),
            schema, np.zeros(1), **options)
    material = constitutive.explicit_user_material(state_schema=schema, update=reference,
        identity={'test': failure}, density=1.)
    def reject(n, f0, f1, old, props, dt, rho, stress, state, work, status):
        status[0] = 1
    kernel = elements.native_explicit_hex_kernel(material, abi(reject), properties=[],
        density=1., identity={'test': failure}, fallback=reference)
    geometry = elements.ExplicitHexGeometry(np.zeros((1, 8, 3)),
        np.zeros((1, 8, 4)), np.ones(1), np.zeros(1))
    expected = ExplicitMaterialCutbackRequest if failure == 'cutback' else ValueError
    with pytest.raises(expected):
        kernel.evaluate(material, request, geometry, np.zeros((1, 8, 3)))
    np.testing.assert_array_equal(request.state_old, 0.)
    kernel.close()


def test_native_executor_alternates_batch_sizes_and_recovers_validation_failure():
    native = pytest.importorskip('agentfem._body_pipeline')
    for invalid in (0, 17, True):
        with pytest.raises(ValueError):
            native.create_pool(invalid)
    pool = native.create_pool(4)
    values = np.arange(81., dtype=float).reshape(27, 3)
    for count, workers in ((9001,4), (1,1), (0,4), (17003,3), (9001,2)):
        indices = np.arange(count*8, dtype=np.int32).reshape(count,8) % len(values)
        output, = native.auxiliary(2, (values, indices), workers, pool)
        np.testing.assert_array_equal(np.frombuffer(output, dtype=float).reshape(count,8,3), values[indices])
    with pytest.raises(ValueError):
        native.auxiliary(2, (values, np.full((9001,8), 50, dtype=np.int32)), 4, pool)
    with pytest.raises(RuntimeError, match='undersized'):
        native.auxiliary(2, (values, indices), 5, pool)
    expected, = native.auxiliary(2, (values, indices), 4, pool)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match='closed'):
        native.auxiliary(2, (values, indices), 4, pool)
    np.testing.assert_array_equal(np.frombuffer(expected, dtype=float).reshape(count,8,3), values[indices])


def test_native_executor_rejects_close_and_reentry_during_material_callback():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    native = pytest.importorskip('agentfem._body_pipeline')
    real = ctypes.POINTER(ctypes.c_double)
    abi = ctypes.CFUNCTYPE(None, ctypes.c_int, real, real, real, real,
        ctypes.c_double, ctypes.c_double, real, real, real, ctypes.POINTER(ctypes.c_int))
    started, release = Event(), Event()
    def callback(n, old_f, new_f, old_state, properties, dt, rho, stress, state, work, status):
        started.set()
        if not release.wait(5):
            return  # untouched sentinel output remains invalid
        for i in range(n):
            for j in range(9): stress[9*i+j]=0.
            state[i]=1.; work[i]=0.
    function = abi(callback)
    pool = native.create_pool(2)
    f = np.tile(np.eye(3), (2,1,1))
    inputs = (f,f,np.zeros((2,1)),np.empty(0),np.zeros((2,8,3)),
              np.zeros((2,8,3)),np.zeros((2,8,4)),np.ones(2),np.zeros(2))
    def run():
        return native.run(ctypes.cast(function,ctypes.c_void_p).value,
                          inputs,.01,1.,2,1,0,lambda _:None,pool)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(run)
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match='busy'): pool.close()
            with pytest.raises(RuntimeError, match='busy'): run()
        finally:
            release.set()
        output=future.result(timeout=5)
    pool.close()
    np.testing.assert_array_equal(np.frombuffer(output[1],dtype=float),1.)


def test_native_executor_fork_is_refused_without_inherited_lock_wait():
    import os
    import subprocess
    import sys
    pytest.importorskip('agentfem._body_pipeline')
    if not hasattr(os, 'fork'):
        pytest.skip('POSIX native executor')
    source = '''
import os
import numpy as np
from agentfem import _body_pipeline as native
pool = native.create_pool(2)
values = np.zeros((1, 3))
indices = np.zeros((9001, 8), dtype=np.int32)
pid = os.fork()
if pid == 0:
    try:
        native.auxiliary(2, (values, indices), 2, pool)
    except RuntimeError as error:
        success = 'after fork' in str(error)
    else:
        success = False
    pool.close()
    os._exit(0 if success else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
result, = native.auxiliary(2, (values, indices), 2, pool)
assert len(result) == 9001 * 8 * 3 * 8
pool.close()
'''
    subprocess.run([sys.executable, '-c', source], check=True, timeout=15)
