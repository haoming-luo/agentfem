# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Parallel arithmetic is byte-identical; transport is detached and bounded."""
import ctypes
from types import SimpleNamespace
import numpy as np
import pytest

from agentfem.elements.body_auxiliary import auxiliary_callbacks, gather_callback
from agentfem.elements._hex_validity import _coordinate_edges


def inputs(n):
    rng = np.random.default_rng(703)
    geometry = SimpleNamespace(coordinates=rng.normal(size=(n,8,3)),
        average_gradient=rng.normal(scale=.05, size=(n,8,3)),
        reference_perturbation_radius=np.full(n, .7),
        lumped_mass=rng.uniform(.5, 2, (n,8)), volume=rng.uniform(.3, 1, n))
    return geometry, rng.normal(scale=.01, size=(n,8,3))


@pytest.mark.parametrize('n', [0, 7, 8201])
def test_auxiliary_parallel_is_exact_and_retains_reference(n):
    native = pytest.importorskip('agentfem._body_pipeline')
    geometry, u = inputs(n)
    # Call the unchanged serial arithmetic directly as an independent dispatch
    # oracle, rather than comparing two new implementations of the formula.
    library = ctypes.CDLL(native.__file__)
    real = np.ctypeslib.ndpointer(dtype=float, flags=('C_CONTIGUOUS', 'ALIGNED'))
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags=('C_CONTIGUOUS', 'ALIGNED'))
    gradient = library.hex_gradient
    gradient.argtypes = [ctypes.c_int, real, real, real, real, integer, real, integer]
    gradient.restype = None
    f = np.empty((n,3,3)); flags = np.empty(n, dtype=np.int32)
    edges = np.ascontiguousarray(_coordinate_edges(), dtype=np.int32)
    gradient(n, geometry.coordinates, u, geometry.average_gradient,
        geometry.reference_perturbation_radius, edges, f, flags)
    coefficients = {name: np.full(n, value) for name, value in
        [('density', 2.), ('wave_speed', 3.), ('characteristic_length', .4),
         ('linear', .06), ('quadratic', 1.2)]}
    bulk = library.hex_bulk_response
    bulk.argtypes = [ctypes.c_int]+[real]*15+[integer]
    bulk.restype = None
    expected = [np.empty((n,8,3))]+[np.empty(n) for _ in range(4)]
    status = np.empty(n, dtype=np.int32)
    bulk(n, f, geometry.average_gradient, u, geometry.lumped_mass,
        geometry.volume, *coefficients.values(), *expected, status)
    assert not status.any()
    for workers in (1, 2, 4):
        kin, visc = auxiliary_callbacks(workers=workers)
        actual_f, actual_flags = kin(geometry, u)
        np.testing.assert_array_equal(actual_f, f)
        np.testing.assert_array_equal(actual_flags, flags.astype(bool))
        with pytest.raises(ValueError):
            actual_f.setflags(write=True)
        actual = visc(geometry, actual_f, u, coefficients,
                      lambda: pytest.fail('Unexpected reference fallback'))
        for name, value in zip(('internal_force', 'artificial_power', 'volumetric_rate',
                               'isotropic_stress', 'mass_scaled_damping_bound'), expected, strict=True):
            np.testing.assert_array_equal(getattr(actual, name), value)
            with pytest.raises(ValueError):
                getattr(actual, name).setflags(write=True)


@pytest.mark.parametrize('n', [0, 7, 8201])
def test_native_gather_is_exact_detached_and_reentrant(n):
    pytest.importorskip('agentfem._body_pipeline')
    rng = np.random.default_rng(912)
    nodes = rng.integers(0, 61, (n,8), dtype=np.int32)
    values = rng.normal(size=(61,3))
    expected = values[nodes].copy()
    for workers in (1, 4):
        gather = gather_callback(workers=workers)
        first = gather(values, nodes)
        np.testing.assert_array_equal(first, expected)
        second = gather(values+1, nodes)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected+1)
        assert not np.shares_memory(first, second)
        with pytest.raises(ValueError):
            first.setflags(write=True)
        if n:
            bad = nodes.copy(); bad[-1,-1] = 61
            with pytest.raises(ValueError, match='outside'):
                gather(values, bad)
            bad[-1,-1] = -1
            with pytest.raises(ValueError, match='outside'):
                gather(values, bad)
        np.testing.assert_array_equal(first, expected)


def test_auxiliary_failure_contracts():
    native = pytest.importorskip('agentfem._body_pipeline')
    geometry, u = inputs(5)
    kin, visc = auxiliary_callbacks(workers=4)
    f, _ = kin(geometry, u)
    f = f.copy(); f[-1] = -np.eye(3)
    coefficients = {name: np.ones(5) for name in
                    ('density','wave_speed','characteristic_length','linear','quadratic')}
    sentinel = object()
    assert visc(geometry, f, u, coefficients, lambda: sentinel) is sentinel
    with pytest.raises(RuntimeError, match='reference rejected'):
        visc(geometry, f, u, coefficients,
             lambda: (_ for _ in ()).throw(RuntimeError('reference rejected')))
    for workers in (True, 0, 17, 2.5):
        with pytest.raises(ValueError, match='workers'):
            auxiliary_callbacks(workers=workers)
    for nodes in (np.zeros((1,8), dtype=np.int64), np.zeros((1,7), dtype=np.int32)):
        with pytest.raises(ValueError):
            native.auxiliary(2, (np.zeros((8,3)), nodes), 4)
    with pytest.raises(ValueError):
        native.auxiliary(2, (np.zeros((8,3), dtype=np.float32), np.zeros((1,8), dtype=np.int32)), 4)
    with pytest.raises(ValueError):
        native.auxiliary(99, (), 1)


def test_gather_provider_cannot_retain_solver_storage():
    from dataclasses import replace
    from agentfem import elements
    from test_response_material_step import contact_step
    pytest.importorskip('agentfem._body_pipeline')
    kernel = elements.explicit_hex_kernel(backend='native', cell_workers=2)
    observed = []
    original = kernel.gather_cell_vectors
    def gather(values, nodes):
        observed.append((values, nodes))
        with pytest.raises(ValueError):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            nodes.setflags(write=True)
        return original(values, nodes)
    step = contact_step(execution_kernel=replace(kernel, gather_cell_vectors=gather))
    step.run(until_step=1)
    retained = observed[-1][0]
    old = retained.copy()
    step.run(until_step=2)
    np.testing.assert_array_equal(retained, old)
