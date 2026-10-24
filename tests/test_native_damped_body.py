# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused execution preserves separately observable body and artificial damping."""
import ctypes

import numpy as np
import pytest


@pytest.fixture
def native():
    module = pytest.importorskip("agentfem._body_pipeline")
    if not hasattr(module, "run_damped_checked"):
        pytest.skip("requires the candidate damped dispatcher")
    return module


def inputs(n=37, *, rejected=False, ill_conditioned=False):
    rng = np.random.default_rng(517)
    f = np.tile(np.eye(3), (n, 1, 1))
    f[:, 0, 1] = .03
    if ill_conditioned:
        f[:, 0, 0] = 1e-12
    state = np.arange(n * 3, dtype=float).reshape(n, 3)
    nodes = np.arange(n * 8, dtype=np.int32).reshape(n, 8)
    v = rng.normal(size=(n * 8, 3))
    g = rng.normal(scale=.1, size=(n, 8, 3))
    body = (f, f, state, np.empty(0), rng.normal(size=(n, 8, 3)), g,
            rng.normal(size=(n, 8, 4)), np.ones(n), np.full(n, .1))
    damping = (v, nodes, np.ones((n, 8)), np.ones(n), np.full(n, 2.),
               np.full(n, .3), np.full(n, .06), np.full(n, 1.2))
    real = ctypes.POINTER(ctypes.c_double)
    abi = ctypes.CFUNCTYPE(None, ctypes.c_int, real, real, real, real,
                          ctypes.c_double, ctypes.c_double, real, real, real,
                          ctypes.POINTER(ctypes.c_int))

    def material(count, f0, f1, old, props, dt, rho, stress, new, work, status):
        old = np.ctypeslib.as_array(old, shape=(count * 3,)).reshape(count, 3)
        np.ctypeslib.as_array(stress, shape=(count * 9,))[:] = np.tile(np.eye(3), (count, 1, 1)).ravel()
        np.ctypeslib.as_array(new, shape=(count * 3,))[:] = (old + 1.).ravel()
        np.ctypeslib.as_array(work, shape=(count,))[:] = .2
        if rejected:
            np.ctypeslib.as_array(status, shape=(count,))[:] = (old[:, 0] % 2 == 0)

    function = abi(material)

    def repair(raw):
        selected = np.frombuffer(raw, dtype=np.int64)
        return np.tile(np.eye(3), (len(selected), 1, 1)), state[selected] + 1., np.full(len(selected), .2)
    return function, body, damping, repair


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("tile", [0, 7])
@pytest.mark.parametrize("rejected", [False, True])
def test_fused_body_damping_equals_separate_cells(native, workers, tile, rejected):
    function, body, damp, repair = inputs(rejected=rejected)
    address = ctypes.cast(function, ctypes.c_void_p).value
    executor = native.create_pool(workers) if workers > 1 else None
    try:
        before = [value.copy() for value in (*body, *damp)]
        reference = native.run_checked(address, body, .001, 1., workers, 3, 0, repair, executor, tile)
        old_bulk = native.auxiliary(1, (body[1], body[5], damp[0][damp[1]], damp[2], body[7], *damp[3:]), workers, executor)
        candidate = native.run_damped_checked(address, body, .001, 1., workers, 3, 0, repair, executor, tile, damp)
        assert candidate == (*reference, *old_bulk)
        for value, original in zip((*body, *damp), before, strict=True):
            np.testing.assert_array_equal(value, original)
    finally:
        if executor is not None:
            executor.close()


def test_viscosity_conditioning_keeps_explicit_fallback_status(native):
    function, body, damp, repair = inputs(ill_conditioned=True)
    output = native.run_damped_checked(ctypes.cast(function, ctypes.c_void_p).value,
        body, .001, 1., 1, 3, 0, repair, None, 0, damp)
    assert np.frombuffer(output[-1], dtype=np.int32).all()
    assert np.isnan(np.frombuffer(output[6], dtype=float)).all()
    assert np.isfinite(np.frombuffer(output[4], dtype=float)).all()


@pytest.mark.parametrize("fault", ["node", "velocity", "mass", "repair"])
def test_damped_failure_publishes_no_response(native, fault):
    function, body, damp, repair = inputs(rejected=fault == "repair")
    if fault == "node":
        damp[1][0, 0] = len(damp[0])
    elif fault == "velocity":
        damp[0][0, 0] = np.nan
    elif fault == "mass":
        damp[2][0, 0] = 0.
    else:
        def repair(_):
            raise RuntimeError("injected exact repair failure")
    before = body[2].copy()
    with pytest.raises((ValueError, RuntimeError)):
        native.run_damped_checked(ctypes.cast(function, ctypes.c_void_p).value,
            body, .001, 1., 3, 3, 0, repair, None, 7, damp)
    np.testing.assert_array_equal(body[2], before)
