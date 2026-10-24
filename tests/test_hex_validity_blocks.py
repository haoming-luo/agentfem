# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded-scratch admission must retain the unblocked sufficient decision."""
import basix
import numpy as np
import pytest
from agentfem.elements._hex_validity import (
    admits_affine_perturbation, _affine_reference,
    reference_perturbation_radius, require_positive_hex_jacobian,
)


@pytest.mark.parametrize('scale', [1e-8, 1., 1e8])
def test_chunk_boundaries_preserve_affine_admission(scale):
    rng = np.random.default_rng(902)
    count = 4101
    x = np.tile(basix.cell.geometry(basix.CellType.hexahedron), (count, 1, 1))*scale
    radius = reference_perturbation_radius(x)
    f = np.tile(np.eye(3), (count, 1, 1))
    angle = np.linspace(0, 3.1, count)
    f[:,0,0] = f[:,1,1] = np.cos(angle)
    f[:,0,1], f[:,1,0] = -np.sin(angle), np.sin(angle)
    f[::7,0,0] += .2
    f[::31,2,2] = -1  # Reflections must not be certified by a Gram bound.
    u = x @ (f-np.eye(3)).swapaxes(1,2)
    u[::11] += rng.normal(0, .03*scale, u[::11].shape)
    expected = _affine_reference(x,u,f,radius)
    actual = admits_affine_perturbation(x,u,f,radius)
    np.testing.assert_array_equal(actual,expected)
    assert np.any(actual) and np.any(~actual)
    assert not np.any(actual[::31])
    assert np.all(require_positive_hex_jacobian((x+u)[actual]) > 0)


def test_empty_affine_batch():
    assert admits_affine_perturbation(np.empty((0,8,3)),np.empty((0,8,3)),
                                    np.empty((0,3,3)),np.empty(0)).shape == (0,)


def test_native_screen_is_sufficient_and_fallback_is_retained(monkeypatch):
    import agentfem.elements._hex_validity as module
    if module._native_affine_screen is None:
        pytest.skip('Optional native extension not built')
    x=np.tile(basix.cell.geometry(basix.CellType.hexahedron),(6,1,1))
    f=np.tile(np.eye(3),(6,1,1))
    f[1,0,0]=-1
    f[2]=np.diag([2.,1.,1.])
    u=x@(f-np.eye(3)).swapaxes(1,2)
    u[3,0,0]=3.
    f[4,0,0]=np.nan
    radius=reference_perturbation_radius(x)
    reference=module._affine_reference(x,u,f,radius)
    expected_fallback=[]
    original=module._affine_reference
    def tracked(*args):
        expected_fallback.append(len(args[0]))
        return original(*args)
    monkeypatch.setattr(module,'_affine_reference',tracked)
    np.testing.assert_array_equal(module.admits_affine_perturbation(x,u,f,radius),reference)
    # Identities and the positive affine stretch are certified; reflection,
    # nonaffine distortion and nonfinite F still require the reference path.
    assert sum(expected_fallback)==3


def test_native_buffer_rejects_wrong_types():
    import agentfem.elements._hex_validity as module
    if module._native_affine_screen is None:
        pytest.skip('Optional native extension not built')
    with pytest.raises(ValueError,match='float64/int32'):
        module._native_affine_screen(*(np.zeros(24,dtype=np.int64) for _ in range(5)))
