# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._q1_swept_separation import separated_q1_paths


def square():
    return np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])


def test_tiny_parallel_gap_with_common_rigid_translation():
    x = square()
    x[:, 2] = .3*x[:, 0] + .2*x[:, 1]
    point = np.array([[.3, .4, .17 + 1e-12]])
    move = np.array([2., -3., 1.])
    assert separated_q1_paths(x, x+move, [[0,1,2,3]], point, point+move)[0]


def test_crossing_and_touching_are_not_admitted():
    x = square()
    result = separated_q1_paths(x, x, [[0,1,2,3]],
        [[.3,.4,-.1], [.3,.4,0.], [-.1,.4,0.]],
        [[.3,.4,.1], [.3,.4,0.], [1.1,.4,0.]])
    np.testing.assert_array_equal(result, False)


def test_warped_patch_needs_subdivision_not_triangle_flattening():
    x = square(); x[2,2] = .4
    point = [[.5,.5,.15]]
    assert not separated_q1_paths(x,x,[[0,1,2,3]],point,point,max_depth=0)[0]
    assert separated_q1_paths(x,x,[[0,1,2,3]],point,point)[0]
    assert not separated_q1_paths(x,x,[[0,1,2,3]],[[.5,.5,.1]],[[.5,.5,.1]])[0]


def test_moving_patch_crossing_stationary_point_rejected():
    x = square()
    assert not separated_q1_paths(x-[0,0,.1],x+[0,0,.1],[[0,1,2,3]],
                                  [[.3,.4,0.]],[[.3,.4,0.]])[0]


@pytest.mark.parametrize('scale', [1e-3, 1., 1e3])
def test_rotation_scale_and_large_translation(scale):
    rotation, _ = np.linalg.qr(np.random.default_rng(14).normal(size=(3,3)))
    x = square();x[2,2]=.4
    x = scale*x@rotation.T + [100., -100., 100.]
    point = scale*np.array([[.5,.5,.15]])@rotation.T + [100., -100., 100.]
    assert separated_q1_paths(x,x,[[0,1,2,3]],point,point)[0]


def test_invalid_and_empty_queries():
    x=square()
    assert separated_q1_paths(x,x,[[0,1,2,3]],np.empty((0,3)),np.empty((0,3))).shape==(0,)
    with pytest.raises(ValueError):
        separated_q1_paths(x,x,[[0,1,2,4]],[[0,0,0]],[[0,0,0]])
    with pytest.raises(ValueError):
        separated_q1_paths(x,x,[[0,1,2,3]],[[0,0,np.nan]],[[0,0,0]])


def test_manufactured_dyadic_impacts_are_never_certified_separated():
    rng = np.random.default_rng(52)
    for _ in range(40):
        old = square() + rng.integers(-2, 3, (4,3))/32
        new = old + rng.integers(-2, 3, (4,3))/64
        u, v, t = rng.integers(1,4,3)/4
        weights = np.array([(1-u)*(1-v), u*(1-v), u*v, (1-u)*v])
        impact = weights@((1-t)*old+t*new)
        velocity = rng.integers(-2, 3, 3)/8
        p0, p1 = impact-t*velocity, impact+(1-t)*velocity
        assert not separated_q1_paths(old,new,[[0,1,2,3]],[p0],[p1])[0]
