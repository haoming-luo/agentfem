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


@pytest.mark.parametrize('exponent', [-100, 0, 100])
def test_narrow_curved_gap_uses_proof_depth_not_time_subdivision(exponent):
    x = square(); x[2, 2] = .25
    p = np.array([[.3125, .4375, .25*.3125*.4375 + 2.**-28]])
    x *= 2.**exponent; p *= 2.**exponent
    assert not separated_q1_paths(x, x, [[0,1,2,3]], p, p, max_depth=18)[0]
    assert separated_q1_paths(x, x, [[0,1,2,3]], p, p)[0]
    # The true on-surface state remains inadmissible at the larger budget.
    p[0, 2] = .25*.3125*.4375*2.**exponent
    assert not separated_q1_paths(x, x, [[0,1,2,3]], p, p)[0]


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


def test_float_filter_preserves_exact_dyadic_impacts_across_scales():
    from agentfem._q1_swept_separation import _filtered_separating_plane

    rng = np.random.default_rng(871)
    for exponent in (-300, -100, 0, 100, 300):
        for _ in range(100):
            old = (square()+rng.integers(-2, 3, (4, 3))/32)*2.**exponent
            new = old+rng.integers(-2, 3, (4, 3))*2.**(exponent-6)
            u, v, t = rng.integers(1, 4, 3)/4
            weights = np.array([(1-u)*(1-v), u*(1-v), u*v, (1-u)*v])
            impact = weights@((1-t)*old+t*new)
            velocity = rng.integers(-2, 3, 3)*2.**(exponent-3)
            assert not _filtered_separating_plane(
                old, new, impact-t*velocity, impact+(1-t)*velocity)


def test_filtered_positive_sign_agrees_with_exact_control_proof():
    from agentfem._q1_swept_separation import _filtered_separating_plane, _separating_plane, Q

    rng = np.random.default_rng(746)
    passed = 0
    for _ in range(200):
        old = square()+rng.normal(size=(4, 3))*.1
        new = old+rng.normal(size=(4, 3))*.01
        point = rng.normal(size=3)
        end = point+rng.normal(size=3)*.01
        if not _filtered_separating_plane(old, new, point, end):
            continue
        control = np.empty((2, 2, 2, 3), dtype=object)
        for t, (vertices, p) in enumerate(((old, point), (new, end))):
            for k, (u, v) in enumerate(((0,0),(1,0),(1,1),(0,1))):
                control[t,u,v]=[Q(float(x))-Q(float(y)) for x,y in zip(vertices[k],p)]
        # This independent exact test uses the same family of proposal axes;
        # generic random cases are well separated from uncertain signs.
        assert _separating_plane(control)
        passed += 1
    assert passed > 100
