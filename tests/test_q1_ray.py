# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._q1_ray import AmbiguousRay, q1_ray_crossings


def test_warped_intersection_reconstructs_original_q1():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0.4], [0, 1, 0]])
    u, v = 0.37, 0.64
    target = np.array([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]) @ x
    r = np.array([0.13, -0.27, 1.0])
    r /= np.linalg.norm(r)
    hit = q1_ray_crossings(x[None], target - 0.3 * r, r)
    np.testing.assert_allclose(hit, [[0, u, v, 0.3, 1]], atol=1e-12)


def test_two_intersections_of_one_bilinear_patch():
    # z = u*v; line u=v, z=u-.16 intersects at .2 and .8.
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]])
    r = np.ones(3) / np.sqrt(3)
    hits = q1_ray_crossings(x[None], [0, 0, -0.16], r)
    hits = hits[np.argsort(hits[:, 1])]
    np.testing.assert_allclose(hits[:, 1:3], [[0.2, 0.2], [0.8, 0.8]], atol=1e-12)
    np.testing.assert_array_equal(hits[:, 4], [1, -1])


@pytest.mark.parametrize("origin", [[0.5, 0.5, 1], [2, 2, -1]])
def test_no_forward_hit(origin):
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    assert q1_ray_crossings(x[None], origin, [0, 0, 1]).shape == (0, 5)


def test_boundary_tangent_and_coplanar_fail_closed():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    with pytest.raises(AmbiguousRay):
        q1_ray_crossings(x[None], [0, 0.5, -1], [0, 0, 1])
    with pytest.raises(AmbiguousRay):
        q1_ray_crossings(x[None], [-1, 0.5, 0], [1, 0, 0])
    x[2, 2] = 1.0
    with pytest.raises(AmbiguousRay):
        q1_ray_crossings(x[None], [0, 0, -0.25], np.ones(3) / np.sqrt(3))


def test_random_generated_hits_are_not_lost():
    rng = np.random.default_rng(149)
    for _ in range(150):
        x = np.array(
            [[0.0, 0, 0], [1, 0, 0], [1, 1, rng.uniform(-0.5, 0.5)], [0, 1, 0]]
        )
        u, v = rng.uniform(0.1, 0.9, 2)
        target = np.array([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]) @ x
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        answer = q1_ray_crossings(x[None], target - 0.1 * direction, direction)
        assert np.any(np.linalg.norm(answer[:, 1:3] - [u, v], axis=1) < 1e-10)
