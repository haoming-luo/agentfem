# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._q1_ray import AmbiguousRay, q1_ray_crossings
from agentfem._q1_ray_exact import exact_q1_ray_crossings


def plane():
    return np.array([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]])


@pytest.mark.parametrize("coordinate", [1e-12, 1e-16, 1e-30])
def test_exact_kernel_distinguishes_near_edge_and_near_origin(coordinate):
    origin = [coordinate, .3, -coordinate]
    with pytest.raises(AmbiguousRay):
        q1_ray_crossings(plane(), origin, [0., 0., 1.])
    hits = exact_q1_ray_crossings(plane(), origin, [0., 0., 1.])
    np.testing.assert_allclose(hits, [[0., coordinate, .3, coordinate, 1.]], rtol=1e-14, atol=0)


@pytest.mark.parametrize("origin", [[0., .3, -1.], [.3, .3, 0.], [-1., .3, 0.]])
def test_true_seam_origin_or_coplanar_still_rejected(origin):
    direction = [1., 0., 0.] if origin[0] == -1 else [0., 0., 1.]
    with pytest.raises(AmbiguousRay):
        exact_q1_ray_crossings(plane(), origin, direction)


def test_warped_two_roots_and_opposite_crossings():
    x = plane()
    x[0, 2, 2] = 1.
    direction = np.ones(3)/np.sqrt(3)
    hits = exact_q1_ray_crossings(x, [0., 0., -.16], direction)
    hits = hits[np.argsort(hits[:, 1])]
    np.testing.assert_allclose(hits[:, 1:3], [[.2, .2], [.8, .8]], atol=1e-14)
    np.testing.assert_array_equal(hits[:, 4], [1., -1.])


def test_exact_matches_regular_random_warped_intersections():
    rng = np.random.default_rng(713)
    for _ in range(30):
        x = plane()
        x[0, 2, 2] = rng.uniform(-.5, .5)
        u, v = rng.uniform(.1, .9, 2)
        target = np.array([(1-u)*(1-v), u*(1-v), u*v, (1-u)*v]) @ x[0]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = target-.1*direction
        reference = q1_ray_crossings(x, origin, direction)
        actual = exact_q1_ray_crossings(x, origin, direction)
        reference = reference[np.argsort(reference[:, 1])]
        actual = actual[np.argsort(actual[:, 1])]
        np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_forward_misses_and_repeated_root():
    assert exact_q1_ray_crossings(plane(), [.5, .5, 1.], [0., 0., 1.]).shape == (0, 5)
    x = plane()
    x[0, 2, 2] = 1.
    with pytest.raises(AmbiguousRay):
        exact_q1_ray_crossings(x, [0., 0., -.25], np.ones(3)/np.sqrt(3))
