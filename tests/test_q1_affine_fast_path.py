# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Affine acceleration checked against the independent polynomial route."""

import numpy as np
import pytest

from agentfem import _q1_surface_geometry as geometry


def test_random_affine_queries_match_polynomial_oracle():
    rng = np.random.default_rng(314)
    for _ in range(80):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        a, b = rotation[:, 0], 0.7 * rotation[:, 0] + 0.4 * rotation[:, 1]
        vertices = np.array([np.zeros(3), a, a + b, b])
        point = rng.uniform(-0.8, 1.8, size=3)
        fast = geometry.q1_closest_point(vertices, point)
        reference = geometry._q1_closest_point_polynomial(vertices, point)
        for first, second in zip(fast, reference, strict=True):
            np.testing.assert_allclose(first, second, rtol=1e-10, atol=2e-12)


@pytest.mark.parametrize("length", [1e-6, 1.0, 1e6])
def test_affine_interior_edge_corner_and_scale(length):
    vertices = length * np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    for query in ([0.3, 0.4, 0.2], [-0.2, 0.3, 0.1], [-0.3, -0.4, -0.2]):
        q = length * np.array(query)
        actual = geometry.q1_closest_point(vertices, q)
        expected = length * np.array(
            [np.clip(query[0], 0, 1), np.clip(query[1], 0, 1), 0]
        )
        np.testing.assert_allclose(actual[2], expected, atol=length * 1e-14)
        assert actual[-1] == pytest.approx(np.linalg.norm(q - expected))


def test_true_warpage_does_not_take_planar_path(monkeypatch):
    original = geometry._q1_closest_point_polynomial
    called = []

    def record(x, q):
        called.append(True)
        return original(x, q)

    monkeypatch.setattr(geometry, "_q1_closest_point_polynomial", record)
    for corner in ([1.0, 1.0, 1e-10], [1.01, 1.0, 1e-10]):
        x = np.array([[0.0, 0, 0], [1, 0, 0], corner, [0, 1, 0]])
        geometry.q1_closest_point(x, [0.3, 0.4, 0.1])
    assert len(called) == 2


def test_nonaffine_planar_fast_path_matches_independent_polynomial():
    rng = np.random.default_rng(632)
    for _ in range(60):
        x = np.array(
            [[0.0, 0, 0], [1, 0, 0], [rng.uniform(0.8, 1.3), 1, 0], [0, 0.9, 0]]
        )
        p = rng.uniform(-0.5, 1.5, 3)
        fast = geometry.q1_closest_point(x, p)
        reference = geometry._q1_closest_point_polynomial(x, p)
        for first, second in zip(fast, reference, strict=True):
            np.testing.assert_allclose(first, second, rtol=1e-10, atol=2e-12)


def test_affine_degeneracy_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        geometry.q1_closest_point(np.zeros((4, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="Degenerate"):
        geometry.q1_closest_point(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0]], [0, 0, 0]
        )


def test_nonaffine_edge_inverse_is_accurate_enough_for_search_ties():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1.37, 1, 0], [0, 0.93, 0]]) + [11, -9, 2]
    for v in np.linspace(0.07, 0.94, 30):
        p = (1 - v) * x[1] + v * x[2]
        uv, basis, point, _, distance = geometry.q1_closest_point(x, p)
        assert np.all((uv >= 0) & (uv <= 1))
        assert distance < 2e-14
        np.testing.assert_allclose(point, p, atol=2e-14, rtol=0)
