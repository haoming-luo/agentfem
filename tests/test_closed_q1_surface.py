# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._closed_q1_surface import ClosedQ1Surface


def solid(concave=False):
    vertices = []
    lookup = {}
    faces = {}
    local = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    indices = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    for offset in [[0, 0, 0], [1, 0, 0], [0, 1, 0]] if concave else [[0, 0, 0]]:
        nodes = []
        for point in local + offset:
            key = tuple(point)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(point)
            nodes.append(lookup[key])
        for face in indices:
            row = np.array(nodes)[face]
            key = tuple(sorted(row))
            if key in faces:
                del faces[key]
            else:
                faces[key] = row
    return np.array(vertices, dtype=float), np.array(list(faces.values()))


def test_closed_cube_face_edge_corner_sign_and_gradient():
    x, cells = solid()
    search = ClosedQ1Surface(x, cells)
    points = np.array(
        [[0.2, 0.3, 0.1], [-0.1, 0.3, 0.4], [-0.1, -0.2, 0.4], [-0.1, -0.2, -0.3]]
    )
    found = search.project(points, maximum_distance=0.5)
    np.testing.assert_allclose(
        found["signed_gaps"], [-0.1, 0.1, np.sqrt(0.05), np.sqrt(0.14)], atol=1e-12
    )
    np.testing.assert_allclose(
        points - found["points"],
        found["signed_gaps"][:, None] * found["normals"],
        atol=1e-12,
    )


def test_concave_interior_edge_has_central_force_direction():
    x, cells = solid(concave=True)
    answer = ClosedQ1Surface(x, cells).project([[0.9, 0.9, 0.4]], maximum_distance=0.3)
    np.testing.assert_allclose(answer["signed_gaps"], [-np.sqrt(0.02)], atol=1e-12)
    np.testing.assert_allclose(answer["points"], [[1, 1, 0.4]], atol=1e-12)
    np.testing.assert_allclose(
        answer["normals"], [[1 / np.sqrt(2), 1 / np.sqrt(2), 0]], atol=1e-12
    )


def test_warped_closed_surface_sign_retains_q1_not_triangle():
    x, cells = solid()
    x[6, 2] = 1.4
    search = ClosedQ1Surface(x, cells)
    # At (.3,.6), exact top is 1.072, not either triangular surrogate.
    points = [[0.3, 0.6, 1.065], [0.3, 0.6, 1.08]]
    answer = search.project(points, maximum_distance=0.1)
    assert answer["signed_gaps"][0] < 0 < answer["signed_gaps"][1]


def test_rigid_rotation_and_scale():
    x, cells = solid(concave=True)
    rotation, _ = np.linalg.qr(np.random.default_rng(44).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    for scale in (1e-3, 1.0, 1e3):
        y = scale * x @ rotation.T + [3, 4, 5]
        query = scale * np.array([[0.9, 0.9, 0.4]]) @ rotation.T + [3, 4, 5]
        answer = ClosedQ1Surface(y, cells).project(query, maximum_distance=0.3 * scale)
        np.testing.assert_allclose(
            answer["signed_gaps"], [-np.sqrt(0.02) * scale], rtol=1e-10
        )


def test_open_and_reversed_surfaces_rejected():
    x, cells = solid()
    with pytest.raises(ValueError, match="two oppositely"):
        ClosedQ1Surface(x, cells[:-1])
    with pytest.raises(ValueError, match="outward"):
        ClosedQ1Surface(x, cells[:, ::-1]).project(
            [[0.2, 0.3, 0.1]], maximum_distance=0.5
        )


def test_zero_distance_does_not_need_an_arbitrary_sign():
    x, cells = solid()
    result = ClosedQ1Surface(x, cells).project(
        [[0, 0, 0.4], [0, 0, 0]], maximum_distance=0.1
    )
    np.testing.assert_array_equal(result["signed_gaps"], 0)
