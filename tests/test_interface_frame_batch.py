# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from agentfem.interfaces import _interface_frames


def test_vectorized_frames_preserve_scalar_orientation_and_ties():
    rng = np.random.default_rng(120)
    normals = np.concatenate(
        (
            rng.normal(size=(1000, 3)),
            np.eye(3),
            -np.eye(3),
            [[1, 1, 1], [-1, -1, -1], [1, 1, 0]],
        )
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    expected = np.empty((len(normals), 3, 3))
    for index, direction in enumerate(normals):
        seed = np.eye(3)[np.argmin(np.abs(direction))]
        first = np.cross(direction, seed)
        first /= np.linalg.norm(first)
        expected[index] = np.column_stack(
            (direction, first, np.cross(direction, first))
        )
    actual = _interface_frames(normals)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(actual), 1, atol=1e-15)
    np.testing.assert_allclose(
        np.einsum("qji,qjk->qik", actual, actual),
        np.broadcast_to(np.eye(3), actual.shape),
        atol=1e-15,
    )


def test_two_dimensional_and_empty_frames_preserved():
    normal = np.array([[1.0, 0.0], [0.0, -1.0]])
    frame = _interface_frames(normal)
    np.testing.assert_array_equal(frame[:, :, 0], normal)
    np.testing.assert_allclose(np.linalg.det(frame), 1)
    assert _interface_frames(np.empty((0, 3))).shape == (0, 3, 3)
