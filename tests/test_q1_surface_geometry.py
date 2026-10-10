import numpy as np
import pytest

from agentfem._q1_surface_geometry import planar_q1_pull_back


def weights(uv):
    u, v = uv.T
    return np.array([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]).T


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
def test_nonaffine_round_trip_and_linear_field(scale):
    rng = np.random.default_rng(83)
    local = np.concatenate((rng.random((128, 2)), [[0, 0], [1, 0], [1, 1], [0, 1]]))
    quad = np.array([[0, 0, 0], [2, 0, 0], [1.4, 1.3, 0], [0.2, 1, 0]])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    quad = scale * (quad @ rotation.T + [3, -4, 2])
    vertices = np.broadcast_to(quad, (len(local), 4, 3))
    physical = weights(local) @ quad
    recovered, basis = planar_q1_pull_back(vertices, physical)
    np.testing.assert_allclose(recovered, local, atol=2e-10)
    np.testing.assert_allclose(basis, weights(local), atol=2e-10)
    np.testing.assert_allclose(basis.sum(axis=1), 1, atol=1e-14)
    field = quad @ np.array([0.3, -0.2, 1.4])
    np.testing.assert_allclose(
        basis @ field, physical @ [0.3, -0.2, 1.4], atol=scale * 2e-10
    )


@pytest.mark.parametrize(
    "kind",
    [
        "warped",
        "concave",
        "crossed",
        "degenerate",
        "off_surface",
        "outside",
        "nan",
        "complex",
    ],
)
def test_invalid_geometry_or_query_rejected(kind):
    x = np.array([[[0.0, 0, 0], [2, 0, 0], [1.4, 1, 0], [0, 1, 0]]])
    point = x.mean(axis=1)
    if kind == "warped":
        x[0, 2, 2] = 0.01
    elif kind == "concave":
        x[0, 2] = [0.1, 0.1, 0]
    elif kind == "crossed":
        x = x[:, [0, 2, 1, 3]]
    elif kind == "degenerate":
        x[:] = 0
    elif kind == "off_surface":
        point[0, 2] = 0.1
    elif kind == "outside":
        point[0, 0] = 3
    elif kind == "nan":
        point[0, 0] = np.nan
    elif kind == "complex":
        x = x.astype(complex) + 1j
    with pytest.raises(ValueError):
        planar_q1_pull_back(x, point)


def test_empty_and_iteration_exhaustion():
    uv, basis = planar_q1_pull_back(np.empty((0, 4, 3)), np.empty((0, 3)))
    assert uv.shape == (0, 2) and basis.shape == (0, 4)
    x = np.array([[[0.0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 1, 0]]])
    with pytest.raises(ValueError, match="converge"):
        planar_q1_pull_back(x, np.array([[0.1, 0.2, 0]]), max_iterations=1)
