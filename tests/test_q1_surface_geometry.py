import numpy as np
import pytest

from agentfem._q1_surface_geometry import planar_q1_pull_back
from agentfem._q1_surface_geometry import q1_surface_pull_back
from agentfem._q1_surface_geometry import q1_surface_quadrature


def test_true_warped_surface_quadrature_and_orientation():
    # x=(u,v,a*u*v); exact Jacobian sqrt(1+a^2*(u^2+v^2)).
    a = 0.6
    x = np.array([[[0.0, 0, 0], [1, 0, 0], [1, 1, a], [0, 1, 0]]])
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(64)
    u, v = np.meshgrid((reference_nodes + 1) / 2, (reference_nodes + 1) / 2)
    exact = np.sum(
        np.outer(reference_weights, reference_weights)
        / 4
        * np.sqrt(1 + a * a * (u * u + v * v))
    )
    errors = []
    for order in (2, 4, 8):
        points, basis, weights, normals = q1_surface_quadrature(x, order=order)
        errors.append(abs(weights.sum() - exact))
        np.testing.assert_allclose(points[0], basis @ x[0], atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-14)
        np.testing.assert_allclose(
            (weights[..., None] * normals).sum(axis=1),
            [[-a / 2, -a / 2, 1]],
            atol=1e-14,
        )
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-12
    _, _, reverse_weights, reverse_normals = q1_surface_quadrature(
        x[:, [0, 3, 2, 1]], order=8
    )
    np.testing.assert_allclose(reverse_weights.sum(), exact, atol=1e-12)
    np.testing.assert_allclose(
        (reverse_weights[..., None] * reverse_normals).sum(axis=1),
        [[a / 2, a / 2, -1]],
        atol=1e-14,
    )


def test_surface_quadrature_planar_area_and_empty_batch():
    x = np.array([[[0.0, 0, 0], [2, 0, 0], [1.5, 1, 0], [0, 1, 0]]])
    _, basis, weights, normals = q1_surface_quadrature(x)
    assert weights.sum() == pytest.approx(1.75)
    np.testing.assert_allclose(normals, np.broadcast_to([0, 0, 1], normals.shape))
    np.testing.assert_allclose(basis.sum(axis=1), 1)
    assert q1_surface_quadrature(np.empty((0, 4, 3)))[2].shape == (0, 9)


@pytest.mark.parametrize("warpage", [1e-7, 0.1, 0.6])
def test_warped_surface_inverse_preserves_geometry(warpage):
    rng = np.random.default_rng(12)
    local = rng.random((100, 2))
    quad = np.array([[0.0, 0, 0], [2, 0, 0], [1.4, 1, warpage], [0.1, 1, 0]])
    points = weights(local) @ quad
    vertices = np.broadcast_to(quad, (len(local), 4, 3))
    recovered, basis = q1_surface_pull_back(vertices, points)
    np.testing.assert_allclose(recovered, local, atol=2e-10)
    np.testing.assert_allclose(basis @ quad, points, atol=2e-11)
    with pytest.raises(ValueError, match="Warped"):
        planar_q1_pull_back(vertices, points)
    with pytest.raises(ValueError, match="converge"):
        q1_surface_pull_back(vertices, points + [0, 0, 0.01])


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
