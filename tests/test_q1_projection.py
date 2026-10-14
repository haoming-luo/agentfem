import numpy as np
import pytest
from scipy.optimize import minimize

from agentfem._q1_surface_geometry import q1_interior_normal_projection
from agentfem._q1_surface_geometry import q1_closest_point


@pytest.mark.parametrize(
    "query,expected",
    [
        ([0.3, 0.6, 0.4], [0.3, 0.6, 0]),
        ([1.4, 0.6, 0.4], [1, 0.6, 0]),
        ([-1, 2, 0.4], [0, 1, 0]),
    ],
)
def test_closest_point_interior_edges_corners(query, expected):
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    uv, basis, point, normal, distance = q1_closest_point(x, query)
    np.testing.assert_allclose(point, expected, atol=1e-12)
    np.testing.assert_allclose(basis @ x, point, atol=1e-12)
    assert distance == pytest.approx(np.linalg.norm(np.array(query) - expected))
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-12)


def test_closest_point_warped_against_global_numerical_oracle():
    from scipy.optimize import differential_evolution

    x = np.array([[0.0, 0, 0], [2, 0, 0], [1.4, 1, 0.7], [0.1, 1, 0]])
    rng = np.random.default_rng(813)
    for query in rng.uniform([-1, -1, -2], [3, 2, 2], (12, 3)):

        def distance(uv, query=query):
            u, v = uv
            p = np.array([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]) @ x
            return np.sum((p - query) ** 2)

        result = q1_closest_point(x, query)
        reference = differential_evolution(
            distance, [(0, 1), (0, 1)], seed=41, tol=1e-10
        )
        assert result[-1] ** 2 == pytest.approx(reference.fun, abs=2e-9)


def test_near_edge_interior_minimum_is_not_ambiguous():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    result = q1_closest_point(x, [0.6, 1 - 2e-8, 0.001])
    np.testing.assert_allclose(result[0], [0.6, 1 - 2e-8], atol=1e-12)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_warped_normal_projection_objectivity_and_original_basis(scale):
    rng = np.random.default_rng(37)
    uv = rng.uniform(0.15, 0.85, (60, 2))
    u, v = uv.T
    # Independent graph x=(u,v,0.4uv), with analytical normals.
    surface = np.column_stack((u, v, 0.4 * u * v))
    normal = np.column_stack((-0.4 * v, -0.4 * u, np.ones(len(u))))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    gap = rng.uniform(-0.03, 0.03, len(u))
    quad = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0.4], [0, 1, 0]])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.linalg.det(rotation) > 0
    x = scale * (quad @ rotation.T + [3, -2, 4])
    queries = scale * ((surface + gap[:, None] * normal) @ rotation.T + [3, -2, 4])
    local, basis, projected, normals, gaps = q1_interior_normal_projection(
        np.broadcast_to(x, (len(u), 4, 3)), queries
    )
    np.testing.assert_allclose(local, uv, atol=1e-10)
    np.testing.assert_allclose(projected, basis @ x, atol=scale * 1e-12)
    np.testing.assert_allclose(normals, normal @ rotation.T, atol=1e-10)
    np.testing.assert_allclose(gaps, gap * scale, atol=scale * 1e-11)


def test_projection_against_independent_constrained_optimizer():
    quad = np.array([[0.0, 0, 0], [2, 0, 0], [1.4, 1, 0.3], [0.1, 1, 0]])

    def position(uv):
        u, v = uv
        return np.array([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]) @ quad

    query = position([0.31, 0.62]) + [0.01, -0.01, 0.03]
    local, _, projected, _, _ = q1_interior_normal_projection(quad[None], query[None])
    reference = minimize(
        lambda uv: np.sum((position(uv) - query) ** 2),
        [0.5, 0.5],
        method="Nelder-Mead",
        bounds=[(0, 1), (0, 1)],
        options={"xatol": 1e-13, "fatol": 1e-16},
    )
    assert reference.success
    np.testing.assert_allclose(local[0], reference.x, atol=1e-8)
    np.testing.assert_allclose(projected[0], position(reference.x), atol=2e-8)


@pytest.mark.parametrize("query", [[1.2, 0.5, 0.1], [0, 0.5, 0.1], [np.nan, 0.5, 0.1]])
def test_outside_edge_or_nonfinite_rejected(query):
    x = np.array([[[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    with pytest.raises(ValueError):
        q1_interior_normal_projection(x, np.array([query]))


def test_empty_and_exhaustion():
    result = q1_interior_normal_projection(np.empty((0, 4, 3)), np.empty((0, 3)))
    assert result[-1].shape == (0,)
    x = np.array([[[0.0, 0, 0], [1, 0, 0], [1, 1, 0.6], [0, 1, 0]]])
    with pytest.raises(ValueError, match="converge"):
        q1_interior_normal_projection(x, [[0.2, 0.7, 0.2]], max_iterations=1)


def test_nonpositive_hessian_is_not_a_nearest_point_claim():
    x = np.array([[[0.0, 0, 0], [1, 0, 0], [1, 1, 0.6], [0, 1, 0]]])
    with pytest.raises(ValueError, match="Hessian"):
        q1_interior_normal_projection(x, [[0.5, 0.5, 10.0]])


def test_batch_scalar_and_orientation():
    x = np.array([[[0.0, 0, 0], [2, 0, 0], [1.5, 1, 0.1], [0, 1, 0]]])
    queries = np.array([[0.2, 0.3, 0.04], [0.8, 0.6, -0.02]])
    batch = q1_interior_normal_projection(np.repeat(x, 2, axis=0), queries)
    for i in range(2):
        scalar = q1_interior_normal_projection(x, queries[i : i + 1])
        for a, b in zip(batch, scalar):
            np.testing.assert_allclose(a[i : i + 1], b, atol=1e-11)
    reverse = q1_interior_normal_projection(
        np.repeat(x[:, [0, 3, 2, 1]], 2, axis=0), queries
    )
    np.testing.assert_allclose(batch[2], reverse[2], atol=1e-11)
    np.testing.assert_allclose(batch[3], -reverse[3], atol=1e-11)
    np.testing.assert_allclose(batch[4], -reverse[4], atol=1e-11)
