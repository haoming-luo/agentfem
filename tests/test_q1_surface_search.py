import numpy as np
import pytest

from agentfem._q1_surface_search import AmbiguousSurfaceProjection, Q1SurfaceSearch


def square():
    return np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])


def test_cross_facet_and_seam_search():
    x = np.array([square(), square() + [1, 0, 0]])
    search = Q1SurfaceSearch(x)
    result = search.project(
        [[0.2, 0.3, 0.01], [1.2, 0.3, 0.01], [1, 0.3, 0.01]], maximum_distance=0.1
    )
    np.testing.assert_array_equal(result["facet_indices"], [0, 1, 0])
    np.testing.assert_allclose(
        result["points"], [[0.2, 0.3, 0], [1.2, 0.3, 0], [1, 0.3, 0]], atol=1e-12
    )
    np.testing.assert_allclose(result["distances"], 0.01)
    for i, facet in enumerate(result["facet_indices"]):
        np.testing.assert_allclose(
            result["weights"][i] @ x[facet], result["points"][i], atol=1e-12
        )


def test_ambiguous_missing_and_budget_rejected():
    search = Q1SurfaceSearch(np.array([square(), square() + [0, 0, 0.1]]))
    with pytest.raises(AmbiguousSurfaceProjection, match="ambiguous") as captured:
        search.project([[0.5, 0.5, 0.05]], maximum_distance=0.2)
    error = captured.value
    assert error.query_index == 0
    assert error.facets == (0, 1)
    np.testing.assert_allclose(error.distances, (.05, .05))
    assert error.point_separation == pytest.approx(.1)
    assert 0 < error.distance_tolerance < 1e-12
    with pytest.raises(ValueError, match="no surface"):
        search.project([[3, 3, 3]], maximum_distance=0.1)
    with pytest.raises(ValueError, match="budget"):
        search.project([[0.5, 0.5, 0.05]], maximum_distance=0.2, maximum_candidates=1)


def test_warped_surface_search_and_empty():
    x = square()
    x[2, 2] = 0.4
    point = np.array([0.3, 0.6, 0.4 * 0.3 * 0.6])
    search = Q1SurfaceSearch(x[None])
    result = search.project([point], maximum_distance=1e-5)
    np.testing.assert_allclose(result["points"], [point], atol=1e-10)
    assert search.project(np.empty((0, 3)), maximum_distance=0.1)["weights"].shape == (
        0,
        4,
    )


@pytest.mark.parametrize("length", [1e-3, 1., 1e3])
@pytest.mark.parametrize("reverse", [False, True])
def test_convex_interior_beats_shared_edge_despite_distance_roundoff(length, reverse):
    vertices = np.array([square(), square() + [1, 0, 0]]) * length
    nodes = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    if reverse:
        vertices, nodes = vertices[::-1], nodes[::-1]
    query = np.array([[1 - 2e-10, .4, 1e-5]]) * length
    answer = Q1SurfaceSearch(vertices, node_ids=nodes).project(
        query, maximum_distance=.01*length, require_unique_normal=False)
    assert answer['facet_indices'][0] == int(reverse)
    np.testing.assert_allclose(answer['points'], query * [1, 1, 0], atol=1e-14*length)
    # Coincident geometry without shared physical DOFs cannot use this screen.
    with pytest.raises(AmbiguousSurfaceProjection):
        Q1SurfaceSearch(vertices, node_ids=np.arange(8).reshape(2,4)).project(
            query, maximum_distance=.01*length, require_unique_normal=False)


@pytest.mark.parametrize("length", [1e-3, 1.0, 1e3])
def test_resolvable_near_seam_is_not_a_false_ambiguity(length):
    surface = np.array([square(), square() + [1, 0, 0]]) * length
    query = np.array([[1 - 5e-9, 0.4, 0.0004]]) * length
    result = Q1SurfaceSearch(surface).project(query, maximum_distance=0.01 * length)
    np.testing.assert_array_equal(result["facet_indices"], [0])
    np.testing.assert_allclose(result["points"], query * [1, 1, 0], atol=1e-13 * length)


def test_nonaffine_planar_stationarity_is_polished():
    from agentfem._q1_surface_geometry import q1_closest_point

    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.1, 1.0, 0], [0.0, 0.95, 0]])
    angle = 0.782
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    vertices = vertices @ rotation.T + [5, -5, 2]
    for u, v in ((0.04, 0.08), (0.25, 0.7), (0.9, 0.2)):
        basis = np.array([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v])
        point = basis @ vertices
        result = q1_closest_point(vertices, point + [0, 0, 0.003])
        np.testing.assert_allclose(result[0], [u, v], atol=1e-11)
        np.testing.assert_allclose(result[2], point, atol=1e-12)


def test_near_degree_reduction_keeps_finite_stationary_root():
    from agentfem._q1_surface_geometry import q1_closest_point

    for angle in np.linspace(0.1, 1.4, 17):
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        vertices = np.array(
            [[-1.0, 0, 0], [1.0, 0, 0], [1.1, 1.0, 0], [-1.1, 1.0, 0]]
        ) @ rotation.T + [9, -9, 2]
        basis = np.array([0.15, 0.15, 0.35, 0.35])
        point = basis @ vertices
        answer = q1_closest_point(vertices, point)
        np.testing.assert_allclose(answer[0], [0.5, 0.7], atol=1e-10)
        assert answer[-1] < 1e-12
