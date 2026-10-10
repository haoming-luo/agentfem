import numpy as np
import pytest

from agentfem._q1_surface_search import Q1SurfaceSearch


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
    with pytest.raises(ValueError, match="ambiguous"):
        search.project([[0.5, 0.5, 0.05]], maximum_distance=0.2)
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
