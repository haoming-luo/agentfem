"""Solver-neutral checks of diagnostic P1 reference-face subdivision."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem.boundary_models.contact_reference_refinement import (
    refine_p1_triangle_reference_trace,
)
from agentfem.boundary_models.contact_trace import ContactTrace


def _triangle_trace():
    return ContactTrace(
        point_ids=np.arange(3),
        node_ids=np.tile(np.arange(3), (3, 1)),
        shape_values=np.asarray((
            (2 / 3, 1 / 6, 1 / 6),
            (1 / 6, 2 / 3, 1 / 6),
            (1 / 6, 1 / 6, 2 / 3),
        )),
        weights=np.full(3, 1 / 6),
        source="test_triangle_degree_two_three_point",
    )


@pytest.mark.parametrize("level", range(4))
def test_reference_subdivision_preserves_area_and_quadratic_integral(level):
    original = _triangle_trace()
    refined = refine_p1_triangle_reference_trace(original, level)
    assert refined.point_count == 3 * 4**level
    np.testing.assert_allclose(refined.weights.sum(), 0.5, atol=1e-15)
    points = refined.evaluate(np.asarray(((0, 0), (1, 0), (0, 1)))).query_points
    np.testing.assert_allclose(np.dot(refined.weights, points[:, 0]**2), 1 / 12)
    assert np.all(refined.weights > 0)
    assert np.unique(refined.point_ids).size == refined.point_count


def test_reference_subdivision_rejects_budget_and_wrong_trace():
    original = _triangle_trace()
    with pytest.raises(ValueError, match="point budget"):
        refine_p1_triangle_reference_trace(original, 4, maximum_points=100)
    with pytest.raises(ValueError, match="from 0 to 5"):
        refine_p1_triangle_reference_trace(original, True)
    other = ContactTrace(
        point_ids=original.point_ids,
        node_ids=original.node_ids,
        shape_values=np.tile(original.shape_values[:1], (3, 1)),
        weights=original.weights,
        source="unreviewed_three_point_rule",
    )
    with pytest.raises(ValueError, match="grouping is inconsistent"):
        refine_p1_triangle_reference_trace(other, 1)
