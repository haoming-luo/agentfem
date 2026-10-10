# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem.boundary_models.search import TriangleSurfaceBVH
from test_rigid_surface_search import _grid_surface


def exhaustive_boxes(surface, start, end, padding):
    vertices = surface.vertices[surface.triangles]
    lower, upper = vertices.min(axis=1) - padding, vertices.max(axis=1) + padding
    delta = end - start
    lo, hi = np.zeros(len(lower)), np.ones(len(lower))
    valid = np.ones(len(lower), dtype=bool)
    for j in range(3):
        if delta[j] == 0:
            valid &= (lower[:, j] <= start[j]) & (start[j] <= upper[:, j])
        else:
            a, b = (
                (lower[:, j] - start[j]) / delta[j],
                (upper[:, j] - start[j]) / delta[j],
            )
            lo = np.maximum(lo, np.minimum(a, b))
            hi = np.minimum(hi, np.maximum(a, b))
    return np.flatnonzero(valid & (lo <= hi))


def test_segment_bvh_matches_exhaustive_boxes_and_reversal():
    surface = _grid_surface(12)
    tree = TriangleSurfaceBVH(surface)
    rng = np.random.default_rng(21)
    for _ in range(150):
        start, end = rng.uniform(-0.5, 1.5, (2, 3))
        for padding in (0.0, 0.03):
            found = tree.segment_overlapping_facets(start, end, padding=padding)
            np.testing.assert_array_equal(
                found, exhaustive_boxes(surface, start, end, padding)
            )
            np.testing.assert_array_equal(
                found, tree.segment_overlapping_facets(end, start, padding=padding)
            )


@pytest.mark.parametrize(
    "start,end",
    [
        ([0, 0, 0], [1, 1, 0]),
        ([0.5, 0.5, -1], [0.5, 0.5, 1]),
        ([0.5, 0.5, 0], [0.5, 0.5, 0]),
        ([-1, 0, 0], [0, 0, 0]),
        ([0.5, 0.5, 0.01], [0.5, 0.5, 0.01]),
    ],
)
def test_segment_parallel_zero_length_and_closed_endpoints(start, end):
    surface = _grid_surface(4)
    tree = TriangleSurfaceBVH(surface)
    start, end = np.array(start), np.array(end)
    np.testing.assert_array_equal(
        tree.segment_overlapping_facets(start, end),
        exhaustive_boxes(surface, start, end, 0),
    )


def test_segment_broadphase_reduces_diagonal_box_false_candidates():
    tree = TriangleSurfaceBVH(_grid_surface(50))
    exact = tree.segment_overlapping_facets([0, 0, 0], [1, 1, 0])
    coarse = tree.overlapping_facets([0, 0, 0], [1, 1, 0])
    assert len(exact) < len(coarse) / 10


@pytest.mark.parametrize(
    "start,end,padding",
    [
        ([0, 0, np.nan], [1, 1, 1], 0),
        ([0, 0, 0], [1, 1, 1], -1),
        ([0, 0, 0], [1, 1, 1], 1j),
        ([0, 0], [1, 1, 1], 0),
    ],
)
def test_invalid_segment_rejected(start, end, padding):
    with pytest.raises(ValueError):
        TriangleSurfaceBVH(_grid_surface(1)).segment_overlapping_facets(
            start, end, padding=padding
        )
