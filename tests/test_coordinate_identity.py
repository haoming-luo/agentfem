# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._coordinate_identity import has_duplicate_coordinates, weld_coordinates_first


def reference(points):
    nodes, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind='stable')
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    return nodes[order], renumber[inverse]


@pytest.mark.parametrize('n', [0, 1, 2, 100, 10000])
@pytest.mark.parametrize('strided', [False, True])
def test_exact_first_occurrence_map_matches_original_rows(n, strided):
    rng = np.random.default_rng(741)
    points = rng.integers(-5, 6, (n, 3)).astype(float)
    points[::3] += rng.normal(size=points[::3].shape)*1e-10
    if strided:
        points = np.repeat(points, 2, axis=1)[:, ::2]
    nodes, inverse = weld_coordinates_first(points)
    expected, indices = reference(points)
    np.testing.assert_array_equal(nodes, expected)
    np.testing.assert_array_equal(inverse, indices)
    np.testing.assert_array_equal(nodes[inverse], points)
    assert has_duplicate_coordinates(points) == (len(expected) < n)


def test_signed_zero_and_adjacent_floats_do_not_create_a_weld_tolerance():
    tiny = np.nextafter(0., 1.)
    points = np.array([[-0., 1., 2.], [0., 1., 2.],
                       [tiny, 1., 2.], [0., np.nextafter(1., 2.), 2.],
                       [1e300, -1e300, tiny], [1e300, -1e300, tiny]])
    nodes, inverse = weld_coordinates_first(points)
    expected, indices = reference(points)
    np.testing.assert_array_equal(nodes.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(inverse, indices)
    assert len(nodes) == 4
    assert np.signbit(nodes[0, 0])


@pytest.mark.parametrize('bad', [np.zeros((3,)), np.zeros((2, 2)),
    [[np.nan, 0., 0.]], [[np.inf, 0., 0.]], [[1j, 0., 0.]]])
def test_invalid_coordinates_are_not_silently_identified(bad):
    for operation in (has_duplicate_coordinates, weld_coordinates_first):
        with pytest.raises(ValueError, match='finite real 3D'):
            operation(bad)
