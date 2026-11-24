# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

pv = pytest.importorskip('pyvista')
from agentfem._viewer_smoothing import average_cell_colors
from agentfem._viewer_export import display_frame


def mesh():
    grid = pv.UnstructuredGrid(np.array([3, 0, 1, 2, 3, 1, 3, 2]),
                              np.array([5, 5], dtype=np.uint8),
                              np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [1., 1, 0]]))
    grid.cell_data['stress'] = [2., 8.]
    grid.cell_data['region'] = [1, 2]
    return grid


def test_display_average_and_original_are_independent():
    grid = mesh()
    points, cells = grid.points.copy(), grid.cells.copy()
    result, colors = average_cell_colors(grid, grid.cell_data['stress'])
    np.testing.assert_allclose(colors, [2, 5, 5, 8])
    np.testing.assert_array_equal(grid.cell_data['stress'], [2, 8])
    np.testing.assert_array_equal(grid.points, points)
    np.testing.assert_array_equal(grid.cells, cells)
    assert result.n_cells == grid.n_cells
    assert set(grid.cell_data) == {'stress', 'region'}


def test_region_boundary_is_not_averaged():
    grid = mesh()
    result, colors = average_cell_colors(grid, grid.cell_data['stress'], 'region')
    assert result.n_points == 6
    np.testing.assert_array_equal(np.sort(colors), [2, 2, 2, 8, 8, 8])
    assert result.n_cells == 2


def test_export_uses_same_display_average():
    grid = mesh()
    view = dict(reference_geometry=False, choice=['cell', 'stress', None],
                cell_smoothing=True, smoothing_region='region')
    result, colors = display_frame(None, 0, view, grid=grid)
    expected, expected_colors = average_cell_colors(grid, grid.cell_data['stress'], 'region')
    np.testing.assert_array_equal(colors, expected_colors)
    np.testing.assert_array_equal(result.points, expected.points)


def test_invalid_region_is_not_silently_ignored():
    with pytest.raises(ValueError, match='region'):
        average_cell_colors(mesh(), [2, 8], 'missing')
