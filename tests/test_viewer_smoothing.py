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


@pytest.mark.parametrize('method', ['preserve_jumps', 'edge_aware'])
def test_retired_methods_use_fast_average_without_topology_expansion(method):
    grid = mesh()
    before = grid.points.copy(), grid.cells.copy(), grid.cell_data['stress'].copy()
    result, colors = average_cell_colors(grid, grid.cell_data['stress'], method=method)
    assert result.n_cells == grid.n_cells
    assert result.n_points == grid.n_points
    np.testing.assert_array_equal(colors, [2, 5, 5, 8])
    for actual, original in zip((grid.points, grid.cells, grid.cell_data['stress']), before):
        np.testing.assert_array_equal(actual, original)
    assert colors.min() >= 2 and colors.max() <= 8


def test_legacy_smoothing_preserves_explicit_regions():
    grid = mesh()
    result, colors = average_cell_colors(grid, [2, 8], method='edge_aware')
    assert result.n_cells == 2
    assert colors.min() >= 2 and colors.max() <= 8
    np.testing.assert_array_equal(colors, [2, 5, 5, 8])
    np.testing.assert_array_equal(grid.cell_data['stress'], [2, 8])
    _, isolated = average_cell_colors(grid, [2, 8], 'region', method='edge_aware')
    np.testing.assert_array_equal(np.sort(isolated), [2, 2, 2, 8, 8, 8])


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


@pytest.mark.parametrize('enabled,scale', [(False, 4.), (True, 1.), (True, 2.)])
def test_current_coordinates_obey_deformation_switch(enabled, scale):
    grid = mesh()
    reference = grid.points.copy()
    u = np.tile([.5, .2, 0], (grid.n_points, 1))
    grid.point_data['U'] = u
    grid.points += u
    view = dict(reference_geometry=enabled, current_coordinates=True,
                displacement='U', scale=scale, choice=None)
    result, _ = display_frame(None, 0, view, grid=grid)
    np.testing.assert_allclose(result.points, reference + (scale*u if enabled else 0))
    np.testing.assert_allclose(grid.points, reference + u)


def test_legend_is_transparent_over_full_model_viewport():
    from agentfem._viewer_style import legend_panel
    plotter = pv.Plotter(off_screen=True)
    try:
        panel = legend_panel(plotter)
        assert panel.GetViewport() == plotter.renderer.GetViewport() == (0., 0., 1., 1.)
        assert panel.GetLayer() == 1 and panel.GetPreserveColorBuffer()
    finally:
        plotter.close()
