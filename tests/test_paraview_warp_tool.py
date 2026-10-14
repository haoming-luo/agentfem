# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reader-property normalization needs no optional ParaView installation."""
import numpy as np
import pytest

from tools.check_paraview_warp import _cell_topology, _frame_time, _reader_times, _warp_error


def test_final_frame_does_not_silently_check_undeformed_initial_state():
    assert _frame_time([0., .1], 'last') == .1
    assert _frame_time([0., .1], 'first') == 0.
    assert _frame_time([.7], 'last') == .7
    assert _frame_time([], 'last') == 0.
    with pytest.raises(ValueError, match='Frame'):
        _frame_time([0., .1], 'middle')


@pytest.mark.parametrize('raw,expected', [
    (None, []), ([], []), (.7, [.7]), (np.float64(.7), [.7]),
    ([1., 2.], [1., 2.]), (np.array([1., 2.]), [1., 2.]),
])
def test_scalar_and_sequence_reader_times(raw, expected):
    assert _reader_times(raw) == expected


@pytest.mark.parametrize('raw', [np.nan, np.inf, [0., -np.inf]])
def test_invalid_reader_times_are_not_silently_selected(raw):
    with pytest.raises(ValueError, match='nonfinite'):
        _reader_times(raw)


def test_warp_geometry_check_reports_true_error():
    coordinates = np.arange(12, dtype=float).reshape(4, 3)
    values = np.full_like(coordinates, .1)
    assert _warp_error(coordinates, values, coordinates + 2 * values, 2.) == 0.
    with pytest.raises(AssertionError):
        _warp_error(coordinates, values, coordinates, 2.)


@pytest.mark.parametrize('location', range(3))
@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_invalid_coordinates_or_vectors_never_pass(location, invalid):
    arrays = [np.zeros((4, 3)) for _ in range(3)]
    arrays[location][0, 0] = invalid
    with pytest.raises(ValueError, match='finite'):
        _warp_error(*arrays, 1.)


def test_matching_nan_is_not_success():
    invalid = np.full((4, 3), np.nan)
    with pytest.raises(ValueError, match='finite'):
        _warp_error(invalid, invalid, invalid, 1.)


@pytest.mark.parametrize('shape', [(0, 3), (4, 2), (4,), (1, 3)])
def test_warp_shape_mismatch_does_not_broadcast(shape):
    with pytest.raises(ValueError, match='geometry and vector'):
        _warp_error(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(shape), 1.)


def test_warp_overflow_does_not_pass_as_matching_infinity():
    values = np.full((4, 3), 1.e308)
    with pytest.raises(ValueError, match='finite coordinate range'):
        _warp_error(values, values, values, 2.)


class _Grid:
    def __init__(self, cells):
        self.cells = cells

    def GetNumberOfCells(self):
        return len(self.cells)

    def GetCellType(self, index):
        return self.cells[index][0]

    def GetCell(self, index):
        class Cell:
            def GetCellDimension(inner):
                return self.cells[index][1]
        return Cell()


def test_reader_topology_distinguishes_embedded_surface_from_volume():
    assert _cell_topology(_Grid([(12, 3), (12, 3), (10, 3)]), 3) == {
        "cell_types": {12: {"dimension": 3, "count": 2},
                       10: {"dimension": 3, "count": 1}},
        "cell_dimensions": {3: 3},
    }
    for cells in ([(9, 2)], [(12, 3), (9, 2)]):
        with pytest.raises(ValueError, match="Expected only 3D cells"):
            _cell_topology(_Grid(cells), 3)


def test_dimension_is_explicit_and_does_not_forbid_valid_lower_dimensional_models():
    grid = _Grid([(9, 2)])
    assert _cell_topology(grid, 2)["cell_dimensions"] == {2: 1}
    assert _cell_topology(grid)["cell_dimensions"] == {2: 1}
    with pytest.raises(ValueError, match="nonempty"):
        _cell_topology(_Grid([]), 3)
    with pytest.raises(ValueError, match="must be"):
        _cell_topology(grid, 4)
