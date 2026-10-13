# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reader-property normalization needs no optional ParaView installation."""
import numpy as np
import pytest

from tools.check_paraview_warp import _reader_times, _warp_error


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
