"""Geometry checks independent of DCB, ENF and MMB fixtures."""
import numpy as np
import pytest
from agentfem import interfaces


def test_asymmetric_layers_preserve_area_and_pair_interface():
    split = interfaces.split_rectangle(length=3, lower_height=1, upper_height=2,
                                       cells_along=3, cells_below=2, cells_above=3)
    assert split.cells.shape == (15, 4)
    assert split.positive_cells.size == 9
    xy = split.coordinates[split.cells]
    area = 0.5 * np.sum(xy[:, :, 0] * np.roll(xy[:, :, 1], -1, axis=1)
                        - xy[:, :, 1] * np.roll(xy[:, :, 0], -1, axis=1), axis=1)
    assert np.all(area > 0)
    assert np.sum(area) == pytest.approx(9)
    assert len(split.original_to_duplicate) == 4
    for original, duplicate in split.original_to_duplicate.items():
        np.testing.assert_array_equal(split.coordinates[original], split.coordinates[duplicate])
        assert split.coordinates[original, 1] == 0


@pytest.mark.parametrize("override", [dict(length=0), dict(upper_height=float('nan')),
                                       dict(cells_along=2.5), dict(cells_below=True)])
def test_invalid_rectangle_rejected(override):
    kwargs = dict(length=3, lower_height=1, upper_height=1,
                  cells_along=3, cells_below=1, cells_above=1)
    kwargs.update(override)
    with pytest.raises(ValueError):
        interfaces.split_rectangle(**kwargs)
