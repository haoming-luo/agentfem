# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Display choices preserve association and do not invent stress invariants."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from tools.result_viewer import scalar_choices


def test_scalar_choices_preserve_duplicate_names_and_components():
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.cell_data["damage"] = [0.5]
    grid.point_data["damage"] = np.zeros(grid.n_points)
    grid.cell_data["stress"] = np.arange(9).reshape(1, 9)
    grid.point_data["label"] = np.array(["node"] * grid.n_points)
    choices = scalar_choices(grid)
    assert ("cell", "damage", None) in choices
    assert ("point", "damage", None) in choices
    assert [("cell", "stress", i) for i in range(9)] == [
        choice for choice in choices if choice[1] == "stress"
    ]
    assert all(choice[1] != "label" for choice in choices)
    assert set(grid.cell_data) == {"damage", "stress"}
