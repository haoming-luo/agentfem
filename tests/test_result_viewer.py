# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Display choices preserve association and do not invent stress invariants."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from agentfem.viewer import scalar_choices


def test_scalar_choices_preserve_duplicate_names_and_components():
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.cell_data["damage"] = [0.5]
    grid.point_data["U"] = np.zeros((grid.n_points, 3))
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


def test_browser_open_switch_and_failed_open_preserves_current(tmp_path):
    pytest.importorskip("trame")
    from trame.app import get_server
    from agentfem.viewer import build_viewer

    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.cell_data["damage"] = [0.5]
    grid.point_data["U"] = np.zeros((grid.n_points, 3))
    grid.save(tmp_path / "first.vtu")
    grid.cell_data["damage"] = [0.75]
    grid.save(tmp_path / "second.vtu")
    (tmp_path / "nested").mkdir()
    server = get_server(f"test-{tmp_path.name}", client_type="vue3")
    build_viewer(tmp_path, server=server)
    # The actual browser exposed a backend-class mismatch missed by file tests.
    from trame_vtk.modules.vtk.serializers.registry import class_name
    assert class_name(pv.Property()) == "vtkProperty"
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        state.file_selection = "first.vtu"
        ctrl.open_result()
        assert state.error == ""
        assert state.result_name == "first.vtu"
        assert state.frame_count == 1
        assert "U" in state.displacement_items
        with state:
            state.reference_geometry = True
            state.displacement_field = "U"
            state.scale = 1.
        assert state.error == ""
        state.file_selection = "second.vtu"
        ctrl.open_result()
        assert state.result_name == "second.vtu"
        assert state.error == ""
        assert not state.reference_geometry
        assert state.displacement_field is None
        state.file_selection = "missing.vtu"
        ctrl.open_result()
        assert state.error
        assert state.result_name == "second.vtu"
        state.file_selection = "nested"
        ctrl.open_result()
        assert state.folder == "nested"
        ctrl.parent_folder()
        assert state.folder == "."
    finally:
        ctrl.on_server_exited(trame__scripts=[], arbitrary_state="ignored")
