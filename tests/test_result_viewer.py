# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Display choices preserve association and do not invent stress invariants."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from agentfem.viewer import scalar_choices, component_values, component_label


@pytest.mark.parametrize("port", [None, 0, 8097])
def test_launch_uses_atomic_automatic_port_and_loopback(tmp_path, monkeypatch, port):
    from agentfem import viewer
    calls = []

    class App:
        def start(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(viewer, "build_viewer", lambda *args, **kwargs: App())
    argv = [str(tmp_path), "--no-browser"]
    if port is not None:
        argv.extend(["--port", str(port)])
    viewer.main(argv)
    assert calls == [{"host": "127.0.0.1", "port": port or 0, "open_browser": False}]


@pytest.mark.parametrize("port", [-1, 65536])
def test_launch_rejects_invalid_port(tmp_path, port):
    from agentfem.viewer import main
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path), "--port", str(port)])
    assert exc.value.code == 2


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
    assert ("point", "U", "magnitude") in choices
    assert component_label(grid, ("point", "U", 0)) == "X"
    assert component_label(grid, ("point", "U", 2)) == "Z"
    assert component_label(grid, ("cell", "stress", 0)) == "Component 1"
    grid.point_data["U"][:] = [3., 4., 0.]
    np.testing.assert_array_equal(component_values(grid, ("point", "U", "magnitude")), 5.)
    assert [("cell", "stress", i) for i in range(9)] == [
        choice for choice in choices if choice[1] == "stress"
    ]
    assert all(choice[1] != "label" for choice in choices)
    assert set(grid.cell_data) == {"damage", "stress"}


def test_project_run_navigation_opens_saved_result(tmp_path):
    pytest.importorskip("trame")
    from trame.app import get_server
    from agentfem.viewer import build_viewer

    project = tmp_path / "beam"
    run = project / "custom/002-full"
    run.mkdir(parents=True)
    (project / "agentfem.toml").write_text('[project]\nname="Beam"\n[run]\noutput_directory="custom"\n')
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.save(run / "fields.vtu")
    server = get_server(f"project-{tmp_path.name}", client_type="vue3")
    build_viewer(tmp_path, server=server)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        with state:
            state.project_selection = "beam"
        assert state.run_items[0]["title"] == "002-full"
        with state:
            state.run_selection = "beam/custom/002-full"
        assert state.folder == "beam/custom/002-full"
        state.file_selection = state.file_items[0]["value"]
        ctrl.open_result()
        assert state.result_name == "fields.vtu"
        assert state.frame_count == 1 and state.error == ""
    finally:
        ctrl.on_server_exited()


def test_play_pause_advances_only_saved_frames(tmp_path, monkeypatch):
    import asyncio
    pytest.importorskip("trame")
    from trame.app import get_server
    from agentfem.viewer import build_viewer

    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    for i in range(3):
        grid.cell_data["damage"] = [i / 10]
        grid.save(tmp_path / f"frame-{i}.vtu")
    index = tmp_path / "series.pvd"
    index.write_text('<VTKFile type="Collection"><Collection>' + ''.join(
        f'<DataSet timestep="{time}" file="frame-{i}.vtu"/>'
        for i, time in enumerate((0., .2, .9))) + '</Collection></VTKFile>')
    server = get_server(f"play-{tmp_path.name}", client_type="vue3")
    build_viewer(index, server=server)
    state, ctrl = server.state, server.controller
    state.ready()

    async def exercise():
        tick = asyncio.Event()
        original_sleep = asyncio.sleep

        async def clock_tick(_delay):
            await tick.wait()
            tick.clear()

        monkeypatch.setattr(asyncio, "sleep", clock_tick)
        ctrl.on_server_ready()
        assert state.scale == 1. and state.fixed_range
        with state:
            state.quantity = 0
        assert state.component_items == [{"title": "Value", "value": 0}]
        ctrl.fit_series_range()
        assert state.range_min == 0. and state.range_max == .2
        ctrl.toggle_playback()  # Last frame restarts at the first saved frame.
        assert state.frame == 0 and state.playing
        await original_sleep(0)
        tick.set()
        await original_sleep(0)
        assert state.frame == 1 and "coordinate 0.2" in state.coordinate
        ctrl.stop_playback()
        tick.set()
        await original_sleep(0)
        assert state.frame == 1 and not state.playing
        assert not state.error

    try:
        asyncio.run(exercise())
    finally:
        ctrl.on_server_exited()


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
        assert state.error == ""
        ctrl.parent_folder()
        assert state.folder == "."
    finally:
        ctrl.on_server_exited(trame__scripts=[], arbitrary_state="ignored")


def test_refresh_continuation_preserves_station_and_field(tmp_path, monkeypatch):
    pytest.importorskip("trame")
    from trame.app import get_server
    from agentfem.viewer import build_viewer

    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    for index in range(3):
        grid.cell_data["damage"] = [index*.2]
        grid.save(tmp_path / f"frame-{index}.vtu")
    index_path = tmp_path / "results.pvd"

    def publish(count):
        # Same atomic-index pattern as a running solver; accepted frame files
        # are immutable, and refresh must not manufacture intermediate times.
        temporary = tmp_path / "pending.pvd"
        entries = "".join(f'<DataSet timestep="{t}" file="frame-{i}.vtu"/>'
                          for i, t in enumerate((0., .2, .7)[:count]))
        temporary.write_text('<VTKFile type="Collection"><Collection>'+entries+
                             '</Collection></VTKFile>')
        temporary.replace(index_path)

    publish(2)
    server = get_server(f"continuation-{tmp_path.name}", client_type="vue3")
    build_viewer(index_path, server=server)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        with state:
            state.frame = 0
            state.field = 0
            state.fixed_range = True
        assert state.error == ""
        publish(3)
        from agentfem import viewer
        original_read = viewer.read_visualization_frame

        def unavailable_selected_frame(path, *, frame):
            if frame == 0:
                raise ValueError("selected historical frame is unavailable")
            return original_read(path, frame=frame)

        with monkeypatch.context() as patched:
            patched.setattr(viewer, "read_visualization_frame", unavailable_selected_frame)
            ctrl.refresh_result()
        assert "unavailable" in state.error
        assert state.frame_count == 2 and state.frame == 0 and state.field == 0
        ctrl.refresh_result()
        assert state.frame_count == 3
        assert state.frame == 0 and state.field == 0
        assert state.fixed_range
        assert "coordinate 0" in state.coordinate
        with state:
            state.frame = 2
        assert "coordinate 0.7" in state.coordinate
        assert state.error == ""
        index_path.write_text("<broken")
        ctrl.refresh_result()
        assert state.error
        assert state.frame_count == 3 and state.frame == 2
    finally:
        ctrl.on_server_exited()
