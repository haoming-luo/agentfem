# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Display choices preserve association and do not invent stress invariants."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from agentfem.viewer import scalar_choices, component_values, component_label


def test_smooth_animation_waits_for_browser_scene(tmp_path):
    import asyncio
    from agentfem.viewer import build_viewer
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    for i in range(2):
        grid.point_data['U'] = grid.points*(i*.1)
        grid.save(tmp_path / f'f{i}.vtu')
    path = tmp_path / 'motion.pvd'
    path.write_text('<VTKFile type="Collection"><Collection><DataSet timestep="0" file="f0.vtu"/><DataSet timestep="1" file="f1.vtu"/></Collection></VTKFile>')
    server = build_viewer(path)
    state, ctrl = server.state, server.controller
    state.ready()
    async def exercise():
        ctrl.on_server_ready()
        with state:
            state.reference_geometry = True
        with state:
            state.smooth = True
            state.playback_duration = 1.
        ctrl.scene_loaded()
        ctrl.toggle_playback()
        await asyncio.sleep(.15)
        before = state.frame, state.blend
        await asyncio.sleep(.15)
        assert (state.frame, state.blend) == before  # no renderer acknowledgement
        ctrl.scene_loaded()
        await asyncio.sleep(.15)
        assert state.blend > before[1]
        assert state.playing and not state.error
        ctrl.stop_playback()
        await asyncio.sleep(0)
    try:
        asyncio.run(exercise())
    finally:
        ctrl.on_server_exited()


def test_manual_limits_are_committed_only_on_apply(tmp_path):
    from agentfem.viewer import build_viewer
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.point_data['value'] = np.arange(grid.n_points, dtype=float)
    path = tmp_path / 'limits.vtu'
    grid.save(path)
    server = build_viewer(path)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        before = state.range_min, state.range_max
        with state:
            state.range_max_draft = '1e-'
        assert not state.error
        assert (state.range_min, state.range_max) == before
        with state:
            state.range_min_draft = '1e-12'
            state.range_max_draft = '1e-10'
        ctrl.apply_range_drafts()
        assert not state.error
        assert (state.range_min, state.range_max) == (1e-12, 1e-10)
        with state:
            state.range_max_draft = 'nan'
        ctrl.apply_range_drafts()
        assert state.error
        assert (state.range_min, state.range_max) == (1e-12, 1e-10)
    finally:
        ctrl.on_server_exited()


def test_range_scan_is_responsive_and_cancel_preserves_limits(tmp_path, monkeypatch):
    import asyncio
    import threading
    import agentfem.viewer as viewer
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.point_data['value'] = np.arange(grid.n_points, dtype=float)
    path = tmp_path / 'scan.vtu'
    grid.save(path)
    server = viewer.build_viewer(path)
    state, ctrl = server.state, server.controller
    state.ready()
    release = threading.Event()
    started = threading.Event()
    original = viewer.read_visualization_frame
    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    async def exercise():
        ctrl.on_server_ready()
        with state:
            state.field = 0
        before = state.range_min, state.range_max
        monkeypatch.setattr(viewer, 'read_visualization_frame', blocked)
        task = asyncio.create_task(ctrl.fit_series_range())
        for _ in range(100):
            await asyncio.sleep(.001)
            if started.is_set():
                break
        assert started.is_set() and state.range_busy
        assert state.range_notice == 'Reading saved frames…'
        task.cancel()
        await task
        assert not state.range_busy
        assert 'cancelled' in state.range_notice
        assert (state.range_min, state.range_max) == before
        release.set()
    try:
        asyncio.run(exercise())
    finally:
        release.set()
        ctrl.on_server_exited()


def test_dolfinx_shared_mesh_fields_read_as_one_dataset(tmp_path):
    from agentfem.results import read_visualization_frame
    pointer = "xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology or self::Geometry])"
    text = '''<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude"><Domain>
      <Grid GridType="Uniform"><Topology TopologyType="Triangle" NumberOfElements="1">
      <DataItem Dimensions="1 3" NumberType="Int" Format="XML">0 1 2</DataItem></Topology>
      <Geometry GeometryType="XYZ"><DataItem Dimensions="3 3" Format="XML">0 0 0 1 0 0 0 1 0</DataItem></Geometry></Grid>'''
    for name in ('U', 'Velocity'):
        text += f'''<Grid GridType="Collection" CollectionType="Temporal"><Grid GridType="Uniform">
          <xi:include xpointer="{pointer}"/><Time Value="0"/>
          <Attribute Name="{name}" Center="Node" AttributeType="Vector">
          <DataItem Dimensions="3 3" Format="XML">0 0 0 1 0 0 0 1 0</DataItem></Attribute></Grid></Grid>'''
    text += '</Domain></Xdmf>'
    source = tmp_path / 'original.xdmf'
    source.write_text(text)
    grid = read_visualization_frame(source)
    assert grid.n_cells == 1 and grid.n_points == 3
    assert set(grid.point_data) == {'U', 'Velocity'}
    np.testing.assert_allclose(grid.point_data['U'], grid.point_data['Velocity'])
    assert source.read_text() == text


def test_named_displacement_with_units_and_other_vectors(tmp_path):
    from agentfem.viewer import build_viewer
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.point_data['Displacement_m'] = grid.points * .01
    grid.point_data['Velocity'] = grid.points * 2
    path = tmp_path / 'fields.vtu'
    grid.save(path)
    server = build_viewer(path)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        with state:
            state.reference_geometry = True
        assert state.displacement_field == 'Displacement_m'
        assert not state.error
        with state:
            state.displacement_field = 'missing'
        assert state.displacement_field == 'Displacement_m'
        assert not state.error
    finally:
        ctrl.on_server_exited()


def test_export_start_returns_before_cancel(tmp_path, monkeypatch):
    import asyncio
    from agentfem.viewer import build_viewer
    from agentfem._viewer_jobs import ExportJob
    async def waiting(self, request, progress):
        while not self.cancelled:
            await asyncio.sleep(.001)
        return None
    monkeypatch.setattr(ExportJob, 'run', waiting)
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    path = tmp_path / 'fields.vtu'
    grid.save(path)
    server = build_viewer(path)
    state, ctrl = server.state, server.controller
    state.ready()
    async def exercise():
        ctrl.on_server_ready()
        assert ctrl.start_export('image') is None
        await asyncio.sleep(.01)
        assert state.export_busy
        ctrl.cancel_export()
        await asyncio.sleep(.02)
        assert not state.export_busy
        assert 'cancelled' in state.export_progress
        assert not state.export_video
    try:
        asyncio.run(exercise())
    finally:
        ctrl.on_server_exited()


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
    assert calls == [{"host": "127.0.0.1", "port": port or 0, "open_browser": False, "timeout": 120}]


@pytest.mark.parametrize("port", [-1, 65536])
def test_launch_rejects_invalid_port(tmp_path, port):
    from agentfem.viewer import main
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path), "--port", str(port)])
    assert exc.value.code == 2


def test_saved_view_roundtrip_restores_display_not_source_data(tmp_path, monkeypatch):
    pytest.importorskip("trame")
    from trame.app import get_server
    from agentfem.viewer import build_viewer
    plotters = []
    original_plotter = pv.Plotter
    def capture_plotter(*args, **kwargs):
        plotter = original_plotter(*args, **kwargs)
        plotters.append(plotter)
        return plotter
    monkeypatch.setattr(pv, "Plotter", capture_plotter)
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.point_data["U"] = np.ones((grid.n_points, 3)) * .1
    grid.cell_data["damage"] = [.3]
    path = tmp_path / "frame.vtu"
    grid.save(path)
    original = path.read_bytes()
    server = get_server(f"view-save-{tmp_path.name}", client_type="vue3")
    build_viewer(path, server=server)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        assert any(item["title"] == "damage" for item in state.quantity_items)
        with state:
            state.field = 0
            state.reference_geometry = True
            state.displacement_field = "U"
            state.scale = 2.
            state.preset = "publication"
        with state:
            state.range_min, state.range_max = 0., 1.
        ctrl.save_view()
        assert len(state.view_items) == 1 and not state.error
        state.view_selection = state.view_items[0]["value"]
        with state:
            state.field = -1
            state.scale = 0.
            state.preset = "presentation"
        ctrl.restore_view()
        assert not state.error
        assert state.field == 0 and state.scale == 2.
        displayed = [actor.mapper.dataset for actor in plotters[0].actors.values()
                     if hasattr(actor, "mapper") and hasattr(actor.mapper, "dataset")]
        assert any(np.allclose(mesh.points, grid.points + .2) for mesh in displayed
                   if mesh.n_points == grid.n_points)
        assert state.preset == "publication" and state.palette == "ocean"
        with state:
            state.palette = "iris"
        with state:
            state.preset = "scientific"
        assert state.palette == "iris"
        assert (state.range_min, state.range_max) == (0., 1.)
        assert path.read_bytes() == original
        saved = tmp_path / "viewer-exports" / state.view_selection
        media = saved.parent / "figure.png"
        media.write_bytes(b"existing export")
        ctrl.remove_view()
        assert not state.error and not state.view_items and not saved.exists()
        assert media.read_bytes() == b"existing export"
        ctrl.undo_remove_view()
        assert not state.error and saved.exists() and len(state.view_items) == 1
        ctrl.restore_view()
        assert not state.error and state.scale == 2.
        state.view_selection = "../outside/view.json"
        ctrl.remove_view()
        assert state.error and saved.exists()
        assert path.read_bytes() == original
    finally:
        ctrl.on_server_exited()


def test_independent_viewers_do_not_share_controls(tmp_path):
    pytest.importorskip("trame")
    from agentfem.viewer import build_viewer
    first, second = build_viewer(tmp_path), build_viewer(tmp_path)
    try:
        assert first is not second and first.state is not second.state
        first.state.scale = 7.
        first.state.preset = "presentation"
        assert second.state.scale == 1.
        assert second.state.preset == "scientific"
    finally:
        first.controller.on_server_exited()
        second.controller.on_server_exited()


@pytest.mark.parametrize("quality,size", [("720p", [1280, 720]), ("1080p", [1920, 1080]), ("4k", [3840, 2160])])
def test_export_quality_and_stale_download_reset(tmp_path, monkeypatch, quality, size):
    import asyncio
    from agentfem.viewer import build_viewer
    from agentfem._viewer_jobs import ExportJob
    requests = []

    async def cancelled(self, request, progress):
        requests.append(request)
        return None

    monkeypatch.setattr(ExportJob, "run", cancelled)
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    path = tmp_path / "frame.vtu"
    grid.save(path)
    server = build_viewer(path)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        state.export_resolution = quality
        state.export_image, state.export_video = "old.png", "old.mp4"
        asyncio.run(ctrl.export_result("image"))
        assert requests[0]["size"] == size
        assert state.export_step == 3 and not state.export_busy
        assert not state.export_image and not state.export_video
        assert "cancelled" in state.export_progress
        state.export_resolution = "unbounded"
        asyncio.run(ctrl.export_result("image"))
        assert len(requests) == 1 and "720p" in state.error
    finally:
        ctrl.on_server_exited()


def test_zero_centered_colors_follow_component_and_preserve_override(tmp_path):
    from agentfem.viewer import build_viewer
    grid = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    grid.point_data["U"] = np.tile([-4., 2., 0.], (grid.n_points, 1))
    grid.point_data["temperature"] = np.arange(grid.n_points, dtype=float) - 2
    path = tmp_path / "signed.vtu"
    grid.save(path)
    server = build_viewer(path, displacement="U", reference_geometry=True)
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        choices = scalar_choices(grid)
        with state:
            state.field = choices.index(("point", "U", 0))
        assert not state.error
        assert state.color_mode == "zero" and state.palette == "azure-rose"
        assert state.range_min == -state.range_max
        import asyncio
        asyncio.run(ctrl.fit_series_range())
        assert state.range_min == -state.range_max
        with state:
            state.range_max = 2.
        assert state.range_min == -2.
        with state:
            state.preset = "publication"
        assert state.palette == "azure-rose"
        ctrl.save_view()
        assert not state.error
        state.view_selection = state.view_items[0]["value"]
        with state:
            state.field = choices.index(("point", "U", "magnitude"))
        assert state.color_mode == "sequential"
        ctrl.restore_view()
        assert not state.error
        assert state.color_mode == "zero" and (state.range_min, state.range_max) == (-2., 2.)
        with state:
            state.color_mode = "sequential"
        with state:
            state.edges = True
        assert state.color_mode == "sequential"
        with state:
            state.field = choices.index(("point", "temperature", None))
        assert state.color_mode == "sequential"  # Not inferred from signs/name.
        for palette in ("azure-rose", "lagoon-rose"):
            with state:
                state.color_mode = "zero"
            with state:
                state.palette = palette
            with state:
                state.preset = "presentation"
            assert state.palette == palette
            with state:
                state.color_mode = "sequential"
            with state:
                state.color_mode = "zero"
            assert state.palette == palette
            ctrl.save_view()
            assert not state.error
            state.view_selection = state.view_items[-1]["value"]
        with state:
            state.path_input = str(path)
        ctrl.open_path()
        assert not state.error and not state.open_panel
        with state:
            state.path_input = "missing.vtu"
        ctrl.open_path()
        assert state.error and state.loaded_source == path.name
    finally:
        ctrl.on_server_exited()


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
        await ctrl.fit_series_range()
        assert state.range_min == 0. and state.range_max == .2
        ctrl.toggle_playback()  # Last frame restarts at the first saved frame.
        assert state.frame == 0 and state.playing
        await original_sleep(0)
        tick.set()
        await original_sleep(0)
        assert state.frame == 1 and "Step value 0.2" in state.coordinate
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
        assert state.displacement_field == "U"
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
        assert "Step value 0" in state.coordinate
        with state:
            state.frame = 2
        assert "Step value 0.7" in state.coordinate
        assert state.error == ""
        index_path.write_text("<broken")
        ctrl.refresh_result()
        assert state.error
        assert state.frame_count == 3 and state.frame == 2
    finally:
        ctrl.on_server_exited()
