# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional local result viewer; consumes saved artifacts, never executes a case."""

import argparse
import asyncio
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from agentfem.results import (
    deform_visualization_frame, inspect_visualization, read_visualization_frame,
)
from agentfem._viewer_files import ResultDirectory


def scalar_choices(grid):
    """Keep association explicit; do not invent tensor invariants or smooth cells."""
    choices = []
    for association, data in (("cell", grid.cell_data), ("point", grid.point_data)):
        for name in data.keys():
            values = np.asarray(data[name])
            if values.dtype.kind not in "fiu":
                continue
            if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
                choices.append((association, name, None))
            elif values.ndim == 2:
                choices.extend((association, name, index) for index in range(values.shape[1]))
    return choices


def build_viewer(path, *, displacement=None, reference_geometry=False, server=None):
    import pyvista as pv
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import vtk, vuetify3 as v3
    from trame_vtk.modules.vtk.serializers.registry import register_js_class

    # VTK's OpenGL property implements the ordinary surface-property contract.
    # Send that contract to vtk.js, not its backend-specific OpenGL class.
    # Otherwise trame-vtk 2.11.17 creates an object without getOpacity().
    register_js_class("vtkOpenGLProperty", "vtkProperty")

    requested = Path(path).expanduser().resolve(strict=True)
    directory = ResultDirectory(requested if requested.is_dir() else requested.parent)
    path = None if requested.is_dir() else directory.result(requested)
    info = inspect_visualization(path) if path else {
        "frame": 0, "frame_count": 0, "saved_times": [], "information": []}
    server = server or get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    state.trame__title = "AgentFEM · Local results"
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("#f3f5f7")
    cache = {"frame": None, "grid": None, "choices": [], "actor": None, "bar": None}
    state.frame = info["frame"]
    state.field = -1
    state.scale = 0.
    state.edges = False
    state.error = ""
    state.field_items = [{"title": "Solid color", "value": -1}]
    state.coordinate = ""
    state.source_status = ""
    state.frame_count = info["frame_count"]
    state.result_name = path.name if path else "Open a local result"
    state.loaded_source = str(path.relative_to(directory.root)) if path else ""
    state.folder = "."
    state.file_items = directory.entries()
    state.file_selection = None
    state.project_items = directory.projects()
    state.project_selection = None
    state.run_items = []
    state.run_selection = None
    state.playing = False
    state.fixed_range = False
    state.range_min = 0.
    state.range_max = 1.
    state.mesh_summary = ""
    state.reference_geometry = bool(reference_geometry)
    state.displacement_field = displacement
    state.displacement_items = []
    playback = {"task": None}

    def update(**_):
        if path is None:
            return
        try:
            index = int(state.frame)
            changed = cache["frame"] != index
            grid = read_visualization_frame(path, frame=index) if changed else cache["grid"]
            choices = scalar_choices(grid)
            vectors = [name for name, values in grid.point_data.items()
                       if np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == 3
                       and np.asarray(values).dtype.kind in "fiu"]
            state.displacement_items = vectors
            selected = int(state.field)
            if changed:
                # Changing schema must not silently reuse an index for another field.
                previous = cache["choices"][selected] if 0 <= selected < len(cache["choices"]) else None
                selected = choices.index(previous) if previous in choices else -1
            if not -1 <= selected < len(choices):
                raise ValueError("Select a field available in this frame.")
            displayed = (deform_visualization_frame(grid, state.displacement_field,
                         scale=float(state.scale), reference_geometry=True)
                         if state.reference_geometry and float(state.scale) != 0.
                         else grid.copy(deep=True))
            scalar = None
            if selected >= 0:
                association, name, component = choices[selected]
                data = displayed.cell_data if association == "cell" else displayed.point_data
                values = np.asarray(data[name])
                values = values.reshape(-1) if component is None else values[:, component]
                if not np.isfinite(values).all():
                    raise ValueError("Selected field contains nonfinite values.")
                # A view-only scalar array never replaces source tensor/field data.
                key = "__view_scalar__"
                while key in data:
                    key += "_"
                data[key] = values.copy()
                label = name if component is None else f"{name}[{component}]"
                bounds = None
                if state.fixed_range:
                    bounds = float(state.range_min), float(state.range_max)
                    if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                        raise ValueError("Color minimum must be finite and smaller than maximum.")
                scalar = (values, key, association, f"{label} ({association})", bounds)
            first_view = cache["grid"] is None
            camera = plotter.camera_position
            # Keep scene identities stable across interactive updates. Repeated
            # clear/add can recycle VTK addresses into a different object type
            # while vtk.js still holds the old serialized identity.
            if cache["actor"] is None:
                cache["actor"] = plotter.add_mesh(displayed, color="#648aa6",
                                                  show_scalar_bar=False)
            actor = cache["actor"]
            actor.mapper.dataset = displayed
            actor.prop.show_edges = bool(state.edges)
            actor.mapper.scalar_visibility = scalar is not None
            if scalar is not None:
                values, key, association, label, bounds = scalar
                actor.mapper.set_scalars(values, key, preference=association,
                                         cmap="viridis", clim=bounds)
                if cache["bar"] is None:
                    cache["bar"] = plotter.add_scalar_bar(label, mapper=actor.mapper)
                cache["bar"].SetTitle(label)
                cache["bar"].SetLookupTable(actor.mapper.lookup_table)
                cache["bar"].SetVisibility(True)
            elif cache["bar"] is not None:
                cache["bar"].SetVisibility(False)
            if cache["grid"] is None:
                plotter.view_isometric()
                plotter.reset_camera()
            else:
                plotter.camera_position = camera
            cache.update(frame=index, grid=grid, choices=choices)
            with state:
                state.field_items = [{"title": "Solid color", "value": -1}] + [
                    {"title": f"{name}{'' if component is None else '['+str(component)+']'} · {association}",
                     "value": i} for i, (association, name, component) in enumerate(choices)]
                state.field = selected
                time = info["saved_times"][index] if info["saved_times"] else None
                state.coordinate = f"Frame {index+1}/{info['frame_count']}" + (f" · saved coordinate {time:.6g}" if time is not None else "")
                state.mesh_summary = f"{grid.n_points:,} points · {grid.n_cells:,} cells"
                notices = [f"{item['name']}: {item['value']}" for item in info["information"]
                           if item["name"] in {"status", "agentfem_output_status", "source_status", "temporal_coverage"}]
                state.source_status = " · ".join(notices)
                state.error = ""
            ctrl.view_update()
            if first_view:
                ctrl.view_camera()
                ctrl.view_reset()
        except (ValueError, IndexError, TypeError, OSError, ET.ParseError) as exc:
            state.error = str(exc)
            state.playing = False

    def stop_playback(**_):
        state.playing = False
        task = playback["task"]
        if task is not None and not task.done():
            task.cancel()
        playback["task"] = None

    async def play():
        try:
            while state.playing and int(state.frame) < info["frame_count"] - 1:
                await asyncio.sleep(.5)
                if state.playing:
                    with state:
                        state.frame = int(state.frame) + 1
        finally:
            with state:
                state.playing = False

    def toggle_playback(**_):
        if state.playing:
            stop_playback()
        elif path is not None and info["frame_count"] > 1:
            with state:
                if int(state.frame) == info["frame_count"] - 1:
                    state.frame = 0
                state.playing = True
            playback["task"] = asyncio.create_task(play())

    def open_selection(**_):
        nonlocal path
        stop_playback()
        try:
            selected = directory.resolve(state.file_selection or state.folder)
            if selected.is_dir():
                with state:
                    state.file_items = directory.entries(selected)
                    state.folder = str(selected.relative_to(directory.root))
                    state.file_selection = None
                    state.error = ""
                return
            candidate = directory.result(selected)
            # Read before replacing the current view: failed opens preserve it.
            latest = inspect_visualization(candidate)
            path = candidate
            info.clear()
            info.update(latest)
            cache.update(frame=None, grid=None, choices=[])
            with state:
                state.result_name = path.name
                state.loaded_source = str(path.relative_to(directory.root))
                state.frame_count = latest["frame_count"]
                state.frame = latest["frame"]
                state.field = -1
                state.scale = 0.
                # Coordinate semantics belong to this file, not the previously
                # opened file. Never accidentally warp current coordinates twice.
                state.reference_geometry = False
                state.displacement_field = None
                state.fixed_range = False
                state.error = ""
            update()
        except (ValueError, IndexError, TypeError, OSError, ET.ParseError) as exc:
            state.error = str(exc)

    def parent_folder(**_):
        try:
            current = directory.resolve(state.folder)
            if current != directory.root:
                entries = directory.entries(current.parent)
                with state:
                    state.folder = str(current.parent.relative_to(directory.root))
                    state.file_items = entries
                    state.file_selection = None
                    state.error = ""
        except (ValueError, OSError) as exc:
            state.error = str(exc)

    def select_project(**_):
        stop_playback()
        if state.project_selection is None:
            return
        try:
            runs = directory.runs(state.project_selection)
            with state:
                state.run_items = runs
                state.run_selection = None
                state.file_selection = state.project_selection
            open_selection()
        except (ValueError, TypeError, OSError) as exc:
            state.error = str(exc)

    def select_run(**_):
        if state.run_selection is not None:
            state.file_selection = state.run_selection
            open_selection()
            files = [item for item in state.file_items if not item["directory"]]
            if not state.error and len(files) == 1:
                state.file_selection = files[0]["value"]

    def refresh(**_):
        # Reopen only the atomically published index; never inspect a writer's
        # temporary files or interpolate across a restart gap.
        stop_playback()
        try:
            state.project_items = directory.projects()
            if state.project_selection is not None:
                state.run_items = directory.runs(state.project_selection)
            if path is None:
                state.file_items = directory.entries(state.folder)
                state.error = ""
                return
            directory.result(path)
            latest = inspect_visualization(path)
            current_time = (info["saved_times"][int(state.frame)]
                            if info["saved_times"] else None)
            index = (latest["saved_times"].index(current_time)
                     if current_time in latest["saved_times"] else latest["frame"])
            # Inspect reads the latest frame, which may be valid even when the
            # user's selected older frame has become unreadable. Preflight that
            # frame before publishing new navigation metadata. Refresh is an
            # explicit action, so one extra frame read is preferable to losing
            # the last usable view after a partially available continuation.
            if index != latest["frame"]:
                read_visualization_frame(path, frame=index)
            info.update(latest)
            cache["frame"] = None
            with state:
                state.frame_count = info["frame_count"]
                state.frame = index
            update()
        except (ValueError, IndexError, TypeError, OSError, ET.ParseError) as exc:
            state.error = str(exc)

    with SinglePageWithDrawerLayout(server, width=320) as ui:
        ui.title.set_text("AgentFEM · {{ result_name }}")
        ui.icon.title = "Show or hide result controls"
        with ui.toolbar:
            v3.VSpacer()
            v3.VBtn("Refresh frames", click=refresh)
            v3.VBtn("Reset view", click=lambda: ctrl.view_reset())
        with ui.drawer:
            with v3.VContainer(classes="pa-4 d-flex flex-column ga-3"):
                v3.VLabel("LOCAL RESULTS", style="font-size:12px; letter-spacing:1px")
                v3.VSelect(v_model=("project_selection", None), items=("project_items",),
                           label="Project", v_if="project_items.length", hide_details=True,
                           density="compact")
                v3.VSelect(v_model=("run_selection", None), items=("run_items",),
                           label="Run", v_if="project_selection !== null", hide_details=True,
                           density="compact")
                v3.VSelect(v_model=("file_selection", None), items=("file_items",),
                           label="Result or folder", hide_details=True, density="compact")
                with v3.VRow(classes="ma-0 ga-2"):
                    v3.VBtn("Open", click=open_selection, color="primary", variant="tonal", prepend_icon="mdi-folder-open-outline")
                    v3.VBtn("Up", click=parent_folder, disabled=("folder === '.'",), variant="text")
                v3.VLabel("Folder: {{ folder }}", style="font-size:12px; white-space:normal; overflow-wrap:anywhere")
                v3.VDivider()
                v3.VSelect(v_model=("field", -1), items=("field_items",),
                           label="Field · association", hide_details=True, density="compact")
                v3.VCheckbox(v_model=("edges", False), label="Show mesh edges", hide_details=True, density="compact")
                v3.VCheckbox(v_model=("fixed_range", False), label="Fixed color range", hide_details=True, density="compact")
                with v3.VRow(classes="ma-0 ga-2 flex-nowrap"):
                    v3.VTextField(v_model=("range_min", 0.), label="Minimum", type="number",
                                  disabled=("!fixed_range",), hide_details=True, density="compact", style="min-width:0")
                    v3.VTextField(v_model=("range_max", 1.), label="Maximum", type="number",
                                  disabled=("!fixed_range",), hide_details=True, density="compact", style="min-width:0")
                v3.VDivider()
                v3.VCheckbox(v_model=("reference_geometry", bool(reference_geometry)),
                             label="Coordinates are undeformed", hide_details=True, density="compact")
                v3.VSelect(v_model=("displacement_field", displacement), items=("displacement_items",),
                           label="Displacement · point vector", disabled=("!reference_geometry",),
                           hide_details=True, density="compact")
                v3.VTextField(v_model=("scale", 0.), label="Warp scale", type="number",
                              disabled=("!reference_geometry || !displacement_field",), hide_details=True, density="compact")
                v3.VLabel("Warp changes the view, not the saved geometry.",
                          style="font-size:12px; white-space:normal")
        with ui.content:
            with v3.VContainer(fluid=True, classes="pa-0 d-flex flex-column",
                               style="height:calc(100vh - 96px); min-height:320px"):
                v3.VAlert("{{ error }}", v_if="error", type="error", density="compact")
                with v3.VRow(classes="ma-0 px-3 align-center ga-3", style="flex:0 0 auto; width:100%"):
                    v3.VBtn("{{ playing ? 'Pause' : 'Play' }}", click=toggle_playback,
                            disabled=("frame_count <= 1",), variant="text",
                            prepend_icon=("playing ? 'mdi-pause' : 'mdi-play'",))
                    v3.VSlider(v_model=("frame", info["frame"]), min=0,
                               max=("frame_count - 1",), step=1, label="Saved frame",
                               disabled=("frame_count <= 1",), hide_details=True, style="min-width:160px")
                v3.VLabel("{{ coordinate }}", classes="px-4")
                v3.VLabel("Viewing: {{ loaded_source }}", v_if="loaded_source", classes="px-4",
                          style="font-size:12px; overflow-wrap:anywhere")
                v3.VLabel("{{ mesh_summary }}", classes="px-4")
                v3.VLabel("{{ source_status }}", v_if="source_status", classes="px-4")
                # VtkLocalView manages its initial synchronization. Its ready
                # event is not a user change: rebuilding the scene there can
                # form a ready -> update -> ready loop on large meshes.
                view = vtk.VtkLocalView(plotter.ren_win,
                                        style="flex:1; min-height:0; width:100%")
                ctrl.view_update = view.update
                ctrl.view_reset = view.reset_camera
                ctrl.view_camera = view.push_camera
                v3.VLabel("Local preview · original field association · no averaging · {{ fixed_range ? 'fixed range' : 'range per frame' }}", classes="px-4 pb-2")
    state.change("frame", "field", "scale", "edges", "fixed_range", "range_min", "range_max",
                 "reference_geometry", "displacement_field")(update)
    state.change("project_selection")(select_project)
    state.change("run_selection")(select_run)
    ctrl.on_server_ready.add(update)
    ctrl.on_server_exited.add(stop_playback)
    def close_view(**_):
        # Trame supplies state keyword arguments to lifecycle callbacks.
        plotter.close()
    ctrl.on_server_exited.add(close_view)
    # The controller callbacks are also exercised without an LLM or browser.
    ctrl.open_result = open_selection
    ctrl.refresh_result = refresh
    ctrl.parent_folder = parent_folder
    ctrl.select_project = select_project
    ctrl.select_run = select_run
    ctrl.toggle_playback = toggle_playback
    ctrl.stop_playback = stop_playback
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path.cwd(),
                        help="Result file or directory to browse (default: current directory)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--displacement")
    parser.add_argument("--reference-geometry", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if bool(args.displacement) != args.reference_geometry:
        parser.error("Use --displacement and --reference-geometry together to enable warp.")
    try:
        app = build_viewer(args.path, displacement=args.displacement,
                           reference_geometry=args.reference_geometry)
    except ImportError as exc:
        parser.error(f"Install the optional viewer: python -m pip install 'agentfem[viewer]' ({exc}).")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    app.start(host="127.0.0.1", port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
