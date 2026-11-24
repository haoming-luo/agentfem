# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional local result viewer; consumes saved artifacts, never executes a case."""

import argparse
import asyncio
import json
import time as clock
from pathlib import Path
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np

from agentfem.results import (
    deform_visualization_frame, inspect_visualization, read_visualization_frame,
)
from agentfem._viewer_files import ResultDirectory, result_browser_root
from agentfem._viewer_export import (PRESETS, palette_items,
    apply_camera, validate_view, display_frame, color_map, centered_range)
from agentfem._viewer_jobs import ExportJob
from agentfem._atomic_text import atomic_write_text
from agentfem._viewer_motion import automatic_scale, displacement_metrics, timeline_sample, interpolate_displacement


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
                if values.shape[1] == 3:
                    choices.append((association, name, "magnitude"))
                choices.extend((association, name, index) for index in range(values.shape[1]))
    return choices


def component_values(grid, choice):
    association, name, component = choice
    data = grid.cell_data if association == "cell" else grid.point_data
    values = np.asarray(data[name])
    if component == "magnitude":
        return np.linalg.norm(values, axis=1)
    return values.reshape(-1) if component is None else values[:, component]


def component_label(grid, choice):
    association, name, component = choice
    if component is None:
        return "Value"
    if component == "magnitude":
        return "Magnitude"
    data = grid.cell_data if association == "cell" else grid.point_data
    # Only three-component vectors have an unambiguous XYZ display here.
    # Tensor ordering needs a declared result contract, not a guessed Voigt map.
    if np.asarray(data[name]).shape[1] == 3:
        return ("X", "Y", "Z")[component]
    return f"Component {component + 1}"


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
    directory = ResultDirectory(result_browser_root(requested))
    path = None if requested.is_dir() else directory.result(requested)
    info = inspect_visualization(path) if path else {
        "frame": 0, "frame_count": 0, "saved_times": [], "information": []}
    server = server or get_server("viewer-" + uuid4().hex, client_type="vue3")
    server.enable_module({"serve": {"viewer-assets": str(Path(__file__).parent / "_viewer_assets")},
                          "scripts": ["viewer-assets/viewer.js"], "vue_use": ["AgentFEMViewer"]})
    state, ctrl = server.state, server.controller
    state.trame__title = "AgentFEM · Local results"
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("#f3f5f7")
    cache = {"frame": None, "grid": None, "choices": [], "actor": None, "bar": None}
    state.frame = info["frame"]
    state.field = -1
    state.scale = 1.
    state.edges = False
    state.error = ""
    state.field_items = [{"title": "Solid color", "value": -1}]
    state.quantity = -1
    state.quantity_items = [{"title": "Solid color", "value": -1}]
    state.component_items = []
    state.browse_root = str(directory.root)
    state.coordinate = ""
    state.station_label = ""
    state.time_label = ""
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
    state.smooth = False
    state.smooth_available = False
    state.blend = 0.
    state.playback_duration = 10.
    state.scale_status = ""
    state.scale_busy = False
    state.details = False
    state.fixed_range = True
    state.color_mode = "sequential"
    state.range_min = 0.
    state.range_max = 1.
    state.mesh_summary = ""
    state.reference_geometry = bool(reference_geometry)
    state.displacement_field = displacement
    state.displacement_items = []
    state.preset = "scientific"
    state.palette = PRESETS["scientific"]["cmap"]
    state.palette_items = palette_items("sequential")
    state.open_panel = path is None
    state.session_panel = False
    state.playback_panel = False
    state.path_input = ""
    state.advanced = False
    state.figure_title = ""
    state.field_unit = ""
    state.time_unit = ""
    state.transparent = False
    state.export_busy = False
    state.export_panel = False
    state.export_progress = ""
    state.export_image = ""
    state.export_video = ""
    state.export_record = ""
    state.closed = False
    state.close_confirmation = False
    state.view_selection = None
    state.view_items = []
    export_root = directory.root / "viewer-exports"
    # Only derived exports are served, never the project directory or H5 files.
    export_root.mkdir(exist_ok=True)
    if export_root.is_symlink():
        raise ValueError("The viewer export directory must not be a symlink.")
    server.serve["viewer-exports"] = str(export_root)
    job = ExportJob(export_root)
    playback = {"task": None}
    range_owner = {"choice": None}
    preferred_palette = {"zero": "coolwarm", "sequential": state.palette}

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
            if (state.reference_geometry and state.displacement_field is None and len(vectors) == 1
                    and vectors[0] in {"Displacement", "displacement", "U"}):
                state.displacement_field = vectors[0]
            selected = int(state.field)
            if changed:
                # Changing schema must not silently reuse an index for another field.
                previous = cache["choices"][selected] if 0 <= selected < len(cache["choices"]) else None
                selected = choices.index(previous) if previous in choices else -1
            if not -1 <= selected < len(choices):
                raise ValueError("Select a field available in this frame.")
            choice = choices[selected] if selected >= 0 else None
            eligible = bool(state.reference_geometry and state.displacement_field and
                            (choice is None or choice[:2] == ("point", state.displacement_field)))
            state.smooth_available = eligible and info["frame_count"] > 1
            if not eligible:
                if state.smooth and state.playing:
                    stop_playback()
                state.blend = 0.
                state.smooth = False
            motion_grid = grid
            if float(state.blend) > 0:
                if cache.get("next_index") != index+1:
                    cache["next_grid"] = read_visualization_frame(path, frame=index+1)
                    cache["next_index"] = index+1
                motion_grid = interpolate_displacement(grid, cache["next_grid"], state.displacement_field,
                                                       float(state.blend), choice)
            displayed = (deform_visualization_frame(motion_grid, state.displacement_field,
                         scale=float(state.scale), reference_geometry=True)
                         if state.reference_geometry and float(state.scale) != 0.
                         else motion_grid.copy(deep=True))
            scalar = None
            if selected >= 0:
                association, name, component = choices[selected]
                data = displayed.cell_data if association == "cell" else displayed.point_data
                values = component_values(displayed, choices[selected])
                if not np.isfinite(values).all():
                    raise ValueError("Selected field contains nonfinite values.")
                # A view-only scalar array never replaces source tensor/field data.
                key = "__view_scalar__"
                while key in data:
                    key += "_"
                data[key] = values.copy()
                label = name if component is None else f"{name} · {component_label(grid, choices[selected])}"
                bounds = None
                if range_owner["choice"] != choices[selected]:
                    # Only the explicitly selected displacement vector has known
                    # semantics here. Never guess stress/temperature from names.
                    state.color_mode = ("zero" if association == "point" and
                                        name == state.displacement_field and
                                        type(component) is int else "sequential")
                if state.fixed_range:
                    if range_owner["choice"] != choices[selected]:
                        low, high = float(values.min()), float(values.max())
                        if low == high:
                            pad = max(abs(low)*.01, 1.e-12)
                            low, high = low-pad, high+pad
                        if state.color_mode == "zero":
                            low, high = centered_range(low, high)
                        range_owner["choice"] = choices[selected]
                        state.range_min, state.range_max = low, high
                    if state.color_mode == "zero":
                        if float(state.range_max) <= 0:
                            raise ValueError("Color limit must be positive.")
                        state.range_min = -float(state.range_max)
                    bounds = float(state.range_min), float(state.range_max)
                    if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                        raise ValueError("Color minimum must be finite and smaller than maximum.")
                elif state.color_mode == "zero":
                    bounds = centered_range(values.min(), values.max())
                range_owner["choice"] = choices[selected]
                scalar = (values, key, association, label, bounds)
            first_view = cache["grid"] is None
            camera = plotter.camera_position
            # Keep scene identities stable across interactive updates. Repeated
            # clear/add can recycle VTK addresses into a different object type
            # while vtk.js still holds the old serialized identity.
            if cache["actor"] is None:
                cache["actor"] = plotter.add_mesh(displayed, color="#648aa6",
                                                  show_scalar_bar=False)
            actor = cache["actor"]
            preset = PRESETS[state.preset]
            plotter.set_background(preset["background"])
            actor.prop.color = preset["solid"]
            actor.prop.ambient = .55
            actor.prop.diffuse = .45
            actor.prop.specular = 0.
            actor.mapper.dataset = displayed
            actor.prop.show_edges = bool(state.edges)
            actor.mapper.scalar_visibility = scalar is not None
            if scalar is not None:
                values, key, association, label, bounds = scalar
                actor.mapper.set_scalars(values, key, preference=association,
                                         cmap=color_map(state.palette), clim=bounds)
                if cache["bar"] is None:
                    cache["bar"] = plotter.add_scalar_bar(label, mapper=actor.mapper)
                cache["bar"].SetTitle(label)
                cache["bar"].SetLookupTable(actor.mapper.lookup_table)
                cache["bar"].SetVisibility(True)
                cache["bar"].GetTitleTextProperty().SetColor(pv.Color(preset["foreground"]).float_rgb)
                cache["bar"].GetLabelTextProperty().SetColor(pv.Color(preset["foreground"]).float_rgb)
            elif cache["bar"] is not None:
                cache["bar"].SetVisibility(False)
            if cache["grid"] is None:
                plotter.view_isometric()
                plotter.reset_camera()
            else:
                plotter.camera_position = camera
            cache.update(frame=index, grid=grid, choices=choices)
            with state:
                groups = list(dict.fromkeys((a, n) for a, n, _ in choices))
                state.quantity_items = [{"title": "Solid color", "value": -1}] + [
                    {"title": f"{n} · {'Elements' if a == 'cell' else 'Nodes'}", "value": i}
                    for i, (a, n) in enumerate(groups)]
                state.quantity = groups.index(choices[selected][:2]) if selected >= 0 else -1
                state.component_items = [
                    {"title": component_label(grid, choice), "value": i}
                    for i, choice in enumerate(choices)
                    if selected >= 0 and choice[:2] == choices[selected][:2]]
                state.field_items = [{"title": "Solid color", "value": -1}] + [
                    {"title": f"{name}{'' if component is None else '['+str(component)+']'} · {association}",
                     "value": i} for i, (association, name, component) in enumerate(choices)]
                state.field = selected
                time = info["saved_times"][index] if info["saved_times"] else None
                if float(state.blend) > 0 and time is not None:
                    time += float(state.blend)*(info["saved_times"][index+1]-time)
                position = (f"Interpolated · frames {index+1}–{index+2}" if float(state.blend) > 0
                            else f"Saved frame {index+1}/{info['frame_count']}")
                label = "Time" if state.time_unit else "Coordinate"
                state.coordinate = position + (f" · {label} {time:.6g} {state.time_unit}" if time is not None else "")
                state.station_label = position
                state.time_label = f"{label} {time:.6g} {state.time_unit}" if time is not None else "No saved time coordinate"
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
            if state.smooth:
                duration = float(state.playback_duration)
                if not np.isfinite(duration) or not 1 <= duration <= 60:
                    raise ValueError("Playback duration must be 1–60 seconds.")
                times = info["saved_times"]
                timeline_sample(times, 0.)
                start_index = int(state.frame)
                start_time = times[start_index]
                if start_index + 1 < len(times):
                    start_time += float(state.blend)*(times[start_index+1]-times[start_index])
                start_fraction = (start_time-times[0])/(times[-1]-times[0])
                started = clock.monotonic() - start_fraction*duration
                while state.playing:
                    fraction = min(1., (clock.monotonic()-started)/duration)
                    index, blend, _ = timeline_sample(times, fraction)
                    with state:
                        state.frame = index if fraction < 1 else info["frame_count"]-1
                        state.blend = blend if fraction < 1 else 0.
                    if fraction >= 1:
                        break
                    await asyncio.sleep(.05)
                return
            while state.playing and int(state.frame) < info["frame_count"] - 1:
                await asyncio.sleep(.5)
                if state.playing:
                    with state:
                        state.frame = int(state.frame) + 1
                        state.blend = 0.
        except (ValueError, KeyError, IndexError, OSError) as exc:
            state.error = str(exc)
        finally:
            if asyncio.current_task() is playback["task"]:
                with state:
                    state.playing = False

    def toggle_playback(**_):
        if state.playing:
            stop_playback()
        elif path is not None and info["frame_count"] > 1:
            with state:
                if int(state.frame) == info["frame_count"] - 1:
                    state.frame = 0
                    state.blend = 0.
                elif not state.smooth:
                    state.blend = 0.
                state.playing = True
            playback["task"] = asyncio.create_task(play())

    def seek_frame(**_):
        stop_playback()
        state.blend = 0.
        update()

    def change_smoothing(**_):
        if not state.smooth:
            stop_playback()
            state.blend = 0.

    async def fit_deformation(**_):
        stop_playback()
        if path is None or state.scale_busy:
            return
        source, count, name = path, info["frame_count"], state.displacement_field
        if not state.reference_geometry or not name:
            state.error = "Select reference geometry and displacement before Auto scale."
            return
        state.scale_busy = True
        try:
            metrics = []
            for i in range(count):
                directory.result(source)
                metrics.append(displacement_metrics(read_visualization_frame(source, frame=i), name))
                with state:
                    state.scale_status = f"Sizing saved frame {i+1}/{count}…"
                await asyncio.sleep(0)
                if state.closed or path != source or state.displacement_field != name or not state.reference_geometry:
                    raise ValueError("Result selection changed; Auto scale was not applied.")
            scale = automatic_scale(metrics)
            with state:
                state.scale = scale
                state.scale_status = f"Suggested ×{scale:.3g} from {count} saved frames · fixed during playback"
                state.error = ""
        except (ValueError, KeyError, OSError, ET.ParseError) as exc:
            state.error = str(exc)
        finally:
            state.scale_busy = False

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
            cache.update(frame=None, grid=None, choices=[], next_index=None, next_grid=None)
            with state:
                state.result_name = path.name
                state.loaded_source = str(path.relative_to(directory.root))
                state.frame_count = latest["frame_count"]
                state.frame = latest["frame"]
                state.field = -1
                state.scale = 1.
                state.blend = 0.
                state.smooth = False
                state.scale_status = ""
                state.time_unit = ""
                state.field_unit = ""
                # Coordinate semantics belong to this file, not the previously
                # opened file. Never accidentally warp current coordinates twice.
                state.reference_geometry = False
                state.displacement_field = None
                state.fixed_range = True
                range_owner["choice"] = None
                state.error = ""
            update()
            if not state.error:
                state.open_panel = False
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

    def select_quantity(**_):
        groups = list(dict.fromkeys(choice[:2] for choice in cache["choices"]))
        index = int(state.quantity)
        if index < 0:
            state.field = -1
        elif index < len(groups):
            current = int(state.field)
            if not (0 <= current < len(cache["choices"]) and cache["choices"][current][:2] == groups[index]):
                state.field = next(i for i, choice in enumerate(cache["choices"]) if choice[:2] == groups[index])

    def fit_series_range(**_):
        stop_playback()
        try:
            selected = int(state.field)
            if path is None or selected < 0:
                return
            choice = cache["choices"][selected]
            low, high = float("inf"), float("-inf")
            for i in range(info["frame_count"]):
                frame_grid = read_visualization_frame(path, frame=i)
                values = component_values(frame_grid, choice)
                if not np.isfinite(values).all():
                    raise ValueError("Selected field contains nonfinite values.")
                low, high = min(low, float(values.min())), max(high, float(values.max()))
            if low == high:
                pad = max(abs(low)*.01, 1.e-12)
                low, high = low-pad, high+pad
            if state.color_mode == "zero":
                low, high = centered_range(low, high)
            range_owner["choice"] = choice
            with state:
                state.fixed_range = True
                state.range_min, state.range_max = low, high
                state.error = ""
        except (KeyError, ValueError, IndexError, OSError, ET.ParseError) as exc:
            state.error = str(exc)

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
            cache["next_index"] = None
            with state:
                state.frame_count = info["frame_count"]
                state.frame = index
                state.blend = 0.
            update()
        except (ValueError, IndexError, TypeError, OSError, ET.ParseError) as exc:
            state.error = str(exc)

    def sync_camera(camera=None, **_):
        if camera:
            apply_camera(plotter, camera)

    def camera_record():
        camera = plotter.camera
        return dict(position=list(camera.position), focalPoint=list(camera.focal_point),
                    viewUp=list(camera.up), parallelProjection=bool(camera.parallel_projection),
                    parallelScale=camera.parallel_scale, viewAngle=camera.view_angle)

    def snapshot(camera=None):
        if path is None:
            raise ValueError("Open a result first.")
        if float(state.blend) > 0:
            raise ValueError("Select a saved frame on the slider before saving a still view. Smooth videos retain their interpolation recipe.")
        sync_camera(camera)
        selected = int(state.field)
        view = dict(schema="agentfem.view.v1", source=str(path.relative_to(directory.root)),
                    frame=int(state.frame), choice=list(cache["choices"][selected]) if selected >= 0 else None,
                    range=[float(state.range_min), float(state.range_max)], preset=state.preset,
                    palette=state.palette, color_mode=state.color_mode, scale=float(state.scale), reference_geometry=bool(state.reference_geometry),
                    displacement=state.displacement_field, edges=bool(state.edges), camera=camera_record(),
                    title=state.figure_title, unit=state.field_unit, time_unit=state.time_unit)
        if selected >= 0 and not state.fixed_range:
            raise ValueError("Keep colors consistent before saving or exporting a view.")
        return validate_view(view)

    def list_views():
        items = []
        for p in sorted(export_root.glob("*/view.json"), reverse=True):
            if p.parent.name.startswith(".") or p.is_symlink() or p.parent.is_symlink():
                continue
            try:
                if p.stat().st_size > 1_000_000:
                    continue
                record = validate_view(json.loads(p.read_text()))
                title = record.get("title") or Path(record["source"]).stem
                label = f"{title} · Frame {record['frame'] + 1} · {p.parent.name[-6:]}"
                items.append({"title": label, "value": str(p.relative_to(export_root))})
            except (ValueError, TypeError, KeyError, OSError):
                continue
        state.view_items = items

    def save_view(camera=None, **_):
        try:
            view = snapshot(camera)
            target = export_root / ("view-" + uuid4().hex[:12]) / "view.json"
            atomic_write_text(target, json.dumps(view, indent=2, allow_nan=False))
            state.export_record = "viewer-exports/" + str(target.relative_to(export_root))
            state.export_progress = "View saved. Select it below to restore this scene."
            list_views()
        except (ValueError, TypeError, OSError) as exc:
            state.error = str(exc)

    def restore_view(**_):
        try:
            allowed = {entry["value"] for entry in state.view_items}
            if state.view_selection not in allowed:
                raise ValueError("Select a saved view.")
            selected = (export_root / state.view_selection).resolve(strict=True)
            selected.relative_to(export_root.resolve())
            view_record = validate_view(json.loads(selected.read_text()))
            source = directory.result(view_record["source"])
            display_frame(source, view_record["frame"], view_record)
            grid = read_visualization_frame(source, frame=view_record["frame"])
            choice = tuple(view_record["choice"]) if view_record["choice"] is not None else None
            choices = scalar_choices(grid)
            if choice is not None and choice not in choices:
                raise ValueError("Saved field is no longer available.")
            # Validate before changing the currently displayed source.
            state.file_selection = str(source.relative_to(directory.root))
            open_selection()
            with state:
                state.frame = view_record["frame"]
            with state:
                state.field = choices.index(choice) if choice is not None else -1
                state.preset = view_record["preset"]
                state.reference_geometry = view_record["reference_geometry"]
                state.displacement_field = view_record["displacement"]
                state.scale = view_record["scale"]
                state.edges = view_record["edges"]
                state.figure_title = view_record.get("title", "")
                state.field_unit = view_record.get("unit", "")
                state.time_unit = view_record.get("time_unit", "")
            range_owner["choice"] = choice
            with state:
                state.color_mode = view_record.get("color_mode", "sequential")
            with state:
                state.palette = view_record["palette"]
                if choice is not None:
                    state.range_min, state.range_max = view_record["range"]
            sync_camera(view_record.get("camera"))
            ctrl.view_camera()
            prefix = "viewer-exports/" + str(selected.parent.relative_to(export_root)) + "/"
            state.export_record = prefix + "view.json"
            state.export_image = prefix + "figure.png" if (selected.parent / "figure.png").is_file() else ""
            state.export_video = prefix + "animation.mp4" if (selected.parent / "animation.mp4").is_file() else ""
            state.error = ""
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            state.error = str(exc)

    async def export(mode, camera=None, **_):
        if state.export_busy:
            return
        stop_playback()
        if mode == "process":
            state.blend = 0.
        state.export_busy = True
        try:
            request = dict(root=str(directory.root), view=snapshot(camera), mode=mode,
                           size=[1920, 1080], fps=24, transparent=state.transparent,
                           smooth=bool(state.smooth), duration=float(state.playback_duration))
            def progress(done, total):
                with state:
                    state.export_progress = f"Rendering {done}/{total}"
            with state:
                state.export_progress = "Preparing export…"
                state.error = ""
            target = await job.run(request, progress)
            with state:
                if target is None:
                    state.export_progress = "Export cancelled. No partial file published."
                else:
                    prefix = "viewer-exports/" + target.name + "/"
                    if mode == "image":
                        state.export_image = prefix + "figure.png"
                    else:
                        state.export_video = prefix + "animation.mp4"
                    state.export_record = prefix + "view.json"
                    state.export_progress = "Ready to download · saved in viewer-exports"
                    list_views()
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            state.error = str(exc)
        finally:
            with state:
                state.export_busy = False

    async def close_session(**_):
        stop_playback()
        job.close()
        with state:
            state.closed = True
            state.close_confirmation = False
        await asyncio.sleep(.25)
        await server.stop()

    def change_preset(**_):
        if state.preset in PRESETS:
            state.palette = preferred_palette["zero"] if state.color_mode == "zero" else PRESETS[state.preset]["cmap"]

    def change_color_mode(**_):
        with state:
            if state.color_mode == "zero":
                state.range_min, state.range_max = centered_range(state.range_min, state.range_max)
            state.palette_items = palette_items(state.color_mode)
            state.palette = preferred_palette[state.color_mode]

    def remember_palette(**_):
        if state.palette in {item["value"] for item in palette_items(state.color_mode)}:
            preferred_palette[state.color_mode] = state.palette

    def open_path(**_):
        state.file_selection = state.path_input
        open_selection()

    list_views()

    with SinglePageWithDrawerLayout(server, width=300, show_drawer=False) as ui:
        ui.title.set_text("AgentFEM")
        ui.title.style = "flex:0 0 auto; font-size:16px; margin:0 12px"
        ui.icon.hide()
        ui.drawer.permanent = False
        ui.drawer.temporary = True
        ui.drawer.scrim = False
        with ui.toolbar:
            v3.VBtn("Open", prepend_icon="mdi-folder-open-outline", click="open_panel = true", variant="text")
            v3.VBtn("Display", prepend_icon="mdi-tune", click="main_drawer = !main_drawer", variant="text")
            v3.VSpacer()
            v3.VBtn("Export", prepend_icon="mdi-export-variant", variant="tonal",
                    disabled=("frame_count === 0 && view_items.length === 0",), click="export_panel = true")
            v3.VBtn(icon="mdi-fit-to-screen-outline", title="Fit model to view", click=lambda: ctrl.view_reset())
        with v3.VDialog(v_model=("open_panel", False), max_width=580):
            with v3.VCard(classes="pa-5 d-flex flex-column ga-3", title="Open local results"):
                v3.VSelect(v_model=("project_selection", None), items=("project_items",),
                           label="Project", v_if="project_items.length", hide_details=True,
                           density="compact")
                v3.VSelect(v_model=("run_selection", None), items=("run_items",),
                           label="Run", v_if="project_selection !== null", hide_details=True,
                           density="compact")
                v3.VSelect(v_model=("file_selection", None), items=("file_items",),
                           label="Result or folder", hide_details=True, density="compact")
                with v3.VRow(classes="ma-0 ga-2"):
                    v3.VBtn("{{ file_items.some(item => item.value === file_selection && item.directory) ? 'Browse folder' : 'Show result' }}", click=open_selection, color="primary", variant="tonal", prepend_icon="mdi-folder-open-outline")
                    v3.VBtn("Back one folder", click=parent_folder, disabled=("folder === '.'",), variant="text")
                v3.VLabel("Folder: {{ folder === '.' ? browse_root : browse_root + '/' + folder }}", style="font-size:12px; white-space:normal; overflow-wrap:anywhere")
                v3.VTextField(v_model=("path_input", ""), label="Or paste a result path inside this location",
                              hide_details=True, density="compact", keydown_enter=open_path)
                v3.VBtn("Show path", click=open_path, disabled=("!path_input",), variant="text")
                v3.VLabel("Choose a folder to browse, or a result file to display. XDMF companion files stay on your computer.", style="font-size:12px; white-space:normal")
                v3.VAlert("{{ error }}", v_if="error", type="error", density="compact")
                v3.VBtn("Back to viewer", click="open_panel = false", variant="text")
        with ui.drawer:
            with v3.VContainer(classes="pa-4 d-flex flex-column ga-3"):
                v3.VBtn("Done", prepend_icon="mdi-check", click="main_drawer = false", variant="tonal")
                v3.VLabel("DISPLAY", style="font-size:12px; letter-spacing:1px")
                v3.VDivider()
                v3.VSelect(v_model=("preset", "scientific"),
                           items=("preset_items", [{"title": name.title(), "value": name} for name in PRESETS]), label="Style",
                           hide_details=True, density="compact")
                v3.VSelect(v_model=("quantity", -1), items=("quantity_items",),
                           label="Field", hide_details=True, density="compact")
                v3.VSelect(v_model=("field", -1), items=("component_items",),
                           label="Component", v_if="quantity >= 0", hide_details=True, density="compact")
                v3.VCheckbox(v_model=("edges", False), label="Show mesh edges", hide_details=True, density="compact")
                v3.VSelect(v_model=("color_mode", "sequential"),
                           items=("color_modes", [{"title": "Sequential", "value": "sequential"},
                                                  {"title": "Center on zero", "value": "zero"}]),
                           label="Color scale", hide_details=True, density="compact")
                v3.VCheckbox(v_model=("fixed_range", True), label="Keep colors consistent", hide_details=True, density="compact")
                v3.VBtn("Fit all saved frames", click=fit_series_range, disabled=("field < 0",), variant="text")
                v3.VSelect(v_model=("palette", "ocean"), items=("palette_items",), label="Palette",
                           hide_details=True, density="compact")
                v3.VCheckbox(v_model=("advanced", False), label="Custom limits and units", hide_details=True, density="compact")
                with v3.VRow(v_if="advanced", classes="ma-0 ga-2 flex-nowrap"):
                    v3.VTextField(v_model=("range_min", 0.), label="Minimum", type="number",
                                  v_if="color_mode !== 'zero'", disabled=("!fixed_range",), hide_details=True, density="compact", style="min-width:0")
                    v3.VTextField(v_model=("range_max", 1.), label=("color_mode === 'zero' ? 'Limit ±' : 'Maximum'",), type="number",
                                  disabled=("!fixed_range",), hide_details=True, density="compact", style="min-width:0")
                v3.VDivider()
                v3.VCheckbox(v_model=("reference_geometry", bool(reference_geometry)),
                             label="Deform reference mesh", hide_details=True, density="compact")
                v3.VSelect(v_model=("displacement_field", displacement), items=("displacement_items",),
                           label="Displacement · point vector", disabled=("!reference_geometry",),
                           hide_details=True, density="compact")
                v3.VTextField(v_model=("scale", 1.), label="Deformation scale", type="number",
                              disabled=("!reference_geometry || !displacement_field",), hide_details=True, density="compact")
                with v3.VRow(classes="ma-0 ga-2"):
                    v3.VBtn("Auto scale", click=fit_deformation, variant="tonal", loading=("scale_busy",),
                            disabled=("!reference_geometry || !displacement_field || scale_busy",))
                    v3.VBtn("Actual ×1", click="scale = 1; scale_status = ''", variant="text")
                v3.VLabel("{{ scale_status }}", v_if="scale_status", style="font-size:12px; white-space:normal")
                v3.VLabel("Use undeformed output geometry. Scale 1 shows actual displacement; saved data are unchanged.",
                          style="font-size:12px; white-space:normal")
        with v3.VDialog(v_model=("export_panel", False), max_width=500):
            with v3.VCard(classes="pa-5 d-flex flex-column ga-3", title="Export & saved views"):
                v3.VLabel("1920 × 1080 · fixed colors · {{ smooth ? 'interpolated displacement video' : 'saved states' }}", style="font-size:12px")
                v3.VTextField(v_model=("figure_title", ""), label="Figure title", hide_details=True, density="compact")
                v3.VTextField(v_model=("field_unit", ""), label="Field unit (as recorded)", v_if="advanced", hide_details=True, density="compact")
                v3.VTextField(v_model=("time_unit", ""), label="Time unit (as recorded)", v_if="advanced", hide_details=True, density="compact")
                v3.VCheckbox(v_model=("transparent", False), label="Transparent image", hide_details=True, density="compact")
                v3.VBtn("Save image", prepend_icon="mdi-image-outline", variant="tonal",
                        disabled=("frame_count === 0 || export_busy",),
                        click=(export, "['image', trame.refs.result_view.getCamera()]"))
                v3.VBtn("Save process video", prepend_icon="mdi-movie-open-outline", variant="text",
                        disabled=("frame_count < 2 || export_busy",),
                        click=(export, "['process', trame.refs.result_view.getCamera()]"))
                v3.VBtn("Save turntable video", variant="text", disabled=("frame_count === 0 || export_busy",),
                        click=(export, "['orbit', trame.refs.result_view.getCamera()]"))
                v3.VBtn("Cancel export", v_if="export_busy", click=job.cancel)
                v3.VLabel("{{ export_progress }}", style="white-space:normal; font-size:12px")
                # Blob downloads do not navigate the live websocket page. Direct
                # href downloads cause some embedded WebViews to unload it.
                for label, key, filename, mime in (
                    ("Download image", "export_image", "figure.png", "image/png"),
                    ("Download video", "export_video", "animation.mp4", "video/mp4"),
                    ("Download view settings", "export_record", "view.json", "application/json"),
                ):
                    v3.VBtn(label, v_if=key, variant="text", click=(
                        f"$viewer.download({key}, '{filename}', '{mime}').catch(e => {{ error = e.message; }})"))
                v3.VDivider()
                v3.VBtn("Remember this view", disabled=("frame_count === 0",),
                        click=(save_view, "[trame.refs.result_view.getCamera()]"), variant="text")
                v3.VSelect(v_model=("view_selection", None), items=("view_items",), label="Saved views",
                           hide_details=True, density="compact")
                v3.VBtn("Restore view", disabled=("!view_selection",), click=restore_view, variant="text")
                v3.VBtn("Back to viewer", click="export_panel = false", variant="tonal")
        with ui.content:
            with v3.VContainer(fluid=True, classes="pa-0 d-flex flex-column",
                               style="height:calc(100vh - 96px); min-height:320px"):
                v3.VAlert("{{ error }}", v_if="error", type="error", density="compact")
                v3.VAlert("Viewer closed. Your results and exports are saved; this tab can be closed.", v_if="closed", type="success")
                with v3.VDialog(v_model=("close_confirmation", False), max_width=420):
                    with v3.VCard(title="Close this viewer?", text="This stops this session, including its other tabs and any active export. Saved results and completed exports stay intact."):
                        with v3.VCardActions():
                            v3.VBtn("Keep viewing", click="close_confirmation = false")
                            v3.VBtn("Close viewer", click=close_session)
                # VtkLocalView manages its initial synchronization. Its ready
                # event is not a user change: rebuilding the scene there can
                # form a ready -> update -> ready loop on large meshes.
                view = vtk.VtkLocalView(plotter.ren_win, ref="result_view",
                                        interactor_events=("camera_events", ["EndAnimation"]),
                                        EndAnimation=(sync_camera, "[trame.refs.result_view.getCamera()]"),
                                        reset_camera=(sync_camera, "[trame.refs.result_view.getCamera()]"),
                                        style="flex:1; min-height:0; width:100%")
                ctrl.view_update = view.update
                ctrl.view_reset = view.reset_camera
                ctrl.view_camera = view.push_camera
                with v3.VSheet(classes="px-3 py-2 d-flex flex-column", style="flex:0 0 auto; background:#f5f7f9; color:#243746; font-variant-numeric:tabular-nums"):
                    with v3.VRow(classes="ma-0 align-center ga-2 flex-nowrap", style="width:100%"):
                        v3.VBtn("{{ playing ? 'Pause' : 'Play' }}", click=toggle_playback,
                                disabled=("frame_count <= 1",), variant="tonal", style="min-width:96px",
                                prepend_icon=("playing ? 'mdi-pause' : 'mdi-play'",))
                        v3.VSlider(v_model=("frame", info["frame"]), min=0, max=("frame_count - 1",), step=1,
                                   aria_label="Saved frame", end=seek_frame, disabled=("frame_count <= 1 || playing",),
                                   hide_details=True, style="min-width:0; flex:1")
                        v3.VBtn("Playback", click="playback_panel = true", variant="text", size="small")
                    v3.VLabel("{{ station_label }} · {{ time_label }}", style="font-size:12px; height:24px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis")
                    with v3.VRow(classes="ma-0 align-center ga-2"):
                        v3.VLabel("{{ reference_geometry ? 'Deformation ×' + Number(Number(scale).toPrecision(4)) : 'Saved geometry' }} · {{ fixed_range ? 'Fixed colors' : 'Colors per frame' }}", style="font-size:12px")
                        v3.VSpacer()
                        v3.VBtn("Result details", click="details = !details", variant="text", size="small")
                    v3.VLabel("{{ loaded_source }} · {{ mesh_summary }}", v_if="details", style="font-size:12px; white-space:normal; overflow-wrap:anywhere")
                    v3.VLabel("Coordinates have no declared physical unit. Playback seconds describe the video, not the simulation.",
                              v_if="details && !time_unit", style="font-size:12px; white-space:normal")
                    v3.VLabel("{{ source_status }}", v_if="source_status", style="font-size:12px; white-space:normal")
        with v3.VDialog(v_model=("playback_panel", False), max_width=420):
            with v3.VCard(classes="pa-5 d-flex flex-column ga-3", title="Playback"):
                v3.VCheckbox(v_model=("smooth", False), label="Smooth displacement", disabled=("!smooth_available || playing",), hide_details=True)
                v3.VLabel("Available for displacement or solid color. Damage and stress show saved states only.", style="font-size:13px; white-space:normal")
                v3.VTextField(v_model=("playback_duration", 10.), label="Playback duration (s)", type="number",
                              v_if="smooth", disabled=("playing",), hide_details=True, density="compact")
                v3.VLabel("Display interpolation · not additional computed states", v_if="smooth", style="font-size:12px; white-space:normal")
                v3.VBtn("Done", click="playback_panel = false", variant="tonal")
        with v3.VDialog(v_model=("session_panel", False), max_width=440):
            with v3.VCard(classes="pa-5 d-flex flex-column ga-3", title="Local viewer"):
                v3.VLabel("Closing a browser tab does not delete results. The local service stops after its configured idle timeout once all tabs disconnect (default: 2 minutes).", style="font-size:13px; white-space:normal")
                v3.VBtn("Refresh saved frames", prepend_icon="mdi-refresh", click=refresh, variant="tonal")
                v3.VBtn("Stop viewer service…", click="session_panel = false; close_confirmation = true", variant="text")
                v3.VLabel("Powered by trame / Kitware", style="font-size:12px")
                v3.VBtn("Back to viewer", click="session_panel = false", variant="text")
        ui.footer.clear()
        with ui.footer:
            v3.VLabel("{{ result_name }}", style="font-size:12px; overflow:hidden; text-overflow:ellipsis")
            v3.VSpacer()
            v3.VBtn("Session", click="session_panel = true", variant="text", size="x-small")
    state.change("frame", "field", "scale", "edges", "fixed_range", "range_min", "range_max",
                 "reference_geometry", "displacement_field", "preset", "palette", "blend", "time_unit")(update)
    state.change("preset")(change_preset)
    state.change("color_mode")(change_color_mode)
    state.change("palette")(remember_palette)
    state.change("smooth")(change_smoothing)
    state.change("project_selection")(select_project)
    state.change("run_selection")(select_run)
    state.change("quantity")(select_quantity)
    ctrl.on_server_ready.add(update)
    ctrl.on_server_exited.add(stop_playback)
    ctrl.on_server_exited.add(job.close)
    def close_view(**_):
        # Trame supplies state keyword arguments to lifecycle callbacks.
        plotter.close()
    ctrl.on_server_exited.add(close_view)
    # The controller callbacks are also exercised without an LLM or browser.
    ctrl.open_result = open_selection
    ctrl.open_path = open_path
    ctrl.refresh_result = refresh
    ctrl.parent_folder = parent_folder
    ctrl.select_project = select_project
    ctrl.select_run = select_run
    ctrl.toggle_playback = toggle_playback
    ctrl.stop_playback = stop_playback
    ctrl.fit_series_range = fit_series_range
    ctrl.fit_deformation = fit_deformation
    ctrl.seek_frame = seek_frame
    ctrl.export_result = export
    ctrl.save_view = save_view
    ctrl.restore_view = restore_view
    ctrl.close_session = close_session
    ctrl.cancel_export = job.cancel
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path.cwd(),
                        help="Result file or directory to browse (default: current directory)")
    parser.add_argument("--port", type=int, default=0,
                        help="Local port (default: automatically select an available port)")
    parser.add_argument("--displacement")
    parser.add_argument("--reference-geometry", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=120,
                        help="Stop after this many seconds with no browser connected (default: 120; 0 disables)")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0 (automatic) or between 1 and 65535")
    if bool(args.displacement) != args.reference_geometry:
        parser.error("Use --displacement and --reference-geometry together to enable warp.")
    if args.idle_timeout < 0:
        parser.error("idle-timeout must be nonnegative")
    try:
        app = build_viewer(args.path, displacement=args.displacement,
                           reference_geometry=args.reference_geometry)
    except ImportError as exc:
        parser.error(f"Install the optional viewer: python -m pip install 'agentfem[viewer]' ({exc}).")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    # Let the operating system allocate the listening port atomically. Probing
    # and releasing a free socket first races with other viewer launches.
    app.start(host="127.0.0.1", port=args.port, open_browser=not args.no_browser, timeout=args.idle_timeout)


if __name__ == "__main__":
    main()
