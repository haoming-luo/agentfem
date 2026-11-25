# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, reproducible rendering of saved results; no solver or field recovery.

Runs in a disposable process so rendering/encoding cannot block the viewer and
cancellation can release native VTK resources. Final publication belongs to the
caller, after this process succeeds. A request contains JSON, never Python code.
"""

import json
from pathlib import Path
import sys

import numpy as np

from agentfem._atomic_text import atomic_write_text
from agentfem._viewer_files import ResultDirectory
from agentfem._viewer_motion import timeline_sample, interpolate_displacement, reference_frame
from agentfem._viewer_smoothing import average_cell_colors
from agentfem._viewer_style import legend_panel, scalar_bar, background, draw_axes
from agentfem._viewer_camera import fit_camera, orbit_pose, text_size
from agentfem.results import read_visualization_frame, deform_visualization_frame, inspect_visualization


PRESETS = {
    "scientific": {"background": "#f4f6f8", "foreground": "#263444", "cmap": "ocean", "solid": "#527e91"},
    "publication": {"background": "#ffffff", "foreground": "#243746", "cmap": "ocean", "solid": "#638b92"},
    "presentation": {"background": "#0b1220", "foreground": "#e7edf3", "cmap": "ocean", "solid": "#82bcc4"},
}
# Sequential presentation palettes: pale low values, saturated high values.
# These are our palettes, not a claimed Nature/Apple palette or a calibrated
# perceptually-uniform map. Keep the established scientific maps available.
COLOR_STOPS = {
    "ocean": ("#e7f3f3", "#b9dedc", "#7ac2c5", "#399eae", "#23748f", "#254c78"),
    "iris": ("#f0eef9", "#d2d0ef", "#aca9e1", "#8380cc", "#615cb2", "#43418e"),
    "garnet": ("#f7eef1", "#e9c8d3", "#d99caf", "#bf6d8b", "#984565", "#712d4b"),
    "jade": ("#eef5ef", "#c6dfd3", "#93c3b0", "#5ca58f", "#34836f", "#205c50"),
}
DIVERGING_STOPS = {
    "azure-rose": ("#3d469b", "#5667bf", "#738fd9", "#9eb9e9", "#ccdff1", "#f6f4f5",
                   "#f2cee0", "#eaa0c7", "#d870ab", "#bd4689", "#96326b"),
    "lagoon-rose": ("#126d80", "#218f9b", "#4aafb9", "#83cdd3", "#bfe6e9", "#f5f4f2",
                    "#f4d2e1", "#eeaccb", "#dd7bac", "#bf4e88", "#943465"),
}
SEQUENTIAL_PALETTES = (*COLOR_STOPS, "viridis", "cividis", "magma")
DIVERGING_PALETTES = (*DIVERGING_STOPS, "coolwarm")
PALETTES = (*SEQUENTIAL_PALETTES, *DIVERGING_PALETTES)


def palette_items(mode):
    # Keep a small everyday menu; legacy saved palettes remain supported.
    names = DIVERGING_PALETTES if mode == "zero" else ('ocean', 'viridis', 'cividis', *DIVERGING_PALETTES)
    labels = {"coolwarm": "Blue / Red", "azure-rose": "Azure / Rose", "lagoon-rose": "Lagoon / Rose"}
    return [{"title": labels.get(name, name.title()), "value": name} for name in names]


def centered_range(low, high):
    """Symmetric display limits, including an exactly zero field."""
    if not np.isfinite([low, high]).all():
        raise ValueError("Color limits must be finite.")
    extent = max(abs(float(low)), abs(float(high)), 1.e-12)
    return -extent, extent


def color_map(name):
    """Resolve a stable saved palette name for both interactive and file views."""
    stops = COLOR_STOPS.get(name, DIVERGING_STOPS.get(name))
    if stops is not None:
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(name, stops, N=257)
    return name


def figure_annotations(view, coordinate, index, size):
    """Independent anchors prevent changing time widths from moving other text."""
    if coordinate is None:
        time = f"Frame {index+1}"
    elif view.get("time_unit"):
        time = f"Time {coordinate:.6g} {view['time_unit']}"
    else:
        time = f"Frame {index+1}"
    geometry = f"Deformation ×{view['scale']:g}" if view["reference_geometry"] else "Reference shape"
    return (
        ("time", time, (int(size[0]*.045), int(size[1]*.72))),
        ("geometry", geometry, (int(size[0]*.045), int(size[1]*.68))),
    )


def validate_view(view):
    """Validate a view record before using it, including records restored from disk."""
    if view.get("schema") != "agentfem.view.v1":
        raise ValueError("Unsupported saved view schema.")
    if view.get("preset") not in PRESETS or view.get("palette") not in PALETTES:
        raise ValueError("Select a supported style and palette.")
    if view.get("color_mode", "sequential") not in {"sequential", "zero"}:
        raise ValueError("Select sequential or zero-centered colors.")
    if view.get("color_mode") == "zero" and view["palette"] not in DIVERGING_PALETTES:
        raise ValueError("Zero-centered colors require a diverging palette.")
    if type(view.get("frame")) is not int or view["frame"] < 0:
        raise ValueError("Select a saved frame.")
    if not isinstance(view.get("source"), str):
        raise ValueError("A saved view needs a source result.")
    if type(view.get("reference_geometry")) is not bool or type(view.get("edges")) is not bool:
        raise ValueError("Geometry and edge settings must be boolean.")
    if not np.isfinite(view.get("scale", np.nan)) or abs(view["scale"]) > 1.e6:
        raise ValueError("Deformation scale must be finite and bounded.")
    if view["reference_geometry"] and not isinstance(view.get("displacement"), str):
        raise ValueError("Choose the displacement field before enabling deformation.")
    choice = view.get("choice")
    if type(view.get("cell_smoothing", False)) is not bool:
        raise ValueError("Cell display averaging must be boolean.")
    if view.get('smoothing_method', 'average') not in {'average', 'preserve_jumps', 'edge_aware'}:
        raise ValueError('Choose Average color smoothing.')
    if 'smoothing_method' in view:
        view = dict(view, smoothing_method='average')
    if not isinstance(view.get("smoothing_region", ""), str):
        raise ValueError("A display region must be a field name.")
    if choice is not None:
        if (not isinstance(choice, list) or len(choice) != 3 or choice[0] not in {"cell", "point"}
                or not isinstance(choice[1], str)):
            raise ValueError("Invalid field association or component.")
        component = choice[2]
        if not (component is None or component == "magnitude" or
                (type(component) is int and component >= 0)):
            raise ValueError("Invalid field component.")
        bounds = np.asarray(view.get("range"), dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Export needs a finite fixed color range.")
        if view.get("color_mode") == "zero" and bounds[0] != -bounds[1]:
            raise ValueError("Zero-centered colors require symmetric limits.")
    for name in ("title", "unit", "time_unit"):
        if not isinstance(view.get(name, ""), str) or len(view.get(name, "")) > 160:
            raise ValueError("Titles and units must be short text.")
    camera = view.get("camera")
    if camera is not None:
        for name in ("position", "focalPoint", "viewUp"):
            vector = np.asarray(camera.get(name), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("Invalid camera coordinates.")
        direction = np.asarray(camera["position"]) - camera["focalPoint"]
        if np.linalg.norm(np.cross(direction, camera["viewUp"])) < 1.e-12:
            raise ValueError("Camera direction and up vector must be independent.")
        for name in ("parallelScale", "viewAngle"):
            if not np.isfinite(camera.get(name, np.nan)) or camera[name] <= 0:
                raise ValueError("Invalid camera scale.")
    return view


def apply_camera(plotter, camera):
    if camera is not None:
        plotter.camera_position = [camera["position"], camera["focalPoint"], camera["viewUp"]]
        plotter.camera.parallel_projection = bool(camera.get("parallelProjection", False))
        plotter.camera.parallel_scale = camera["parallelScale"]
        plotter.camera.view_angle = camera["viewAngle"]
        plotter.reset_camera_clipping_range()


def display_frame(source, index, view, *, grid=None, right=None, alpha=0.):
    from agentfem.viewer import component_values, scalar_choices
    grid = read_visualization_frame(source, frame=index) if grid is None else grid
    if view.get("displacement"):
        grid = reference_frame(grid, view["displacement"], view.get("current_coordinates", False))
        if right is not None:
            right = reference_frame(right, view["displacement"], view.get("current_coordinates", False))
    if right is not None:
        if not view["reference_geometry"]:
            raise ValueError("Smooth display requires declared reference geometry and displacement.")
        grid = interpolate_displacement(grid, right, view["displacement"], alpha, view["choice"])
    if view["reference_geometry"]:
        grid = deform_visualization_frame(grid, view["displacement"], scale=view["scale"], reference_geometry=True)
    choice = tuple(view["choice"]) if view["choice"] is not None else None
    scalar = None
    if choice is not None:
        if choice not in scalar_choices(grid):
            raise ValueError(f"Selected field/component is unavailable at frame {index + 1}.")
        scalar = component_values(grid, choice)
        if not np.isfinite(scalar).all():
            raise ValueError("Cannot render nonfinite field values.")
        if view.get("cell_smoothing", False) and choice[0] == "cell":
            grid, scalar = average_cell_colors(grid, scalar, view.get("smoothing_region") or None, view.get('smoothing_method', 'average'))
    return grid, scalar


def render(request, destination, progress=None):
    """Render PNG or MP4 to a caller-owned staging directory, one frame at a time."""
    import pyvista as pv
    from agentfem.viewer import component_label

    view = validate_view(request["view"])
    averaged = bool(view.get("cell_smoothing") and view["choice"] and view["choice"][0] == "cell")
    directory = ResultDirectory(request["root"])
    source = directory.result(view["source"])
    info = inspect_visualization(source)
    if view["frame"] >= info["frame_count"]:
        raise ValueError("Saved frame is no longer available.")
    mode = request.get("mode", "image")
    if mode not in {"image", "process", "orbit"}:
        raise ValueError("Unknown export mode.")
    size = request.get("size", [1920, 1080])
    if (len(size) != 2 or any(type(n) is not int or not 256 <= n <= 3840 or n % 2 for n in size)):
        raise ValueError("Use even image dimensions from 256 to 3840 pixels.")
    if mode == "process" and info["frame_count"] > 500:
        raise ValueError("Interactive export supports up to 500 saved frames per video.")
    frames = list(range(info["frame_count"])) if mode == "process" else [view["frame"]]
    framing = request.get("framing", "fit")
    motion = request.get("camera_motion", "showcase")
    if framing not in {"fit", "current"} or motion not in {"showcase", "turntable"}:
        raise ValueError("Select valid framing and camera motion.")
    fps = request.get("fps", 24)
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("Video frame rate must be between 1 and 60.")
    if mode == 'orbit':
        frames = [view['frame']]*(fps*10)
    smooth = request.get("smooth", False) and mode == "process"
    samples = None
    if smooth:
        duration = float(request.get("duration", 10.))
        if not np.isfinite(duration) or not 1 <= duration <= 60:
            raise ValueError("Video duration must be between 1 and 60 seconds.")
        samples = [timeline_sample(info["saved_times"], f)
                   for f in np.linspace(0., 1., max(2, round(fps*duration)))]
        frames = [sample[0] for sample in samples]
    transparent = bool(request.get("transparent", False)) and mode == "image"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    preset = PRESETS[view["preset"]]
    plotter = pv.Plotter(off_screen=True, window_size=size)
    legend = legend_panel(plotter)
    movie = mode != "image"
    output = destination / ("animation.mp4" if movie else "figure.png")
    last_index, actor = None, None
    raw_frames = {}
    try:
        bounds = None
        if framing == "fit":
            indices = list(range(info['frame_count'])) if mode == 'process' else [view['frame']]
            for index in indices:
                candidate, _ = display_frame(source, index, view)
                box = np.asarray(candidate.bounds).reshape(3, 2)
                if bounds is None:
                    bounds = box.copy()
                else:
                    bounds[:, 0] = np.minimum(bounds[:, 0], box[:, 0])
                    bounds[:, 1] = np.maximum(bounds[:, 1], box[:, 1])
        background(plotter, legend, preset)
        plotter.enable_anti_aliasing("ssaa")
        if movie:
            plotter.open_movie(output, framerate=fps, quality=9, macro_block_size=2)
        for step, index in enumerate(frames):
            if index != last_index or smooth:
                # Revalidate XML references each read, including a live index.
                directory.result(source)
                if smooth:
                    for i in (index, index+1):
                        if i not in raw_frames:
                            raw_frames[i] = read_visualization_frame(source, frame=i)
                    raw_frames = {i: raw_frames[i] for i in (index, index+1)}
                    grid, scalar = display_frame(source, index, view, grid=raw_frames[index],
                                                 right=raw_frames[index+1], alpha=samples[step][1])
                else:
                    grid, scalar = display_frame(source, index, view)
                # Geometry is untouched; cell color averaging is explicit and recorded.
                kwargs = dict(color=preset["solid"], show_edges=view["edges"],
                              show_scalar_bar=False, ambient=.55, diffuse=.45, specular=0.,
                              reset_camera=False)
                if scalar is not None:
                    kwargs.update(scalars=scalar, preference="point" if averaged else view["choice"][0],
                                  cmap=color_map(view["palette"]), clim=view["range"])
                if actor is None:
                    actor = plotter.add_mesh(grid, **kwargs)
                else:
                    actor.mapper.dataset = grid
                    if scalar is not None:
                        actor.mapper.set_scalars(scalar, "__view_scalar__", preference="point" if averaged else view["choice"][0],
                                                 cmap=color_map(view["palette"]), clim=view["range"])
                if step == 0:
                    plotter.view_isometric()
                    plotter.reset_camera()
                    apply_camera(plotter, view.get("camera"))
                    if bounds is not None:
                        fit_camera(plotter.camera, bounds, size[0]/size[1])
                    base_camera = (tuple(plotter.camera.position), tuple(plotter.camera.focal_point), tuple(plotter.camera.up))
                    base_parallel_scale = plotter.camera.parallel_scale
                    if scalar is not None:
                        name = view["choice"][1].replace("_", " ")
                        name = name[:1].upper() + name[1:]
                        label_grid = read_visualization_frame(source, frame=index) if averaged else grid
                        component = component_label(label_grid, tuple(view["choice"]))
                        unit = f" [{view['unit']}]" if view.get("unit") else ""
                        label = f"{name}{'' if component == 'Value' else ' / '+component}{unit}"
                        if averaged:
                            label += " · averaged display"
                        scalar_bar(plotter, legend, label, actor.mapper, preset['foreground'])
                elif scalar is not None:
                    for bar in plotter.scalar_bars.values():
                        bar.SetLookupTable(actor.mapper.lookup_table)
                last_index = index
            if mode == "orbit":
                plotter.camera_position = orbit_pose(*base_camera, step/max(1, len(frames)-1), motion)
                if plotter.camera.parallel_projection:
                    plotter.camera.parallel_scale = base_parallel_scale*(
                        np.linalg.norm(np.asarray(plotter.camera.position)-plotter.camera.focal_point)/
                        np.linalg.norm(np.asarray(base_camera[0])-base_camera[1]))
            plotter.reset_camera_clipping_range()
            draw_axes(plotter, view['preset'] == 'presentation')
            title = view.get("title") or ""
            plotter.add_text(title, position=(int(size[0]*.045), int(size[1]*.78)),
                             font_size=text_size(24, size), color=preset["foreground"], font="arial", name="title")
            coordinate = info["saved_times"][index] if info["saved_times"] else None
            if smooth:
                coordinate = samples[step][2]
            for name, label, position in figure_annotations(view, coordinate, index, size):
                plotter.add_text(label, position=position, font_size=text_size(18, size),
                                 color=preset["foreground"], font="arial", name=name)
            if smooth:
                plotter.add_text("Interpolated display", position=(int(size[0]*.045), int(size[1]*.64)),
                                 font_size=text_size(11, size), color=preset["foreground"], font="arial", name="interpolation")
            if movie:
                # Actual saved states only. Holding a frame gives a readable
                # cadence without fabricating interpolated damage evolution.
                repeats = max(1, round(fps * .4)) if mode == "process" and not smooth else 1
                for _ in range(repeats):
                    plotter.write_frame()
            else:
                plotter.screenshot(output, transparent_background=transparent)
            if progress:
                progress(step + 1, len(frames))
        record = dict(view)
        record.update(export={"mode": mode, "size": size, "fps": fps if movie else None,
                      "framing": framing, "camera_motion": motion if mode == 'orbit' else None,
                      "framing_bounds": bounds.tolist() if bounds is not None else None,
                      "saved_frames": frames, "saved_times": info["saved_times"],
                      "interpolation": "linear_displacement" if smooth else "none",
                      "samples": [{"left": i, "right": i+1, "weight": a, "coordinate": t}
                                  for i, a, t in samples] if samples else None,
                      "seconds_per_saved_state": max(1, round(fps*.4))/fps if mode == "process" and not smooth else None,
                      "transparent": transparent, "vtk_version": pv.vtk_version_info.__str__()})
        atomic_write_text(destination / "view.json", json.dumps(record, indent=2, allow_nan=False))
    finally:
        plotter.close()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Renderer produced no output.")
    return output


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    def progress(done, total):
        atomic_write_text(request_path.parent / "progress.json", json.dumps({"done": done, "total": total}))
    render(request, request_path.parent / "result", progress)


if __name__ == "__main__":
    main()
