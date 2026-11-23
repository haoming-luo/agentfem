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
from agentfem.results import read_visualization_frame, deform_visualization_frame, inspect_visualization


PRESETS = {
    "scientific": {"background": "#f4f6f8", "foreground": "#263444", "cmap": "viridis", "solid": "#527e91"},
    "publication": {"background": "#ffffff", "foreground": "#243746", "cmap": "cividis", "solid": "#638b92"},
    "presentation": {"background": "#121e2b", "foreground": "#e7edf3", "cmap": "magma", "solid": "#82bcc4"},
}
PALETTES = ("viridis", "cividis", "magma", "coolwarm")


def validate_view(view):
    """Validate a view record before using it, including records restored from disk."""
    if view.get("schema") != "agentfem.view.v1":
        raise ValueError("Unsupported saved view schema.")
    if view.get("preset") not in PRESETS or view.get("palette") not in PALETTES:
        raise ValueError("Select a supported style and palette.")
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


def display_frame(source, index, view):
    from agentfem.viewer import component_values, scalar_choices
    grid = read_visualization_frame(source, frame=index)
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
    return grid, scalar


def render(request, destination, progress=None):
    """Render PNG or MP4 to a caller-owned staging directory, one frame at a time."""
    import pyvista as pv
    from agentfem.viewer import component_label

    view = validate_view(request["view"])
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
    if mode == "orbit":
        frames = [view["frame"]] * 120
    fps = request.get("fps", 24)
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("Video frame rate must be between 1 and 60.")
    transparent = bool(request.get("transparent", False)) and mode == "image"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    preset = PRESETS[view["preset"]]
    plotter = pv.Plotter(off_screen=True, window_size=size)
    movie = mode != "image"
    output = destination / ("animation.mp4" if movie else "figure.png")
    last_index, actor = None, None
    try:
        plotter.set_background(preset["background"])
        plotter.enable_anti_aliasing("ssaa")
        if movie:
            plotter.open_movie(output, framerate=fps, quality=8, macro_block_size=2)
        for step, index in enumerate(frames):
            if index != last_index:
                # Revalidate XML references each read, including a live index.
                directory.result(source)
                grid, scalar = display_frame(source, index, view)
                # Do not average cells or smooth geometry to beautify output.
                kwargs = dict(color=preset["solid"], show_edges=view["edges"],
                              show_scalar_bar=False, ambient=.32, diffuse=.68, specular=.08,
                              reset_camera=False)
                if scalar is not None:
                    kwargs.update(scalars=scalar, preference=view["choice"][0],
                                  cmap=view["palette"], clim=view["range"])
                if actor is None:
                    actor = plotter.add_mesh(grid, **kwargs)
                else:
                    actor.mapper.dataset = grid
                    if scalar is not None:
                        actor.mapper.set_scalars(scalar, "__view_scalar__", preference=view["choice"][0],
                                                 cmap=view["palette"], clim=view["range"])
                if step == 0:
                    plotter.view_isometric()
                    plotter.reset_camera()
                    apply_camera(plotter, view.get("camera"))
                    if scalar is not None:
                        name = view["choice"][1]
                        component = component_label(grid, tuple(view["choice"]))
                        unit = f" [{view['unit']}]" if view.get("unit") else ""
                        label = f"{name}{'' if component == 'Value' else ' / '+component}{unit}"
                        plotter.add_text(label, position=(int(size[0]*.25), int(size[1]*.155)),
                                         font_size=15, color=preset["foreground"], font="arial", name="field_label")
                        plotter.add_scalar_bar("", mapper=actor.mapper, color=preset["foreground"],
                            vertical=False, position_x=.25, position_y=.065, width=.50, height=.065,
                            title_font_size=18, label_font_size=15, n_labels=5, fmt="%.3g",
                            font_family="arial", bold=False, shadow=False)
                elif scalar is not None:
                    for bar in plotter.scalar_bars.values():
                        bar.SetLookupTable(actor.mapper.lookup_table)
                last_index = index
            if mode == "orbit" and step:
                plotter.camera.Azimuth(360 / len(frames))
                plotter.reset_camera_clipping_range()
            title = view.get("title") or source.stem.replace("_", " ")
            plotter.add_text(title, position=(int(size[0]*.045), int(size[1]*.92)),
                             font_size=24, color=preset["foreground"], font="arial", name="title")
            coordinate = info["saved_times"][index] if info["saved_times"] else None
            if coordinate is None:
                time = f"Frame {index+1}"
            elif view.get("time_unit"):
                time = f"Time {coordinate:.6g} {view['time_unit']}"
            else:
                time = f"Saved coordinate {coordinate:.6g}"
            geometry = f"Deformation ×{view['scale']:g}" if view["reference_geometry"] else "Saved geometry"
            subtitle = f"{time}   /   {geometry}   /   {'Element values' if view.get('choice') and view['choice'][0] == 'cell' else 'Nodal values' if view.get('choice') else 'Geometry'}"
            plotter.add_text(subtitle, position=(int(size[0]*.045), int(size[1]*.875)),
                             font_size=13, color=preset["foreground"], font="arial", name="subtitle")
            if movie:
                # Actual saved states only. Holding a frame gives a readable
                # cadence without fabricating interpolated damage evolution.
                repeats = max(1, round(fps * .4)) if mode == "process" else 1
                for _ in range(repeats):
                    plotter.write_frame()
            else:
                plotter.screenshot(output, transparent_background=transparent)
            if progress:
                progress(step + 1, len(frames))
        record = dict(view)
        record.update(export={"mode": mode, "size": size, "fps": fps if movie else None,
                      "saved_frames": frames, "saved_times": info["saved_times"],
                      "interpolation": "none", "seconds_per_saved_state": max(1, round(fps*.4))/fps if mode == "process" else None,
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
