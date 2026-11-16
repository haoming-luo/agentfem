# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional local result viewer; consumes saved artifacts, never executes a case."""

import argparse
from pathlib import Path

import numpy as np

from agentfem.results import (
    deform_visualization_frame, inspect_visualization, read_visualization_frame,
)


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


def build_viewer(path, *, displacement=None, reference_geometry=False):
    import pyvista as pv
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import vtk, vuetify3 as v3

    path = Path(path)
    info = inspect_visualization(path)
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("#f3f5f7")
    cache = {"frame": None, "grid": None, "choices": []}
    state.frame = info["frame"]
    state.field = -1
    state.scale = 0.
    state.edges = False
    state.error = ""
    state.field_items = [{"title": "Solid color", "value": -1}]
    state.coordinate = ""
    state.source_status = ""
    state.frame_count = info["frame_count"]
    warp_enabled = reference_geometry and displacement is not None

    def update(**_):
        try:
            index = int(state.frame)
            changed = cache["frame"] != index
            grid = read_visualization_frame(path, frame=index) if changed else cache["grid"]
            choices = scalar_choices(grid)
            selected = int(state.field)
            if changed:
                # Changing schema must not silently reuse an index for another field.
                previous = cache["choices"][selected] if 0 <= selected < len(cache["choices"]) else None
                selected = choices.index(previous) if previous in choices else -1
            if not -1 <= selected < len(choices):
                raise ValueError("Select a field available in this frame.")
            displayed = (deform_visualization_frame(grid, displacement,
                         scale=float(state.scale), reference_geometry=True)
                         if warp_enabled else grid.copy(deep=True))
            options = {"color": "#648aa6"}
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
                options = {"scalars": key, "preference": association, "cmap": "viridis",
                           "scalar_bar_args": {"title": f"{label} ({association})"}}
            camera = plotter.camera_position
            plotter.clear()
            plotter.add_mesh(displayed, show_edges=bool(state.edges), **options)
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
                notices = [f"{item['name']}: {item['value']}" for item in info["information"]
                           if item["name"] in {"status", "agentfem_output_status", "source_status", "temporal_coverage"}]
                state.source_status = " · ".join(notices)
                state.error = ""
            ctrl.view_update()
        except (ValueError, IndexError, TypeError, OSError) as exc:
            state.error = str(exc)

    def refresh(**_):
        # Reopen only the atomically published index; never inspect a writer's
        # temporary files or interpolate across a restart gap.
        try:
            latest = inspect_visualization(path)
            current_time = (info["saved_times"][int(state.frame)]
                            if info["saved_times"] else None)
            index = (latest["saved_times"].index(current_time)
                     if current_time in latest["saved_times"] else latest["frame"])
            info.update(latest)
            cache["frame"] = None
            with state:
                state.frame_count = info["frame_count"]
                state.frame = index
            update()
        except (ValueError, IndexError, TypeError, OSError) as exc:
            state.error = str(exc)

    with SinglePageLayout(server) as ui:
        ui.title.set_text(f"AgentFEM · {path.name}")
        with ui.toolbar:
            v3.VSpacer()
            v3.VBtn("Refresh frames", click=refresh)
            v3.VBtn("Reset view", click=lambda: ctrl.view_reset())
        with ui.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height d-flex flex-column"):
                with v3.VRow(classes="ma-0 pa-3", style="flex: 0 0 auto; width:100%"):
                    with v3.VCol(cols=12, md=4):
                        v3.VSelect(v_model=("field", -1), items=("field_items",), label="Field · association", hide_details=True)
                    with v3.VCol(cols=8, md=4):
                        v3.VSlider(v_model=("frame", info["frame"]), min=0,
                                   max=("frame_count - 1",), step=1, label="Saved frame",
                                   disabled=("frame_count <= 1",), hide_details=True)
                    with v3.VCol(cols=4, md=2):
                        v3.VTextField(v_model=("scale", 0.), label="Warp scale", type="number",
                                      disabled=not warp_enabled, hide_details=True)
                    with v3.VCol(cols=12, md=2):
                        v3.VCheckbox(v_model=("edges", False), label="Edges", hide_details=True)
                v3.VAlert("{{ error }}", v_if="error", type="error", density="compact")
                v3.VLabel("{{ coordinate }}", classes="px-4")
                v3.VLabel("{{ source_status }}", v_if="source_status", classes="px-4")
                view = vtk.VtkLocalView(plotter.ren_win, style="flex: 1; min-height:400px; width:100%")
                ctrl.view_update = view.update
                ctrl.view_reset = view.reset_camera
                v3.VLabel("Local preview · original field association · no averaging · range per frame", classes="px-4 pb-2")
    state.change("frame", "field", "scale", "edges")(update)
    ctrl.on_server_ready.add(update)
    ctrl.on_server_exited.add(plotter.close)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--displacement")
    parser.add_argument("--reference-geometry", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if bool(args.displacement) != args.reference_geometry:
        parser.error("Use --displacement and --reference-geometry together to enable warp.")
    try:
        app = build_viewer(args.path, displacement=args.displacement,
                           reference_geometry=args.reference_geometry)
    except ImportError as exc:
        parser.error(f"Viewing requires optional pyvista, trame, trame-vtk and trame-vuetify ({exc}).")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    app.start(host="127.0.0.1", port=args.port, open_browser=not args.no_browser)
