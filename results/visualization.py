"""Optional, reproducible visual checks for simulation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_deformation_comparison(
    undeformed_path,
    deformed_path,
    output_path,
    *,
    scalar: str = "DisplacementMagnitude",
) -> Path:
    """Render side-by-side undeformed/deformed surfaces with PyVista."""

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PNG deformation rendering requires the optional dependency `pyvista`."
        ) from exc

    undeformed = pv.read(str(undeformed_path)).extract_surface(
        algorithm="dataset_surface"
    )
    deformed = pv.read(str(deformed_path)).extract_surface(
        algorithm="dataset_surface"
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        shape=(1, 2),
        off_screen=True,
        window_size=(1800, 800),
        border=False,
    )
    plotter.subplot(0, 0)
    plotter.add_text("Imported C3D10H mesh", font_size=13)
    plotter.add_mesh(
        undeformed,
        color="#d9e4f5",
        show_edges=True,
        edge_color="#334155",
        line_width=0.35,
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.subplot(0, 1)
    plotter.add_text("Deformed mesh (scale = 1)", font_size=13)
    plotter.add_mesh(
        deformed,
        scalars=scalar,
        cmap="viridis",
        show_edges=True,
        edge_color="#253247",
        line_width=0.25,
        scalar_bar_args={"title": "|u|"},
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.link_views()
    plotter.show(screenshot=str(output), auto_close=True)
    return output


def render_deformation_animation(
    undeformed_path,
    snapshots,
    nodes,
    output_path,
    *,
    fps: int = 2,
) -> Path:
    """Render scale-one deformation history as GIF or MP4."""

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Deformation animation requires the optional dependency `pyvista`."
        ) from exc
    from ..mesh.abaqus import displacement_in_source_order

    selected = tuple(snapshots)
    if len(selected) < 2:
        raise ValueError("Animation requires at least the initial and final snapshots.")
    if int(fps) <= 0:
        raise ValueError("Animation fps must be positive.")
    output = Path(output_path)
    if output.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("Animation output must use .gif or .mp4.")
    output.parent.mkdir(parents=True, exist_ok=True)
    grid = pv.read(str(undeformed_path))
    original_points = np.asarray(grid.points).copy()
    histories = [
        displacement_in_source_order(item.solution, nodes)
        for item in selected
    ]
    magnitudes = [np.linalg.norm(values, axis=1) for values in histories]
    maximum = max(float(np.max(values)) for values in magnitudes)
    grid.point_data["Displacement"] = histories[0]
    grid.point_data["DisplacementMagnitude"] = magnitudes[0]
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
    plotter.set_background("white")
    plotter.add_mesh(
        grid,
        scalars="DisplacementMagnitude",
        cmap="viridis",
        clim=(0.0, maximum if maximum > 0.0 else 1.0),
        show_edges=True,
        edge_color="#253247",
        line_width=0.25,
        scalar_bar_args={"title": "|u|"},
    )
    plotter.add_axes()
    plotter.view_isometric()
    gif_frames = []
    if output.suffix.lower() == ".mp4":
        plotter.open_movie(str(output), framerate=int(fps), quality=7)
    for snapshot, displacement, magnitude in zip(
        selected,
        histories,
        magnitudes,
    ):
        grid.points = original_points + displacement
        grid.point_data["Displacement"] = displacement
        grid.point_data["DisplacementMagnitude"] = magnitude
        plotter.add_text(
            f"load factor = {snapshot.load_factor:.3f}",
            position="upper_left",
            font_size=13,
            color="black",
            name="load_factor",
        )
        if output.suffix.lower() == ".gif":
            frame = plotter.screenshot(return_img=True)
            gif_frames.append(frame)
        else:
            plotter.write_frame()
    plotter.close()
    if gif_frames:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "GIF deformation output requires the optional dependency `Pillow`."
            ) from exc
        images = [Image.fromarray(frame) for frame in gif_frames]
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / int(fps))),
            loop=0,
        )
    return output


def render_vtk_series_animation(
    frame_paths,
    output_path,
    *,
    scalar: str = "UMAG",
    fps: int = 2,
) -> Path:
    """Render a GIF directly from a combined-field deformed VTU series."""

    try:
        import pyvista as pv
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "VTK-series animation requires optional `pyvista` and `Pillow`."
        ) from exc
    selected = tuple(Path(path) for path in frame_paths)
    if len(selected) < 2:
        raise ValueError("VTK-series animation requires at least two frames.")
    output = Path(output_path)
    if output.suffix.lower() != ".gif":
        raise ValueError("render_vtk_series_animation currently writes GIF.")
    grids = [pv.read(path) for path in selected]
    arrays = []
    for grid in grids:
        if scalar in grid.point_data:
            arrays.append(np.asarray(grid.point_data[scalar]))
        elif scalar in grid.cell_data:
            arrays.append(np.asarray(grid.cell_data[scalar]))
        else:
            raise KeyError(f"Scalar {scalar!r} is absent from {selected[0]}.")
    clim = (
        min(float(np.min(values)) for values in arrays),
        max(float(np.max(values)) for values in arrays),
    )
    images = []
    for index, grid in enumerate(grids):
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
        plotter.set_background("white")
        plotter.add_mesh(
            grid,
            scalars=scalar,
            clim=clim,
            cmap="viridis",
            show_edges=True,
            edge_color="#253247",
            line_width=0.25,
            scalar_bar_args={"title": scalar},
        )
        plotter.add_text(
            f"frame {index} / {len(grids) - 1}",
            position="upper_left",
            font_size=13,
            color="black",
        )
        plotter.add_axes()
        plotter.view_isometric()
        images.append(Image.fromarray(plotter.screenshot(return_img=True)))
        plotter.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / int(fps))),
        loop=0,
    )
    return output


def render_unified_xdmf_animation(
    xdmf_path,
    output_path,
    *,
    scalar: str = "UMAG",
    fps: int = 2,
) -> Path:
    """Render a GIF or MP4 from AgentFEM's single XDMF/HDF5 series."""

    from .output import read_unified_xdmf_series

    try:
        from PIL import Image
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Unified-XDMF animation requires optional `pyvista` and `Pillow`."
        ) from exc
    grids = read_unified_xdmf_series(xdmf_path)
    if len(grids) < 2:
        raise ValueError("Unified-XDMF animation requires at least two frames.")
    output = Path(output_path)
    suffix = output.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("Unified-XDMF animation must use .gif or .mp4.")
    arrays = []
    for grid in grids:
        if scalar in grid.point_data:
            arrays.append(np.asarray(grid.point_data[scalar]))
        elif scalar in grid.cell_data:
            arrays.append(np.asarray(grid.cell_data[scalar]))
        else:
            raise KeyError(f"Scalar {scalar!r} is absent from {xdmf_path}.")
    clim = (
        min(float(np.min(values)) for values in arrays),
        max(float(np.max(values)) for values in arrays),
    )
    images = []
    for index, grid in enumerate(grids):
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
        plotter.set_background("white")
        plotter.add_mesh(
            grid,
            scalars=scalar,
            clim=clim,
            cmap="viridis",
            show_edges=True,
            edge_color="#253247",
            line_width=0.25,
            scalar_bar_args={"title": scalar},
        )
        load_factor = float(grid.field_data["load_factor"][0])
        plotter.add_text(
            f"load factor = {load_factor:.3f}",
            position="upper_left",
            font_size=13,
            color="black",
        )
        plotter.add_axes()
        plotter.view_isometric()
        images.append(Image.fromarray(plotter.screenshot(return_img=True)))
        plotter.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / int(fps))),
            loop=0,
        )
    else:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                "MP4 output requires optional `imageio` with an ffmpeg backend."
            ) from exc
        iio.imwrite(
            output,
            np.stack([np.asarray(image) for image in images]),
            fps=int(fps),
        )
    return output


def render_unified_xdmf_comparison(
    xdmf_path,
    output_path,
    *,
    scalar: str = "UMAG",
) -> Path:
    """Render the first and final grids from a unified XDMF series."""

    from .output import read_unified_xdmf_series

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Unified-XDMF comparison requires optional `pyvista`."
        ) from exc
    grids = read_unified_xdmf_series(xdmf_path)
    if len(grids) < 2:
        raise ValueError("Unified-XDMF comparison requires at least two frames.")
    first = grids[0]
    final = grids[-1]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        shape=(1, 2),
        off_screen=True,
        window_size=(1800, 800),
        border=False,
    )
    plotter.subplot(0, 0)
    plotter.add_text("Initial configuration", font_size=13)
    plotter.add_mesh(
        first,
        color="#d9e4f5",
        show_edges=True,
        edge_color="#334155",
        line_width=0.35,
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.subplot(0, 1)
    plotter.add_text("Final configuration (scale = 1)", font_size=13)
    plotter.add_mesh(
        final,
        scalars=scalar,
        cmap="viridis",
        show_edges=True,
        edge_color="#253247",
        line_width=0.25,
        scalar_bar_args={"title": scalar},
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.link_views()
    plotter.show(screenshot=str(output), auto_close=True)
    return output
