# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only, optional VTK inspection of saved visualization frames."""

from numbers import Integral, Real
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _reader(path):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("Result viewing requires optional pyvista: install agentfem[visualization].") from exc
    source = Path(path)
    if source.suffix.lower() not in {".xdmf", ".xmf", ".vtu", ".vtk", ".pvd"}:
        raise ValueError("Select a local XDMF, VTU, VTK or PVD visualization artifact.")
    if not source.is_file():
        raise FileNotFoundError(source)
    reader = pv.get_reader(source, force_ext=".xdmf" if source.suffix.lower() == ".xmf" else None)
    times = tuple(float(value) for value in getattr(reader, "time_values", ()))
    if times and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Saved frame times must be finite and strictly increasing.")
    return pv, reader, times


def _read(path, frame):
    if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, Integral):
        raise ValueError("frame must be an integer saved-frame index.")
    pv, reader, times = _reader(path)
    count = len(times) or 1
    index = int(frame) if frame >= 0 else count+int(frame)
    if not 0 <= index < count:
        raise IndexError(f"Saved-frame index {frame} is outside {count} available frames.")
    if times:
        reader.set_active_time_point(index)
    grid = reader.read()
    while isinstance(grid, pv.MultiBlock):
        if len(grid) != 1 or grid[0] is None:
            raise ValueError("Expected one dataset per time; select the recommended combined-field artifact. Blocks are not merged automatically.")
        grid = grid[0]
    if not isinstance(grid, pv.DataSet) or grid.n_cells == 0 or grid.n_points == 0:
        raise ValueError("The selected frame contains no usable mesh.")
    if not np.isfinite(grid.points).all():
        raise ValueError("The selected frame contains nonfinite coordinates.")
    return grid.copy(deep=True), times, index


def read_visualization_frame(path, *, frame=-1):
    """Read one saved frame as a detached PyVista dataset, without solving.

    Negative indices follow Python convention. No interpolation, node merging,
    field recovery or geometry deformation is performed. Use the result's
    recommended visualization artifact, not an arbitrary per-field XDMF file.
    VTK owns format-specific connectivity and high-order cell interpretation.
    """
    return _read(path, frame)[0]


def inspect_visualization(path, *, frame=-1):
    """Describe saved times and selected-frame arrays without loading all frames.

    VTK cell identities describe display topology, not solver formulation or
    numerical qualification. A time coordinate has no inferred physical unit.
    """
    grid, times, index = _read(path, frame)
    def arrays(data):
        return [{"name": name, "shape": list(data[name].shape),
                 "dtype": str(data[name].dtype)} for name in data.keys()]
    # VTK, not a second connectivity table, owns display cell identities.
    unstructured = grid.cast_to_unstructured_grid()
    cell_types, counts = np.unique(unstructured.celltypes, return_counts=True)
    information = []
    if Path(path).suffix.lower() in {".xdmf", ".xmf"}:
        information = [{"name": item.get("Name"), "value": item.get("Value")}
                       for item in ET.parse(path).findall("Domain/Information")]
    return {"schema": "agentfem.visualization-inspection.v1", "source": str(Path(path)),
            "frame_count": len(times) or 1, "frame": index,
            "saved_times": list(times), "time": times[index] if times else None,
            "points": grid.n_points, "cells": grid.n_cells,
            "dataset_type": grid.GetClassName(),
            "information": information,
            "cell_types": [{"vtk_id": int(kind), "count": int(count)}
                           for kind, count in zip(cell_types, counts)],
            "point_fields": arrays(grid.point_data), "cell_fields": arrays(grid.cell_data),
            "scope": "Saved visualization data only; no solver support or scientific validation inferred"}


def deform_visualization_frame(grid, displacement, *, scale=1., reference_geometry=False):
    """Explicit display-only warp from a declared reference-geometry frame.

    Returns a deep copy with unchanged topology and point/cell arrays. The
    caller must establish reference geometry from its output contract: names
    such as U do not prove that displacement has not already been applied.
    """
    if reference_geometry is not True:
        raise ValueError("Warp requires an explicit reference-geometry declaration.")
    if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real)
            or not np.isfinite(scale)):
        raise ValueError("Deformation scale must be finite and real.")
    if displacement not in grid.point_data:
        raise ValueError("Displacement must be a node-associated field on this frame.")
    values = np.asarray(grid.point_data[displacement])
    if values.shape != (grid.n_points, 3) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Displacement must contain three finite components per node.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            points = np.asarray(grid.points, dtype=float)+float(scale)*values
        except FloatingPointError as exc:
            raise ValueError("Deformed coordinates are not representable.") from exc
    if not np.isfinite(points).all():
        raise ValueError("Deformed coordinates are not finite.")
    result = grid.copy(deep=True)
    result.points = points
    return result
