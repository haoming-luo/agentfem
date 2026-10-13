# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent installed-ParaView check; run with pvpython, not solver Python.

Example: pvpython tools/check_paraview_warp.py fields.xdmf --vector displacement
This checks presentation geometry, not the physical validity of a solution.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _reader_times(value):
    """Readers may expose one saved time as a scalar, not a sequence."""
    values = () if value is None else (value,) if np.isscalar(value) else value
    times = [float(item) for item in values]
    if not np.isfinite(times).all():
        raise ValueError("Reader returned nonfinite time coordinates.")
    return times


def _warp_error(coordinates, values, actual, scale):
    """Do not let NaN equality or broadcasting turn invalid output into a pass."""
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3
            or not len(coordinates) or values.shape != coordinates.shape
            or actual.shape != coordinates.shape):
        raise ValueError("Warp must preserve a nonempty (points, 3) geometry and vector.")
    if (not np.isfinite(scale)
            or any(not np.isfinite(array).all()
                   for array in (coordinates, values, actual))):
        raise ValueError("Warp coordinates, vector and scale must be finite.")
    with np.errstate(over="ignore", invalid="ignore"):
        expected = coordinates + scale * values
    if not np.isfinite(expected).all():
        raise ValueError("Warp exceeds the finite coordinate range.")
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8, equal_nan=False)
    return float(np.max(np.abs(actual - expected)))


def leaves(data):
    if data.IsA("vtkCompositeDataSet"):
        iterator = data.NewIterator()
        iterator.SkipEmptyNodesOn()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            item = iterator.GetCurrentDataObject()
            if item is not None and item.IsA("vtkDataSet"):
                yield item
            iterator.GoToNextItem()
    else:
        yield data


def check(path, vector, scale=1.):
    from paraview import servermanager
    from paraview.simple import OpenDataFile, WarpByVector, Delete
    from vtkmodules.util.numpy_support import vtk_to_numpy

    if not np.isfinite(scale):
        raise ValueError("Warp scale must be finite.")
    source = OpenDataFile(str(Path(path).resolve()))
    if source is None:
        raise RuntimeError("ParaView did not admit the result file.")
    warped = None
    try:
        source.UpdatePipelineInformation()
        times = _reader_times(getattr(source, "TimestepValues", ()))
        # A continuation segment may begin at t>0. The pipeline default of
        # zero is not necessarily a saved frame and may omit its attributes.
        selected_time = float(times[0]) if times else 0.
        source.UpdatePipeline(time=selected_time)
        grids = list(leaves(servermanager.Fetch(source)))
        if len(grids) != 1:
            raise ValueError(f"Expected one mesh per time, got {len(grids)} blocks.")
        grid = grids[0]
        field = grid.GetPointData().GetArray(vector)
        if field is None or field.GetNumberOfComponents() != 3:
            raise ValueError("Select an existing three-component point vector.")
        coordinates = vtk_to_numpy(grid.GetPoints().GetData()).copy()
        values = vtk_to_numpy(field).copy()
        if not len(coordinates) or values.shape != coordinates.shape:
            raise ValueError("Vector must cover every point of a nonempty mesh.")
        warped = WarpByVector(Input=source)
        warped.Vectors = ["POINTS", vector]
        warped.ScaleFactor = scale
        warped.UpdatePipeline(time=selected_time)
        output = list(leaves(servermanager.Fetch(warped)))
        if len(output) != 1 or output[0].GetNumberOfCells() != grid.GetNumberOfCells():
            raise ValueError("Warp introduced or lost geometry blocks/cells.")
        actual = vtk_to_numpy(output[0].GetPoints().GetData())
        error = _warp_error(coordinates, values, actual, scale)
        return {"schema": "agentfem.paraview-warp-check.v1", "passed": True,
                "reader": source.SMProxy.GetXMLName(), "path": str(Path(path).resolve()),
                "blocks": 1, "points": len(coordinates), "cells": grid.GetNumberOfCells(),
                "vector": vector, "scale": scale,
                "checked_time": selected_time, "available_times": times,
                "maximum_coordinate_error": error,
                "scope": "presentation only; not scientific validation"}
    finally:
        if warped is not None:
            Delete(warped)
        Delete(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.path, args.vector, args.scale)
    except Exception as error:
        report = {"schema": "agentfem.paraview-warp-check.v1", "passed": False,
                  "path": str(args.path.resolve()),
                  "error": {"code": "paraview_warp_check_failed",
                            "type": type(error).__name__, "message": str(error)}}
    if args.output:
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)
