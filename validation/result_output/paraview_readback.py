"""Run in ParaView pvpython; read all frames, Warp and render final fields."""

from pathlib import Path
import sys, json
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from paraview.simple import (
    OpenDataFile,
    UpdatePipeline,
    WarpByVector,
    Show,
    GetActiveViewOrCreate,
    Render,
    SaveScreenshot,
    Delete,
    ColorBy,
    ResetCamera,
)
from paraview import servermanager

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
records = []
reference_values = {}
max_difference = 0.0
for suffix in (".xdmf", ".pvd"):
    reader = OpenDataFile(str(root / ("fields" + suffix)))
    times = list(reader.TimestepValues)
    assert len(times) == 5
    for time in times:
        reader.UpdatePipeline(time)
        data = servermanager.Fetch(reader)
        if data.IsA("vtkMultiBlockDataSet"):
            assert data.GetNumberOfBlocks() == 1
            data = data.GetBlock(0)
        print(suffix, time, data.GetClassName(), data.GetNumberOfPoints(), flush=True)
        assert data.GetNumberOfPoints() > 0
        assert data.GetPointData().GetArray("U") is not None
        assert data.GetCellData().GetArray("Energy") is not None
        coordinates = np.round(vtk_to_numpy(data.GetPoints().GetData()), decimals=12)
        assert len(np.unique(coordinates, axis=0)) == len(coordinates)
        order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
        values = vtk_to_numpy(data.GetPointData().GetArray("U"))[order].copy()
        if suffix == ".xdmf":
            reference_values[time] = values
        else:
            np.testing.assert_allclose(
                values, reference_values[time], rtol=1e-12, atol=1e-12
            )
            max_difference = max(
                max_difference, float(np.max(np.abs(values - reference_values[time])))
            )
    warp = WarpByVector(Input=reader)
    warp.Vectors = ["POINTS", "U"]
    warp.ScaleFactor = 1
    UpdatePipeline(time=times[-1], proxy=warp)
    view = GetActiveViewOrCreate("RenderView")
    view.ViewTime = times[-1]
    display = Show(warp, view)
    ColorBy(display, ("POINTS", "U", "Magnitude"))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.Representation = "Surface With Edges"
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    ResetCamera(view)
    Render(view)
    SaveScreenshot(
        str(out / ("warp" + suffix.replace(".", "_") + ".png")),
        view,
        ImageResolution=[1000, 600],
    )
    records.append(
        dict(
            format=suffix,
            frames=len(times),
            points=data.GetNumberOfPoints(),
            warp=True,
            max_displacement_difference_vs_xdmf=max_difference,
        )
    )
    Delete(warp)
    Delete(reader)
(out / "paraview_readback.json").write_text(json.dumps(records, indent=2) + "\n")
print(records)
