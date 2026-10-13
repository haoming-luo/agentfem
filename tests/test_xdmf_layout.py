# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""No duplicate geometry or partial displacement after external reader/warp."""
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from dolfinx import fem, io, mesh
from mpi4py import MPI
from agentfem.io import ResultWriter
from agentfem.results import consolidate_xdmf_fields


def fields():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2,
                                  cell_type=mesh.CellType.hexahedron)
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))), name="displacement")
    u.interpolate(lambda x: .1*x)
    damage = fem.Function(fem.functionspace(domain, ("DG", 0)), name="damage")
    damage.x.array[:] = .25
    return domain, u, damage


def check_warp(path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutputDataObject(0)
    assert grid.IsA("vtkUnstructuredGrid"), grid.GetClassName()
    assert grid.GetNumberOfCells() == 8
    points = vtk_to_numpy(grid.GetPoints().GetData()).copy()
    assert grid.GetPointData().GetArray("displacement").GetNumberOfTuples() == len(points)
    np.testing.assert_allclose(vtk_to_numpy(grid.GetCellData().GetArray("damage")), .25)
    warp = vtk.vtkWarpVector()
    warp.SetInputConnection(reader.GetOutputPort())
    warp.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "displacement")
    warp.SetScaleFactor(1.)
    warp.Update()
    warped = warp.GetOutput()
    assert warped.GetNumberOfCells() == 8
    np.testing.assert_allclose(vtk_to_numpy(warped.GetPoints().GetData()), points*1.1, atol=1e-7)


def test_result_writer_single_grid_and_warp(tmp_path):
    domain, u, damage = fields()
    path = tmp_path / "fields.xdmf"
    with ResultWriter(path, domain, (u, damage)) as writer:
        writer.write(.1)
    check_warp(path)


def test_result_writer_rejects_same_size_different_mesh(tmp_path):
    domain, u, _ = fields()
    _, _, other = fields()
    with pytest.raises(ValueError, match="mesh"):
        with ResultWriter(tmp_path / "bad.xdmf", domain) as writer:
            writer.write(.1, u, other)
    assert not (tmp_path / "bad.xdmf").exists()


def test_unified_writer_rejects_mesh_switch_before_publishing(tmp_path):
    from agentfem.results import UnifiedXDMFTimeSeries
    _, u, damage = fields()
    _, other, _ = fields()
    with pytest.raises(ValueError, match="mesh changed"):
        with UnifiedXDMFTimeSeries(tmp_path / "bad.xdmf") as writer:
            writer.write_fields(.1, u, damage)
            writer.write_fields(.2, other)
    assert not (tmp_path / "bad.xdmf").exists()


def archive(tmp_path):
    domain, u, damage = fields()
    path = tmp_path / "legacy.xdmf"
    with io.XDMFFile(domain.comm, str(path), "w") as writer:
        writer.write_mesh(domain)
        writer.write_function(u, .1)
        writer.write_function(damage, .1)
    return path


def test_consolidate_preserves_arrays_and_original(tmp_path):
    path = archive(tmp_path)
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (path, path.with_suffix('.h5'))}
    output = consolidate_xdmf_fields(path, tmp_path / "view" / "fields.xdmf")
    check_warp(output)
    for p, digest in originals.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        consolidate_xdmf_fields(path, path)
    with pytest.raises(FileExistsError):
        consolidate_xdmf_fields(path, output)


@pytest.mark.parametrize("fault", ["asynchronous", "external", "duplicate", "mesh"])
def test_consolidate_rejects_ambiguous_archives(tmp_path, fault):
    path = archive(tmp_path)
    tree = ET.parse(path)
    root = tree.getroot()
    if fault == "asynchronous":
        root.findall(".//Time")[-1].set("Value", ".2")
    elif fault == "external":
        next(item for item in root.iter() if item.tag.endswith('}include')).set("href", "other.xdmf")
    elif fault == "duplicate":
        attrs = root.findall(".//Attribute")
        attrs[-1].attrib.update(attrs[0].attrib)
    else:
        root.find("Domain").append(ET.Element("Grid", GridType="Uniform"))
    tree.write(path)
    output = tmp_path / "repaired.xdmf"
    with pytest.raises(ValueError):
        consolidate_xdmf_fields(path, output)
    assert not output.exists()
