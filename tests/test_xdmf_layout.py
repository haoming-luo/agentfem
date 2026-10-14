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
    damage.interpolate(lambda x: .1*(x[0]+2*x[1]+4*x[2]))
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
    damage = vtk_to_numpy(grid.GetCellData().GetArray("damage"))
    for index in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(index)
        center = np.mean([points[cell.GetPointId(i)] for i in range(8)], axis=0)
        assert damage[index] == pytest.approx(.1*np.dot(center, [1, 2, 4]))
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


@pytest.mark.parametrize("fault", ["time_nan", "time_repeat", "time_backwards",
                                  "nonfinite", "duplicate", "reserved", "unsafe_name",
                                  "schema", "mesh_motion"])
def test_frame_rejection_leaves_no_partial_frame(tmp_path, fault):
    import h5py
    domain, u, damage = fields()
    path = tmp_path / "frames.xdmf"
    with ResultWriter(path, domain, (u, damage)) as writer:
        writer.write(1.)
        time = {"time_nan": np.nan, "time_repeat": 1., "time_backwards": .5}.get(fault, 2.)
        selected = [u, damage]
        saved = damage.x.array.copy()
        coordinates = domain.geometry.x.copy()
        if fault == "nonfinite":
            damage.x.array[0] = np.inf
        elif fault == "duplicate":
            selected.append(damage)
        elif fault == "reserved":
            damage.name = "UMAG"
        elif fault == "unsafe_name":
            damage.name = "nested/damage"
        elif fault == "schema":
            selected = [u]
        elif fault == "mesh_motion":
            domain.geometry.x[0, 0] += .1
        with pytest.raises(ValueError):
            writer.write(time, *selected)
        assert list(writer._series._h5["Frames"]) == ["0000"]
        damage.name = "damage"
        damage.x.array[:] = saved
        domain.geometry.x[:] = coordinates
        writer.write(2.)
    with h5py.File(path.with_suffix(".h5")) as store:
        assert list(store["Frames"]) == ["0000", "0001"]
        # Reference geometry is stored once, not duplicated at every frame.
        assert store["Frames/0000/Geometry"].id == store["Frames/0001/Geometry"].id
        assert store["Frames/0000/Geometry"].id == store["Mesh/ReferenceGeometry"].id
    assert [item.get("Value") for item in ET.parse(path).findall(".//Time")] == ["1", "2"]


def test_invalid_first_frame_does_not_open_storage(tmp_path):
    domain, u, damage = fields()
    damage.x.array[0] = np.nan
    path = tmp_path / "invalid.xdmf"
    with pytest.raises(ValueError, match="finite"):
        with ResultWriter(path, domain, (u, damage)) as writer:
            writer.write(0.)
    assert not path.exists()
    assert not path.with_suffix(".h5").exists()


@pytest.mark.parametrize("path,mode", [("fields.vtu", "w"), ("fields.h5", "w"),
                                     ("fields.xdmf", "r")])
def test_result_writer_rejects_misleading_format_or_mode(tmp_path, path, mode):
    domain, _, _ = fields()
    with pytest.raises(ValueError, match="ResultWriter"):
        ResultWriter(tmp_path / path, domain, mode=mode)


def test_storage_failure_poisons_writer(tmp_path, monkeypatch):
    import h5py
    domain, u, damage = fields()
    path = tmp_path / "failed.xdmf"
    original = h5py.Group.create_dataset
    def fail(group, name, *args, **kwargs):
        if name == "damage":
            raise OSError("injected storage failure")
        return original(group, name, *args, **kwargs)
    with pytest.raises(RuntimeError, match="incomplete artifact"):
        with ResultWriter(path, domain, (u, damage)) as writer:
            monkeypatch.setattr(h5py.Group, "create_dataset", fail)
            with pytest.raises(OSError, match="injected"):
                writer.write(0.)
            with pytest.raises(RuntimeError, match="writer failed"):
                writer.write(1.)
    assert not path.exists()


def test_equal_size_different_nodal_variants_are_interpolated():
    import basix
    import basix.ufl
    from agentfem.results.output import _unified_field_values
    domain, _, _ = fields()
    primary_space = fem.functionspace(domain, basix.ufl.element(
        "Lagrange", "hexahedron", 3, lagrange_variant=basix.LagrangeVariant.gll_warped))
    other_space = fem.functionspace(domain, basix.ufl.element(
        "Lagrange", "hexahedron", 3, lagrange_variant=basix.LagrangeVariant.equispaced))
    other = fem.Function(other_space, name="temperature")
    other.interpolate(lambda x: x[0]**3 + 2*x[1])
    coordinates = primary_space.tabulate_dof_coordinates()
    assert other.x.array.size == len(coordinates)
    assert not np.allclose(other.x.array, coordinates[:, 0]**3 + 2*coordinates[:, 1])
    center, values = _unified_field_values(other, solution_space=primary_space,
        point_count=len(coordinates), cell_count=8)
    assert center == "Node"
    np.testing.assert_allclose(values, coordinates[:, 0]**3 + 2*coordinates[:, 1], atol=2e-14)


def test_reconstructed_primary_space_keeps_layout_but_not_object_identity(tmp_path):
    import h5py
    domain, u, _ = fields()
    copy_space = fem.functionspace(domain, u.function_space.ufl_element())
    assert copy_space != u.function_space
    other = fem.Function(copy_space, name=u.name)
    other.interpolate(lambda x: .2*x)
    incompatible = fem.Function(fem.functionspace(domain, ("DG", 1, (3,))), name=u.name)
    path = tmp_path/"equivalent-space.xdmf"
    with ResultWriter(path, domain) as writer:
        writer.write(0., u)
        with pytest.raises(ValueError, match="function space layout"):
            writer.write(.5, incompatible)
        writer.write(1., other)
    with h5py.File(path.with_suffix('.h5')) as data:
        np.testing.assert_array_equal(data[f'Frames/0001/Point/{u.name}'][:], other.x.array.reshape(-1,3))


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
