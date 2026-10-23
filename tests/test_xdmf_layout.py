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


@pytest.mark.parametrize("storage", ["single_file", "frame_files"])
@pytest.mark.parametrize("abort", [False, True])
def test_closed_writer_cannot_reopen_or_republish_accepted_data(tmp_path, storage, abort):
    domain, u, damage = fields()
    path = tmp_path / "closed.xdmf"
    writer = ResultWriter(path, domain, (u, damage), storage=storage)
    writer.__enter__()
    writer.write(0.)
    error = RuntimeError("simulation interrupted") if abort else None
    writer.__exit__(type(error) if error is not None else None, error, None)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(1.)
    with pytest.raises(RuntimeError, match="closed"):
        writer.__enter__()
    # Cleanup may be repeated, but must never promote an interrupted series
    # or reopen/truncate its heavy data on an accidental post-context write.
    writer.__exit__(None, None, None)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    if abort:
        index = path if storage == "frame_files" else path.with_name("closed.partial.xdmf")
        assert ET.parse(index).find("Domain/Information").get("Value") == "incomplete"


@pytest.mark.parametrize("storage", ["single_file", "frame_files"])
def test_writer_failed_close_cannot_be_reported_as_success(tmp_path, monkeypatch, storage):
    from agentfem.results.output import UnifiedXDMFTimeSeries
    domain, u, damage = fields()
    writer = ResultWriter(tmp_path / 'failed-close.xdmf', domain, (u, damage), storage=storage)
    writer.__enter__()
    writer.write(0.)
    def fail(*args, **kwargs):
        raise OSError('index publication failure')
    monkeypatch.setattr(UnifiedXDMFTimeSeries, '_publish_index', fail)
    with pytest.raises(OSError, match='index publication failure'):
        writer.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match='close failed') as caught:
        writer.__exit__(None, None, None)
    assert isinstance(caught.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match='closed'):
        writer.write(1.)


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


def test_completed_frame_storage_survives_exit_between_frames(tmp_path):
    """An abrupt process exit must not strand completed HDF5 metadata in RAM.

    This is deliberately not a checkpoint or a claim about interruption inside
    a write, filesystem failure, or power loss. The final XDMF stays unpublished.
    """
    import os
    import subprocess
    import sys
    import textwrap
    import h5py

    path = tmp_path / "interrupted.xdmf"
    script = textwrap.dedent("""
        import os
        import sys
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from agentfem.results import UnifiedXDMFTimeSeries

        domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1,
                                      cell_type=mesh.CellType.hexahedron)
        u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
        damage = fem.Function(fem.functionspace(domain, ("DG", 0)), name="damage")
        with UnifiedXDMFTimeSeries(sys.argv[1]) as writer:
            writer.write_fields(0., u, damage)
            u.x.array[:] = .1
            damage.x.array[:] = .25
            writer.write_fields(1., u, damage)
            os._exit(23)  # No context exit, destructor, or MPI finalization.
    """)
    child = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=45,
    )
    assert child.returncode == 23, child.stderr
    assert not path.exists()  # Incomplete run is not presented as final output.
    with h5py.File(path.with_suffix(".h5"), "r") as data:
        assert list(data["Frames"]) == ["0000", "0001"]
        np.testing.assert_array_equal(data["Frames/0000/Point/U"][:], 0.)
        np.testing.assert_array_equal(data["Frames/0001/Point/U"][:], .1)
        np.testing.assert_array_equal(data["Frames/0001/Cell/damage"][:], .25)
        assert data["Frames/0000/Geometry"].id == data["Frames/0001/Geometry"].id
    partial = tmp_path / "interrupted.partial.xdmf"
    assert ET.parse(partial).find("Domain/Information").get("Value") == "incomplete"
    meshio = pytest.importorskip("meshio")
    with meshio.xdmf.TimeSeriesReader(partial) as reader:
        points, cells = reader.read_points_cells()
        assert cells[0].type == "hexahedron"
        assert len(cells[0].data) == 1
        assert reader.num_steps == 2
        time, point, cell = reader.read_data(1)
        assert time == 1.
        assert point["U"].shape == points.shape
        np.testing.assert_array_equal(point["U"], .1)
        np.testing.assert_array_equal(cell["damage"][0], .25)


def test_partial_index_removed_only_after_successful_publication(tmp_path):
    from agentfem.results import UnifiedXDMFTimeSeries

    _, u, damage = fields()
    path = tmp_path / "frames.xdmf"
    with UnifiedXDMFTimeSeries(path) as writer:
        writer.write_fields(0., u, damage)
        assert writer.partial_path.is_file()
        assert not path.exists()
    assert path.is_file()
    assert not writer.partial_path.exists()


def test_existing_partial_index_is_not_overwritten(tmp_path):
    from agentfem.results import UnifiedXDMFTimeSeries

    path = tmp_path / "frames.xdmf"
    partial = tmp_path / "frames.partial.xdmf"
    partial.write_text("prior output index")
    with pytest.raises(FileExistsError, match="partial output"):
        with UnifiedXDMFTimeSeries(path):
            pass
    assert partial.read_text() == "prior output index"
    assert not path.with_suffix(".h5").exists()


def test_failed_final_index_preserves_partial_index(tmp_path, monkeypatch):
    from agentfem import checkpointing
    from agentfem.results import UnifiedXDMFTimeSeries

    _, u, damage = fields()
    path = tmp_path / "frames.xdmf"
    original = checkpointing.atomic_write_text
    def fail_final(target, content):
        if target == path:
            raise OSError("injected final-index failure")
        return original(target, content)
    monkeypatch.setattr(checkpointing, "atomic_write_text", fail_final)
    with pytest.raises(OSError, match="final-index failure"):
        with UnifiedXDMFTimeSeries(path) as writer:
            writer.write_fields(0., u, damage)
    assert not path.exists()
    assert writer.partial_path.is_file()


def test_failed_partial_index_poisons_writer(tmp_path, monkeypatch):
    from agentfem import checkpointing
    from agentfem.results import UnifiedXDMFTimeSeries

    _, u, damage = fields()
    path = tmp_path / "frames.xdmf"
    def fail_index(target, content):
        raise OSError("injected partial-index failure")
    with pytest.raises(RuntimeError, match="incomplete artifact"):
        with UnifiedXDMFTimeSeries(path) as writer:
            monkeypatch.setattr(checkpointing, "atomic_write_text", fail_index)
            with pytest.raises(OSError, match="partial-index failure"):
                writer.write_fields(0., u, damage)
            with pytest.raises(RuntimeError, match="writer failed"):
                writer.write_fields(1., u, damage)
    assert not path.exists()


def test_frame_flush_failure_does_not_publish_output(tmp_path, monkeypatch):
    import h5py
    from agentfem.results import UnifiedXDMFTimeSeries

    _, u, damage = fields()
    path = tmp_path / "flush-failed.xdmf"
    with pytest.raises(RuntimeError, match="incomplete artifact"):
        with UnifiedXDMFTimeSeries(path) as writer:
            def fail_flush(store):
                raise OSError("injected flush failure")
            with monkeypatch.context() as patch:
                patch.setattr(h5py.File, "flush", fail_flush)
                with pytest.raises(OSError, match="flush failure"):
                    writer.write_fields(0., u, damage)
            with pytest.raises(RuntimeError, match="writer failed"):
                writer.write_fields(1., u, damage)
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
