# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Real VTK inspection preserves volume cells and field association."""

from hashlib import sha256

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
from agentfem.results import (  # noqa: E402
    deform_visualization_frame, inspect_visualization, read_visualization_frame,
)


def fixture(tmp_path):
    grid = pv.ImageData(dimensions=(3, 2, 2)).cast_to_unstructured_grid()
    grid.point_data["U"] = .1*grid.points
    grid.cell_data["damage"] = np.array([.2, .8])
    path = tmp_path/"fields.vtu"
    grid.save(path)
    return path, grid


def test_vtu_inspection_and_warp_leave_scientific_data_unchanged(tmp_path):
    path, original = fixture(tmp_path)
    digest = sha256(path.read_bytes()).hexdigest()
    info = inspect_visualization(path)
    assert info["frame_count"] == 1 and info["time"] is None
    assert info["cells"] == 2
    assert info["cell_types"] == [{"vtk_id": 11, "count": 2}]
    assert info["point_fields"][0]["name"] == "U"
    assert info["cell_fields"][0]["name"] == "damage"
    grid = read_visualization_frame(path)
    deformed = deform_visualization_frame(grid, "U", reference_geometry=True)
    np.testing.assert_allclose(deformed.points, 1.1*original.points)
    np.testing.assert_array_equal(grid.points, original.points)
    np.testing.assert_array_equal(deformed.cells, original.cells)
    np.testing.assert_array_equal(deformed.cell_data["damage"], original.cell_data["damage"])
    assert sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("frame,error", [(True, ValueError), (1.5, ValueError), (1, IndexError), (-2, IndexError)])
def test_saved_frame_indices(tmp_path, frame, error):
    path, _ = fixture(tmp_path)
    with pytest.raises(error):
        read_visualization_frame(path, frame=frame)


def test_warp_requires_geometry_and_nodal_vector(tmp_path):
    _, grid = fixture(tmp_path)
    with pytest.raises(ValueError, match="reference-geometry"):
        deform_visualization_frame(grid, "U")
    with pytest.raises(ValueError, match="node-associated"):
        deform_visualization_frame(grid, "damage", reference_geometry=True)
    for scale in (True, np.inf, 1j):
        with pytest.raises(ValueError):
            deform_visualization_frame(grid, "U", scale=scale, reference_geometry=True)


def test_multiple_blocks_are_not_silently_merged(tmp_path, monkeypatch):
    from agentfem.results import inspection
    from types import SimpleNamespace
    _, grid = fixture(tmp_path)
    reader = SimpleNamespace(read=lambda: pv.MultiBlock([grid, grid]))
    monkeypatch.setattr(inspection, "_reader", lambda p: (pv, reader, ()))
    with pytest.raises(ValueError, match="not merged"):
        read_visualization_frame("anything")


def test_native_reader_failure_is_an_addressable_input_error(monkeypatch):
    from agentfem.results import inspection
    from types import SimpleNamespace

    def fail():
        raise RuntimeError("unavailable HDF dataset")

    monkeypatch.setattr(inspection, "_reader", lambda p: (pv, SimpleNamespace(read=fail), ()))
    with pytest.raises(ValueError, match="VTK could not read"):
        read_visualization_frame("anything")


@pytest.mark.parametrize("cell_name", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("degree", [1, 2])
def test_real_output_cell_order_and_selected_times(tmp_path, cell_name, degree):
    from dolfinx import fem, mesh
    from mpi4py import MPI
    from agentfem import results
    from types import SimpleNamespace
    factory = mesh.create_unit_square if cell_name in {"triangle", "quadrilateral"} else mesh.create_unit_cube
    counts = (2, 2) if cell_name in {"triangle", "quadrilateral"} else (2, 2, 2)
    domain = factory(MPI.COMM_SELF, *counts, cell_type=getattr(mesh.CellType, cell_name))
    u = fem.Function(fem.functionspace(domain, ("Lagrange", degree, (3,))), name="U")
    u.interpolate(lambda x: .1*x)
    path = results.write_unified_xdmf_series(tmp_path/"result.xdmf",
        (SimpleNamespace(solution=u, load_factor=.25), SimpleNamespace(solution=u, load_factor=.75)),
        ((), ()), deformation_scale=0.)
    info = inspect_visualization(path, frame=0)
    assert info["saved_times"] == [.25, .75]
    assert info["time"] == .25 and info["frame_count"] == 2
    grid = read_visualization_frame(path, frame=-1)
    assert grid.n_cells == domain.topology.index_map(domain.topology.dim).size_local
    assert grid.GetCell(0).GetCellDimension() == domain.topology.dim
    np.testing.assert_allclose(grid.point_data["U"], .1*grid.points, atol=1e-14)
    warped = deform_visualization_frame(grid, "U", reference_geometry=True)
    np.testing.assert_allclose(warped.points, 1.1*grid.points, atol=1e-14)


def test_joined_restart_frames_retain_fields_and_sparse_times(tmp_path):
    from agentfem import results
    from test_xdmf_segments import segment
    a = segment(tmp_path / "initial", [0., 1.])
    b = segment(tmp_path / "restart", [1., 3.])
    joined = results.join_xdmf_segments([a, b], tmp_path / "view.xdmf")
    assert inspect_visualization(joined)["saved_times"] == [0., 1., 3.]
    for index, time in enumerate([0., 1., 3.]):
        grid = read_visualization_frame(joined, frame=index)
        assert grid.n_cells == 1
        np.testing.assert_allclose(grid.point_data["u"], time)


def test_xmf_alias_uses_xdmf_reader(tmp_path):
    from test_xdmf_segments import segment
    path = segment(tmp_path / "source", [0., 1.])
    alias = path.with_suffix(".xmf")
    alias.write_bytes(path.read_bytes())
    assert inspect_visualization(alias)["saved_times"] == [0., 1.]


def test_pvd_reads_only_saved_frames_and_preserves_cell_fields(tmp_path):
    import xml.etree.ElementTree as ET
    path, grid = fixture(tmp_path)
    second = tmp_path / "second.vtu"
    grid.cell_data["damage"] *= 2
    grid.save(second)
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    for time, frame in ((0., path), (3., second)):
        ET.SubElement(collection, "DataSet", timestep=str(time), group="", part="0", file=frame.name)
    index = tmp_path / "results.pvd"
    ET.ElementTree(root).write(index)
    assert inspect_visualization(index)["saved_times"] == [0., 3.]
    np.testing.assert_allclose(read_visualization_frame(index, frame=0).cell_data["damage"], [.2, .8])
    np.testing.assert_allclose(read_visualization_frame(index, frame=1).cell_data["damage"], [.4, 1.6])


def test_reopen_published_index_finds_new_frames_before_writer_close(tmp_path):
    from agentfem.io import ResultWriter
    from test_xdmf_layout import fields
    domain, u, damage = fields()
    path = tmp_path / "results.xdmf"
    with ResultWriter(path, domain, (u, damage), storage="frame_files") as writer:
        writer.write(0.)
        assert inspect_visualization(path)["frame_count"] == 1
        assert {"name": "agentfem_output_status", "value": "incomplete"} in inspect_visualization(path)["information"]
        u.x.array[:] *= 2
        writer.write(1.)
        assert inspect_visualization(path)["saved_times"] == [0., 1.]
        grid = read_visualization_frame(path)
        assert "damage" in grid.cell_data
        warped = deform_visualization_frame(grid, "displacement", reference_geometry=True)
        np.testing.assert_allclose(warped.points, 1.2 * grid.points)
