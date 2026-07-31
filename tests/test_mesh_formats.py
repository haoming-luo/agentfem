from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agentfem import mesh as mesh_api
from agentfem.mesh import formats


class _Mesh:
    def __init__(
        self,
        points,
        cells,
        *,
        cell_data=None,
        point_data=None,
        field_data=None,
        cell_sets=None,
        point_sets=None,
    ):
        self.points = np.asarray(points)
        self.cells = [
            SimpleNamespace(type=kind, data=np.asarray(values, dtype=np.int64))
            for kind, values in cells
        ]
        self.cell_data = {} if cell_data is None else cell_data
        self.point_data = {} if point_data is None else point_data
        self.field_data = {} if field_data is None else field_data
        self.cell_sets = {} if cell_sets is None else cell_sets
        self.point_sets = {} if point_sets is None else point_sets


class _MeshIO:
    Mesh = _Mesh

    def __init__(self, source):
        self.source = source
        self.written = None

    def read(self, _path, **_kwargs):
        return self.source

    def write(self, path, mesh):
        self.written = (path, mesh)


def _source_mesh():
    return _Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        cells=[
            ("line", [[0, 1], [1, 3]]),
            ("triangle", [[0, 1, 2], [1, 3, 2]]),
        ],
        cell_sets={
            "matrix": [None, np.array([0])],
            "inclusion": [None, np.array([1])],
            "loaded_nodes_as_elements": [np.array([1]), None],
        },
        point_sets={"fixed": np.array([0, 2])},
    )


def test_external_mesh_inventory_exposes_blocks_and_named_sets():
    summary = formats.summarize_external_mesh("model.inp", _source_mesh())

    assert summary.point_count == 4
    assert summary.cell_types == ("line", "triangle")
    assert summary.cell_sets["matrix"] == {"triangle": 1}
    assert summary.cell_sets["loaded_nodes_as_elements"] == {"line": 1}
    assert summary.point_sets == {"fixed": 2}


def test_conversion_preserves_selected_cell_sets_and_writes_a_manifest(
    tmp_path,
    monkeypatch,
):
    meshio = _MeshIO(_source_mesh())
    monkeypatch.setattr(formats, "require_meshio", lambda: meshio)
    source = tmp_path / "model.inp"
    output = tmp_path / "model.xdmf"

    converted = formats.convert_to_xdmf(
        source,
        output,
        cell_type="triangle",
        prune_z=True,
    )
    written = meshio.written[1]
    tags = written.cell_data["agentfem_region"][0]
    manifest = json.loads(converted.manifest_path.read_text(encoding="utf-8"))

    assert written.points.shape == (4, 2)
    np.testing.assert_array_equal(tags, [2, 1])
    assert converted.region_tags == {"inclusion": 1, "matrix": 2}
    assert manifest["output"]["cell_set_members"]["matrix"] == [0]
    assert "line" in converted.warnings[0]


def test_conversion_can_preserve_boundary_sets_in_a_separate_xdmf(
    tmp_path,
    monkeypatch,
):
    meshio = _MeshIO(_source_mesh())
    writes = []
    meshio.write = lambda path, converted: writes.append((path, converted))
    monkeypatch.setattr(formats, "require_meshio", lambda: meshio)

    converted = formats.convert_to_xdmf(
        tmp_path / "model.inp",
        tmp_path / "model.xdmf",
        cell_type="triangle",
        facet_type="line",
        prune_z=True,
    )
    manifest = json.loads(converted.manifest_path.read_text(encoding="utf-8"))

    assert len(writes) == 2
    assert converted.facet_path.name == "model_facets.xdmf"
    assert converted.boundary_tags == {"loaded_nodes_as_elements": 1}
    assert writes[1][1].cell_data["agentfem_boundary"][0].tolist() == [0, 1]
    assert manifest["output"]["selected_facet_type"] == "line"
    assert manifest["output"]["boundary_set_members"] == {
        "loaded_nodes_as_elements": [1]
    }
    assert converted.warnings == ()


def test_real_abaqus_mesh_conversion_is_readable_by_dolfinx(tmp_path):
    meshio = pytest.importorskip("meshio")
    from mpi4py import MPI

    source = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=[
            ("line", np.array([[0, 1], [1, 2], [2, 3], [3, 0]])),
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
        ],
        cell_sets={
            "outer": [np.array([0, 1, 2, 3]), np.array([], dtype=int)],
            "domain": [np.array([], dtype=int), np.array([0, 1])],
        },
    )
    abaqus_path = tmp_path / "square.inp"
    meshio.write(abaqus_path, source)
    conversion = formats.convert_abaqus_inp_to_xdmf(
        abaqus_path,
        tmp_path / "square.xdmf",
        cell_type="triangle",
        facet_type="line",
        prune_z=True,
    )

    converted_mesh = mesh_api.read_converted_xdmf(
        conversion,
        comm=MPI.COMM_SELF,
    )

    assert converted_mesh.cell_tags.values.tolist() == [
        conversion.region_tags["domain"]
    ] * 2
    assert converted_mesh.facet_tags.values.tolist() == [
        conversion.boundary_tags["outer"]
    ] * 4
