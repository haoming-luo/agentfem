# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models import rigid


@dataclass
class _CellBlock:
    type: str
    data: object


@dataclass
class _Mesh:
    points: object
    cells: object


class _MeshIO:
    def __init__(self, mesh):
        self.mesh = mesh
        self.calls = []

    def read(self, path, *, file_format=None):
        self.calls.append((path, file_format))
        return self.mesh


def _duplicated_tetrahedron(*, reversed_blocks: bool = False):
    triangles = (
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    if reversed_blocks:
        triangles = tuple(reversed(triangles))
    points = np.asarray(triangles, dtype=float).reshape((-1, 3))
    cells = np.arange(points.shape[0], dtype=np.int64).reshape((-1, 3))
    return _Mesh(points=points, cells=(_CellBlock("triangle", cells),))


def _install_meshio(monkeypatch, mesh):
    meshio = _MeshIO(mesh)
    monkeypatch.setattr(rigid.dependencies, "require", lambda *args, **kwargs: meshio)
    return meshio


def test_triangle_import_welds_stl_vertices_scales_and_records_provenance(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "tool.stl"
    source.write_bytes(b"reviewed-binary-or-ascii-source")
    meshio = _install_meshio(monkeypatch, _duplicated_tetrahedron())

    surface = boundary_models.triangulated_rigid_surface_from_mesh(
        source,
        coordinate_scale=1.0e-3,
        input_format="stl",
        name="forming_tool",
    )

    assert meshio.calls == [(source, "stl")]
    assert surface.vertices.shape == (4, 3)
    assert surface.triangles.shape == (4, 3)
    assert np.max(surface.vertices) == pytest.approx(1.0e-3)
    assert surface.summary()["closed"] is True
    assert surface.summary()["orientation_consistent"] is True
    assert surface.summary()["source"] == {
        "format": "stl",
        "sha256": "1e252f7223f717002922833076747be5ff4e0863a00a201ce02fdda89088db8c",
        "coordinate_scale": 1.0e-3,
        "path_in_identity": False,
    }


def test_triangle_import_geometry_identity_ignores_source_order(tmp_path, monkeypatch):
    first_path = tmp_path / "first.stl"
    second_path = tmp_path / "second.stl"
    first_path.write_bytes(b"first ordering")
    second_path.write_bytes(b"second ordering")

    _install_meshio(monkeypatch, _duplicated_tetrahedron())
    first = boundary_models.triangulated_rigid_surface_from_mesh(
        first_path,
        coordinate_scale=1.0,
    )
    _install_meshio(monkeypatch, _duplicated_tetrahedron(reversed_blocks=True))
    second = boundary_models.triangulated_rigid_surface_from_mesh(
        second_path,
        coordinate_scale=1.0,
    )

    assert first.geometry_fingerprint == second.geometry_fingerprint
    np.testing.assert_array_equal(first.facet_ids, second.facet_ids)
    assert first.summary()["source"]["sha256"] != second.summary()["source"]["sha256"]


def test_triangle_import_can_explicitly_flip_all_normals(tmp_path, monkeypatch):
    source = tmp_path / "tool.stl"
    source.write_bytes(b"same source")
    _install_meshio(monkeypatch, _duplicated_tetrahedron())
    forward = boundary_models.triangulated_rigid_surface_from_mesh(
        source,
        coordinate_scale=1.0,
    )
    reverse = boundary_models.triangulated_rigid_surface_from_mesh(
        source,
        coordinate_scale=1.0,
        flip_normals=True,
    )

    np.testing.assert_allclose(reverse.facet_normals, -forward.facet_normals)
    assert reverse.geometry_fingerprint != forward.geometry_fingerprint


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_triangle_import_rejects_ambiguous_coordinate_scale(
    tmp_path,
    monkeypatch,
    scale,
):
    source = tmp_path / "tool.stl"
    source.write_bytes(b"surface")
    _install_meshio(monkeypatch, _duplicated_tetrahedron())

    with pytest.raises(ValueError, match="coordinate_scale"):
        boundary_models.triangulated_rigid_surface_from_mesh(
            source,
            coordinate_scale=scale,
        )


def test_triangle_import_rejects_mixed_topology_instead_of_dropping_it(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "mixed.vtk"
    source.write_bytes(b"mixed topology")
    mesh = _duplicated_tetrahedron()
    mesh.cells = (*mesh.cells, _CellBlock("line", np.asarray(((0, 1),))))
    _install_meshio(monkeypatch, mesh)

    with pytest.raises(ValueError, match="triangle-only.*line"):
        boundary_models.triangulated_rigid_surface_from_mesh(
            source,
            coordinate_scale=1.0,
        )


def test_triangle_import_requires_explicit_boolean_normal_flip(tmp_path, monkeypatch):
    source = tmp_path / "tool.stl"
    source.write_bytes(b"surface")
    _install_meshio(monkeypatch, _duplicated_tetrahedron())

    with pytest.raises(TypeError, match="explicit boolean"):
        boundary_models.triangulated_rigid_surface_from_mesh(
            source,
            coordinate_scale=1.0,
            flip_normals="false",
        )


def test_triangle_import_rejects_source_mutation_during_read(tmp_path, monkeypatch):
    source = tmp_path / "tool.stl"
    source.write_bytes(b"before")
    mesh = _duplicated_tetrahedron()

    class _MutatingMeshIO:
        @staticmethod
        def read(path, *, file_format=None):
            path.write_bytes(b"changed while reading")
            return mesh

    monkeypatch.setattr(
        rigid.dependencies,
        "require",
        lambda *args, **kwargs: _MutatingMeshIO(),
    )
    with pytest.raises(RuntimeError, match="changed while it was being read"):
        boundary_models.triangulated_rigid_surface_from_mesh(
            source,
            coordinate_scale=1.0,
        )


def test_imported_provenance_cannot_be_partially_declared():
    with pytest.raises(ValueError, match="requires format, SHA-256"):
        boundary_models.TriangulatedRigidSurface(
            vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            triangles=((0, 1, 2),),
            source_format="stl",
        )


def test_real_meshio_binary_stl_enters_reviewed_surface_contract(tmp_path):
    pytest.importorskip("meshio")
    source = tmp_path / "single_triangle_binary.stl"
    source.write_bytes(
        b"AgentFEM reviewed binary STL".ljust(80, b"\0")
        + struct.pack("<I", 1)
        + struct.pack(
            "<12fH",
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0,
        )
    )

    surface = boundary_models.triangulated_rigid_surface_from_mesh(
        source,
        coordinate_scale=1.0e-3,
    )
    search = boundary_models.triangle_surface_bvh(surface)
    projection = search.project(((0.5e-3, 0.5e-3, 0.25e-3),))

    assert projection.all_valid
    assert projection.entity_ids.tolist() == [0]
    assert projection.signed_gaps.tolist() == pytest.approx([0.25e-3])
    assert surface.summary()["source"]["format"] == "stl"


def test_imported_source_path_does_not_change_rigid_body_identity(
    tmp_path,
    monkeypatch,
):
    first_path = tmp_path / "first" / "tool.stl"
    second_path = tmp_path / "second" / "renamed.stl"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_bytes(b"identical source bytes")
    second_path.write_bytes(first_path.read_bytes())
    mesh = _duplicated_tetrahedron()
    _install_meshio(monkeypatch, mesh)
    first = boundary_models.triangulated_rigid_surface_from_mesh(
        first_path,
        coordinate_scale=1.0,
        name="tool",
    )
    _install_meshio(monkeypatch, mesh)
    second = boundary_models.triangulated_rigid_surface_from_mesh(
        second_path,
        coordinate_scale=1.0,
        name="tool",
    )

    first_body = boundary_models.rigid_body(first, name="body")
    second_body = boundary_models.rigid_body(second, name="body")
    assert first_body.scientific_identity == second_body.scientific_identity
