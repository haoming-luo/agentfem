from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import cli
from agentfem import mesh as mesh_api
from agentfem import results
from agentfem.mesh import abaqus
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


def _high_order_external_case(cell_type):
    if cell_type == "triangle6":
        return (
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.5, 0.5, 0.0),
                    (0.0, 0.5, 0.0),
                )
            ),
            True,
            2,
            "P",
        )
    if cell_type in {"quad8", "quad9"}:
        points = [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ]
        if cell_type == "quad9":
            points.append((0.0, 0.0, 0.0))
        return np.asarray(points), True, 2, (
            "serendipity" if cell_type == "quad8" else "P"
        )
    if cell_type == "tetra10":
        return (
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.5, 0.0, 0.0),
                    (0.5, 0.5, 0.0),
                    (0.0, 0.5, 0.0),
                    (0.0, 0.0, 0.5),
                    (0.5, 0.0, 0.5),
                    (0.0, 0.5, 0.5),
                )
            ),
            False,
            3,
            "P",
        )
    if cell_type in {"hexahedron20", "hexahedron27"}:
        points = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (0.0, -1.0, -1.0),
            (1.0, 0.0, -1.0),
            (0.0, 1.0, -1.0),
            (-1.0, 0.0, -1.0),
            (0.0, -1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0),
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ]
        if cell_type == "hexahedron27":
            # meshio/VTK face order: x-, x+, y-, y+, z-, z+, center.
            points.extend(
                (
                    (-1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.0),
                )
            )
        return np.asarray(points), False, 3, (
            "serendipity" if cell_type == "hexahedron20" else "P"
        )
    raise AssertionError(f"missing high-order fixture for {cell_type}")


def test_periodic_square_uses_native_mesh_without_gmsh(monkeypatch):
    from mpi4py import MPI

    def fail_if_requested():
        raise AssertionError("periodic_square must not require optional Gmsh")

    monkeypatch.setattr(mesh_api, "require_gmsh", fail_if_requested)
    imported = mesh_api.from_spec(
        {
            "type": "periodic_square",
            "geometry_params": {"bounds": [-2.0, 3.0, -1.0, 4.0]},
        },
        resolution=3,
        comm=MPI.COMM_SELF,
    )

    coordinates = imported.domain.geometry.x
    assert np.isclose(coordinates[:, 0].min(), -2.0)
    assert np.isclose(coordinates[:, 0].max(), 3.0)
    assert np.isclose(coordinates[:, 1].min(), -1.0)
    assert np.isclose(coordinates[:, 1].max(), 4.0)


def test_external_mesh_inventory_exposes_blocks_and_named_sets():
    summary = formats.summarize_external_mesh("model.inp", _source_mesh())

    assert summary.point_count == 4
    assert summary.cell_types == ("line", "triangle")
    assert summary.cell_sets["matrix"] == {"triangle": 1}
    assert summary.cell_sets["loaded_nodes_as_elements"] == {"line": 1}
    assert summary.point_sets == {"fixed": 2}
    assert summary.as_dict()["cell_blocks"][1]["compatibility"] == {
        **mesh_api.describe_cell_compatibility("triangle").summary()
    }


def test_cell_compatibility_separates_geometry_from_source_formulation():
    verified = mesh_api.describe_cell_compatibility("hexahedron")
    high_order = mesh_api.describe_cell_compatibility("hexahedron20")
    conditional = mesh_api.describe_cell_compatibility("quad8")
    unknown = mesh_api.describe_cell_compatibility("polyhedron42")

    assert verified.solver_ready
    assert verified.topology == "hexahedron"
    assert verified.quality_metric == "sampled_scaled_jacobian"
    assert high_order.solver_ready
    assert high_order.import_maturity == "verified"
    assert high_order.geometry_basis == "serendipity"
    assert conditional.import_maturity == "conditional"
    assert conditional.geometry_basis == "serendipity"
    assert not conditional.solver_ready
    assert unknown.import_maturity == "blocked"
    assert unknown.topology is None


def test_runtime_topology_compatibility_does_not_claim_source_formulation():
    topology = mesh_api.describe_topology_compatibility("hexahedron")

    assert topology.release_ready
    assert topology.quality_metric == "sampled_scaled_jacobian"
    assert set(topology.source_cell_types) >= {
        "hexahedron",
        "hexahedron20",
        "hexahedron27",
    }
    assert any("formulation" in item for item in topology.limitations)


def test_cli_inspect_mesh_reports_blocks_without_converting(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(formats, "require_meshio", lambda: _MeshIO(_source_mesh()))

    assert cli.main(["inspect-mesh", str(tmp_path / "model.inp"), "--json"]) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "agentfem.external-mesh-inspection"
    assert record["cell_blocks"][1]["cell_type"] == "triangle"
    assert record["cell_blocks"][1]["compatibility"]["solver_ready"] is True


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


def test_gmsh_physical_volume_and_surface_groups_lower_to_shared_interface(
    tmp_path, monkeypatch
):
    source = _Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        cells=[
            ("tetra", [[0, 1, 2, 3], [0, 2, 1, 4]]),
            ("triangle", [[0, 1, 2]]),
        ],
        cell_sets={
            "positive": [np.array([1]), None],
            "weak_interface": [None, np.array([0])],
        },
    )
    monkeypatch.setattr(formats, "require_meshio", lambda: _MeshIO(source))

    split = formats.split_gmsh_physical_interface(
        tmp_path / "model.msh",
        positive_group="positive",
        interface_group="weak_interface",
        cell_type="tetra",
        facet_type="triangle",
    )

    assert split.summary()["geometric_dimension"] == 3
    np.testing.assert_array_equal(split.negative_facets, [[0, 1, 2]])


def test_gmsh_planar_2d_physical_interface_prunes_constant_z_automatically(
    tmp_path, monkeypatch
):
    source = _Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        cells=[
            ("triangle", [[0, 1, 2], [1, 3, 2]]),
            ("line", [[1, 2]]),
        ],
        cell_sets={
            "positive": [np.array([1]), None],
            "weak_interface": [None, np.array([0])],
        },
    )
    monkeypatch.setattr(formats, "require_meshio", lambda: _MeshIO(source))

    split = formats.split_gmsh_physical_interface(
        tmp_path / "planar.msh",
        positive_group="positive",
        interface_group="weak_interface",
        cell_type="triangle",
        facet_type="line",
    )

    assert split.summary()["geometric_dimension"] == 2
    np.testing.assert_array_equal(split.negative_facets, [[1, 2]])


def test_gmsh_physical_names_are_disambiguated_by_topological_dimension():
    source = _Mesh(
        points=np.zeros((5, 3)),
        cells=[
            ("tetra", [[0, 1, 2, 3], [0, 2, 1, 4]]),
            ("triangle", [[0, 1, 2]]),
        ],
        field_data={
            "bulk_tag_one": np.array([1, 3]),
            "surface_tag_one": np.array([1, 2]),
        },
        cell_data={
            "gmsh:physical": [np.array([1, 2]), np.array([1])],
        },
    )

    bulk = formats._named_physical_members(source, "tetra")
    surface = formats._named_physical_members(source, "triangle")

    assert set(bulk) == {"bulk_tag_one"}
    assert set(surface) == {"surface_tag_one"}


def test_multi_topology_bundle_keeps_each_block_as_an_explicit_domain(tmp_path, monkeypatch):
    meshio = _MeshIO(_source_mesh())
    monkeypatch.setattr(formats, "require_meshio", lambda: meshio)

    bundle = formats.convert_topology_bundle(
        tmp_path / "model.inp",
        tmp_path / "converted",
        cell_types=("triangle", "line"),
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert tuple(item.cell_type for item in bundle.conversions) == ("triangle", "line")
    assert [item["cell_type"] for item in manifest["domains"]] == ["triangle", "line"]


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


@pytest.mark.parametrize(
    ("cell_type", "points", "cells", "prune_z"),
    (
        (
            "quad",
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 1, 2, 3]],
            True,
        ),
        (
            "hexahedron",
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            False,
        ),
    ),
)
def test_verified_tensor_product_import_is_readable_and_quality_audited(
    tmp_path, cell_type, points, cells, prune_z
):
    meshio = pytest.importorskip("meshio")
    from mpi4py import MPI

    source = tmp_path / f"{cell_type}.vtu"
    meshio.write(
        source,
        meshio.Mesh(
            points=np.asarray(points, dtype=float),
            cells=[(cell_type, np.asarray(cells, dtype=int))],
        ),
    )
    conversion = formats.convert_to_xdmf(
        source,
        tmp_path / f"{cell_type}.xdmf",
        cell_type=cell_type,
        prune_z=prune_z,
    )
    imported = mesh_api.read_converted_xdmf(conversion, comm=MPI.COMM_SELF)

    assert mesh_api.audit_quality(imported.domain, strict=True).acceptable
    manifest = json.loads(conversion.manifest_path.read_text(encoding="utf-8"))
    assert manifest["output"]["compatibility"]["solver_ready"] is True


@pytest.mark.parametrize(
    "cell_type",
    ("triangle6", "quad9", "tetra10", "hexahedron20", "hexahedron27"),
)
def test_verified_high_order_import_preserves_geometry_quality_and_p2_patch(
    tmp_path, cell_type
):
    meshio = pytest.importorskip("meshio")
    points, prune_z, dimension, expected_basis = _high_order_external_case(
        cell_type
    )
    source = tmp_path / f"{cell_type}.vtu"
    meshio.write(
        source,
        meshio.Mesh(
            points=points,
            cells=[
                (
                    cell_type,
                    np.arange(len(points), dtype=int).reshape(1, -1),
                )
            ],
        ),
    )
    conversion = formats.convert_to_xdmf(
        source,
        tmp_path / f"{cell_type}.xdmf",
        cell_type=cell_type,
        prune_z=prune_z,
    )
    imported = mesh_api.read_converted_xdmf(conversion, comm=MPI.COMM_SELF)

    geometry = mesh_api.describe_geometry(imported.domain)
    quality = mesh_api.audit_quality(
        imported.domain,
        threshold=0.8,
        strict=True,
    )
    assert geometry.degree == 2
    assert geometry.basis_family == expected_basis
    assert geometry.nodes_per_cell == len(points)
    assert quality.acceptable
    assert quality.minimum > 0.8

    space = fem.functionspace(imported.domain, ("Lagrange", 2))
    field = fem.Function(space)
    if dimension == 2:
        field.interpolate(lambda x: x[0] + 2.0 * x[1] + 0.25)
        exact_gradient = ufl.as_vector((1.0, 2.0))
    else:
        field.interpolate(
            lambda x: x[0] + 2.0 * x[1] - 0.5 * x[2] + 0.25
        )
        exact_gradient = ufl.as_vector((1.0, 2.0, -0.5))
    gradient_error = ufl.grad(field) - exact_gradient
    error = fem.assemble_scalar(
        fem.form(ufl.inner(gradient_error, gradient_error) * ufl.dx)
    )
    assert error == pytest.approx(0.0, abs=5.0e-13)

    manifest = json.loads(conversion.manifest_path.read_text(encoding="utf-8"))
    assert manifest["output"]["compatibility"]["solver_ready"] is True
    assert not any(
        "conditional neutral-geometry" in warning
        for warning in conversion.warnings
    )


def test_quad8_conversion_is_inspectable_but_solver_read_fails_with_context(
    tmp_path,
):
    meshio = pytest.importorskip("meshio")
    points, prune_z, _dimension, _basis = _high_order_external_case("quad8")
    source = tmp_path / "quad8.vtu"
    meshio.write(
        source,
        meshio.Mesh(
            points=points,
            cells=[
                (
                    "quad8",
                    np.arange(len(points), dtype=int).reshape(1, -1),
                )
            ],
        ),
    )
    conversion = formats.convert_to_xdmf(
        source,
        tmp_path / "quad8.xdmf",
        cell_type="quad8",
        prune_z=prune_z,
    )

    assert any(
        "conditional neutral-geometry" in warning
        for warning in conversion.warnings
    )
    with pytest.raises(
        RuntimeError,
        match="quad8.*conditional geometry route",
    ):
        mesh_api.read_converted_xdmf(conversion, comm=MPI.COMM_SELF)


def test_verified_linear_prism_import_preserves_patch_quality_and_output(
    tmp_path,
):
    meshio = pytest.importorskip("meshio")
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    )
    source = tmp_path / "wedge.vtu"
    meshio.write(
        source,
        meshio.Mesh(
            points=points,
            cells=[("wedge", np.arange(6, dtype=int).reshape(1, -1))],
        ),
    )
    conversion = formats.convert_to_xdmf(
        source,
        tmp_path / "wedge.xdmf",
        cell_type="wedge",
    )
    imported = mesh_api.read_converted_xdmf(conversion, comm=MPI.COMM_SELF)

    geometry = mesh_api.describe_geometry(imported.domain)
    quality = mesh_api.audit_quality(
        imported.domain,
        threshold=0.9,
        strict=True,
    )
    assert geometry.cell_type == "prism"
    assert geometry.degree == 1
    assert geometry.nodes_per_cell == 6
    assert quality.minimum == pytest.approx(1.0)

    scalar_space = fem.functionspace(imported.domain, ("Lagrange", 1))
    scalar = fem.Function(scalar_space)
    scalar.interpolate(lambda x: x[0] + 2.0 * x[1] - 0.5 * x[2] + 0.25)
    gradient_error = ufl.grad(scalar) - ufl.as_vector((1.0, 2.0, -0.5))
    error = fem.assemble_scalar(
        fem.form(ufl.inner(gradient_error, gradient_error) * ufl.dx)
    )
    assert error == pytest.approx(0.0, abs=2.0e-14)

    displacement_space = fem.functionspace(
        imported.domain,
        ("Lagrange", 1, (3,)),
    )
    displacement = fem.Function(displacement_space, name="U")
    displacement.interpolate(
        lambda x: np.vstack((0.01 * x[0], 0.02 * x[1], -0.01 * x[2]))
    )
    output = results.write_unified_xdmf_series(
        tmp_path / "wedge_fields.xdmf",
        (SimpleNamespace(solution=displacement, load_factor=1.0),),
        ((),),
        deformation_scale=0.0,
    )
    topology = ET.parse(output).find(
        ".//Grid[@GridType='Uniform']/Topology"
    )
    assert topology is not None
    assert topology.attrib["TopologyType"] == "Wedge"
    assert topology.find("DataItem").attrib["Dimensions"] == "1 6"

    manifest = json.loads(conversion.manifest_path.read_text(encoding="utf-8"))
    assert manifest["output"]["compatibility"]["solver_ready"] is True
    assert not conversion.warnings

    quadratic = fem.Function(
        fem.functionspace(imported.domain, ("Lagrange", 2, (3,))),
        name="U",
    )
    with pytest.raises(NotImplementedError, match="prism.*18 nodes"):
        results.write_unified_xdmf_series(
            tmp_path / "unsupported_wedge_p2_fields.xdmf",
            (SimpleNamespace(solution=quadratic, load_factor=1.0),),
            ((),),
        )


def test_unknown_external_cell_type_is_not_silently_converted(tmp_path, monkeypatch):
    source = _Mesh(
        points=np.zeros((42, 3)),
        cells=[("polyhedron42", [list(range(42))])],
    )
    monkeypatch.setattr(formats, "require_meshio", lambda: _MeshIO(source))

    with pytest.raises(ValueError, match="no declared AgentFEM"):
        formats.convert_to_xdmf(
            tmp_path / "unknown.mesh",
            tmp_path / "unknown.xdmf",
            cell_type="polyhedron42",
        )


def test_c3d10h_conversion_keeps_hybrid_identity_beside_tetra10(tmp_path):
    pytest.importorskip("meshio")
    source = tmp_path / "one_hybrid_tet.inp"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., 1.",
                "5, .5, 0., 0.",
                "6, .5, .5, 0.",
                "7, 0., .5, 0.",
                "8, 0., 0., .5",
                "9, .5, 0., .5",
                "10, 0., .5, .5",
                "*Element, type=C3D10H, elset=SOLID",
                "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            )
        ),
        encoding="utf-8",
    )

    conversion = formats.convert_abaqus_inp_to_xdmf(
        source,
        tmp_path / "hybrid.xdmf",
        cell_type="tetra10",
    )
    manifest = json.loads(conversion.manifest_path.read_text(encoding="utf-8"))

    assert conversion.cell_type == "tetra10"
    definition = conversion.source_metadata["abaqus_element_definitions"][0]
    assert definition["source_type"] == "C3D10H"
    assert definition["pressure_interpolation"] == "constant"
    assert manifest["source"]["format_metadata"] == conversion.source_metadata
    assert manifest["source"]["fingerprint"]["algorithm"] == "sha256"
    assert "tetra10 geometry alone is not C3D10H equivalence" in conversion.warnings[0]

    cached = formats.reusable_conversion(
        source, tmp_path / "hybrid.xdmf", cell_type="tetra10",
    )
    assert cached is not None

    source.write_text(source.read_text(encoding="utf-8") + "\n** changed\n", encoding="utf-8")
    assert formats.reusable_conversion(
        source, tmp_path / "hybrid.xdmf", cell_type="tetra10",
    ) is None


def test_c3d10h_conversion_keeps_formulation_derivation_provenance(tmp_path):
    pytest.importorskip("meshio")
    source = tmp_path / "one_tet.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., 1.",
                "5, .5, 0., 0.",
                "6, .5, .5, 0.",
                "7, 0., .5, 0.",
                "8, 0., 0., .5",
                "9, .5, 0., .5",
                "10, 0., .5, .5",
                "*Element, type=C3D10, elset=SOLID",
                "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            )
        ),
        encoding="utf-8",
    )
    derived = tmp_path / "one_tet_hybrid.inp"
    abaqus.derive_element_formulation(
        source, derived, source_type="C3D10", target_type="C3D10H",
    )

    conversion = formats.convert_abaqus_inp_to_xdmf(
        derived, tmp_path / "derived.xdmf", cell_type="tetra10",
    )

    provenance = conversion.source_metadata[
        "abaqus_element_formulation_derivation"
    ]
    assert provenance["source_type"] == "C3D10"
    assert provenance["target_type"] == "C3D10H"
    assert provenance["nodes_and_connectivity_preserved"] is True
