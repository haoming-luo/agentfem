from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import cli, constraints, fields, mesh

from agentfem.constraints.affine import (
    _build_reduction,
    _expand_semantic_relations,
)
from agentfem.mesh import abaqus
from agentfem.mesh import abaqus_lowering, abaqus_migration


CASE_INPUT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "abaqus_c3d10h_periodic_cell"
    / "input"
)


def test_versioned_periodic_cell_is_a_direct_c3d10h_source():
    source = CASE_INPUT / "periodic_cell_c3d10h.dat"
    definitions = abaqus.read_element_definitions(source)
    equations = abaqus.read_equations(
        CASE_INPUT / "periodic_cell_equations.mpc"
    )

    assert [item.source_type for item in definitions] == ["C3D10H"]
    assert definitions[0].formulation == "hybrid"
    assert definitions[0].pressure_interpolation == "constant"
    assert definitions[0].additional_pressure_variables == 1
    assert len(equations.equations) == 4212


def test_abaqus_node_reader_preserves_nonconsecutive_labels(tmp_path):
    source = tmp_path / "mesh.dat"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node, Nset=ALLN",
                "10, 0., 0., 0.",
                "25, 1., 0., 0.",
                "*Element, type=C3D10",
                "1, 10, 25, 10, 25, 10, 25, 10, 25, 10, 25",
            )
        ),
        encoding="utf-8",
    )

    nodes = abaqus.read_node_table(source)

    assert nodes.labels.tolist() == [10, 25]
    np.testing.assert_allclose(nodes.coordinate(25), [1.0, 0.0, 0.0])


def test_abaqus_element_reader_preserves_c3d10h_formulation_identity(tmp_path):
    source = tmp_path / "hybrid.inp"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Element, type=C3D10H, elset=RUBBER",
                "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            )
        ),
        encoding="utf-8",
    )

    definitions = abaqus.read_element_definitions(source)

    assert len(definitions) == 1
    assert definitions[0].source_type == "C3D10H"
    assert definitions[0].topology == "tetra10"
    assert definitions[0].formulation == "hybrid"
    assert definitions[0].pressure_interpolation == "constant"
    assert definitions[0].additional_pressure_variables == 1


def test_abaqus_source_graph_tracks_nested_includes_and_content_identity(tmp_path):
    root = tmp_path / "model.inp"
    mesh_source = tmp_path / "parts" / "mesh.inp"
    sets_source = tmp_path / "parts" / "sets.inp"
    mesh_source.parent.mkdir()
    root.write_text("*Heading\n*Include, input='parts/mesh.inp'\n", encoding="utf-8")
    mesh_source.write_text(
        "*Node\n1,0,0,0\n*Include, input=sets.inp\n", encoding="utf-8"
    )
    sets_source.write_text("*Nset, nset=FIXED\n1\n", encoding="utf-8")

    graph = abaqus.read_source_graph(root)
    first_fingerprint = graph.fingerprint

    assert graph.complete is True
    assert [item.logical_path for item in graph.files] == [
        "model.inp",
        "parts/mesh.inp",
        "parts/sets.inp",
    ]
    assert [item.status for item in graph.edges] == ["resolved", "resolved"]
    sets_source.write_text("*Nset, nset=FIXED\n1,2\n", encoding="utf-8")
    assert abaqus.read_source_graph(root).fingerprint != first_fingerprint


def test_abaqus_source_graph_reports_missing_and_recursive_includes(tmp_path):
    root = tmp_path / "model.inp"
    child = tmp_path / "child.inp"
    root.write_text(
        "*Include, input=child.inp\n*Include, input=missing.inp\n",
        encoding="utf-8",
    )
    child.write_text("*Include, input=model.inp\n", encoding="utf-8")

    report = abaqus.inspect_input(root)
    summary = report.summary()["source_graph"]

    assert summary["complete"] is False
    assert [item["status"] for item in summary["edges"]] == [
        "resolved",
        "cycle",
        "missing",
    ]
    assert any("AFM-ABAQUS-INCLUDE-002" in item for item in summary["issues"])
    assert any("AFM-ABAQUS-INCLUDE-003" in item for item in summary["issues"])


def test_c3d10h_derivation_changes_only_element_formulation_keyword(tmp_path):
    source = tmp_path / "quadratic.dat"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node",
                "1, 0., 0., 0.",
                "*Element, type=C3D10, elset=SOLID",
                "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
                "*Elset, elset=KEEP_C3D10_TEXT",
                "1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    derived = tmp_path / "quadratic_c3d10h.dat"

    evidence = abaqus.derive_element_formulation(
        source,
        derived,
        source_type="C3D10",
        target_type="C3D10H",
    )

    source_lines = source.read_text(encoding="utf-8").splitlines()
    derived_lines = derived.read_text(encoding="utf-8").splitlines()
    changed = [
        (before, after)
        for before, after in zip(source_lines, derived_lines)
        if before != after
    ]
    assert changed == [
        (
            "*Element, type=C3D10, elset=SOLID",
            "*Element, type=C3D10H, elset=SOLID",
        )
    ]
    assert abaqus.read_element_table(source).elements[0].connectivity == (
        abaqus.read_element_table(derived).elements[0].connectivity
    )
    assert evidence.rewritten_declarations == 1
    assert evidence.source_sha256 != evidence.derived_sha256
    assert abaqus.read_element_formulation_derivation(derived)[
        "topology_preserved"
    ] is True


def test_element_formulation_derivation_rejects_topology_change(tmp_path):
    source = tmp_path / "mesh.inp"
    source.write_text("*Element, type=C3D10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identical topology"):
        abaqus.derive_element_formulation(
            source,
            tmp_path / "bad.inp",
            source_type="C3D10",
            target_type="C3D8",
        )


def test_abaqus_element_table_preserves_connectivity_and_official_face_order(tmp_path):
    source = tmp_path / "solid.inp"
    source.write_text(
        "\n".join(
            (
                "*Element, type=C3D10H, elset=SOLID",
                "42, 11, 12, 13, 14, 15, 16,",
                "17, 18, 19, 20",
            )
        ),
        encoding="utf-8",
    )

    element = abaqus.read_element_table(source).element(42)

    assert element.connectivity == tuple(range(11, 21))
    assert element.face_corner_labels("S1") == (11, 12, 13)
    assert element.face_corner_labels("S2") == (11, 14, 12)
    assert element.face_corner_labels("S3") == (12, 14, 13)
    assert element.face_corner_labels("S4") == (13, 14, 11)


def test_abaqus_semantics_preserve_sets_and_element_face_surfaces(tmp_path):
    source = tmp_path / "sets.inp"
    source.write_text(
        "\n".join(
            (
                "*Node, nset=ALLN",
                "10, 0., 0., 0.",
                "20, 1., 0., 0.",
                "*Nset, nset=FIXED",
                "10",
                "*Elset, elset=SOLID, generate",
                "1, 5, 2",
                "*Surface, name=LOAD_FACE, type=ELEMENT",
                "SOLID, S2",
            )
        ),
        encoding="utf-8",
    )

    semantics = abaqus.read_model_semantics(source)

    assert semantics.node_sets[0].name == "ALLN"
    assert semantics.node_sets[1].labels == (10,)
    assert semantics.element_sets[0].labels == (1, 3, 5)
    assert semantics.surfaces[0].entries[0].reference == "SOLID"
    assert semantics.surfaces[0].entries[0].face == "S2"

    imported = abaqus.AbaqusMeshImport(
        fem_mesh=SimpleNamespace(), nodes=SimpleNamespace(),
        conversion=SimpleNamespace(source_metadata={"abaqus_model_semantics": semantics.summary()}),
    )
    assert imported.surface_faces("LOAD_FACE") == ((1, "S2"), (3, "S2"), (5, "S2"))


def test_abaqus_elset_and_internal_surface_lower_to_three_dimensional_cohesive_interface(
    tmp_path,
):
    source = tmp_path / "cohesive_tetra.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., -1.",
                "5, 0., 0., 1.",
                "*Element, type=C3D4, elset=ALL",
                "10, 1, 2, 3, 4",
                "20, 1, 3, 2, 5",
                "*Elset, elset=POSITIVE",
                "20",
                "*Elset, elset=INTERFACE_OWNER",
                "10",
                "*Surface, name=WEAK_INTERFACE, type=ELEMENT",
                "INTERFACE_OWNER, S1",
            )
        ),
        encoding="utf-8",
    )
    semantics = abaqus.read_model_semantics(source)
    imported = abaqus.AbaqusMeshImport(
        fem_mesh=SimpleNamespace(),
        nodes=abaqus.read_node_table(source),
        conversion=SimpleNamespace(
            source_path=source,
            source_metadata={"abaqus_model_semantics": semantics.summary()},
        ),
    )

    split = imported.cohesive_interface(
        positive_elset="POSITIVE", surface="WEAK_INTERFACE"
    )

    assert split.summary()["geometric_dimension"] == 3
    assert split.negative_facets.shape == (1, 3)
    np.testing.assert_allclose(
        split.coordinates[split.negative_facets],
        split.coordinates[split.positive_facets],
    )


def test_abaqus_surface_becomes_a_load_ready_boundary_region(tmp_path):
    pytest.importorskip("meshio")
    source = tmp_path / "one_tetra.inp"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., 1.",
                "*Nset, nset=FIXED",
                "1, 4",
                "*Element, type=C3D4, elset=SOLID",
                "1, 1, 2, 3, 4",
                "*Surface, name=LOADED, type=ELEMENT",
                "SOLID, S1",
            )
        ),
        encoding="utf-8",
    )
    imported = mesh.read_abaqus_mesh(
        source,
        tmp_path / "converted.xdmf",
        comm=MPI.COMM_SELF,
        cell_type="tetra",
        reuse_conversion=False,
    )

    loaded = imported.boundary("LOADED", tag=17)
    fixed_nodes = imported.node_set("FIXED")
    fixed = constraints.fixed(fields.displacement(imported.domain), on=fixed_nodes)
    evidence = loaded.audit(strict=True)

    assert loaded.name == "LOADED"
    assert evidence["global_tagged_facets"] == 1
    assert evidence["measure"] == pytest.approx(0.5)
    assert fixed_nodes.summary()["global_nodes"] == 2
    assert len(fixed.bcs) == 3


def test_abaqus_node_set_retains_c3d10_midside_nodes_for_p2_constraints(tmp_path):
    pytest.importorskip("meshio")
    source = tmp_path / "quadratic_tetra.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., 1.",
                "5, 0.5, 0., 0.",
                "6, 0.5, 0.5, 0.",
                "7, 0., 0.5, 0.",
                "8, 0., 0., 0.5",
                "9, 0.5, 0., 0.5",
                "10, 0., 0.5, 0.5",
                "*Nset, nset=MIDSIDE",
                "5",
                "*Element, type=C3D10, elset=SOLID",
                "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            )
        ),
        encoding="utf-8",
    )
    imported = mesh.read_abaqus_mesh(
        source,
        tmp_path / "quadratic_tetra.xdmf",
        comm=MPI.COMM_SELF,
        cell_type="tetra10",
        reuse_conversion=False,
    )
    midside = imported.node_set("MIDSIDE")
    fixed = constraints.fixed(
        fields.displacement(imported.domain, degree=2), on=midside
    )

    assert midside.summary()["global_nodes"] == 1
    assert midside.marker(np.asarray(((0.5,), (0.0,), (0.0,)))).tolist() == [True]
    assert all(len(bc.dof_indices()[0]) == 1 for bc in fixed.bcs)


def test_imported_hybrid_identity_cannot_be_silently_consumed_as_displacement_only():
    imported = abaqus.AbaqusMeshImport(
        fem_mesh=SimpleNamespace(),
        nodes=SimpleNamespace(),
        conversion=SimpleNamespace(
            source_metadata={
                "abaqus_element_definitions": [
                    abaqus.describe_element_type("C3D10H").summary()
                ]
            }
        ),
    )

    with pytest.raises(NotImplementedError, match="C3D10H.*hybrid"):
        imported.require_formulation(
            "displacement",
            operation="ordinary hyperelastic step",
        )


def test_abaqus_equation_reader_supports_continued_term_lines(tmp_path):
    source = tmp_path / "periodic.mpc"
    source.write_text(
        "\n".join(
            (
                "*Equation",
                "4",
                "20, 1, -1., 10, 1, 1.,",
                "7, 1, 1., 1, 1, -1.",
                "2",
                "30, 2, 1., 11, 2, -1.",
            )
        ),
        encoding="utf-8",
    )

    equations = abaqus.read_equations(source)

    assert len(equations.equations) == 2
    assert equations.equations[0].slave == (20, 1)
    assert [term.coefficient for term in equations.equations[0].terms] == [
        -1.0,
        1.0,
        1.0,
        -1.0,
    ]


def test_affine_reduction_resolves_chains_and_prescribed_offsets():
    # u0 = 2, u2 = u1 + u0, u3 = u2 - u0.  Only u1 is independent.
    reduction = _build_reduction(
        4,
        relations={
            2: {1: 1.0, 0: 1.0},
            3: {2: 1.0, 0: -1.0},
        },
        prescribed={0: 2.0},
    )

    np.testing.assert_allclose(reduction.reconstruct([5.0]), [2.0, 5.0, 7.0, 5.0])
    assert reduction.independent_full_dofs.tolist() == [1]
    assert reduction.eliminated_count == 3


def test_affine_reduction_rejects_constraint_cycles():
    try:
        _build_reduction(2, relations={0: {1: 1.0}, 1: {0: 1.0}}, prescribed={})
    except ValueError as exc:
        assert "Cyclic" in str(exc)
    else:
        raise AssertionError("A cyclic affine relation must be rejected.")


def test_semantic_chain_can_resolve_to_a_zero_correction():
    relations = {
        (20, 0): {(10, 0): 1.0, (1, 0): -1.0},
        (10, 0): {(1, 0): 1.0},
    }

    expanded = _expand_semantic_relations(relations, {(1, 0)})

    assert expanded[(10, 0)] == {}
    assert expanded[(20, 0)] == {}


def test_periodic_cell_volume_uses_control_node_lattice():
    nodes = abaqus.AbaqusNodeTable(
        labels=np.asarray([1, 7, 9, 4]),
        coordinates=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.5, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ),
    )

    assert abaqus.periodic_cell_volume(
        nodes,
        anchor_node=1,
        reference_nodes=(7, 9, 4),
    ) == pytest.approx(24.0)


def test_reviewed_abaqus_native_draft_runs_to_structured_result(tmp_path, capsys):
    pytest.importorskip("meshio")
    source = tmp_path / "legacy-static.inp"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node",
                "1,0,0,0",
                "2,1,0,0",
                "3,0,1,0",
                "4,0,0,1",
                "5,1,1,1",
                "*Nset, nset=FIXED",
                "1,2,3",
                "*Nset, nset=MOVED",
                "5",
                "*Element, type=C3D4, elset=SOLID",
                "1,1,2,3,4",
                "2,2,3,4,5",
                "*Material, name=STEEL",
                "*Elastic",
                "210000.,0.3",
                "*Density",
                "7.85e-9",
                "*Solid Section, elset=SOLID, material=STEEL",
                "*Step, name=PULL",
                "*Static",
                "*Boundary",
                "FIXED,1,3,0.",
                "MOVED,1,2,0.",
                "MOVED,3,3,0.01",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    project = tmp_path / "native-project"
    abaqus_migration.create_project(source, project, created_with="test")
    abaqus_lowering.lower_project(
        project,
        reviewed_by="Regression Test",
        unit_system="mm-N-s",
        activate=True,
    )

    assert cli.main(["run", "--project", str(project), "--json"]) == 0
    execution = capsys.readouterr().out
    assert '"status": "completed"' in execution
    latest = project / "outputs" / "native-project" / "latest.json"
    assert latest.is_file()


def test_reviewed_abaqus_surface_pressure_lowers_and_runs(tmp_path, capsys):
    pytest.importorskip("meshio")
    source = tmp_path / "pressure.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1,0,0,0",
                "2,1,0,0",
                "3,1,1,0",
                "4,0,1,0",
                "5,0,0,1",
                "6,1,0,1",
                "7,1,1,1",
                "8,0,1,1",
                "*Nset, nset=FIXED",
                "1,2,3,4",
                "*Element, type=C3D8, elset=SOLID",
                "1,1,2,3,4,5,6,7,8",
                "*Surface, name=LOADED, type=ELEMENT",
                "SOLID,S2",
                "*Material, name=MAT",
                "*Elastic",
                "1000.,0.3",
                "*Density",
                "1.0",
                "*Solid Section, elset=SOLID, material=MAT",
                "*Step, name=PRESSURE",
                "*Static",
                "*Boundary",
                "FIXED,1,3,0.",
                "*Dsload",
                "LOADED,P,1.0",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    project = tmp_path / "pressure-project"
    abaqus_migration.create_project(source, project, created_with="test")
    abaqus_lowering.lower_project(
        project,
        reviewed_by="Regression Test",
        unit_system="SI",
        activate=True,
    )

    assert cli.main(["run", "--project", str(project), "--json"]) == 0
    execution = capsys.readouterr().out
    assert '"status": "completed"' in execution
    native = (project / "case.native.py").read_text(encoding="utf-8")
    assert "cell.boundary('LOADED'" in native
    assert "model.pressure(" in native
