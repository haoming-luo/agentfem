from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
from mpi4py import MPI

from agentfem import mesh as fem_mesh
from agentfem.mesh import abaqus
from agentfem.mesh import abaqus_lowering
from agentfem.mesh import abaqus_migration


FIXTURES = Path(__file__).parent / "fixtures" / "abaqus_migration"
NATIVE_FIXTURES = Path(__file__).parent / "fixtures" / "abaqus_native_lowering"
GOLDENS = Path(__file__).parent / "goldens"


def _migration_signature(plan):
    summary = plan.summary()
    return {
        "schema": summary["schema"],
        "status": summary["status"],
        "ready_to_solve": summary["ready_to_solve"],
        "source_complete": summary["source_graph"]["complete"],
        "source_files": [
            item["logical_path"] for item in summary["source_graph"]["files"]
        ],
        "parts": [item["name"] for item in summary["parts"]],
        "instances": [
            {key: item[key] for key in ("name", "part", "scope")}
            for item in summary["instances"]
        ],
        "regions": [item["key"] for item in summary["regions"]],
        "element_blocks": [
            {
                key: item[key]
                for key in (
                    "scope",
                    "region",
                    "source_type",
                    "topology",
                    "solver_capability",
                )
            }
            for item in summary["element_blocks"]
        ],
        "materials": [
            {
                "name": item["name"],
                "translation_status": item["translation_status"],
                "constructor": item["native_candidate"].get("constructor"),
            }
            for item in summary["materials"]
        ],
        "sections": [
            {key: item[key] for key in ("scope", "region", "material", "status")}
            for item in summary["sections"]
        ],
        "effective_assignments": [
            {
                key: item[key]
                for key in (
                    "target_scope",
                    "region",
                    "material",
                    "inherited_from_part",
                )
            }
            for item in summary["effective_assignments"]
        ],
        "issue_codes": [item["code"] for item in summary["issues"]],
        "pending_assets": [
            {key: item[key] for key in ("category", "keyword", "scope", "step", "rows")}
            for item in summary["pending_assets"]
        ],
    }


def _native_signature(assessment):
    summary = assessment.summary()
    return {
        "status": summary["status"],
        "dimension": summary["dimension"],
        "assumption": summary["assumption"],
        "topology": summary["topology"],
        "degree": summary["degree"],
        "material_name": summary["material_name"],
        "material": summary["material"],
        "step_name": summary["step_name"],
        "boundaries": [
            {key: item[key] for key in ("region", "components", "value")}
            for item in summary["boundaries"]
        ],
        "pressures": summary["pressures"],
        "gravities": summary["gravities"],
        "finding_codes": [item["code"] for item in summary["findings"]],
    }


def test_scope_aware_migration_plan_matches_golden_contract():
    plan = abaqus_migration.plan(FIXTURES / "model.inp")
    expected = json.loads(
        (GOLDENS / "abaqus_migration_plan.json").read_text(encoding="utf-8")
    )

    assert _migration_signature(plan) == expected


def test_native_lowering_matches_versioned_golden_contract():
    assessment = abaqus_lowering.assess(
        abaqus_migration.plan(NATIVE_FIXTURES / "static.inp")
    )
    expected = json.loads(
        (GOLDENS / "abaqus_native_lowering.json").read_text(encoding="utf-8")
    )

    assert _native_signature(assessment) == expected


def test_element_catalog_separates_topology_from_formulation():
    reduced = abaqus.describe_element_type("C3D8R")
    hybrid = abaqus.describe_element_type("C3D10H")
    plane = abaqus.describe_element_type("CPE8R")
    thermal = abaqus.describe_element_type("DC3D20")
    shell = abaqus.describe_element_type("S4R")

    assert reduced.topology == "hexahedron"
    assert reduced.integration == "reduced"
    assert reduced.solver_capability == "topology_only"
    assert reduced.neutral_conversion == "meshio_reader"
    assert "hourglass" in reduced.notes[0].lower()
    assert hybrid.topology == "tetra10"
    assert hybrid.pressure_interpolation == "constant"
    assert hybrid.solver_capability == "native_mixed_analogue"
    assert plane.kinematics == "plane_strain"
    assert plane.node_count == 8
    assert thermal.physics == "heat_transfer"
    assert thermal.topology == "hexahedron20"
    assert shell.family == "shell"
    assert shell.solver_capability == "topology_only"
    assert abaqus.describe_element_type("C3D27H").neutral_conversion == "not_verified"


def test_supported_element_types_can_be_filtered_by_family():
    solids = abaqus.supported_element_types(family="continuum_solid")
    cohesive = abaqus.supported_element_types(family="cohesive")

    assert {"C3D4", "C3D8R", "C3D10H", "C3D20R", "CPE4R", "CAX8R"} <= set(solids)
    assert {"COH2D4", "COH3D6", "COH3D8"} <= set(cohesive)


def test_broad_element_connectivity_is_retained_without_claiming_equivalence(tmp_path):
    source = tmp_path / "mixed.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1, 0., 0., 0.",
                "*Element, type=C3D20R, elset=SOLID",
                "10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,",
                "11, 12, 13, 14, 15, 16, 17, 18, 19, 20",
                "*Element, type=COH3D8, elset=INTERFACE",
                "20, 1, 2, 3, 4, 5, 6, 7, 8",
            )
        ),
        encoding="utf-8",
    )

    table = abaqus.read_element_table(source)

    assert len(table.element(10).connectivity) == 20
    assert len(table.element(20).connectivity) == 8
    assert (
        abaqus.read_element_definitions(source)[0].solver_capability == "topology_only"
    )


def test_inspection_reports_preserved_and_not_lowered_assets(tmp_path):
    source = tmp_path / "model.inp"
    source.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node, nset=ALL",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 1., 1., 0.",
                "4, 0., 1., 0.",
                "5, 0., 0., 1.",
                "6, 1., 0., 1.",
                "7, 1., 1., 1.",
                "8, 0., 1., 1.",
                "*Element, type=C3D8R, elset=SOLID",
                "1, 1, 2, 3, 4, 5, 6, 7, 8",
                "*Surface, name=LOAD, type=ELEMENT",
                "SOLID, S2",
                "*Material, name=STEEL",
                "*Elastic",
                "2.0e11, 0.3",
                "*Solid Section, elset=SOLID, material=STEEL",
                "*Step",
                "*Static",
                "*Boundary",
                "ALL, 1, 1, 0.",
                "*End Step",
            )
        ),
        encoding="utf-8",
    )

    report = abaqus.inspect_input(source)
    summary = report.summary()

    assert report.node_count == 8
    assert report.element_count == 1
    assert report.topology_only_elements == ("C3D8R",)
    assert summary["source_sha256"]
    statuses = {item[0]: item[2] for item in report.keyword_inventory}
    assert statuses["*NODE"] == "preserved"
    assert statuses["*MATERIAL"] == "recognized_not_lowered"
    assert "C3D8R" in report.text()
    assert any("explicit migration decision" in item for item in report.warnings)


def test_inspection_accepts_part_scoped_duplicate_labels_and_reports_scope(tmp_path):
    source = tmp_path / "assembly.inp"
    source.write_text(
        "\n".join(
            (
                "*Part, name=LEFT",
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., 1.",
                "*Element, type=C3D4",
                "1, 1, 2, 3, 4",
                "*End Part",
                "*Part, name=RIGHT",
                "*Node",
                "1, 1., 0., 0.",
                "2, 2., 0., 0.",
                "3, 1., 1., 0.",
                "4, 1., 0., 1.",
                "*Element, type=C3D4",
                "1, 1, 2, 3, 4",
                "*End Part",
                "*Assembly, name=A",
                "*Instance, name=LEFT-1, part=LEFT",
                "*End Instance",
                "*Instance, name=RIGHT-1, part=RIGHT",
                "*End Instance",
                "*End Assembly",
            )
        ),
        encoding="utf-8",
    )

    report = abaqus.inspect_input(source)

    assert report.node_count == 8
    assert report.element_count == 2
    assert report.part_names == ("LEFT", "RIGHT")
    assert report.instance_names == ("LEFT-1", "RIGHT-1")
    assert any("instance-aware" in item for item in report.warnings)


def test_inspection_resolves_include_graph_without_flattening_scopes(tmp_path):
    source = tmp_path / "root.inp"
    source.write_text(
        "*Heading\n*Include, input=mesh.inc\n*Include, input=missing.inc\n",
        encoding="utf-8",
    )
    (tmp_path / "mesh.inc").write_text("*Node\n1, 0., 0., 0.\n", encoding="utf-8")

    report = abaqus.inspect_input(source)

    assert report.include_files == ("mesh.inc", "missing.inc")
    assert [item.logical_path for item in report.source_graph.files] == [
        "root.inp",
        "mesh.inc",
    ]
    assert [item.status for item in report.source_graph.edges] == [
        "resolved",
        "missing",
    ]
    assert report.source_graph.complete is False
    assert any("not flattened" in item for item in report.warnings)
    assert any("missing.inc" in item for item in report.warnings)


def test_inspection_does_not_execute_or_reject_scoped_equation_terms(tmp_path):
    source = tmp_path / "scoped-equation.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1, 0., 0., 0.",
                "*Equation",
                "2",
                "LEFT_FACE, 1, 1.0, RIGHT_FACE, 1, -1.0",
            )
        ),
        encoding="utf-8",
    )

    report = abaqus.inspect_input(source)

    assert report.equation_count is None
    assert any("scoped set/instance" in item for item in report.warnings)


def test_scope_aware_plan_resolves_part_section_material_and_instance(tmp_path):
    root = tmp_path / "model.inp"
    mesh_source = tmp_path / "mesh.inc"
    material_source = tmp_path / "steel.inc"
    root.write_text(
        "\n".join(
            (
                "*Part, name=BRACKET",
                "*Include, input=mesh.inc",
                "*Solid Section, elset=SOLID, material=STEEL",
                "*End Part",
                "*Material, name=STEEL",
                "*Include, input=steel.inc",
                "*Assembly, name=ASSEMBLY",
                "*Instance, name=BRACKET-1, part=BRACKET",
                "1.0, 2.0, 3.0",
                "*End Instance",
                "*End Assembly",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    mesh_source.write_text(
        "*Node\n1,0,0,0\n*Element, type=C3D4, elset=SOLID\n1,1,1,1,1\n",
        encoding="utf-8",
    )
    material_source.write_text(
        "*Elastic\n210000.,0.3\n*Density\n7.85e-9\n",
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(root)

    assert plan.blocked is False
    assert plan.ready_to_solve is False
    assert [item.name for item in plan.parts] == ["BRACKET"]
    assert plan.instances[0].part == "BRACKET"
    assert plan.instances[0].positioning == (("1.0", "2.0", "3.0"),)
    assert plan.sections[0].scope == "part:BRACKET"
    assert plan.sections[0].region_resolved is True
    assert plan.sections[0].material_resolved is True
    assert plan.effective_assignments[0].target_scope == ("instance:ASSEMBLY/BRACKET-1")
    assert plan.effective_assignments[0].region == "BRACKET-1.SOLID"
    assert plan.effective_assignments[0].inherited_from_part is True
    assert plan.materials[0].translation_status == "native_candidate"
    assert plan.materials[0].native_candidate["young"] == 210000.0
    assert plan.materials[0].native_candidate["density"] == 7.85e-9
    assert plan.element_blocks[0].definition.source_type == "C3D4"
    assert plan.element_blocks[0].scope == "part:BRACKET"
    assert plan.source_graph.fingerprint


def test_plan_preserves_step_load_boundary_and_output_rows_for_lowering(tmp_path):
    source = tmp_path / "loaded.inp"
    source.write_text(
        "\n".join(
            (
                "*Step, name=PULL",
                "*Static",
                "0.1, 1.0, 1e-5, 0.1",
                "*Boundary",
                "FIXED, 1, 3, 0.0",
                "*Cload, amplitude=RAMP",
                "LOAD, 2, -100.0",
                "*Output, field",
                "*Node Output",
                "U, RF",
                "*End Step",
            )
        ),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)

    assert [item.category for item in plan.pending_assets] == [
        "step",
        "procedure",
        "boundary_condition",
        "load",
        "output_request",
        "output_request",
    ]
    assert all(item.step == "PULL" for item in plan.pending_assets)
    assert plan.pending_assets[1].rows == (("0.1", "1.0", "1e-5", "0.1"),)
    assert plan.pending_assets[3].options == {"AMPLITUDE": "RAMP"}
    assert plan.pending_assets[3].rows == (("LOAD", "2", "-100.0"),)


def test_plan_does_not_leak_material_context_into_surface_or_equation(tmp_path):
    source = tmp_path / "interfaces.inp"
    source.write_text(
        "\n".join(
            (
                "*Material, name=STEEL",
                "*Elastic",
                "210000.0, 0.3",
                "*Density",
                "7.85e-9",
                "*Surface, name=LOAD, type=ELEMENT",
                "SOLID, S2",
                "*Equation",
                "2",
                "LEFT, 1, 1.0, RIGHT, 1, -1.0",
            )
        ),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)

    assert [item.keyword for item in plan.materials[0].blocks] == [
        "*ELASTIC",
        "*DENSITY",
    ]
    assert [item.category for item in plan.pending_assets] == [
        "region_definition",
        "constraint",
    ]
    assert plan.pending_assets[0].rows == (("SOLID", "S2"),)
    assert plan.pending_assets[1].rows[-1] == (
        "LEFT",
        "1",
        "1.0",
        "RIGHT",
        "1",
        "-1.0",
    )


def test_plan_preserves_user_material_contract_without_claiming_execution(tmp_path):
    source = tmp_path / "umat.inp"
    source.write_text(
        "\n".join(
            (
                "*Material, name=LEGACY_UMAT",
                "*User Material, constants=2",
                "1000.0, 0.3",
                "*Depvar",
                "4",
            )
        ),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)
    material = plan.materials[0]

    assert material.translation_status == "review_required_user_material"
    assert material.native_candidate == {}
    assert material.blocks[0].options == {"CONSTANTS": "2"}
    assert material.blocks[0].rows == (("1000.0", "0.3"),)
    assert material.blocks[1].rows == (("4",),)
    assert {item.code for item in plan.issues} == {"AFM-ABAQUS-MATERIAL-004"}


def test_plan_marks_topology_only_element_without_discarding_suffix(tmp_path):
    source = tmp_path / "reduced.inp"
    source.write_text(
        "*Element, type=C3D8R, elset=SOLID\n1,1,2,3,4,5,6,7,8\n",
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)

    assert plan.blocked is False
    assert plan.element_blocks[0].definition.source_type == "C3D8R"
    assert plan.element_blocks[0].definition.solver_capability == "topology_only"
    assert {item.code for item in plan.issues} == {"AFM-ABAQUS-ELEMENT-002"}


def test_plan_blocks_element_declaration_without_type(tmp_path):
    source = tmp_path / "missing-type.inp"
    source.write_text("*Element, elset=SOLID\n1,1,2,3,4\n", encoding="utf-8")

    plan = abaqus_migration.plan(source)

    assert plan.blocked is True
    assert plan.element_blocks[0].definition.source_type == "<UNSPECIFIED>"
    assert {item.code for item in plan.issues} == {"AFM-ABAQUS-ELEMENT-001"}


def test_scope_aware_plan_does_not_confuse_equal_set_names_across_parts(tmp_path):
    source = tmp_path / "two-parts.inp"
    source.write_text(
        "\n".join(
            (
                "*Part, name=LEFT",
                "*Element, type=C3D4, elset=SOLID",
                "1,1,2,3,4",
                "*Solid Section, elset=SOLID, material=LEFT_MAT",
                "*End Part",
                "*Part, name=RIGHT",
                "*Element, type=C3D4, elset=SOLID",
                "1,1,2,3,4",
                "*Solid Section, elset=SOLID, material=RIGHT_MAT",
                "*End Part",
                "*Material, name=LEFT_MAT",
                "*Elastic",
                "1.0,0.2",
                "*Material, name=RIGHT_MAT",
                "*Elastic",
                "2.0,0.3",
            )
        ),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)

    assert plan.blocked is False
    assert {item.key for item in plan.regions} == {
        "part:LEFT/elset:SOLID",
        "part:RIGHT/elset:SOLID",
    }
    assert all(item.status == "references_resolved" for item in plan.sections)
    assert all(
        item.translation_status == "review_required_missing_density"
        for item in plan.materials
    )
    assert {item.code for item in plan.issues} == {"AFM-ABAQUS-MATERIAL-002"}


def test_scope_aware_plan_blocks_unknown_section_references(tmp_path):
    source = tmp_path / "invalid.inp"
    source.write_text(
        "*Solid Section, elset=MISSING, material=UNKNOWN\n",
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)

    assert plan.blocked is True
    assert plan.sections[0].status == "blocked_unresolved_reference"
    assert {item.code for item in plan.issues} >= {
        "AFM-ABAQUS-SECTION-001",
        "AFM-ABAQUS-SECTION-002",
    }


def test_composite_section_is_preserved_for_review_without_fake_material(tmp_path):
    source = tmp_path / "composite.inp"
    source.write_text(
        "\n".join(
            (
                "*Element, type=C3D8, elset=LAYUP",
                "1,1,2,3,4,5,6,7,8",
                "*Solid Section, elset=LAYUP, composite",
                "0.5, MAT_A, 0.0",
                "0.5, MAT_B, 90.0",
            )
        ),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)

    assert plan.blocked is False
    assert plan.sections[0].status == "review_required_composite"
    assert plan.sections[0].flags == ("COMPOSITE",)
    assert plan.sections[0].rows == (
        ("0.5", "MAT_A", "0.0"),
        ("0.5", "MAT_B", "90.0"),
    )
    assert any(item.code == "AFM-ABAQUS-SECTION-003" for item in plan.issues)


def _write_native_static_deck(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "*Heading",
                "*Node",
                "1, 0., 0., 0.",
                "2, 1., 0., 0.",
                "3, 0., 1., 0.",
                "4, 0., 0., 1.",
                "*Nset, nset=FIXED",
                "1, 2, 3",
                "*Nset, nset=MOVED",
                "4",
                "*Element, type=C3D4, elset=SOLID",
                "1, 1, 2, 3, 4",
                "*Material, name=STEEL",
                "*Elastic",
                "210000., 0.3",
                "*Density",
                "7.85e-9",
                "*Solid Section, elset=SOLID, material=STEEL",
                "*Step, name=PULL",
                "*Static",
                "*Boundary",
                "FIXED, 1, 3, 0.",
                "MOVED, 1, 2, 0.",
                "MOVED, 3, 3, 0.01",
                "*Node Output",
                "U, RF",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_native_lowering_assessment_accepts_narrow_linear_static_route(tmp_path):
    source = tmp_path / "static.inp"
    _write_native_static_deck(source)

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))

    assert assessment.eligible is True
    assert assessment.dimension == 3
    assert assessment.assumption is None
    assert assessment.topology == "tetra"
    assert assessment.degree == 1
    assert assessment.material_name == "STEEL"
    assert [item.region for item in assessment.boundaries] == [
        "FIXED",
        "MOVED",
        "MOVED",
    ]
    assert {item.code for item in assessment.findings} == {
        "AFM-ABAQUS-LOWER-ASSET-101"
    }


def test_native_lowering_emits_reviewed_draft_without_erasing_guard(tmp_path):
    source = tmp_path / "static.inp"
    _write_native_static_deck(source)
    project = tmp_path / "migrated"
    abaqus_migration.create_project(source, project, created_with="test")

    record = abaqus_lowering.lower_project(
        project,
        reviewed_by="Test Engineer",
        unit_system="mm-N-s",
    )

    assert record["status"] == "drafted"
    assert "fail-closed migration scaffold" in (project / "case.py").read_text()
    native = (project / "case.native.py").read_text(encoding="utf-8")
    assert "studies.static_solid" in native
    assert "cell.node_set('FIXED')" in native
    assert "young=210000.0" in native
    assert 'entrypoint = "case.py"' in (project / "agentfem.toml").read_text()
    lowering = json.loads((project / "lowering.json").read_text(encoding="utf-8"))
    assert lowering["review"]["reviewed_by"] == "Test Engineer"
    assert lowering["review"]["unit_system"] == "mm-N-s"
    assert lowering["review"]["source_values_reinterpreted"] is False
    assert lowering["expanded_source_sha256"]

    activated = abaqus_lowering.lower_project(
        project,
        reviewed_by="Test Engineer",
        unit_system="mm-N-s",
        activate=True,
        force=True,
    )
    assert activated["status"] == "activated"
    assert activated["decision_fingerprint"] == record["decision_fingerprint"]
    assert 'entrypoint = "case.native.py"' in (
        project / "agentfem.toml"
    ).read_text()


def test_native_lowering_blocks_reduced_integration_and_unlowered_load(tmp_path):
    source = tmp_path / "unsupported.inp"
    _write_native_static_deck(source)
    text = source.read_text(encoding="utf-8")
    text = text.replace("type=C3D4", "type=C3D8R")
    text = text.replace(
        "*End Step", "*Cload\nMOVED, 3, 10.0\n*End Step"
    )
    source.write_text(text, encoding="utf-8")

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))
    codes = {item.code for item in assessment.findings if item.severity == "error"}

    assert assessment.eligible is False
    assert "AFM-ABAQUS-LOWER-ELEMENT-002" in codes
    assert "AFM-ABAQUS-LOWER-LOAD-001" in codes


def test_native_lowering_does_not_collapse_temperature_dependent_material_table(
    tmp_path,
):
    source = tmp_path / "temperature-dependent.inp"
    _write_native_static_deck(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "210000., 0.3\n*Density",
            "210000., 0.3, 20.\n190000., 0.3, 500.\n*Density",
        ),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)
    assessment = abaqus_lowering.assess(plan)

    assert plan.materials[0].translation_status == "review_required"
    assert assessment.eligible is False
    assert "AFM-ABAQUS-LOWER-MATERIAL-001" in {
        item.code for item in assessment.findings
    }


def test_native_lowering_rejects_nonphysical_native_material_candidate(tmp_path):
    source = tmp_path / "invalid-material.inp"
    _write_native_static_deck(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace("210000., 0.3", "-1., 0.6"),
        encoding="utf-8",
    )

    plan = abaqus_migration.plan(source)
    assessment = abaqus_lowering.assess(plan)

    assert plan.materials[0].translation_status == "review_required"
    assert assessment.eligible is False


def test_native_lowering_requires_section_to_cover_every_element_block(tmp_path):
    source = tmp_path / "partially-assigned.inp"
    _write_native_static_deck(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "*Material, name=STEEL",
            "*Element, type=C3D4, elset=UNASSIGNED\n2, 1, 2, 3, 4\n"
            "*Material, name=STEEL",
        ),
        encoding="utf-8",
    )

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))

    assert assessment.eligible is False
    assert "AFM-ABAQUS-LOWER-SECTION-002" in {
        item.code for item in assessment.findings
    }


def test_native_lowering_refuses_to_discard_nonunit_two_dimensional_thickness(
    tmp_path,
):
    source = tmp_path / "plane-stress.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1,0,0",
                "2,1,0",
                "3,0,1",
                "*Nset, nset=FIXED",
                "1,2",
                "*Element, type=CPS3, elset=SOLID",
                "1,1,2,3",
                "*Material, name=MAT",
                "*Elastic",
                "1000.,0.3",
                "*Density",
                "1.0",
                "*Solid Section, elset=SOLID, material=MAT",
                "2.5",
                "*Step, name=PULL",
                "*Static",
                "*Boundary",
                "FIXED,1,2,0.",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))

    assert assessment.eligible is False
    assert "AFM-ABAQUS-LOWER-SECTION-003" in {
        item.code for item in assessment.findings
    }


def test_native_lowering_requires_explicit_step_and_surface_semantics(tmp_path):
    source = tmp_path / "ambiguous-semantics.inp"
    _write_native_static_deck(source)
    text = source.read_text(encoding="utf-8")
    text = text.replace("*Static", "*Static, stabilize")
    text = text.replace("*Boundary", "*Surface, name=LOAD, type=NODE\nMOVED\n*Boundary, op=NEW")
    text = text.replace("*Node Output", "*Dsload\nLOAD,P,1.0\n*Node Output")
    source.write_text(text, encoding="utf-8")

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))
    codes = {item.code for item in assessment.findings}

    assert assessment.eligible is False
    assert {
        "AFM-ABAQUS-LOWER-STEP-004",
        "AFM-ABAQUS-LOWER-BC-007",
        "AFM-ABAQUS-LOWER-SURFACE-001",
        "AFM-ABAQUS-LOWER-LOAD-004",
    } <= codes


def test_native_lowering_rejects_conflicting_prescribed_dofs(tmp_path):
    source = tmp_path / "conflicting-boundary.inp"
    _write_native_static_deck(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "MOVED, 3, 3, 0.01",
            "MOVED, 3, 3, 0.01\nMOVED, 3, 3, 0.02",
        ),
        encoding="utf-8",
    )

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))

    assert assessment.eligible is False
    assert "AFM-ABAQUS-LOWER-BC-008" in {
        item.code for item in assessment.findings
    }


def test_native_lowering_refuses_source_mutation_after_migration(tmp_path):
    source = tmp_path / "static.inp"
    _write_native_static_deck(source)
    project = tmp_path / "migrated"
    abaqus_migration.create_project(source, project, created_with="test")
    copied = project / "source" / "static.inp"
    copied.write_text(copied.read_text() + "** changed\n", encoding="utf-8")

    try:
        abaqus_lowering.lower_project(
            project,
            reviewed_by="Test Engineer",
            unit_system="SI",
        )
    except ValueError as exc:
        assert "changed after migration planning" in str(exc)
    else:
        raise AssertionError("Mutated source must invalidate native lowering.")


def test_native_lowering_flattens_one_positioned_part_instance(tmp_path):
    source = tmp_path / "assembly.inp"
    source.write_text(
        "\n".join(
            (
                "*Part, name=BRACKET",
                "*Node",
                "1,0,0,0",
                "2,1,0,0",
                "3,0,1,0",
                "4,0,0,1",
                "*Nset, nset=FIXED",
                "1,2,3",
                "*Nset, nset=MOVED",
                "4",
                "*Element, type=C3D4, elset=SOLID",
                "1,1,2,3,4",
                "*Solid Section, elset=SOLID, material=STEEL",
                "*End Part",
                "*Assembly, name=A",
                "*Instance, name=BRACKET-1, part=BRACKET",
                "10.,20.,30.",
                "0.,0.,0.,0.,0.,1.,90.",
                "*End Instance",
                "*End Assembly",
                "*Material, name=STEEL",
                "*Elastic",
                "210000.,0.3",
                "*Density",
                "7.85e-9",
                "*Step, name=PULL",
                "*Static",
                "*Boundary",
                "BRACKET-1.FIXED,1,3,0.",
                "BRACKET-1.MOVED,1,2,0.",
                "BRACKET-1.MOVED,3,3,0.01",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    project = tmp_path / "migrated"
    abaqus_migration.create_project(source, project, created_with="test")

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))
    assert assessment.eligible is True
    assert {item.region for item in assessment.boundaries} == {"FIXED", "MOVED"}
    abaqus_lowering.lower_project(
        project,
        reviewed_by="Test Engineer",
        unit_system="mm-N-s",
    )
    derived = project / "mesh" / "abaqus-expanded.inp"
    nodes = abaqus.read_node_table(derived)

    np.testing.assert_allclose(nodes.coordinate(1), [-20.0, 10.0, 30.0])
    np.testing.assert_allclose(nodes.coordinate(2), [-20.0, 11.0, 30.0])
    text = derived.read_text(encoding="utf-8")
    assert "*Part" not in text
    assert "*Instance" not in text
    assert "*Nset, nset=FIXED" in text


def test_native_lowering_maps_whole_material_gravity_without_density_duplication(
    tmp_path,
):
    source = tmp_path / "gravity.inp"
    _write_native_static_deck(source)
    text = source.read_text(encoding="utf-8").replace(
        "*End Step", "*Dload\nSOLID, GRAV, 9.81, 0., 0., -1.\n*End Step"
    )
    source.write_text(text, encoding="utf-8")
    project = tmp_path / "migrated"
    abaqus_migration.create_project(source, project, created_with="test")

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))
    assert assessment.eligible is True
    assert assessment.gravities[0].acceleration == (0.0, 0.0, -9.81)
    abaqus_lowering.lower_project(
        project,
        reviewed_by="Test Engineer",
        unit_system="SI",
    )
    native = (project / "case.native.py").read_text(encoding="utf-8")
    assert "model.gravity(" in native
    assert "(0.0, 0.0, -9.81)" in native


def test_native_lowering_maps_multiple_solid_sections_to_material_regions(tmp_path):
    source = tmp_path / "two-materials.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1,0,0,0",
                "2,1,0,0",
                "3,0,1,0",
                "4,0,0,1",
                "5,2,0,0",
                "6,3,0,0",
                "7,2,1,0",
                "8,2,0,1",
                "*Nset, nset=FIXED",
                "1,2,3,4,5,6,7,8",
                "*Element, type=C3D4, elset=SOFT",
                "1,1,2,3,4",
                "*Element, type=C3D4, elset=STIFF",
                "2,5,6,7,8",
                "*Material, name=RUBBER",
                "*Elastic",
                "10.,0.3",
                "*Density",
                "1.0",
                "*Material, name=STEEL",
                "*Elastic",
                "1000.,0.25",
                "*Density",
                "2.0",
                "*Solid Section, elset=SOFT, material=RUBBER",
                "*Solid Section, elset=STIFF, material=STEEL",
                "*Step, name=HOLD",
                "*Static",
                "*Boundary",
                "FIXED,1,3,0.",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    project = tmp_path / "migrated"
    abaqus_migration.create_project(source, project, created_with="test")

    assessment = abaqus_lowering.assess(abaqus_migration.plan(source))

    assert assessment.eligible is True
    assert assessment.material_name is None
    assert assessment.material == {}
    assert [item.region for item in assessment.material_assignments] == [
        "SOFT",
        "STIFF",
    ]
    assert [item.material_name for item in assessment.material_assignments] == [
        "RUBBER",
        "STEEL",
    ]

    abaqus_lowering.lower_project(
        project,
        reviewed_by="Test Engineer",
        unit_system="SI",
    )
    native = (project / "case.native.py").read_text(encoding="utf-8")
    assert "region=cell.element_set('SOFT')" in native
    assert "region=cell.element_set('STIFF')" in native

    imported = fem_mesh.read_abaqus_mesh(
        project / "mesh" / "abaqus-expanded.inp",
        project / "mesh" / "two-materials.xdmf",
        comm=MPI.COMM_SELF,
        cell_type="tetra",
        reuse_conversion=False,
    )
    assert imported.element_set("soft").name == "soft"
    assert imported.element_set("STIFF").name == "STIFF"

    namespace = runpy.run_path(str(project / "case.native.py"))
    result = namespace["main"]()
    assert result.status == "completed"


def test_abaqus_element_set_refuses_partial_tag_from_overlapping_sets(tmp_path):
    source = tmp_path / "overlap.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1,0,0,0",
                "2,1,0,0",
                "3,0,1,0",
                "4,0,0,1",
                "5,2,0,0",
                "6,3,0,0",
                "7,2,1,0",
                "8,2,0,1",
                "*Element, type=C3D4, elset=SOLID",
                "1,1,2,3,4",
                "2,5,6,7,8",
                "*Elset, elset=FIRST",
                "1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    imported = fem_mesh.read_abaqus_mesh(
        source,
        tmp_path / "overlap.xdmf",
        comm=MPI.COMM_SELF,
        cell_type="tetra",
        reuse_conversion=False,
    )

    try:
        imported.element_set("SOLID")
    except ValueError as exc:
        assert "overlaps another ELSET" in str(exc)
    else:
        raise AssertionError("A partial overlapping ELSET tag must be rejected.")
