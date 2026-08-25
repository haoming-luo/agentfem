from __future__ import annotations

import json
from pathlib import Path

from agentfem.mesh import abaqus
from agentfem.mesh import abaqus_migration


FIXTURES = Path(__file__).parent / "fixtures" / "abaqus_migration"
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


def test_scope_aware_migration_plan_matches_golden_contract():
    plan = abaqus_migration.plan(FIXTURES / "model.inp")
    expected = json.loads(
        (GOLDENS / "abaqus_migration_plan.json").read_text(encoding="utf-8")
    )

    assert _migration_signature(plan) == expected


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
