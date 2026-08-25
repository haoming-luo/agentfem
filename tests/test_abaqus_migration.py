from __future__ import annotations

from agentfem.mesh import abaqus


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
    assert abaqus.read_element_definitions(source)[0].solver_capability == "topology_only"


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


def test_inspection_reports_include_dependencies_without_expanding_them(tmp_path):
    source = tmp_path / "root.inp"
    source.write_text(
        "*Heading\n*Include, input=mesh.inc\n*Include, input=missing.inc\n",
        encoding="utf-8",
    )
    (tmp_path / "mesh.inc").write_text("*Node\n1, 0., 0., 0.\n", encoding="utf-8")

    report = abaqus.inspect_input(source)

    assert report.include_files == ("mesh.inc", "missing.inc")
    assert any("not expanded" in item for item in report.warnings)
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
