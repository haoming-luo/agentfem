from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from agentfem.constraints.affine import (
    _build_reduction,
    _expand_semantic_relations,
)
from agentfem.mesh import abaqus


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
