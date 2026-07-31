from __future__ import annotations

import numpy as np
import pytest

from agentfem.constraints.affine import _build_reduction
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
