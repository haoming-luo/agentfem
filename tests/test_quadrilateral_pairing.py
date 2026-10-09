# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._interface_quadrilateral import (
    QuadrilateralReferenceTrace,
    quadrilateral_overlap_pairing,
)
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler
from agentfem.interfaces import elastic_cohesive


def _surface(n, *, reverse=False):
    vertices = np.array(
        [[i / n, j / n, 0.0] for j in range(n + 1) for i in range(n + 1)]
    )
    quads = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            quads.append([a, a + 1, a + n + 2, a + n + 1])
    quads = np.array(quads)
    return QuadrilateralReferenceTrace(
        vertices, quads[:, ::-1] if reverse else quads, tolerance=1e-10
    )


@pytest.mark.parametrize("n,m", [(1, 1), (1, 3), (3, 2)])
def test_q1_bilinear_field_is_retained_and_integrated_exactly(n, m):
    a, b = _surface(n), _surface(m, reverse=True)
    pair = quadrilateral_overlap_pairing(a, b, tolerance=1e-10)
    un = np.zeros_like(a.vertices)
    un[:, 2] = a.vertices[:, 0] * a.vertices[:, 1]
    up = np.zeros_like(b.vertices)
    # Integral of (x*y)^2 over the unit square is 1/9. P1 substitution fails this.
    jump = pair.jump(un, up)
    assert np.sum(pair.weights * jump[:, 2] ** 2) == pytest.approx(1 / 9, rel=1e-12)
    assert pair.negative_nodes.shape[1] == pair.positive_nodes.shape[1] == 4
    for report in pair.constant_traction_audit()["sides"].values():
        assert report["relative_nodal_measure_error_l2"] < 1e-12
    assembler = FixedReferenceCohesiveAssembler(
        pair,
        elastic_cohesive(normal_stiffness=10, tangential_stiffness=5),
        tangential="mixed",
    )
    response = assembler.begin(un, up)
    assert response.stored_energy == pytest.approx(5 / 9)
    blocks = list(assembler.tangent_blocks(un, up))
    assert all(matrix.shape == (24, 24) for _, _, matrix in blocks)


def test_q1_work_moment_and_side_swap_are_consistent():
    a, b = _surface(2), _surface(3, reverse=True)
    pair = quadrilateral_overlap_pairing(a, b, tolerance=1e-10)
    reverse = quadrilateral_overlap_pairing(b, a, tolerance=1e-10)
    rng = np.random.default_rng(38)
    un, up = rng.normal(size=a.vertices.shape), rng.normal(size=b.vertices.shape)
    jump = pair.jump(un, up)
    rn, rp = pair.residual(jump)
    sn, sp = reverse.residual(reverse.jump(up, un))
    np.testing.assert_allclose(rn, sp, atol=1e-12)
    np.testing.assert_allclose(rp, sn, atol=1e-12)
    assert np.sum(rn * un) + np.sum(rp * up) == pytest.approx(
        np.sum(pair.weights[:, None] * jump**2)
    )
    np.testing.assert_allclose(rn.sum(axis=0) + rp.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(a.vertices, rn).sum(axis=0) + np.cross(b.vertices, rp).sum(axis=0),
        0,
        atol=1e-12,
    )


def test_q1_nonaffine_geometry_and_budget_are_explicitly_rejected():
    a, b = _surface(1), _surface(2, reverse=True)
    points = a.vertices.copy()
    points[-1, 0] += 0.1
    with pytest.raises(NotImplementedError, match="parallelograms"):
        QuadrilateralReferenceTrace(points, a.quadrilaterals, tolerance=1)
    with pytest.raises(ValueError, match="maximum_points"):
        quadrilateral_overlap_pairing(a, b, tolerance=1e-10, maximum_points=2)
