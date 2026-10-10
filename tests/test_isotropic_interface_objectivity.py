# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Objective special case: isotropic reference-area separation potential."""

import numpy as np
import pytest

from agentfem import interfaces
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler
from test_nonmatching_pairing import surface


def traces(n, *, reverse, quadrilateral):
    if not quadrilateral:
        return surface(n, reverse=reverse)
    vertices = surface(n).vertices
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells.append([a, a + 1, a + n + 2, a + n + 1])
    cells = np.asarray(cells)
    if reverse:
        cells = cells[:, ::-1]
    return interfaces.reference_trace(
        vertices, cells, topology="quadrilateral", tolerance=1e-10
    )


@pytest.mark.parametrize("quadrilateral", [False, True])
def test_open_isotropic_interface_rotates_force_and_preserves_energy(quadrilateral):
    a = traces(2, reverse=False, quadrilateral=quadrilateral)
    b = traces(3, reverse=True, quadrilateral=quadrilateral)
    pair = interfaces.pair_reference_traces(a, b, tolerance=1e-10)
    law = interfaces.elastic_cohesive(normal_stiffness=100, tangential_stiffness=100)
    assembler = FixedReferenceCohesiveAssembler(pair, law, tangential="mixed")
    rng = np.random.default_rng(380)
    un = rng.normal(scale=0.1, size=a.vertices.shape)
    up = rng.normal(scale=0.1, size=b.vertices.shape) + [0.1, 0.2, 0.3]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.linalg.det(q) > 0
    shift = [0.3, -0.4, 0.5]
    rotated_n = (a.vertices + un) @ q.T + shift - a.vertices
    rotated_p = (b.vertices + up) @ q.T + shift - b.vertices
    reference = assembler.evaluate(un, up)
    rotated = assembler.evaluate(rotated_n, rotated_p)
    assert reference.stored_energy > 1
    assert rotated.stored_energy == pytest.approx(reference.stored_energy, rel=1e-12)
    np.testing.assert_allclose(
        rotated.negative_residual, reference.negative_residual @ q.T, atol=1e-12
    )
    np.testing.assert_allclose(
        rotated.positive_residual, reference.positive_residual @ q.T, atol=1e-12
    )
    current_moment = np.cross(a.vertices + un, reference.negative_residual).sum(
        axis=0
    ) + np.cross(b.vertices + up, reference.positive_residual).sum(axis=0)
    np.testing.assert_allclose(current_moment, 0, atol=1e-12)
    dn, dp = rng.normal(size=un.shape), rng.normal(size=up.shape)
    an, ap = assembler.tangent_action(un, up, dn, dp)
    bn, bp = assembler.tangent_action(rotated_n, rotated_p, dn @ q.T, dp @ q.T)
    np.testing.assert_allclose(bn, an @ q.T, atol=1e-12)
    np.testing.assert_allclose(bp, ap @ q.T, atol=1e-12)


def test_unequal_normal_tangent_stiffness_cannot_use_fixed_frame_at_large_rotation():
    a, b = surface(1), surface(2, reverse=True)
    pair = interfaces.pair_reference_traces(a, b, tolerance=1e-10)
    law = interfaces.elastic_cohesive(normal_stiffness=100, tangential_stiffness=10)
    assembler = FixedReferenceCohesiveAssembler(pair, law, tangential="mixed")
    un, up = np.zeros_like(a.vertices), np.tile([0, 0, 0.2], (len(b.vertices), 1))
    q = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    original = assembler.evaluate(un, up)
    rotated = assembler.evaluate(
        (a.vertices + un) @ q.T - a.vertices, (b.vertices + up) @ q.T - b.vertices
    )
    assert rotated.stored_energy != pytest.approx(original.stored_energy)
