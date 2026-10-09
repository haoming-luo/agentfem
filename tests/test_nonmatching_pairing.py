# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._interface_pairing import (
    fixed_reference_pairing,
    FixedReferenceCohesiveAssembler,
)
from agentfem.interfaces import bilinear_cohesive
from agentfem.boundary_models.rigid import TriangulatedRigidSurface


def surface(n, *, reverse=False, offset=0):
    vertices = np.array(
        [[i / n, j / n, offset] for j in range(n + 1) for i in range(n + 1)]
    )
    triangles = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]])
    triangles = np.array(triangles)
    if reverse:
        triangles = triangles[:, ::-1]
    return TriangulatedRigidSurface(vertices, triangles)


@pytest.mark.parametrize("n,m", [(1, 1), (2, 1), (3, 2), (1, 3)])
def test_affine_patch_and_work_conjugacy(n, m):
    a, b = surface(n), surface(m, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    gradient = np.array([[0.1, 0.2, 0.3], [0.4, -0.2, 0.6], [0.5, 0.7, -0.1]])
    un, up = a.vertices @ gradient.T, b.vertices @ gradient.T
    np.testing.assert_allclose(pair.jump(un, up), 0, atol=1e-15)
    rng = np.random.default_rng(28)
    t = rng.normal(size=(len(pair.weights), 3))
    rn, rp = pair.residual(t)
    np.testing.assert_allclose(rn.sum(axis=0) + rp.sum(axis=0), 0, atol=1e-15)
    moment = np.cross(a.vertices, rn).sum(axis=0) + np.cross(b.vertices, rp).sum(axis=0)
    np.testing.assert_allclose(moment, 0, atol=1e-15)
    dn, dp = rng.normal(size=un.shape), rng.normal(size=up.shape)
    assert np.sum(rn * dn) + np.sum(rp * dp) == pytest.approx(
        np.sum(pair.weights[:, None] * t * pair.jump(dn, dp)), abs=1e-14
    )
    assert pair.weights.sum() == pytest.approx(1)


def test_common_finite_rotation_preserves_coincident_trace():
    a, b = surface(3), surface(2, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    un = a.vertices @ rotation.T - a.vertices + [1, 2, 3]
    up = b.vertices @ rotation.T - b.vertices + [1, 2, 3]
    np.testing.assert_allclose(pair.jump(un, up), 0, atol=1e-15)


def test_reject_bad_geometry_and_field():
    a = surface(1)
    with pytest.raises(ValueError, match="normals"):
        fixed_reference_pairing(a, surface(2), tolerance=1e-10)
    with pytest.raises(ValueError, match="projection"):
        fixed_reference_pairing(
            a, surface(2, reverse=True, offset=0.1), tolerance=1e-10
        )
    with pytest.raises(ValueError, match="tolerance"):
        fixed_reference_pairing(a, surface(2, reverse=True), tolerance=np.nan)
    pair = fixed_reference_pairing(a, surface(2, reverse=True), tolerance=1e-10)
    with pytest.raises(ValueError, match="traction"):
        pair.residual(np.full((len(pair.weights), 3), np.nan))
    with pytest.raises(ValueError):
        pair.weights[0] = 0


def test_identity_and_uniform_opening():
    a, b = surface(2), surface(3, reverse=True)
    p = fixed_reference_pairing(a, b, tolerance=1e-10)
    q = fixed_reference_pairing(a, b, tolerance=1e-10)
    assert p.fingerprint == q.fingerprint
    jump = p.jump(
        np.zeros_like(a.vertices), np.tile([0.01, -0.02, 0.03], (len(b.vertices), 1))
    )
    np.testing.assert_allclose(
        jump, np.tile([0.01, -0.02, 0.03], (len(p.weights), 1)), atol=1e-15
    )
    assert p.summary()["global_step_integrated"] is False


def test_tangent_action_matches_energy_gradient_and_residual_difference():
    a, b = surface(3), surface(2, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    rng = np.random.default_rng(3)
    un, up = rng.normal(size=a.vertices.shape), rng.normal(size=b.vertices.shape)
    dn, dp = rng.normal(size=un.shape), rng.normal(size=up.shape)
    d = np.tile(np.diag([2.0, 3.0, 5.0]), (len(pair.weights), 1, 1))

    def response(u, v):
        jump = pair.jump(u, v)
        traction = np.einsum("qij,qj->qi", d, jump)
        return pair.residual(traction), 0.5 * np.sum(
            pair.weights[:, None] * jump * traction
        )

    h = 1e-6
    (rn, rp), _ = response(un, up)
    (plusn, plusp), ep = response(un + h * dn, up + h * dp)
    (minusn, minusp), em = response(un - h * dn, up - h * dp)
    kn, kp = pair.tangent_action(d, dn, dp)
    np.testing.assert_allclose(kn, (plusn - minusn) / (2 * h), atol=1e-9)
    np.testing.assert_allclose(kp, (plusp - minusp) / (2 * h), atol=1e-9)
    assert (ep - em) / (2 * h) == pytest.approx(
        np.sum(rn * dn) + np.sum(rp * dp), abs=1e-8
    )
    with pytest.raises(ValueError, match="tangent"):
        pair.tangent_action(np.full_like(d, np.nan), dn, dp)


def test_existing_law_damage_rollback_and_bound_restart():
    a, b = surface(2), surface(3, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    law = bilinear_cohesive(strength=10, fracture_energy=2, initial_stiffness=1000)
    assembler = FixedReferenceCohesiveAssembler(pair, law)
    un = np.zeros_like(a.vertices)
    up = np.tile([0, 0, 0.2], (len(b.vertices), 1))
    response = assembler.begin(un, up)
    material = law.update(0.2)
    assert response.stored_energy == pytest.approx(float(material.stored_energy))
    assert response.dissipated_energy == pytest.approx(
        float(material.dissipated_energy)
    )
    np.testing.assert_allclose(response.damage, material.damage)
    assembler.rollback()
    np.testing.assert_allclose(assembler.state.committed_maximum, 0)
    assembler.begin(un, up)
    assembler.commit()
    import copy

    snapshot = assembler.snapshot()
    restored = FixedReferenceCohesiveAssembler(pair, law)
    restored.restore(snapshot)
    np.testing.assert_allclose(restored.state.committed_maximum, 0.2)
    broken = copy.deepcopy(snapshot)
    broken["state"]["maximum_opening"][0] = float("nan")
    with pytest.raises(ValueError):
        restored.restore(broken)
    assert restored.snapshot() == snapshot
    broken = copy.deepcopy(snapshot)
    broken["pairing"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        restored.restore(broken)
    assert restored.snapshot() == snapshot
    with pytest.raises(ValueError):
        restored.begin(un, up * np.nan)
    assert restored.snapshot() == snapshot


def test_existing_law_matrix_free_tangent_at_fixed_history():
    a, b = surface(2), surface(1, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    law = bilinear_cohesive(strength=10, fracture_energy=2, initial_stiffness=1000)
    assembler = FixedReferenceCohesiveAssembler(
        pair, law, tangential="tie", tangential_stiffness=300
    )
    un, up = (
        np.zeros_like(a.vertices),
        np.tile([0.002, -0.001, 0.1], (len(b.vertices), 1)),
    )
    rng = np.random.default_rng(9)
    dn, dp = rng.normal(size=un.shape), rng.normal(size=up.shape)
    kn, kp = assembler.tangent_action(un, up, dn, dp)
    h = 1e-7
    plus = assembler.begin(un + h * dn, up + h * dp)
    minus = assembler.begin(un - h * dn, up - h * dp)
    np.testing.assert_allclose(
        kn, (plus.negative_residual - minus.negative_residual) / (2 * h), atol=1e-7
    )
    np.testing.assert_allclose(
        kp, (plus.positive_residual - minus.positive_residual) / (2 * h), atol=1e-7
    )
    np.testing.assert_allclose(assembler.state.committed_maximum, 0)
