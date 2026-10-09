# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._interface_pairing import (
    fixed_reference_pairing,
    FixedReferenceCohesiveAssembler,
)
from agentfem.interfaces import bilinear_cohesive
from agentfem._elastic_cohesive import ElasticCohesiveLaw
from agentfem._interface_overlap import planar_overlap_pairing
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


@pytest.mark.parametrize("n,m", [(1, 1), (1, 3), (3, 2), (2, 5)])
def test_overlap_nodal_patch_and_side_swap(n, m):
    a, b = surface(n), surface(m, reverse=True)
    pair = planar_overlap_pairing(a, b, tolerance=1e-10)
    swapped = planar_overlap_pairing(b, a, tolerance=1e-10)
    for item in pair.constant_traction_audit()["sides"].values():
        assert item["relative_nodal_measure_error_l2"] < 1e-12
    assert pair.weights.sum() == pytest.approx(1)
    rng = np.random.default_rng(16)
    un, up = rng.normal(size=a.vertices.shape), rng.normal(size=b.vertices.shape)
    jump = pair.jump(un, up)
    other = swapped.jump(up, un)
    energy = np.sum(pair.weights[:, None] * jump**2)
    assert np.sum(swapped.weights[:, None] * other**2) == pytest.approx(
        energy, rel=1e-12
    )
    rn, rp = pair.residual(jump)
    sp, sn = swapped.residual(other)
    np.testing.assert_allclose(rn, sn, atol=1e-12)
    np.testing.assert_allclose(rp, sp, atol=1e-12)


def test_overlap_unstructured_trace_and_rotated_plane():
    a, b = surface(3), surface(4, reverse=True)
    vertices = b.vertices.copy()
    inside = np.all((vertices[:, :2] > 0) & (vertices[:, :2] < 1), axis=1)
    vertices[inside, :2] += np.random.default_rng(4).uniform(
        -0.04, 0.04, (inside.sum(), 2)
    )
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(3, 3)))
    shift = np.array([1.3, -2.1, 0.4])
    a = TriangulatedRigidSurface(a.vertices @ rotation.T + shift, a.triangles)
    b = TriangulatedRigidSurface(vertices @ rotation.T + shift, b.triangles)
    pair = planar_overlap_pairing(a, b, tolerance=1e-10)
    for item in pair.constant_traction_audit()["sides"].values():
        assert item["relative_nodal_measure_error_l2"] < 1e-12
    traction = np.tile([0.3, -0.7, 1.2], (len(pair.weights), 1))
    rn, rp = pair.residual(traction)
    np.testing.assert_allclose(rn.sum(axis=0) + rp.sum(axis=0), 0, atol=1e-13)
    np.testing.assert_allclose(
        np.cross(a.vertices, rn).sum(axis=0) + np.cross(b.vertices, rp).sum(axis=0),
        0,
        atol=1e-13,
    )
    np.testing.assert_allclose(pair.jump(a.vertices, b.vertices), 0, atol=1e-13)


def test_overlap_rejects_geometrically_overlapping_disconnected_facets():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.1, 0.1, 0], [0.8, 0.1, 0], [0.1, 0.8, 0]]
    )
    duplicate_area = TriangulatedRigidSurface(
        vertices, np.array([[0, 1, 2], [3, 4, 5]])
    )
    with pytest.raises(ValueError, match="Within-side"):
        planar_overlap_pairing(
            duplicate_area, surface(1, reverse=True), tolerance=1e-10
        )


def test_overlap_rejects_missing_coverage_and_point_budget():
    a, b = surface(2), surface(3, reverse=True)
    with pytest.raises(ValueError, match="maximum_points"):
        planar_overlap_pairing(a, b, tolerance=1e-10, maximum_points=3)
    vertices = b.vertices.copy()
    vertices[:, 0] *= 0.9
    truncated = TriangulatedRigidSurface(vertices, b.triangles)
    with pytest.raises(ValueError, match="coverage"):
        planar_overlap_pairing(a, truncated, tolerance=1e-10)
    with pytest.raises(ValueError, match="coplanar"):
        planar_overlap_pairing(a, surface(1, reverse=True, offset=0.1), tolerance=1e-10)


def test_aabb_overlap_candidates_equal_exhaustive_boxes():
    from agentfem.boundary_models.search import TriangleSurfaceBVH

    s = surface(5)
    tree = TriangleSurfaceBVH(s)
    xyz = s.vertices[s.triangles]
    lower, upper = np.array([0.21, 0.33, 0]), np.array([0.51, 0.72, 0])
    actual = tree.overlapping_facets(lower, upper)
    expected = np.flatnonzero(
        np.all(xyz.max(axis=1) >= lower, axis=1)
        & np.all(xyz.min(axis=1) <= upper, axis=1)
    )
    np.testing.assert_array_equal(np.sort(actual), expected)
    with pytest.raises(ValueError, match="AABB"):
        tree.overlapping_facets(upper, lower)


def test_aabb_random_and_touching_queries_preserve_closed_box_candidates():
    from agentfem.boundary_models.search import TriangleSurfaceBVH

    original = surface(8)
    rotation, _ = np.linalg.qr(np.random.default_rng(79).normal(size=(3, 3)))
    s = TriangulatedRigidSurface(original.vertices @ rotation, original.triangles)
    tree = TriangleSurfaceBVH(s)
    xyz = s.vertices[s.triangles]
    lo, hi = xyz.min(axis=1), xyz.max(axis=1)
    rng = np.random.default_rng(101)
    boxes = rng.uniform(-1.5, 1.5, (100, 2, 3))
    queries = [(pair.min(axis=0), pair.max(axis=0)) for pair in boxes]
    queries += [(point, point) for point in s.vertices]
    for lower, upper in queries:
        actual = tree.overlapping_facets(lower, upper)
        expected = np.flatnonzero(np.all(hi >= lower, axis=1) & np.all(lo <= upper, axis=1))
        np.testing.assert_array_equal(np.sort(actual), expected)


def test_elastic_interface_has_no_hidden_damage_and_reuses_assembler():
    law = ElasticCohesiveLaw(1000, 300, 400)
    jump = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    response = law.update(jump)
    np.testing.assert_allclose(response.traction, jump * [1000, 300, 400])
    np.testing.assert_allclose(response.stored_energy, 29)
    np.testing.assert_array_equal(response.damage, 0)
    np.testing.assert_array_equal(response.dissipated_energy, 0)
    a, b = surface(2), surface(3, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    assembler = FixedReferenceCohesiveAssembler(pair, law, tangential="mixed")
    result = assembler.begin(
        np.zeros_like(a.vertices), np.tile([0, 0, 0.1], (len(b.vertices), 1))
    )
    assert result.stored_energy == pytest.approx(5)
    assert result.dissipated_energy == 0
    assembler.commit()
    saved = assembler.snapshot()
    assembler.begin(
        np.zeros_like(a.vertices), np.tile([0, 0, -0.1], (len(b.vertices), 1))
    )
    assembler.rollback()
    assert assembler.snapshot() == saved
    assembler.restore(saved)
    with pytest.raises(ValueError, match="precracking"):
        assembler.state.initialize(1)


def test_quadrature_refinement_controls_constant_traction_nodal_error():
    a, b = surface(3), surface(2, reverse=True)
    xyz = b.vertices[b.triangles]
    areas = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    expected = np.zeros(len(b.vertices))
    np.add.at(expected, b.triangles.ravel(), np.repeat(areas / 3, 3))
    errors = []
    fingerprints = []
    for level in (0, 1, 2):
        pair = fixed_reference_pairing(
            a, b, tolerance=1e-10, quadrature_refinement=level
        )
        _, residual = pair.residual(np.tile([0.0, 0.0, 1.0], (len(pair.weights), 1)))
        errors.append(
            np.linalg.norm(residual[:, 2] - expected) / np.linalg.norm(expected)
        )
        fingerprints.append(pair.fingerprint)
    assert errors[0] > 0.01  # A balanced total force is not a nodal patch test.
    assert errors[2] < 0.002
    assert (
        max(errors[1:]) < 1e-12
    )  # This nested grid becomes exact; roundoff need not decrease.
    assert len(set(fingerprints)) == 3
    with pytest.raises(ValueError, match="maximum_points"):
        fixed_reference_pairing(
            a, b, tolerance=1e-10, quadrature_refinement=2, maximum_points=10
        )
    with pytest.raises(ValueError, match="refinement"):
        fixed_reference_pairing(a, b, tolerance=1e-10, quadrature_refinement=1.5)


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


def test_constant_traction_audit_exposes_side_sensitivity():
    coarse = fixed_reference_pairing(
        surface(1), surface(3, reverse=True), tolerance=1e-10
    )
    report = coarse.constant_traction_audit()
    assert report["coverage_proven"] is False
    assert report["sides"]["negative"]["relative_nodal_measure_error_l2"] < 1e-14
    assert report["sides"]["positive"]["relative_nodal_measure_error_l2"] > 0.7
    matching = fixed_reference_pairing(
        surface(2), surface(2, reverse=True), tolerance=1e-10
    )
    for result in matching.constant_traction_audit()["sides"].values():
        assert result["relative_nodal_measure_error_l2"] < 1e-14


def test_elastic_failed_trial_cannot_commit_previous_response():
    from agentfem._elastic_cohesive import ElasticCohesiveLaw

    law = ElasticCohesiveLaw(1000, 300)
    state = law.transaction(2)
    state.begin(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        state.begin(np.full((2, 3), np.nan))
    with pytest.raises(RuntimeError, match="No elastic"):
        state.commit()
    state.begin(np.zeros((2, 3)))
    state.commit()
    for invalid in (True, 2.0, float("inf"), float("nan"), 0):
        with pytest.raises(ValueError, match="positive integer"):
            law.transaction(invalid)
    with pytest.raises(ValueError, match="name"):
        ElasticCohesiveLaw(1000, 300, name=None)


@pytest.mark.parametrize("budget", [True, 100.0, float("inf"), float("nan"), 0])
def test_quadrature_budget_rejects_invalid_values(budget):
    with pytest.raises(ValueError, match="positive integer"):
        fixed_reference_pairing(
            surface(1),
            surface(1, reverse=True),
            tolerance=1e-10,
            maximum_points=budget,
        )


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


def test_grouped_blocks_preserve_point_varying_damage_tangents():
    a, b = surface(2), surface(1, reverse=True)
    pair = fixed_reference_pairing(a, b, tolerance=1e-10)
    law = bilinear_cohesive(strength=10, fracture_energy=2, initial_stiffness=1000)
    assembler = FixedReferenceCohesiveAssembler(
        pair, law, tangential="tie", tangential_stiffness=300
    )
    un, up = np.zeros_like(a.vertices), np.zeros_like(b.vertices)
    up[:, 2] = 0.004 + 0.12 * b.vertices[:, 0]
    rng = np.random.default_rng(431)
    dn, dp = rng.normal(size=un.shape), rng.normal(size=up.shape)
    expected_n, expected_p = assembler.tangent_action(un, up, dn, dp)
    assembled_n, assembled_p = np.zeros_like(un), np.zeros_like(up)
    blocks = list(assembler.tangent_blocks(un, up))
    assert len(blocks) < len(pair.weights)
    for negative, positive, block in blocks:
        action = (block @ np.concatenate((dn[negative], dp[positive])).ravel()).reshape(-1, 3)
        np.add.at(assembled_n, negative, action[:len(negative)])
        np.add.at(assembled_p, positive, action[len(negative):])
    np.testing.assert_allclose(assembled_n, expected_n, atol=1e-12)
    np.testing.assert_allclose(assembled_p, expected_p, atol=1e-12)
    np.testing.assert_allclose(assembler.state.committed_maximum, 0)
