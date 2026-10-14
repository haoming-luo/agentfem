import numpy as np
import pytest

from agentfem._elastic_cohesive import ElasticCohesiveLaw
from agentfem._initial_nodal_bond import initial_nodal_q1_pairing
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler
from agentfem._interface_quadrilateral import BilinearReferenceTrace
from agentfem._q1_surface_search import Q1SurfaceSearch


def trace(n):
    points = np.array(
        [[i / n, j / n, 0.2 * i * j / n**2] for j in range(n + 1) for i in range(n + 1)]
    )
    quads = np.array(
        [
            [
                j * (n + 1) + i,
                j * (n + 1) + i + 1,
                (j + 1) * (n + 1) + i + 1,
                (j + 1) * (n + 1) + i,
            ]
            for j in range(n)
            for i in range(n)
        ]
    )
    return BilinearReferenceTrace(points, quads, tolerance=1e-10)


def test_fixed_eligibility_and_true_q1_weights():
    negative, positive = trace(2), trace(1)
    eligible = np.arange(len(negative.vertices)) % 2 == 0
    pair = initial_nodal_q1_pairing(
        negative, positive, eligible_nodes=eligible, tolerance=1e-10
    )
    assert len(pair.weights) == eligible.sum()
    np.testing.assert_allclose(pair.reference_mismatch, 0, atol=1e-12)
    np.testing.assert_allclose(pair.weights, negative.nodal_measures[eligible])
    assert pair.weights.flags.writeable is False
    assert pair.summary()["coverage"] == "integration-points-only-not-overlap-proof"
    different = initial_nodal_q1_pairing(
        negative, positive, eligible_nodes=~eligible, tolerance=1e-10
    )
    assert pair.fingerprint != different.fingerprint


def test_bond_large_rotation_work_and_current_moment():
    negative, positive = trace(2), trace(1)
    pair = initial_nodal_q1_pairing(
        negative, positive, eligible_nodes=np.ones(9, dtype=bool), tolerance=1e-10
    )
    law = ElasticCohesiveLaw(100, 100)
    assembler = FixedReferenceCohesiveAssembler(pair, law, tangential="mixed")
    rng = np.random.default_rng(9)
    un, up = rng.normal(scale=0.02, size=(9, 3)), rng.normal(scale=0.02, size=(4, 3))
    response = assembler.evaluate(un, up)
    np.testing.assert_allclose(
        response.negative_residual.sum(axis=0) + response.positive_residual.sum(axis=0),
        0,
        atol=1e-13,
    )
    moment = np.cross(negative.vertices + un, response.negative_residual).sum(
        axis=0
    ) + np.cross(positive.vertices + up, response.positive_residual).sum(axis=0)
    np.testing.assert_allclose(moment, 0, atol=1e-12)
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = assembler.evaluate(
        (negative.vertices + un) @ rotation.T + 2 - negative.vertices,
        (positive.vertices + up) @ rotation.T + 2 - positive.vertices,
    )
    assert rotated.stored_energy == pytest.approx(response.stored_energy, abs=1e-12)
    np.testing.assert_allclose(
        rotated.negative_residual, response.negative_residual @ rotation.T, atol=1e-12
    )
    dn, dp = rng.normal(size=(9, 3)), rng.normal(size=(4, 3))
    h = 1e-7
    derivative = (
        assembler.evaluate(un + h * dn, up + h * dp).stored_energy
        - assembler.evaluate(un - h * dn, up - h * dp).stored_energy
    ) / (2 * h)
    work = np.sum(response.negative_residual * dn) + np.sum(
        response.positive_residual * dp
    )
    assert derivative == pytest.approx(work, rel=1e-8, abs=1e-9)
    with pytest.raises(NotImplementedError, match="isotropic"):
        FixedReferenceCohesiveAssembler(
            pair, ElasticCohesiveLaw(100, 50), tangential="mixed"
        )


def test_nodal_pairing_never_adjusts_or_invents_eligibility():
    negative, positive = trace(1), trace(1)
    shifted = BilinearReferenceTrace(
        negative.vertices + [0, 0, 0.01], negative.quadrilaterals, tolerance=1e-10
    )
    with pytest.raises(ValueError, match="no surface"):
        initial_nodal_q1_pairing(
            shifted, positive, eligible_nodes=np.ones(4, dtype=bool), tolerance=1e-8
        )
    with pytest.raises(ValueError, match="boolean"):
        initial_nodal_q1_pairing(
            negative, positive, eligible_nodes=[1] * 4, tolerance=1e-8
        )


def test_normal_independent_seam_requires_identical_nodal_trace():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 1], [2, 1, 1]])
    quads = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    search = Q1SurfaceSearch(x[quads], node_ids=quads)
    with pytest.raises(ValueError, match="ambiguous"):
        search.project([[1, 0.5, 0]], maximum_distance=0.1)
    result = search.project(
        [[1, 0.5, 0]], maximum_distance=0.1, require_unique_normal=False
    )
    np.testing.assert_allclose(result["points"], [[1, 0.5, 0]])
    disconnected = Q1SurfaceSearch(x[quads], node_ids=np.arange(8).reshape(2, 4))
    with pytest.raises(ValueError, match="trace interpolation"):
        disconnected.project(
            [[1, 0.5, 0]], maximum_distance=0.1, require_unique_normal=False
        )
