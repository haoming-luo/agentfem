# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from agentfem._deformable_q1_contact import DeformableQ1Penalty
from agentfem._interface_quadrilateral import BilinearReferenceTrace


def setup(warp=0.0):
    primary = BilinearReferenceTrace(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, warp], [0, 1, 0]]),
        np.array([[0, 1, 2, 3]]),
        tolerance=1e-10,
    )
    secondary = BilinearReferenceTrace(
        np.array(
            [[0.2, 0.2, 0.02], [0.4, 0.2, 0.02], [0.4, 0.4, 0.02], [0.2, 0.4, 0.02]]
        ),
        np.array([[0, 1, 2, 3]]),
        tolerance=1e-10,
    )
    operator = DeformableQ1Penalty(
        secondary,
        primary,
        candidate_nodes=np.ones(4, dtype=bool),
        penalty=100.0,
        search_radius=0.5,
    )
    return operator, np.zeros((4, 3)), np.zeros((4, 3))


def test_open_and_closed_analytic_response():
    op, us, up = setup()
    assert op.evaluate(us, up).potential_energy == 0
    us[:, 2] = -0.04
    r = op.evaluate(us, up)
    assert r.potential_energy == pytest.approx(0.5 * 100 * 0.02**2 * 0.2**2)
    np.testing.assert_allclose(
        r.secondary_residual.sum(axis=0), [0, 0, -0.08], atol=1e-14
    )
    np.testing.assert_allclose(r.primary_residual.sum(axis=0), [0, 0, 0.08], atol=1e-14)


@pytest.mark.parametrize("warp", [0.0, 0.3])
def test_current_force_moment_objectivity_and_variation(warp):
    op, us, up = setup(warp)
    us[:, 2] = -0.08
    r = op.evaluate(us, up)
    xs, xp = op.secondary.vertices + us, op.primary.vertices + up
    np.testing.assert_allclose(
        r.secondary_residual.sum(axis=0) + r.primary_residual.sum(axis=0), 0, atol=1e-14
    )
    moment = np.cross(xs, r.secondary_residual).sum(axis=0) + np.cross(
        xp, r.primary_residual
    ).sum(axis=0)
    np.testing.assert_allclose(moment, 0, atol=2e-13)
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rotated = op.evaluate(
        xs @ rotation.T + 2 - op.secondary.vertices,
        xp @ rotation.T + 2 - op.primary.vertices,
    )
    assert rotated.potential_energy == pytest.approx(r.potential_energy, rel=1e-12)
    np.testing.assert_allclose(
        rotated.secondary_residual, r.secondary_residual @ rotation.T, atol=1e-13
    )
    rng = np.random.default_rng(11)
    ds, dp = rng.normal(size=(2, 4, 3))
    h = 1e-7
    derivative = (
        op.evaluate(us + h * ds, up + h * dp).potential_energy
        - op.evaluate(us - h * ds, up - h * dp).potential_energy
    ) / (2 * h)
    assert derivative == pytest.approx(
        np.sum(r.secondary_residual * ds) + np.sum(r.primary_residual * dp),
        rel=2e-8,
        abs=1e-10,
    )


def test_current_projection_crosses_facets_without_creating_bonds():
    op, us, _ = setup()
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]]
    )
    primary = BilinearReferenceTrace(
        vertices, np.array([[0, 1, 4, 3], [1, 2, 5, 4]]), tolerance=1e-10
    )
    op = DeformableQ1Penalty(
        op.secondary,
        primary,
        candidate_nodes=np.ones(4, dtype=bool),
        penalty=100.0,
        search_radius=0.5,
    )
    up = np.zeros_like(vertices)
    us[:, 2] = -0.04
    left = op.evaluate(us, up)
    us[:, 0] = 1.0
    right = op.evaluate(us, up)
    assert left.potential_energy == pytest.approx(right.potential_energy)
    assert np.all(left.primary_nodes == [0, 1, 4, 3])
    assert np.all(right.primary_nodes == [1, 2, 5, 4])


def test_off_normal_edge_and_search_escape_fail_closed():
    op, us, up = setup()
    us[:, 2] = -0.04
    us[:, 0] = -0.5
    with pytest.raises(NotImplementedError, match="normal-cone"):
        op.evaluate(us, up)
    us[:, 0] = 20
    with pytest.raises(ValueError, match="no surface"):
        op.evaluate(us, up)


def test_explicit_candidate_mask_is_immutable_and_validated():
    op, us, up = setup()
    mask = np.array([True, False, False, False])
    subset = DeformableQ1Penalty(
        op.secondary, op.primary, candidate_nodes=mask, penalty=100.0, search_radius=0.5
    )
    mask[:] = False
    us[:, 2] = -0.04
    r = subset.evaluate(us, up)
    assert np.count_nonzero(np.linalg.norm(r.secondary_residual, axis=1)) == 1
    assert subset.identity != op.identity
    with pytest.raises(ValueError, match="boolean"):
        DeformableQ1Penalty(
            op.secondary,
            op.primary,
            candidate_nodes=[1] * 4,
            penalty=100.0,
            search_radius=0.5,
        )


def test_closed_concave_edge_contact_force_moment_and_energy_variation():
    from test_closed_q1_surface import solid

    vertices, cells = solid(concave=True)
    primary = BilinearReferenceTrace(vertices, cells, tolerance=1e-10)
    secondary = BilinearReferenceTrace(
        np.array(
            [[0.90, 0.91, 0.3], [0.92, 0.91, 0.3], [0.92, 0.93, 0.3], [0.90, 0.93, 0.3]]
        ),
        np.array([[0, 1, 2, 3]]),
        tolerance=1e-10,
    )
    op = DeformableQ1Penalty(
        secondary,
        primary,
        candidate_nodes=np.ones(4, dtype=bool),
        penalty=100.0,
        search_radius=0.3,
        geometry_policy="reviewed_closed_solid",
    )
    us, up = np.zeros_like(secondary.vertices), np.zeros_like(vertices)
    response = op.evaluate(us, up)
    assert response.potential_energy > 0
    np.testing.assert_allclose(
        response.secondary_residual.sum(0) + response.primary_residual.sum(0),
        0,
        atol=1e-14,
    )
    moment = np.cross(secondary.vertices, response.secondary_residual).sum(
        0
    ) + np.cross(vertices, response.primary_residual).sum(0)
    np.testing.assert_allclose(moment, 0, atol=1e-14)
    rng = np.random.default_rng(92)
    ds, dp = rng.normal(size=us.shape), rng.normal(size=up.shape)
    h = 1e-7
    gradient = (
        op.evaluate(h * ds, h * dp).potential_energy
        - op.evaluate(-h * ds, -h * dp).potential_energy
    ) / (2 * h)
    expected = np.sum(response.secondary_residual * ds) + np.sum(
        response.primary_residual * dp
    )
    assert gradient == pytest.approx(expected, rel=2e-7, abs=1e-11)
