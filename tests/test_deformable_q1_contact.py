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


def test_closed_solid_keeps_far_candidates_without_fake_projections():
    from test_closed_q1_surface import solid

    vertices, faces = solid()
    near = np.array(
        [[0.2, 0.2, 0.01], [0.4, 0.2, 0.01], [0.4, 0.4, 0.01], [0.2, 0.4, 0.01]]
    )
    secondary = BilinearReferenceTrace(
        np.vstack((near, near + [20, 0, 0])),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        tolerance=1e-10,
    )
    primary = BilinearReferenceTrace(vertices, faces, tolerance=1e-10)
    op = DeformableQ1Penalty(
        secondary,
        primary,
        candidate_nodes=np.ones(8, dtype=bool),
        penalty=100.0,
        search_radius=0.1,
        geometry_policy="reviewed_closed_solid",
    )
    us, up = np.zeros_like(secondary.vertices), np.zeros_like(vertices)
    result = op.evaluate(us, up)
    np.testing.assert_array_equal(result.projection_indices, np.arange(4))
    assert len(result.projection.signed_gaps) == 4
    assert result.path_gaps.shape == (8,)
    assert np.all(result.path_gaps[:4] < 0) and np.all(result.path_gaps[4:] > 0)
    np.testing.assert_array_equal(result.secondary_residual[4:], 0)
    np.testing.assert_allclose(
        result.secondary_residual.sum(0) + result.primary_residual.sum(0), 0, atol=1e-14
    )
    us[:4, 0] = 20
    separated = op.evaluate(us, up)
    assert separated.projection is None
    assert np.all(separated.gap_lower_bounds > 0)
    assert separated.potential_energy == 0


def test_closed_exterior_medial_axis_needs_separation_not_a_chosen_face():
    from test_closed_q1_surface import solid
    from agentfem._q1_surface_search import AmbiguousSurfaceProjection
    from agentfem._closed_q1_surface import ClosedQ1Surface

    vertices, faces = solid(concave=True)
    points = np.array([[1.1, 1.1, .4], [1.2, 1.1, .4],
                       [1.2, 1.1, .6], [1.1, 1.1, .6]])
    surface = ClosedQ1Surface(vertices, faces)
    with pytest.raises(AmbiguousSurfaceProjection):
        surface.project(points, maximum_distance=.3)
    secondary = BilinearReferenceTrace(points, np.array([[0, 1, 2, 3]]), tolerance=1e-10)
    primary = BilinearReferenceTrace(vertices, faces, tolerance=1e-10)
    op = DeformableQ1Penalty(secondary, primary, candidate_nodes=np.ones(4, dtype=bool),
                           penalty=100., search_radius=.3,
                           geometry_policy="reviewed_closed_solid")
    answer = op.evaluate(np.zeros_like(points), np.zeros_like(vertices))
    np.testing.assert_array_equal(answer.secondary_residual, 0)
    np.testing.assert_array_equal(answer.primary_residual, 0)
    assert answer.potential_energy == 0
    assert np.all(answer.gap_lower_bounds > 0)
    np.testing.assert_array_equal(answer.projection_indices, [1, 2])
    # Fresh geometry evaluation agrees, as required during checkpoint restore.
    again = op.evaluate(np.zeros_like(points), np.zeros_like(vertices))
    np.testing.assert_array_equal(answer.gap_lower_bounds, again.gap_lower_bounds)


def test_closed_interior_ambiguous_projection_is_still_rejected():
    from test_closed_q1_surface import solid
    from agentfem._q1_surface_search import AmbiguousSurfaceProjection

    vertices, faces = solid()
    points = np.array([[.5, .5, .5], [.6, .5, .5], [.6, .6, .5], [.5, .6, .5]])
    op = DeformableQ1Penalty(
        BilinearReferenceTrace(points, np.array([[0, 1, 2, 3]]), tolerance=1e-10),
        BilinearReferenceTrace(vertices, faces, tolerance=1e-10),
        candidate_nodes=np.ones(4, dtype=bool), penalty=100., search_radius=.6,
        geometry_policy="reviewed_closed_solid")
    with pytest.raises(AmbiguousSurfaceProjection):
        op.evaluate(np.zeros_like(points), np.zeros_like(vertices))


def test_closed_outside_missing_projection_with_loose_bound_is_force_free(monkeypatch):
    from test_closed_q1_surface import solid
    from agentfem._q1_surface_search import Q1SurfaceSearch

    vertices, faces = solid(concave=True)
    points = np.array([[1.2,1.2,.4], [1.3,1.2,.4],
                       [1.3,1.3,.4], [1.2,1.3,.4]])
    original = Q1SurfaceSearch.distance_lower_bounds
    # Loosening an existing valid lower bound remains conservative. It models
    # the broad-phase bound/radius mismatch without inventing a separation.
    monkeypatch.setattr(Q1SurfaceSearch, "distance_lower_bounds",
        lambda self, queries: np.minimum(original(self, queries), .01))
    op = DeformableQ1Penalty(
        BilinearReferenceTrace(points, np.array([[0,1,2,3]]), tolerance=1e-10),
        BilinearReferenceTrace(vertices, faces, tolerance=1e-10),
        candidate_nodes=np.ones(4,dtype=bool), penalty=100., search_radius=.1,
        geometry_policy="reviewed_closed_solid")
    answer = op.evaluate(np.zeros_like(points), np.zeros_like(vertices))
    assert answer.projection is None
    assert np.all(answer.gap_lower_bounds > 0)
    np.testing.assert_array_equal(answer.secondary_residual, 0)
    assert answer.potential_energy == 0


def test_closed_interior_outside_radius_still_rejected():
    from test_closed_q1_surface import solid
    from agentfem._q1_surface_search import SurfaceOutsideSearchRadius

    vertices, faces = solid()
    points = np.array([[.4,.4,.4],[.6,.4,.4],[.6,.6,.4],[.4,.6,.4]])
    op = DeformableQ1Penalty(
        BilinearReferenceTrace(points, np.array([[0,1,2,3]]), tolerance=1e-10),
        BilinearReferenceTrace(vertices, faces, tolerance=1e-10),
        candidate_nodes=np.ones(4,dtype=bool), penalty=100., search_radius=.1,
        geometry_policy="reviewed_closed_solid")
    with pytest.raises(SurfaceOutsideSearchRadius):
        op.evaluate(np.zeros_like(points), np.zeros_like(vertices))


def test_open_surface_outside_radius_still_rejected():
    from agentfem._q1_surface_search import SurfaceOutsideSearchRadius
    op, us, up = setup()
    us[:,2] += 1.
    with pytest.raises(SurfaceOutsideSearchRadius):
        op.evaluate(us, up)


def test_mixed_path_certificates_only_search_uncertain_points(monkeypatch):
    from test_closed_q1_surface import solid
    from agentfem._closed_q1_surface import ClosedQ1Surface
    from agentfem._contact_path_guard import outside_gap_lower_bounds

    vertices, faces = solid()
    near = np.array([[.2, .2, -.02], [.4, .2, -.02],
                     [.4, .4, -.02], [.2, .4, -.02]])
    points = np.vstack((near, near + [.4, 0, 0]))
    op = DeformableQ1Penalty(
        BilinearReferenceTrace(points, np.arange(8).reshape(2, 4), tolerance=1e-10),
        BilinearReferenceTrace(vertices, faces, tolerance=1e-10),
        candidate_nodes=np.ones(8, dtype=bool), penalty=100., search_radius=.2,
        geometry_policy="reviewed_closed_solid")
    old = np.zeros_like(points)
    up = np.zeros_like(vertices)
    accepted = op.evaluate(old, up)
    moved = old.copy()
    moved[:4, 2] = .03
    moved[4:, 2] = .001
    lower = outside_gap_lower_bounds(accepted.gap_lower_bounds, moved, up)
    assert np.all(lower[:4] < 0) and np.all(lower[4:] > 0)
    exact = op.evaluate(moved, up)
    counts = []
    project = ClosedQ1Surface.project

    def counted(self, query, **kwargs):
        counts.append(len(query))
        return project(self, query, **kwargs)

    monkeypatch.setattr(ClosedQ1Surface, "project", counted)
    bounded = op._evaluate(moved, up, path_lower_bounds=lower)
    assert counts == [4]
    np.testing.assert_array_equal(bounded.projection_indices, np.arange(4))
    np.testing.assert_allclose(bounded.secondary_residual, exact.secondary_residual)
    np.testing.assert_allclose(bounded.primary_residual, exact.primary_residual)
    assert bounded.potential_energy == exact.potential_energy
    assert np.all(bounded.gap_lower_bounds <= exact.path_gaps + 1e-14)
    # Once the formerly separated points approach, their certificates expire.
    newer = moved.copy()
    newer[4:, 2] = .04
    lower2 = outside_gap_lower_bounds(bounded.gap_lower_bounds, newer - moved, up)
    counts.clear()
    next_response = op._evaluate(newer, up, path_lower_bounds=lower2)
    assert counts == [8]
    fresh = op.evaluate(newer, up)
    np.testing.assert_allclose(next_response.primary_residual, fresh.primary_residual)


@pytest.mark.parametrize("bound", [np.ones(3), np.full(4, np.nan),
                                    np.ones(4, dtype=complex)])
def test_path_certificates_reject_invalid_or_open_surface(bound):
    op, us, up = setup()
    with pytest.raises(ValueError, match="Path bounds"):
        op._evaluate(us, up, path_lower_bounds=bound)


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
