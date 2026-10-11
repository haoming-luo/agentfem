# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._q1_exact_order import witness_beats_boundary_minimum
from agentfem._q1_surface_search import Q1SurfaceSearch


def patches():
    return np.array([
        [[4., 0., 2.], [5., 0., 2.], [5., 1., 2.], [4., 1., 2.]],
        [[4., 0., 2.], [4., -1., 2.], [5., -1., 2.], [5., 0., 2.]],
    ])


def test_tiny_interior_edge_margin_uses_exact_fallback(monkeypatch):
    import agentfem._q1_exact_order as module

    v = patches()
    point = np.array([4.25, 3e-12, 2. + 4e-8])
    _, ids = np.unique(v.reshape(-1, 3), axis=0, return_inverse=True)
    calls = []
    original = module.witness_beats_boundary_minimum

    def checked(*args):
        result = original(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "witness_beats_boundary_minimum", checked)
    search = Q1SurfaceSearch(v, node_ids=ids.reshape(2, 4))
    result = search.project([point], maximum_distance=.1, require_unique_normal=False)
    assert result["facet_indices"][0] == 0
    assert calls == [True]
    np.testing.assert_allclose(result["points"][0], [4.25, 3e-12, 2.], atol=1e-15)


@pytest.mark.parametrize("offset", [1e-12, 1e-9, .2])
def test_exact_certificate_orders_feasible_witness(offset):
    v = patches()
    assert witness_beats_boundary_minimum(v[0], [4.25, offset, 2.01],
                                         [.25, offset], v[1])


def test_equal_distance_is_not_resolved_arbitrarily():
    v = patches()
    assert not witness_beats_boundary_minimum(v[0], [4.25, 0., 2.01], [.25, 0.], v[1])


def test_equivalent_shared_boundary_does_not_pay_for_exact_proof(monkeypatch):
    import agentfem._q1_exact_order as module

    def unexpected(*args):
        raise AssertionError('Equivalent trace does not need rational ordering.')

    monkeypatch.setattr(module, 'witness_beats_boundary_minimum', unexpected)
    v = patches()
    _, ids = np.unique(v.reshape(-1, 3), axis=0, return_inverse=True)
    result = Q1SurfaceSearch(v, node_ids=ids.reshape(2, 4)).project(
        [[4.25, 0., 2.01]], maximum_distance=.1, require_unique_normal=False)
    assert result['facet_indices'][0] == 0


def test_other_patch_interior_minimum_is_not_an_edge_certificate():
    v = patches()
    assert not witness_beats_boundary_minimum(v[0], [4.25, -.2, 2.01], [.25, 0.], v[1])


def test_infeasible_witness_and_nonconvex_patch_rejected():
    v = patches()
    assert not witness_beats_boundary_minimum(v[0], [4.25, .2, 2.01], [.25, -.1], v[1])
    folded = v[1].copy()
    folded[2] = folded[0]
    assert not witness_beats_boundary_minimum(v[0], [4.25, .2, 2.01], [.25, .2], folded)


def test_loose_witness_cannot_claim_closer_than_other_patch():
    v = patches()
    assert not witness_beats_boundary_minimum(v[0], [4.25, .2, 2.01], [.9, .9], v[1])


def test_exact_distance_ball_does_not_round_outside_to_inside():
    from agentfem._q1_exact_order import witness_within_distance

    v = patches()[0]
    p = v[0].copy()
    p[2] += .125
    assert witness_within_distance(v, p, [0., 0.], .125)
    assert not witness_within_distance(v, p, [0., 0.], np.nextafter(.125, 0))
    assert not witness_within_distance(v, p, [-1e-16, 0.], 1.)


def test_strong_convex_intervals_resolve_two_interior_near_minima():
    from agentfem._q1_exact_order import convex_minimum_precedes

    first = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    second = first + [0., 0., 2e-11]
    point = np.array([.3, .4, np.nextafter(1e-11, 0.)])
    # Deliberately inaccurate initial witnesses: Newton proposes, the exact
    # strong-convexity enclosure certifies. No root precision assumption.
    assert convex_minimum_precedes(first, point, [.31, .39], second, [.29, .41])
    assert not convex_minimum_precedes(second, point, [.31, .39], first, [.29, .41])
    assert not convex_minimum_precedes(first, [.3, .4, 1e-11], [.3, .4],
                                       second, [.3, .4])
