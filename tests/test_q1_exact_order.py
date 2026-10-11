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
