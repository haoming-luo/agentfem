"""A conic feature-switch tangent needs its moving-boundary contribution."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest


ORACLE = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "tools" / "verify_quadratic_contact_feature_transition.py"
))


def test_parabolic_feature_boundary_is_not_a_straight_clip_line():
    potential, residual, bulk, entry = ORACLE["split_residual_and_bulk_tangent"]()
    height = ORACLE["feature_height"]
    midpoint = 0.5 * (entry + 2.0)
    chord_height = 0.5 * (height(entry) + height(2.0))
    assert 1.5 < entry < 1.6
    assert chord_height - height(midpoint) > 0.02
    assert potential > 0.0
    assert residual.shape == (3, 2)
    assert bulk.shape == (3, 2, 3, 2)


def test_quadratic_bernstein_bound_catches_an_interior_switch_missed_by_vertices():
    triangle = np.asarray((
        (1.0, 0.0), (-0.5, np.sqrt(3.0) / 2.0),
        (-0.5, -np.sqrt(3.0) / 2.0),
    ))

    def circle_difference(points):
        return np.sum(points**2, axis=1) - 0.25

    assert np.all(circle_difference(triangle) > 0.0)
    lower, upper = ORACLE["quadratic_bernstein_bounds"](
        triangle, circle_difference,
    )
    assert lower < circle_difference(np.mean(triangle, axis=0)[None, :])[0] < 0.0
    assert upper > 0.0
    far_triangle = triangle + np.asarray((3.0, 0.0))
    far_lower, far_upper = ORACLE["quadratic_bernstein_bounds"](
        far_triangle, circle_difference,
    )
    assert 0.0 < far_lower < far_upper


def test_conic_oracle_refuses_an_unbracketed_or_reversed_feature_path():
    collapsed = np.asarray(((0.0, 0.0), (0.0, 0.0), (0.0, -6.0)))
    with pytest.raises(ValueError, match="one bracketed crossing"):
        ORACLE["_branch_height"](1.75, collapsed)


@pytest.mark.parametrize("perturbed", [False, True])
def test_moving_parabolic_boundary_matches_independent_nodal_differences(perturbed):
    displacement = (
        np.asarray(((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01)))
        if perturbed else np.zeros((3, 2))
    )
    _, residual, bulk, entry = ORACLE["split_residual_and_bulk_tangent"](
        displacement,
    )
    moving = ORACLE["moving_feature_tangent"](entry, displacement)
    total = bulk + moving
    assert np.allclose(total.reshape(6, 6), total.reshape(6, 6).T, atol=1e-12)
    assert np.max(np.linalg.eigvalsh(moving.reshape(6, 6))) < 1e-12
    step = 1e-5
    numerical = np.zeros((3, 2, 3, 2))
    energy_gradient = np.zeros((3, 2))
    for node in range(3):
        for component in range(2):
            direction = np.zeros((3, 2))
            direction[node, component] = step
            plus = ORACLE["split_residual_and_bulk_tangent"](displacement + direction)
            minus = ORACLE["split_residual_and_bulk_tangent"](displacement - direction)
            numerical[:, :, node, component] = (plus[1] - minus[1]) / (2.0 * step)
            energy_gradient[node, component] = (plus[0] - minus[0]) / (2.0 * step)
    assert np.max(np.abs(energy_gradient - residual)) < 1e-8
    assert np.max(np.abs(numerical - total)) < 1e-8
    assert np.max(np.abs(numerical - bulk)) > 1e-2
