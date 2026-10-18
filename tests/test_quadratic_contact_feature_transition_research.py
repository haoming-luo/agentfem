"""A conic feature-switch tangent needs its moving-boundary contribution."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np


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


def test_moving_parabolic_boundary_matches_independent_nodal_differences():
    _, _, bulk, entry = ORACLE["split_residual_and_bulk_tangent"]()
    moving = ORACLE["moving_feature_tangent"](entry)
    total = bulk + moving
    assert np.allclose(total.reshape(6, 6), total.reshape(6, 6).T, atol=1e-12)
    assert np.max(np.linalg.eigvalsh(moving.reshape(6, 6))) < 1e-12
    step = 1e-5
    numerical = np.zeros((3, 3))
    for node in range(3):
        direction = np.zeros(3)
        direction[node] = step
        plus = ORACLE["split_residual_and_bulk_tangent"](direction)[1]
        minus = ORACLE["split_residual_and_bulk_tangent"](-direction)[1]
        numerical[:, node] = (plus[:, 1] - minus[:, 1]) / (2.0 * step)
    assert np.max(np.abs(numerical - total[:, 1, :, 1])) < 1e-8
    assert np.max(np.abs(numerical - bulk[:, 1, :, 1])) > 1e-2
