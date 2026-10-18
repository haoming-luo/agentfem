"""Analytical counterexample to naively extending planar clipping to edges."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_contact_feature_transition.py"
active_feature_oracle = runpy.run_path(str(_SCRIPT))["active_feature_oracle"]


@pytest.mark.parametrize("width, center", ((0.05, 0.3), (0.2, 0.5), (0.1, 0.8)))
def test_active_feature_interface_cancels_incorrect_bulk_tangent(width, center):
    penalty = 7.0
    value = active_feature_oracle(width=width, center=center, penalty=penalty)
    h = 1e-6
    plus = active_feature_oracle(width=width, center=center + h, penalty=penalty)
    minus = active_feature_oracle(width=width, center=center - h, penalty=penalty)
    assert (plus["potential"] - minus["potential"]) / (2 * h) == pytest.approx(
        value["generalized_force"], rel=1e-9,
    )
    assert (plus["generalized_force"] - minus["generalized_force"]) / (2 * h) == pytest.approx(
        value["balanced_tangent"], abs=1e-12,
    )
    assert value["fixed_branch_bulk_tangent"] > 0.0
    assert value["missing_feature_interface_tangent"] < 0.0
    assert value["balanced_tangent"] == 0.0


def test_active_feature_oracle_rejects_unbounded_strip():
    with pytest.raises(ValueError, match="Require"):
        active_feature_oracle(width=0.5, center=0.2)
