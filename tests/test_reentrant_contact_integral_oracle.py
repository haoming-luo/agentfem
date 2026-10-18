"""Independent L-boundary quadrature counterexample; not a solver claim."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np


ORACLE = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "tools" / "verify_reentrant_contact_integral.py"
))


def test_reentrant_face_can_hide_positive_penalty_energy_from_three_point_rule():
    vertices = ORACLE["SLAVE"]
    barycentric = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                              (1 / 6, 2 / 3, 1 / 6),
                              (1 / 6, 1 / 6, 2 / 3)))
    assert np.all(ORACLE["signed_gap"](barycentric @ vertices) > 0.0)
    assert ORACLE["signed_gap"](vertices[:1])[0] < 0.0
    assert ORACLE["penalty_potential"](subdivisions=None) == 0.0
    coarse = ORACLE["penalty_potential"](subdivisions=128)
    medium = ORACLE["penalty_potential"](subdivisions=256)
    refined = ORACLE["penalty_potential"](subdivisions=512)
    assert 0.0 < coarse < medium < refined
    assert (refined - medium) / refined < 1.0e-3


def test_feature_aware_reference_bounds_hidden_energy_and_total_force():
    reference = ORACLE["penalty_potential"](subdivisions=512)
    bounds = [ORACLE["bounded_potential_reference"](maximum_depth=depth)
              for depth in (6, 8, 10)]
    for result in bounds:
        assert result["lower"] < reference < result["upper"]
        assert result["uncertain_subtriangles"] > 0
        assert result["nodal_force_or_tangent_certified"] is False
    assert bounds[0]["absolute_interval_width"] > bounds[1]["absolute_interval_width"]
    assert bounds[1]["absolute_interval_width"] > bounds[2]["absolute_interval_width"]
    assert bounds[2]["absolute_interval_width"] < 3.2e-6
    assert bounds[2]["processed_subtriangles"] < 25_000
    shift = 1e-4
    numerical_force_x = (
        ORACLE["penalty_potential"](subdivisions=512, shift_x=shift)
        - ORACLE["penalty_potential"](subdivisions=512, shift_x=-shift)
    ) / (2 * shift)
    assert (bounds[2]["translation_force_lower"][0]
            < numerical_force_x
            < bounds[2]["translation_force_upper"][0])


def test_feature_aware_reference_integrates_one_fixed_segment_branch():
    triangle = np.asarray(((0.1, 1.4), (0.2, 1.4), (0.1, 1.5)))
    result = ORACLE["bounded_potential_reference"](
        vertices=triangle, maximum_depth=0,
    )
    barycentric = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                              (1 / 6, 2 / 3, 1 / 6),
                              (1 / 6, 1 / 6, 2 / 3)))
    area = 0.005
    expected = area * np.mean(0.5 * (barycentric @ triangle)[:, 0] ** 2)
    assert result["exact_active_subtriangles"] == 1
    assert result["uncertain_subtriangles"] == 0
    assert result["lower"] <= expected <= result["upper"]
    assert result["absolute_interval_width"] < 1e-12
    expected_force_x = area * np.mean((barycentric @ triangle)[:, 0])
    assert (result["translation_force_lower"][0]
            <= expected_force_x
            <= result["translation_force_upper"][0])
