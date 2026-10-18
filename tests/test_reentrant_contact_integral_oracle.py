"""Independent L-boundary quadrature counterexample; not a solver claim."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest


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
        assert result["nodal_force_interval_only"] is True
        assert result["newton_tangent_certified"] is False
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
    nodal_lower = np.asarray(bounds[2]["reference_p1_nodal_force_lower"])
    nodal_upper = np.asarray(bounds[2]["reference_p1_nodal_force_upper"])
    assert nodal_lower.shape == nodal_upper.shape == (3, 2)
    assert np.all(nodal_lower <= nodal_upper)
    assert np.all(np.sum(nodal_lower, axis=0) <=
                  bounds[2]["translation_force_upper"])
    assert np.all(np.sum(nodal_upper, axis=0) >=
                  bounds[2]["translation_force_lower"])
    sampled_nodal = ORACLE["sampled_reference_p1_nodal_force"](
        subdivisions=512,
    )
    assert np.all(nodal_lower < sampled_nodal)
    assert np.all(sampled_nodal < nodal_upper)
    # An independent directional derivative uses a fixed reference measure:
    # perturb the physical P1 map, not the area of its reference triangle.
    reference_points = ORACLE["triangle_centroids"](256)
    vertices = ORACLE["SLAVE"]
    coordinates = (reference_points - vertices[0]) @ np.linalg.inv(
        np.stack((vertices[1] - vertices[0], vertices[2] - vertices[0])),
    )
    shape = np.column_stack((1.0 - np.sum(coordinates, axis=1), coordinates))
    area = 0.5 * abs(float(np.linalg.det(np.stack((
        vertices[1] - vertices[0], vertices[2] - vertices[0],
    )))))
    step = 1e-4
    for node in range(3):
        for component in range(2):
            direction = np.zeros_like(reference_points)
            direction[:, component] = shape[:, node]
            positive_gap = ORACLE["signed_gap"](reference_points + step * direction)
            negative_gap = ORACLE["signed_gap"](reference_points - step * direction)
            positive = area * np.mean(0.5 * np.maximum(-positive_gap, 0.0) ** 2)
            negative = area * np.mean(0.5 * np.maximum(-negative_gap, 0.0) ** 2)
            directional = (positive - negative) / (2 * step)
            assert nodal_lower[node, component] < directional
            assert directional < nodal_upper[node, component]


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
    # The exact fixed-segment branch gives a P1 weak residual, not just a
    # resultant. Its sum must recover the total translation force.
    nodal_lower = np.asarray(result["reference_p1_nodal_force_lower"])
    nodal_upper = np.asarray(result["reference_p1_nodal_force_upper"])
    expected_nodal_x = area * np.mean(
        barycentric * (barycentric @ triangle)[:, 0, None], axis=0,
    )
    assert np.all(nodal_lower[:, 0] <= expected_nodal_x)
    assert np.all(expected_nodal_x <= nodal_upper[:, 0])
    assert np.allclose(np.sum(expected_nodal_x), expected_force_x)


def test_feature_aware_reference_keeps_unresolved_cells_bounded_at_budget():
    reference = ORACLE["penalty_potential"](subdivisions=512)
    bounded = ORACLE["bounded_potential_reference"](
        maximum_depth=12, maximum_cells=1,
    )
    assert bounded["processed_subtriangles"] == 1
    assert bounded["uncertain_subtriangles"] == 1
    assert bounded["lower"] <= reference <= bounded["upper"]
    nodal_lower = np.asarray(bounded["reference_p1_nodal_force_lower"])
    nodal_upper = np.asarray(bounded["reference_p1_nodal_force_upper"])
    sampled = ORACLE["sampled_reference_p1_nodal_force"](subdivisions=512)
    assert np.all(nodal_lower <= sampled)
    assert np.all(sampled <= nodal_upper)
    with pytest.raises(ValueError, match="nondegenerate"):
        ORACLE["bounded_potential_reference"](
            vertices=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        )
