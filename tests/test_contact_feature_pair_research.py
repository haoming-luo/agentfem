"""The two-feature operator is a mathematical research candidate, not contact."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
import runpy

import numpy as np
import pytest

from agentfem.boundary_models.contact_feature_pair_research import (
    AffineDistanceFeature,
    integrate_two_affine_feature_distances,
)


_ORACLE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "tools"
        / "verify_quadratic_contact_feature_transition.py"
    )
)
_REFERENCE = np.asarray(((1.0, 0.0), (2.0, 0.0), (2.0, 3.0)))
_EDGE = AffineDistanceFeature(np.zeros(2), np.asarray(((1.0,), (0.0,))))
_VERTEX = AffineDistanceFeature(np.asarray((3.0, 1.0)), np.empty((2, 0)))


@pytest.mark.parametrize("perturbed", [False, True])
def test_two_affine_feature_integral_matches_independent_oracle(perturbed):
    displacement = (
        np.asarray(((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01)))
        if perturbed
        else np.zeros((3, 2))
    )
    result = integrate_two_affine_feature_distances(
        _REFERENCE,
        _REFERENCE + displacement,
        _EDGE,
        _VERTEX,
    )
    expected_potential, expected_residual, expected_bulk, entry = _ORACLE[
        "split_residual_and_bulk_tangent"
    ](displacement)
    expected_moving = _ORACLE["moving_feature_tangent"](entry, displacement)
    assert result.potential == pytest.approx(expected_potential, rel=2e-8)
    np.testing.assert_allclose(
        result.nodal_residual, expected_residual, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(result.bulk_tangent, expected_bulk, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        result.moving_boundary_tangent, expected_moving, rtol=2e-8, atol=2e-9
    )
    assert result.branch_areas[0] + result.branch_areas[1] == pytest.approx(1.5)
    assert result.boundary_point_count > 0


def test_two_affine_feature_integral_has_the_full_nodal_variations():
    displacement = np.asarray(((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01)))
    result = integrate_two_affine_feature_distances(
        _REFERENCE,
        _REFERENCE + displacement,
        _EDGE,
        _VERTEX,
    )
    step = 1e-5
    numerical_gradient = np.zeros((3, 2))
    numerical_tangent = np.zeros((3, 2, 3, 2))
    for node in range(3):
        for component in range(2):
            direction = np.zeros((3, 2))
            direction[node, component] = step
            plus = integrate_two_affine_feature_distances(
                _REFERENCE,
                _REFERENCE + displacement + direction,
                _EDGE,
                _VERTEX,
            )
            minus = integrate_two_affine_feature_distances(
                _REFERENCE,
                _REFERENCE + displacement - direction,
                _EDGE,
                _VERTEX,
            )
            numerical_gradient[node, component] = (plus.potential - minus.potential) / (
                2 * step
            )
            numerical_tangent[:, :, node, component] = (
                plus.nodal_residual - minus.nodal_residual
            ) / (2 * step)
    np.testing.assert_allclose(
        result.nodal_residual, numerical_gradient, rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(
        result.nodal_tangent, numerical_tangent, rtol=2e-7, atol=2e-8
    )
    assert np.max(np.abs(result.bulk_tangent - numerical_tangent)) > 0.1


def test_three_dimensional_rigid_frame_covariance():
    reference = np.column_stack((_REFERENCE, np.full(3, 0.2)))
    current = reference + np.asarray(
        ((0.012, -0.006, 0.0), (-0.008, 0.014, 0.0), (0.004, -0.01, 0.0))
    )
    edge = AffineDistanceFeature(
        np.asarray((0.0, 0.0, 0.2)), np.asarray(((1.0,), (0.0,), (0.0,)))
    )
    vertex = AffineDistanceFeature(np.asarray((3.0, 1.0, 0.2)), np.empty((3, 0)))
    original = integrate_two_affine_feature_distances(reference, current, edge, vertex)
    rotation = np.asarray(((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
    shift = np.asarray((0.3, -2.0, 1.4))
    moved_edge = AffineDistanceFeature(
        rotation @ edge.anchor + shift, rotation @ edge.tangent_basis
    )
    moved_vertex = AffineDistanceFeature(
        rotation @ vertex.anchor + shift, np.empty((3, 0))
    )
    moved = integrate_two_affine_feature_distances(
        reference @ rotation.T + shift,
        current @ rotation.T + shift,
        moved_edge,
        moved_vertex,
    )
    assert moved.potential == pytest.approx(original.potential, rel=2e-8)
    np.testing.assert_allclose(
        moved.nodal_residual, original.nodal_residual @ rotation.T, rtol=2e-8, atol=2e-9
    )
    transformed_tangent = np.einsum(
        "pi,aibj,qj->apbq", rotation, original.nodal_tangent, rotation
    )
    np.testing.assert_allclose(
        moved.nodal_tangent, transformed_tangent, rtol=2e-8, atol=2e-9
    )


def test_feature_basis_and_degenerate_reference_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        AffineDistanceFeature(np.zeros(2), np.asarray(((2.0,), (0.0,))))
    with pytest.raises(ValueError, match="positive reference area"):
        integrate_two_affine_feature_distances(
            np.zeros((3, 2)),
            _REFERENCE,
            _EDGE,
            _VERTEX,
        )


def test_separated_feature_has_no_artificial_switch_or_line_tangent():
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    near = AffineDistanceFeature(np.zeros(2), np.empty((2, 0)))
    far = AffineDistanceFeature(np.asarray((10.0, 0.0)), np.empty((2, 0)))
    result = integrate_two_affine_feature_distances(
        reference,
        reference,
        near,
        far,
    )
    assert result.branch_areas == pytest.approx((0.5, 0.0))
    assert result.boundary_point_count == 0
    assert np.all(result.moving_boundary_tangent == 0.0)
    assert result.potential == pytest.approx(1.0 / 12.0)


def test_exactly_tied_features_are_not_silently_assigned_one_side():
    with pytest.raises(ValueError, match="nonzero level set"):
        integrate_two_affine_feature_distances(
            _REFERENCE,
            _REFERENCE,
            _EDGE,
            _EDGE,
        )


def test_triangle_connectivity_permutations_preserve_integral_and_tangent():
    displacement = np.asarray(((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01)))
    baseline = integrate_two_affine_feature_distances(
        _REFERENCE,
        _REFERENCE + displacement,
        _EDGE,
        _VERTEX,
    )
    for ordering in permutations(range(3)):
        indices = list(ordering)
        reordered = integrate_two_affine_feature_distances(
            _REFERENCE[indices],
            (_REFERENCE + displacement)[indices],
            _EDGE,
            _VERTEX,
        )
        assert reordered.potential == pytest.approx(baseline.potential, rel=2e-8)
        np.testing.assert_allclose(
            reordered.nodal_residual,
            baseline.nodal_residual[indices],
            rtol=2e-8,
            atol=2e-9,
        )
        expected_tangent = baseline.nodal_tangent[
            np.ix_(indices, range(2), indices, range(2))
        ]
        np.testing.assert_allclose(
            reordered.nodal_tangent,
            expected_tangent,
            rtol=2e-8,
            atol=2e-9,
        )


def test_pair_resultant_and_moment_are_conjugate_to_rigid_tool_motion():
    current = _REFERENCE + np.asarray(
        ((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01))
    )
    baseline = integrate_two_affine_feature_distances(
        _REFERENCE,
        current,
        _EDGE,
        _VERTEX,
    )

    def moved_potential(translation, angle):
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        edge = AffineDistanceFeature(
            rotation @ _EDGE.anchor + translation,
            rotation @ _EDGE.tangent_basis,
        )
        vertex = AffineDistanceFeature(
            rotation @ _VERTEX.anchor + translation,
            np.empty((2, 0)),
        )
        return integrate_two_affine_feature_distances(
            _REFERENCE,
            current,
            edge,
            vertex,
        ).potential

    step = 1e-5
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = step
        derivative = (moved_potential(shift, 0.0) - moved_potential(-shift, 0.0)) / (
            2 * step
        )
        assert derivative == pytest.approx(
            -baseline.rigid_translation_reaction[axis],
            rel=1e-7,
            abs=1e-8,
        )
    angular_derivative = (
        moved_potential(np.zeros(2), step) - moved_potential(np.zeros(2), -step)
    ) / (2 * step)
    assert angular_derivative == pytest.approx(
        -baseline.rigid_rotation_reaction_about_origin,
        rel=1e-7,
        abs=1e-8,
    )
