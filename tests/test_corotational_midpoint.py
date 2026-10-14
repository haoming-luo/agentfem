# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem.constitutive import (
    polar_strain_increment,
    push_corotational_stress,
)
from test_corotational_kinematics import rotation


def midpoint(old, new, **kwargs):
    return polar_strain_increment(old, new, integration_rule="midpoint", **kwargs)


def test_midpoint_has_explicit_numerical_identity_and_is_not_exact_log():
    old = np.eye(3)[None]
    new = np.diag([2.0, 1.0, 1.0])[None]
    result = midpoint(old, new)
    assert result.strain_increment[0, 0, 0] == pytest.approx(2 / 3)
    assert result.method == "polar_linear_right_stretch_midpoint"
    assert (
        polar_strain_increment(old, new).method
        == "polar_linear_right_stretch_exact_integral"
    )
    assert not np.isclose(result.strain_increment[0, 0, 0], np.log(2))


def test_noncoaxial_path_second_order_against_independent_quadrature():
    u0 = np.diag([0.7, 1.4, 1.1])
    u1 = rotation(0.6) @ np.diag([1.6, 0.9, 1.2]) @ rotation(0.6).T
    points, weights = np.polynomial.legendre.leggauss(64)
    reference = np.zeros((3, 3))
    for s, w in zip((points + 1) / 2, weights / 2):
        rate = (u1 - u0) @ np.linalg.inv(u0 + s * (u1 - u0))
        reference += w * (rate + rate.T) / 2
    errors = []
    for count in (8, 16, 32):
        accumulated = np.zeros((3, 3))
        for k in range(count):
            old = rotation(0.9 * k / count) @ (u0 + (u1 - u0) * k / count)
            new = rotation(0.9 * (k + 1) / count) @ (u0 + (u1 - u0) * (k + 1) / count)
            accumulated += midpoint(old[None], new[None]).strain_increment[0]
        errors.append(np.linalg.norm(accumulated - reference))
    assert errors[1] < 0.26 * errors[0] and errors[2] < 0.26 * errors[1]


def test_rigid_rotation_material_axes_and_virtual_power():
    old = np.array([[[1.3, 0.2, 0], [0.2, 0.9, 0.1], [0, 0.1, 1.1]]])
    axes = rotation(0.2)[None]
    rigid = midpoint(old, rotation(2.9) @ old, reference_frame=axes)
    np.testing.assert_allclose(rigid.strain_increment, 0, atol=3e-15)
    new = old + np.diag([0.01, -0.02, 0.03])[None]
    base = midpoint(old, new, reference_frame=axes)
    moved = midpoint(rotation(-0.6) @ old, rotation(1.4) @ new, reference_frame=axes)
    np.testing.assert_allclose(
        base.strain_increment, moved.strain_increment, atol=4e-15
    )
    stress = np.array([[[3.0, 1, 0], [1, 2, 0], [0, 0, 1]]])
    spatial = push_corotational_stress(stress, moved.frame_new)
    spatial_increment = (
        moved.frame_new @ moved.strain_increment @ moved.frame_new.transpose(0, 2, 1)
    )
    np.testing.assert_allclose(
        np.einsum("nij,nij->n", stress, moved.strain_increment),
        np.einsum("nij,nij->n", spatial, spatial_increment),
        atol=1e-14,
    )


def test_stationary_empty_batch_and_chunk_invariance():
    old = np.tile(np.eye(3), (4, 1, 1))
    new = old.copy()
    new[1:, 0, 0] += 1e-8
    all_points = midpoint(old, new)
    for i in range(4):
        np.testing.assert_array_equal(
            all_points.strain_increment[i],
            midpoint(old[i : i + 1], new[i : i + 1]).strain_increment[0],
        )
    assert not all_points.strain_increment.flags.writeable
    assert midpoint(old[:0], new[:0]).strain_increment.shape == (0, 3, 3)
    assert midpoint(old[:0], new[:0]).method.endswith("midpoint")


@pytest.mark.parametrize(
    "bad",
    [
        np.diag([-1.0, 1, 1]),
        np.diag([0.0, 1, 1]),
        np.diag([1e-14, 1, 1]),
        np.full((3, 3), np.nan),
    ],
)
def test_invalid_gradients_still_reject(bad):
    with pytest.raises(ValueError):
        midpoint(np.eye(3)[None], bad[None])


def test_unknown_rule_rejected():
    with pytest.raises(ValueError):
        polar_strain_increment(
            np.eye(3)[None], np.eye(3)[None], integration_rule="fast"
        )
