# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import mesh, operators


def _rotation():
    axis = np.array((1.0, 2.0, -0.5))
    axis /= np.linalg.norm(axis)
    angle = 0.7
    skew = np.array(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _case(n=6, *, rotation=None):
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        n,
        n,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    gradient = mesh.cell_gradient_operator(domain, rings=2)
    cells = np.arange(gradient.total_cells, dtype=np.int32)
    points = dolfinx_mesh.compute_midpoints(domain, domain.topology.dim, cells)[:, :2]
    angle = 0.8 * points[:, 0]
    directions = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    tangents = np.broadcast_to(
        np.array(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        (gradient.owned_cells, 3, 2),
    ).copy()
    if rotation is not None:
        directions = directions @ rotation.T
        tangents = np.einsum("ij,cjk->cik", rotation, tangents)
    operator = operators.fiber_direction_bending(
        gradient,
        current_tangents=tangents,
        cell_weights=mesh.owned_cell_measures(domain),
        in_plane_stiffness=2.5,
        normal_stiffness=4.0,
    )
    return operator, directions, points


def test_fiber_bending_residual_is_exact_energy_directional_derivative():
    operator, directions, points = _case()
    perturbation = np.column_stack(
        (
            0.2 + points[:, 0],
            -0.3 + points[:, 1] ** 2,
            np.zeros(len(points)),
        )
    )
    epsilon = 1.0e-6
    finite_difference = (
        operator.evaluate(directions + epsilon * perturbation).energy
        - operator.evaluate(directions - epsilon * perturbation).energy
    ) / (2.0 * epsilon)
    response = operator.evaluate(directions)
    exact = np.vdot(response.residual, perturbation)

    assert finite_difference == pytest.approx(exact, rel=2.0e-7, abs=2.0e-9)
    assert response.energy > 0.0
    assert operator.as_dict()["nonlinear_tangent"].startswith("exact_matrix_free")


def test_fiber_bending_tangent_matches_residual_derivative_and_is_symmetric():
    operator, directions, points = _case()
    first = np.column_stack(
        (
            0.2 + points[:, 0],
            -0.3 + points[:, 1] ** 2,
            np.zeros(len(points)),
        )
    )
    second = np.column_stack(
        (
            -0.4 + points[:, 0] ** 2,
            0.1 + points[:, 1],
            np.zeros(len(points)),
        )
    )
    epsilon = 2.0e-6
    finite_difference = (
        operator.evaluate(directions + epsilon * first).residual
        - operator.evaluate(directions - epsilon * first).residual
    ) / (2.0 * epsilon)
    first_action = operator.tangent_action(directions, first)
    second_action = operator.tangent_action(directions, second)

    np.testing.assert_allclose(first_action, finite_difference, rtol=2.0e-7, atol=2.0e-8)
    assert np.vdot(first, second_action) == pytest.approx(
        np.vdot(first_action, second), rel=2.0e-12, abs=2.0e-12
    )


def test_fiber_bending_is_scale_invariant_and_residual_is_radially_orthogonal():
    operator, directions, points = _case()
    scale = 0.7 + points[:, 0] + 0.4 * points[:, 1]
    baseline = operator.evaluate(directions)
    scaled = operator.evaluate(scale[:, None] * directions)

    assert scaled.energy == pytest.approx(baseline.energy, rel=2.0e-13)
    np.testing.assert_allclose(
        scaled.in_plane_curvature,
        baseline.in_plane_curvature,
        atol=2.0e-13,
    )
    radial_work = np.sum(scaled.residual * directions, axis=1)
    np.testing.assert_allclose(radial_work, 0.0, atol=2.0e-13)


def test_fiber_bending_is_objective_and_residual_is_covariant():
    baseline, directions, _ = _case()
    rotation = _rotation()
    rotated, rotated_directions, _ = _case(rotation=rotation)

    first = baseline.evaluate(directions)
    second = rotated.evaluate(rotated_directions)

    assert second.energy == pytest.approx(first.energy, rel=2.0e-13)
    np.testing.assert_allclose(
        second.in_plane_curvature, first.in_plane_curvature, atol=2.0e-13
    )
    np.testing.assert_allclose(second.normal_curvature, first.normal_curvature, atol=2.0e-13)
    np.testing.assert_allclose(second.residual, first.residual @ rotation.T, atol=3.0e-13)
    increment = np.column_stack(
        (
            np.linspace(-0.2, 0.4, len(directions)),
            np.linspace(0.1, -0.3, len(directions)),
            np.zeros(len(directions)),
        )
    )
    np.testing.assert_allclose(
        rotated.tangent_action(rotated_directions, increment @ rotation.T),
        baseline.tangent_action(directions, increment) @ rotation.T,
        atol=8.0e-13,
    )


def test_fiber_bending_constant_direction_has_zero_energy_and_residual():
    operator, directions, _ = _case()
    directions[:] = (1.0, 0.0, 0.0)

    response = operator.evaluate(directions)

    assert response.energy == pytest.approx(0.0, abs=1.0e-25)
    np.testing.assert_allclose(response.residual, 0.0, atol=2.0e-14)


def test_fiber_bending_rejects_nontangent_direction():
    operator, directions, _ = _case()
    directions[0] = (1.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="tangent"):
        operator.evaluate(directions)
