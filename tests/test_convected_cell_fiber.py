# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import mesh, operators


def _rotation():
    axis = np.array((1.0, 2.0, -0.5))
    axis /= np.linalg.norm(axis)
    angle = 0.7
    skew = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _case():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        3,
        2,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    transfer = operators.cell_average_gradient(space)
    reference_tangents = np.array(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    angle = 0.35
    coordinates = np.array((np.cos(angle), np.sin(angle)))
    operator = operators.convected_cell_fiber(
        transfer,
        reference_tangents=reference_tangents,
        reference_tangent_coordinates=coordinates,
    )
    return domain, space, operator, reference_tangents, coordinates


def test_convected_cell_fiber_reproduces_rigid_rotation():
    _, space, operator, reference_tangents, coordinates = _case()
    rotation = _rotation()
    displacement = fem.Function(space, name="U")
    displacement.interpolate(
        lambda x: rotation @ np.vstack((x[:2], np.zeros(x.shape[1])))
        - np.vstack((x[:2], np.zeros(x.shape[1])))
    )

    state = operator.apply(displacement)
    expected_tangents = rotation @ reference_tangents
    expected_direction = rotation @ (reference_tangents @ coordinates)

    np.testing.assert_allclose(
        state.current_tangents,
        np.broadcast_to(expected_tangents, state.current_tangents.shape),
        atol=3.0e-14,
    )
    np.testing.assert_allclose(state.stretch, 1.0, atol=3.0e-14)
    np.testing.assert_allclose(
        state.direction,
        np.broadcast_to(expected_direction, state.direction.shape),
        atol=3.0e-14,
    )


def test_convected_cell_fiber_derivative_and_adjoint_are_exact():
    _, space, operator, _, _ = _case()
    displacement = fem.Function(space, name="U")
    displacement.interpolate(
        lambda x: np.vstack(
            (
                0.2 * x[0] + 0.1 * x[1],
                -0.1 * x[0] + 0.3 * x[1],
                0.15 * x[0],
            )
        )
    )
    increment = fem.Function(space, name="DU")
    increment.interpolate(
        lambda x: np.vstack(
            (-0.3 * x[0] + 0.2 * x[1], 0.25 * x[0], -0.1 * x[1])
        )
    )
    derivative = operator.directional_derivative(displacement, increment)
    epsilon = 1.0e-6
    plus = fem.Function(space)
    minus = fem.Function(space)
    plus.x.array[:] = displacement.x.array + epsilon * increment.x.array
    minus.x.array[:] = displacement.x.array - epsilon * increment.x.array
    plus.x.scatter_forward()
    minus.x.scatter_forward()
    plus_state = operator.apply(plus)
    minus_state = operator.apply(minus)

    np.testing.assert_allclose(
        derivative.tangent_increment,
        (plus_state.current_tangents - minus_state.current_tangents)
        / (2.0 * epsilon),
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        derivative.stretch_increment,
        (plus_state.stretch - minus_state.stretch) / (2.0 * epsilon),
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        derivative.direction_increment,
        (plus_state.direction - minus_state.direction) / (2.0 * epsilon),
        rtol=2.0e-9,
        atol=2.0e-10,
    )

    rng = np.random.default_rng(20260920)
    direction_duals = rng.normal(size=plus_state.direction.shape)
    tangent_duals = rng.normal(size=plus_state.current_tangents.shape)
    stretch_duals = rng.normal(size=plus_state.stretch.shape)
    adjoint = operator.apply_adjoint(
        displacement,
        direction_duals=direction_duals,
        tangent_duals=tangent_duals,
        stretch_duals=stretch_duals,
    )
    output_work = (
        np.vdot(derivative.direction_increment, direction_duals)
        + np.vdot(derivative.tangent_increment, tangent_duals)
        + np.vdot(derivative.stretch_increment, stretch_duals)
    )
    source_work = np.vdot(increment.x.petsc_vec.array_r, adjoint.array_r)
    assert output_work == pytest.approx(source_work, rel=3.0e-13, abs=3.0e-13)
    adjoint.destroy()


def test_displacement_derived_bending_energy_has_exact_fem_residual():
    domain, space, kinematics, _, _ = _case()
    displacement = fem.Function(space, name="U")
    displacement.interpolate(
        lambda x: np.vstack(
            (
                0.08 * x[0] ** 2 + 0.03 * x[1],
                0.06 * x[0] * x[1] - 0.02 * x[0],
                0.05 * x[0] ** 2 + 0.04 * x[1] ** 2,
            )
        )
    )
    increment = fem.Function(space, name="DU")
    increment.interpolate(
        lambda x: np.vstack(
            (
                -0.03 * x[0] + 0.02 * x[1],
                0.04 * x[0] ** 2,
                -0.02 * x[0] * x[1] + 0.01 * x[1],
            )
        )
    )
    neighborhood_gradient = mesh.cell_gradient_operator(domain, rings=2)
    weights = mesh.owned_cell_measures(domain)

    def bending_response(field):
        state = kinematics.apply(field)
        bending = operators.fiber_direction_bending(
            neighborhood_gradient,
            current_tangents=state.current_tangents[
                : neighborhood_gradient.owned_cells
            ],
            cell_weights=weights,
            in_plane_stiffness=2.5,
            normal_stiffness=4.0,
        )
        return state, bending.evaluate(state.direction)

    state, response = bending_response(displacement)
    tangent_duals = np.zeros_like(state.current_tangents)
    tangent_duals[: neighborhood_gradient.owned_cells] = (
        response.surface_tangent_residual
    )
    residual = kinematics.apply_adjoint(
        displacement,
        direction_duals=response.residual,
        tangent_duals=tangent_duals,
    )
    epsilon = 1.0e-6
    plus = fem.Function(space)
    minus = fem.Function(space)
    plus.x.array[:] = displacement.x.array + epsilon * increment.x.array
    minus.x.array[:] = displacement.x.array - epsilon * increment.x.array
    plus.x.scatter_forward()
    minus.x.scatter_forward()
    finite_difference = (
        bending_response(plus)[1].energy - bending_response(minus)[1].energy
    ) / (2.0 * epsilon)
    exact = np.vdot(increment.x.petsc_vec.array_r, residual.array_r)

    assert response.energy > 0.0
    assert finite_difference == pytest.approx(exact, rel=3.0e-7, abs=3.0e-9)
    residual.destroy()
