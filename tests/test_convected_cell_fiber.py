# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import ufl

from agentfem import backends, mesh, operators


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


def test_convected_cell_fiber_adjoint_derivative_is_exact():
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
    state = operator.apply(displacement)
    rng = np.random.default_rng(20260921)
    direction_duals = rng.normal(size=state.direction.shape)
    direction_dual_increments = rng.normal(size=state.direction.shape)
    tangent_duals = rng.normal(size=state.current_tangents.shape)
    tangent_dual_increments = rng.normal(size=state.current_tangents.shape)
    stretch_duals = rng.normal(size=state.stretch.shape)
    stretch_dual_increments = rng.normal(size=state.stretch.shape)
    derivative = operator.apply_adjoint_derivative(
        displacement,
        increment,
        direction_duals=direction_duals,
        direction_dual_increments=direction_dual_increments,
        tangent_duals=tangent_duals,
        tangent_dual_increments=tangent_dual_increments,
        stretch_duals=stretch_duals,
        stretch_dual_increments=stretch_dual_increments,
    )
    epsilon = 1.0e-6
    plus = fem.Function(space)
    minus = fem.Function(space)
    plus.x.array[:] = displacement.x.array + epsilon * increment.x.array
    minus.x.array[:] = displacement.x.array - epsilon * increment.x.array
    plus.x.scatter_forward()
    minus.x.scatter_forward()
    plus_adjoint = operator.apply_adjoint(
        plus,
        direction_duals=direction_duals + epsilon * direction_dual_increments,
        tangent_duals=tangent_duals + epsilon * tangent_dual_increments,
        stretch_duals=stretch_duals + epsilon * stretch_dual_increments,
    )
    minus_adjoint = operator.apply_adjoint(
        minus,
        direction_duals=direction_duals - epsilon * direction_dual_increments,
        tangent_duals=tangent_duals - epsilon * tangent_dual_increments,
        stretch_duals=stretch_duals - epsilon * stretch_dual_increments,
    )
    finite_difference = (plus_adjoint.array_r - minus_adjoint.array_r) / (
        2.0 * epsilon
    )

    np.testing.assert_allclose(
        derivative.array_r,
        finite_difference,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    derivative.destroy()
    plus_adjoint.destroy()
    minus_adjoint.destroy()


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
    composed = operators.displacement_fiber_bending(
        kinematics,
        neighborhood_gradient,
        cell_weights=weights,
        in_plane_stiffness=2.5,
        normal_stiffness=4.0,
    )
    assert isinstance(composed, operators.NonlinearOperatorContribution)

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
        return state, bending, bending.evaluate(state.direction)

    state, bending, response = bending_response(displacement)
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
        bending_response(plus)[2].energy - bending_response(minus)[2].energy
    ) / (2.0 * epsilon)
    exact = np.vdot(increment.x.petsc_vec.array_r, residual.array_r)

    assert response.energy > 0.0
    assert finite_difference == pytest.approx(exact, rel=3.0e-7, abs=3.0e-9)
    residual.destroy()

    kinematic_increment = kinematics.directional_derivative(
        displacement, increment
    )
    response_increment = bending.linearized_response(
        state.direction,
        kinematic_increment.direction_increment,
        surface_tangent_increment=kinematic_increment.tangent_increment[
            : neighborhood_gradient.owned_cells
        ],
    )
    tangent_dual_increments = np.zeros_like(state.current_tangents)
    tangent_dual_increments[: neighborhood_gradient.owned_cells] = (
        response_increment.surface_tangent_residual_increment
    )
    tangent = kinematics.apply_adjoint_derivative(
        displacement,
        increment,
        direction_duals=response.residual,
        direction_dual_increments=(
            response_increment.direction_residual_increment
        ),
        tangent_duals=tangent_duals,
        tangent_dual_increments=tangent_dual_increments,
    )

    def displacement_residual(field):
        trial_state, _, trial_response = bending_response(field)
        trial_tangent_duals = np.zeros_like(trial_state.current_tangents)
        trial_tangent_duals[: neighborhood_gradient.owned_cells] = (
            trial_response.surface_tangent_residual
        )
        return kinematics.apply_adjoint(
            field,
            direction_duals=trial_response.residual,
            tangent_duals=trial_tangent_duals,
        )

    plus_residual = displacement_residual(plus)
    minus_residual = displacement_residual(minus)
    residual_difference = (
        plus_residual.array_r - minus_residual.array_r
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        tangent.array_r,
        residual_difference,
        rtol=8.0e-7,
        atol=5.0e-8,
    )
    composed_residual = composed.residual(displacement)
    composed_tangent = composed.tangent_action(displacement, increment)
    manual_residual = displacement_residual(displacement)
    assert composed.energy(displacement) == pytest.approx(response.energy)
    np.testing.assert_allclose(
        composed_residual.array_r,
        manual_residual.array_r,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        composed_tangent.array_r,
        residual_difference,
        rtol=8.0e-7,
        atol=5.0e-8,
    )
    assert composed.as_dict()["tangent"].startswith("exact_matrix_free")
    tangent.destroy()
    plus_residual.destroy()
    minus_residual.destroy()
    composed_residual.destroy()
    composed_tangent.destroy()
    manual_residual.destroy()


def test_displacement_bending_binds_to_additive_petsc_tangent():
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
    contribution = operators.displacement_fiber_bending(
        kinematics,
        mesh.cell_gradient_operator(domain, rings=2),
        cell_weights=mesh.owned_cell_measures(domain),
        in_plane_stiffness=2.5,
        normal_stiffness=4.0,
    )
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    local_form = fem.form(ufl.inner(trial, test) * ufl.dx)
    local = fem_petsc.assemble_matrix(local_form)
    local.assemble()
    action = backends.fenicsx_tangent_action(contribution, displacement)
    additive = backends.create_additive_tangent_matrix(local, (action,))
    actual = local.createVecLeft()
    additive.operator.mult(increment.x.petsc_vec, actual)
    expected = local.createVecLeft()
    local.mult(increment.x.petsc_vec, expected)
    bending = contribution.tangent_action(displacement, increment)
    expected.axpy(1.0, bending)

    np.testing.assert_allclose(actual.array_r, expected.array_r, atol=2.0e-13)
    assert action.summary()["ghost_update"].startswith("scatter_forward")
    bending.destroy()
    actual.destroy()
    expected.destroy()
    additive.close()
    local.destroy()


def test_displacement_fiber_bending_is_objective_and_hessian_is_symmetric():
    domain, space, kinematics, _, _ = _case()
    gradient = mesh.cell_gradient_operator(domain, rings=2)
    bending = operators.displacement_fiber_bending(
        kinematics,
        gradient,
        cell_weights=mesh.owned_cell_measures(domain),
        in_plane_stiffness=2.5,
        normal_stiffness=4.0,
    )
    displacement = fem.Function(space, name="U")

    def base_values(x):
        return np.vstack(
            (
                0.08 * x[0] ** 2 + 0.03 * x[1],
                0.06 * x[0] * x[1] - 0.02 * x[0],
                0.05 * x[0] ** 2 + 0.04 * x[1] ** 2,
            )
        )

    displacement.interpolate(base_values)
    first = fem.Function(space, name="DU1")
    first.interpolate(
        lambda x: np.vstack(
            (
                -0.03 * x[0] + 0.02 * x[1],
                0.04 * x[0] ** 2,
                -0.02 * x[0] * x[1] + 0.01 * x[1],
            )
        )
    )
    second = fem.Function(space, name="DU2")
    second.interpolate(
        lambda x: np.vstack(
            (
                0.02 * x[1] ** 2,
                -0.03 * x[0] * x[1],
                0.015 * x[0] + 0.025 * x[1],
            )
        )
    )
    first_action = bending.tangent_action(displacement, first)
    second_action = bending.tangent_action(displacement, second)
    assert np.vdot(first.x.petsc_vec.array_r, second_action.array_r) == (
        pytest.approx(
            np.vdot(second.x.petsc_vec.array_r, first_action.array_r),
            rel=2.0e-11,
            abs=2.0e-11,
        )
    )

    rotation = _rotation()
    rotated = fem.Function(space, name="U_rotated")

    def rotated_values(x):
        reference = np.vstack((x[:2], np.zeros(x.shape[1])))
        return rotation @ (reference + base_values(x)) - reference

    rotated.interpolate(rotated_values)
    rotated_first = fem.Function(space, name="DU1_rotated")
    rotated_first.interpolate(
        lambda x: rotation
        @ np.vstack(
            (
                -0.03 * x[0] + 0.02 * x[1],
                0.04 * x[0] ** 2,
                -0.02 * x[0] * x[1] + 0.01 * x[1],
            )
        )
    )
    residual = bending.residual(displacement)
    rotated_residual = bending.residual(rotated)
    rotated_action = bending.tangent_action(rotated, rotated_first)

    assert bending.energy(rotated) == pytest.approx(
        bending.energy(displacement), rel=2.0e-13, abs=2.0e-13
    )
    np.testing.assert_allclose(
        rotated_residual.array_r.reshape((-1, 3)),
        residual.array_r.reshape((-1, 3)) @ rotation.T,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        rotated_action.array_r.reshape((-1, 3)),
        first_action.array_r.reshape((-1, 3)) @ rotation.T,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    first_action.destroy()
    second_action.destroy()
    residual.destroy()
    rotated_residual.destroy()
    rotated_action.destroy()
