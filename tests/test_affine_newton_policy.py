# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""The public Newton policy must mean the same thing after exact reduction."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import solvers
from agentfem._affine_equilibrium import solve_serial_affine_equilibrium
from agentfem.constraints.affine import AffineReduction
from periodic_cube_fixture import periodic_unit_cube


@pytest.mark.parametrize("line_search", ["basic", None, "backtracking"])
def test_affine_cubic_equilibrium_honors_public_full_step_policy(line_search):
    # A real assembled FE residual, but a deliberately elementary algebraic
    # oracle: from u=.1, Newton's full first update is 33.4 and its residual
    # grows. Full Newton then converges to one; Armijo must shorten that step.
    # This distinguishes basic Newton from "backtracking with minimum=1".
    domain = mesh.create_unit_interval(MPI.COMM_SELF, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(space)
    u.x.array[:] = .1
    test = ufl.TestFunction(space)
    residual = (u**3 - 1.) * test * ufl.dx
    jacobian = ufl.derivative(residual, u)
    n = len(u.x.array)
    reduction = AffineReduction(np.arange(n + 1), np.arange(n), np.ones(n),
                                np.zeros(n), np.arange(n))
    transformation = reduction.matrix(MPI.COMM_SELF)
    events = []
    try:
        response = solve_serial_affine_equilibrium(
            fem.form(residual), fem.form(jacobian), u, reduction, transformation,
            u.x.array.copy(), options=solvers.newton(
                line_search=line_search, maximum_iterations=30,
                relative_tolerance=1e-10, absolute_tolerance=1e-11,
            ).for_affine_reduction(), operators=(), transaction=None,
            start_factor=0., target_factor=1., reporter=events.append,
            step_name="cubic_policy", step_number=1, increment_number=1, attempt_number=1,
        )
        assert response.converged
        np.testing.assert_allclose(u.x.array, 1., atol=1e-10)
        if line_search == "backtracking":
            assert response.step_lengths[0] < 1.
        else:
            assert all(alpha == 1. for alpha in response.step_lengths)
            assert events[0].residual_norm > response.initial_norm
    finally:
        transformation.destroy()


def test_explicit_no_search_means_full_step_for_both_backends():
    for value in ("basic", None, "off"):
        policy = solvers.newton(line_search=value)
        assert policy.for_snes().petsc_options()["snes_linesearch_type"] == "basic"
        assert policy.for_affine_reduction().line_search == "basic"
    assert solvers.newton().for_affine_reduction().line_search == "backtracking"
    assert "line_search" not in solvers.AffineNewtonOptions().summary()
    with pytest.raises(ValueError, match="line_search"):
        solvers.AffineNewtonOptions(line_search="unknown")


def test_basic_newton_rejects_invalid_full_trial_without_silent_backtracking():
    """Full Newton is not permission to accept an invalid constitutive state."""
    domain = mesh.create_unit_interval(MPI.COMM_SELF, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(space)
    u.x.array[:] = .1
    test = ufl.TestFunction(space)
    residual = (u**3 - 1.) * test * ufl.dx
    n = len(u.x.array)
    reduction = AffineReduction(np.arange(n + 1), np.arange(n), np.ones(n),
                                np.zeros(n), np.arange(n))
    transformation = reduction.matrix(MPI.COMM_SELF)

    class LimitedTrial:
        def __init__(self):
            self.samples = []

        def refresh_trial(self, **kwargs):
            value = float(u.x.array.max())
            self.samples.append(value)
            if value > 2.:
                raise ValueError("outside the declared material domain")

    transaction = LimitedTrial()
    try:
        response = solve_serial_affine_equilibrium(
            fem.form(residual), fem.form(ufl.derivative(residual, u)), u,
            reduction, transformation, u.x.array.copy(),
            options=solvers.newton(line_search="basic").for_affine_reduction(),
            operators=(), transaction=transaction, start_factor=0., target_factor=1.,
            reporter=None, step_name="invalid_basic", step_number=1,
            increment_number=1, attempt_number=1,
        )
        assert not response.converged and response.step_lengths == ()
        assert "outside the declared material domain" in response.message
        np.testing.assert_allclose(transaction.samples, [.1, 33.4, .1], atol=1e-10)
        np.testing.assert_array_equal(u.x.array, np.full(n, .1))
    finally:
        transformation.destroy()


def test_periodic_cubic_basic_and_backtracking_solve_same_reduced_equilibrium():
    """Same real periodic problem exercises serial or two-rank MPC assembly."""
    from agentfem.problems import AffineNonlinearVariationalProblem

    solved = []
    for mode in ("basic", "backtracking"):
        fixture = periodic_unit_cube(MPI.COMM_WORLD, stretch=1.)
        space = fem.functionspace(fixture.domain, ("Lagrange", 1, (3,)))
        u = fem.Function(space)
        test = ufl.TestFunction(space)
        residual = sum(((u[i] + .1)**3 - 1.) * test[i] for i in range(3)) * ufl.dx
        step = AffineNonlinearVariationalProblem(
            residual, ufl.derivative(residual, u), u, fixture.constraint(u),
            load_factors=(1.,), progress=False,
            solver_options=solvers.newton(line_search=mode, maximum_iterations=40,
                                         relative_tolerance=1e-10, absolute_tolerance=1e-11),
        )
        step.solve()
        record = step.accepted_increments[-1]
        assert record.converged and record.equation_mismatch < 1e-12
        first = next(event for event in step.execution_events if event.kind == "iteration")
        if mode == "basic":
            assert first.residual_norm > record.initial_residual_norm
            assert all(alpha == 1. for alpha in record.accepted_step_lengths)
        else:
            assert record.accepted_step_lengths[0] < 1.
        solved.append(u.x.array.copy())
    np.testing.assert_allclose(solved[0], solved[1], atol=1e-9, rtol=1e-9)


def test_corrected_affine_full_step_policy_is_explicit_in_checkpoint_identity(tmp_path):
    from dataclasses import replace
    import json
    from test_affine_j2_automatic_restart import _automatic_periodic_j2_step

    step = _automatic_periodic_j2_step()
    original = step._checkpoint_identity()
    assert "affine_full_step_policy" not in original
    step.solver_options = replace(step.solver_options, line_search="basic")
    assert step._checkpoint_identity()["affine_full_step_policy"] == "finite_valid_trial_v1"
    step.solve(until=.25)
    checkpoint = step.save_checkpoint(tmp_path / "basic")
    resumed = _automatic_periodic_j2_step()
    resumed.solver_options = replace(resumed.solver_options, line_search="basic")
    resumed.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(resumed.solution.x.array, step.solution.x.array)
    record = json.loads(checkpoint.read_text())
    # A legacy checkpoint using the old Armijo-restricted basic behavior must
    # not silently acquire the corrected execution policy.
    assert "affine_full_step_policy" in record["identity"]
    del record["identity"]["affine_full_step_policy"]
    checkpoint.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        resumed.load_checkpoint(checkpoint)


@pytest.mark.parametrize("mode", ["basic", None])
def test_affine_augmented_full_step_matches_backtracking_with_joint_restart(tmp_path, mode):
    from dataclasses import replace
    from test_affine_normal_augmentation_research import _research_path

    reference = _research_path(moving=True, acceleration="aitken")
    reference.solve()

    def create():
        step = _research_path(moving=True, acceleration="aitken")
        step.solver_options = replace(step.solver_options, line_search=mode)
        return step

    partial = create()
    partial.solve(until=.5)
    saved = partial.save_checkpoint(tmp_path / "full_step_contact")
    resumed = create()
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, reference.solution.x.array, atol=3e-8, rtol=1e-6)
    np.testing.assert_allclose(resumed.response.state.committed_state_vectors(),
                               reference.response.state.committed_state_vectors(), atol=2e-7, rtol=1e-5)
    np.testing.assert_allclose(resumed.contact_provider.normal_response().pressure,
                               reference.contact_provider.normal_response().pressure, atol=1e-5, rtol=2e-5)
    assert all(alpha == 1. for i in resumed.accepted_increments for alpha in i.accepted_step_lengths)
    assert resumed.operator_dual_history.work("interior_plane") == pytest.approx(
        reference.operator_dual_history.work("interior_plane"), abs=1e-8,
    )
