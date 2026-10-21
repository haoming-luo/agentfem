# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-equilibrium lifecycle and recomputation checks for the affine path.

These checks were first run against the unchanged pre-refactor public solver.
The historical comparison now also lives in benchmark_affine_trial_reuse.py;
this regression alone must not be called an independent reference solver.
"""

from dataclasses import replace

import numpy as np
import pytest
from dolfinx import fem

from agentfem import boundary_models, solvers
from agentfem import _affine_equilibrium as candidate
from test_finite_strain_j2_periodic import _periodic_j2_contact_step


def _step(kind):
    kwargs = {}
    if kind == 'moving':
        kwargs['motion'] = boundary_models.prescribed_rigid_motion(
            translation=(-.01, .002, 0.), reference_point=(1.6, .5, .5),
        )
    if kind == 'friction':
        kwargs['surface'] = boundary_models.rigid_plane(
            point=(1.005, 0., 0.), normal=(-1., 0., 0.),
        )
        kwargs['friction_coefficient'] = .2
        kwargs['tangential_penalty'] = 1000.
    return _periodic_j2_contact_step(**kwargs)[2]


def _frozen(step, *, reuse=True, reuse_linear_solver=True, options=None):
    """Advance just the frozen inner solve; never commit the returned trial."""
    start, target = 1/3, 2/3
    constraint, function = step.constraint, step.solution
    previous = constraint.reduction(start)
    reduction = constraint.reduction(target)
    old_affine = constraint.initial_reduced_values(
        previous, constraint.deformation_gradient_at(start),
    )
    new_affine = constraint.initial_reduced_values(
        reduction, constraint.deformation_gradient_at(target),
    )
    values = function.x.array[previous.independent_full_dofs] + new_affine - old_affine
    solvers._set_affine_operator_factor(step.full_space_operators, target)
    solvers._assign_reconstructed(function, reduction, values)
    transformation = reduction.matrix(function.function_space.mesh.comm)
    try:
        return candidate.solve_serial_affine_equilibrium(
            fem.form(step.residual_form), fem.form(step.jacobian_form),
            function, reduction, transformation, values,
            options=options or step.solver_options.for_affine_reduction(),
            operators=step.full_space_operators, transaction=step.state_transaction,
            start_factor=start, target_factor=target, reporter=None,
            step_name='test_frozen_equilibrium', step_number=1,
            increment_number=2, attempt_number=1,
            reuse_accepted_assembly=reuse,
            reuse_linear_solver=reuse_linear_solver,
        )
    finally:
        transformation.destroy()


@pytest.mark.parametrize('kind', ['sphere', 'moving', 'friction'])
def test_candidate_frozen_solver_matches_existing_path_without_committing(kind):
    reference = _step(kind)
    reference.solve(until=2/3)
    expected = reference.accepted_increments[-1]
    records = []
    for reuse in (False, True):
        trial = _step(kind)
        trial.solve(until=1/3)
        committed = trial.response.state.committed_state_vectors().copy()
        accepted = list(trial.accepted_increments)
        response = _frozen(trial, reuse=reuse)
        records.append(response)
        assert response.converged
        assert response.iterations == expected.iterations
        assert response.step_lengths == expected.accepted_step_lengths
        assert response.linear_reasons == expected.linear_converged_reasons
        assert response.residual_norm == pytest.approx(expected.residual_norm, abs=1e-11)
        np.testing.assert_allclose(trial.solution.x.array, reference.solution.x.array,
                                   atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(
            trial.response.state.trial_state_vectors(),
            reference.response.state.committed_state_vectors(), atol=1e-12, rtol=1e-10,
        )
        np.testing.assert_array_equal(trial.response.state.committed_state_vectors(), committed)
        assert trial.accepted_load_factor == 1/3
        assert trial.accepted_increments == accepted
        np.testing.assert_allclose(
            trial.contact_provider.evaluate().assembly.nodal_structural_residual,
            reference.contact_provider.evaluate().assembly.nodal_structural_residual,
            atol=1e-10, rtol=1e-10,
        )
    assert records[0].reused_assemblies == 0
    assert records[1].reused_assemblies == records[1].iterations > 0
    assert records[0].residual_assemblies - records[1].residual_assemblies == records[1].iterations


@pytest.mark.parametrize('kind', ['sphere', 'moving', 'friction'])
def test_scoped_ksp_reuse_refreshes_current_tangent_and_matches_fresh(kind, monkeypatch):
    records, arrays = [], []
    original_create = solvers.create_ksp
    created = []
    def tracked_create(*args, **kwargs):
        solver = original_create(*args, **kwargs)
        created.append(solver)
        return solver
    for reuse in (False, True):
        step = _step(kind)
        step.solve(until=1/3)
        with monkeypatch.context() as scoped:
            scoped.setattr(solvers, 'create_ksp', tracked_create)
            created.clear()
            result = _frozen(step, reuse_linear_solver=reuse)
        assert result.converged and result.iterations > 1
        assert len(created) == (1 if reuse else result.iterations)
        assert all(not solver for solver in created)
        records.append(result)
        arrays.append((step.solution.x.array.copy(),
                       step.response.state.trial_state_vectors().copy()))
    assert records[0].iterations == records[1].iterations
    assert records[0].step_lengths == records[1].step_lengths
    assert records[0].linear_reasons == records[1].linear_reasons
    assert records[0].residual_norm == pytest.approx(records[1].residual_norm, abs=1e-11)
    for a, b in zip(*arrays, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('failure', ['linear', 'tangent', 'material', 'iteration_limit'])
def test_candidate_failure_releases_residuals_and_does_not_commit(monkeypatch, failure):
    step = _step('moving')
    step.solve(until=1/3)
    committed = step.response.state.committed_state_vectors().copy()
    objects = []
    original = candidate._TrialAssembly

    class TrackedAssembly(original):
        def __init__(self, *args):
            super().__init__(*args)
            objects.append(self)

    monkeypatch.setattr(candidate, '_TrialAssembly', TrackedAssembly)
    created = []
    original_create = solvers.create_ksp
    def tracked_create(*args, **kwargs):
        solver = original_create(*args, **kwargs)
        created.append(solver)
        return solver
    monkeypatch.setattr(solvers, 'create_ksp', tracked_create)
    options = step.solver_options.for_affine_reduction()
    if failure == 'linear':
        def failed_linear(*args, **kwargs):
            raise RuntimeError('injected KSP failure')
        monkeypatch.setattr(solvers, 'solve_matrix_system', failed_linear)
    elif failure == 'tangent':
        def failed_tangent(*args, **kwargs):
            raise RuntimeError('injected contact tangent failure')
        monkeypatch.setattr(step.full_space_operators[0], 'add_tangent', failed_tangent)
    elif failure == 'material':
        def failed_material(**kwargs):
            raise ValueError('injected material failure')
        monkeypatch.setattr(step.state_transaction, 'refresh_trial', failed_material)
    else:
        options = replace(options, max_it=1)
    if failure in ('linear', 'tangent'):
        with pytest.raises(RuntimeError, match='injected'):
            _frozen(step, options=options)
    else:
        assert not _frozen(step, options=options).converged
    assert all(item.residual is None and not item.contributions for item in objects)
    assert all(not solver for solver in created)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), committed)
    assert step.state_transaction.accepted_factor == 1/3


def test_public_fixed_increment_failure_preserves_diagnostic_and_accepted_state(monkeypatch):
    step = _step('moving')
    step.solve(until=1/3)
    committed = step.response.state.committed_state_vectors().copy()
    displacement = step.solution.x.array.copy()
    original = solvers.solve_serial_affine_equilibrium

    def one_iteration(*args, **kwargs):
        kwargs["options"] = replace(kwargs["options"], max_it=1)
        return original(*args, **kwargs)

    monkeypatch.setattr(solvers, "solve_serial_affine_equilibrium", one_iteration)
    with pytest.raises(RuntimeError, match=r"fixed increment failed.*threshold="):
        step.solve(until=2/3)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), committed)
    np.testing.assert_array_equal(step.solution.x.array, displacement)
    assert step.accepted_load_factor == 1/3
