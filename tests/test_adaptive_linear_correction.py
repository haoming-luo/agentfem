# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A previous preconditioner never substitutes for the current linear system."""
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import solvers
from agentfem._adaptive_linear_correction import _AdaptiveLinearCorrection


def policy(**changes):
    return replace(solvers.LinearSolverOptions(ksp_type="gmres", pc_type="lu",
                   rtol=1e-11, atol=1e-13, max_it=40), **changes)


@contextmanager
def system(values):
    values = np.asarray(values, dtype=float)
    matrix = PETSc.Mat().createAIJ(values.shape, nnz=len(values), comm=MPI.COMM_SELF)
    matrix.setValues(range(len(values)), range(len(values)), values)
    matrix.assemble()
    rhs, solution = matrix.createVecLeft(), matrix.createVecRight()
    rhs.array[:] = np.arange(len(values)) + 1.
    try:
        yield matrix, rhs, solution
    finally:
        solution.destroy()
        rhs.destroy()
        matrix.destroy()


def test_current_matrix_not_old_factor_and_changed_layout():
    with _AdaptiveLinearCorrection(policy()) as owner:
        with system([[3., 1.], [0., 2.]]) as (matrix, rhs, x):
            assert owner.solve(matrix, rhs, x).converged
            assert not owner.last_evidence.reused
            matrix.setValue(0, 0, 7.)
            matrix.setValue(1, 1, 4.)
            matrix.assemble()
            assert owner.solve(matrix, rhs, x).converged
            assert owner.last_evidence.reused
            np.testing.assert_allclose(x.array, [1/14, .5], atol=1e-12)
            assert owner.last_evidence.true_residual <= owner.last_evidence.threshold
            first = owner.ksp
        with system(np.diag([2., 3., 4.])) as (matrix, rhs, x):
            assert owner.solve(matrix, rhs, x).converged
            assert not owner.last_evidence.reused
            assert not first
            np.testing.assert_allclose(x.array, [1/2, 2/3, 3/4], atol=1e-12)
        last = owner.ksp
    assert not last
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.__enter__()


def test_failed_old_preconditioner_rebuilds_once():
    with system(np.eye(3)) as (matrix, rhs, x), _AdaptiveLinearCorrection(policy(max_it=1)) as owner:
        assert owner.solve(matrix, rhs, x).converged
        matrix.setValues(range(3), range(3), np.diag([2., 3., 4.]))
        matrix.assemble()
        assert owner.solve(matrix, rhs, x).converged
        assert owner.last_evidence.reused and owner.last_evidence.rebuilt
        assert owner.last_evidence.total_iterations == 2
        np.testing.assert_allclose(x.array, [1/2, 2/3, 3/4], atol=1e-12)


def test_expensive_iteration_rebuilds_next_preconditioner():
    with system(np.eye(3)) as (matrix, rhs, x), _AdaptiveLinearCorrection(
            policy(), reuse_iteration_limit=1) as owner:
        assert owner.solve(matrix, rhs, x).converged
        matrix.setValues(range(3), range(3), np.diag([2., 3., 4.]))
        matrix.assemble()
        assert owner.solve(matrix, rhs, x).converged
        assert owner.last_evidence.reused and not owner.last_evidence.rebuilt
        assert owner.last_evidence.total_iterations > 1
        assert not owner.can_reuse
        assert owner.solve(matrix, rhs, x).converged
        assert not owner.last_evidence.reused


def test_nonfinite_rhs_is_rejected_before_solve(monkeypatch):
    with system(np.eye(2)) as (matrix, rhs, x), _AdaptiveLinearCorrection(policy()) as owner:
        owner.solve(matrix, rhs, x)
        rhs.set(np.nan)
        monkeypatch.setattr(solvers, "solve_matrix_system", lambda *a, **kw: pytest.fail("nonfinite solve"))
        with pytest.raises(ValueError, match="AFM-LINEAR-NONFINITE"):
            owner.solve(matrix, rhs, x)
        assert not owner.can_reuse and owner.last_evidence is None


def test_fresh_failure_is_reported_without_unbounded_retries(monkeypatch):
    with system(np.eye(2)) as (matrix, rhs, x), _AdaptiveLinearCorrection(policy()) as owner:
        calls = []
        def failed(*args, **kwargs):
            calls.append(1)
            return solvers.LinearSolveInfo(converged_reason=-3, iterations=40,
                                           residual_norm=float(rhs.norm()))
        monkeypatch.setattr(solvers, "solve_matrix_system", failed)
        assert not owner.solve(matrix, rhs, x).converged
        assert len(calls) == 1
        assert not owner.can_reuse
        assert not owner.last_evidence.rebuilt
        assert owner.last_evidence.true_residual > owner.last_evidence.threshold


def test_old_factor_backend_error_gets_one_fresh_attempt(monkeypatch):
    with system(np.eye(2)) as (matrix, rhs, x), _AdaptiveLinearCorrection(policy()) as owner:
        assert owner.solve(matrix, rhs, x).converged
        solve = solvers.solve_matrix_system
        calls = []
        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise PETSc.Error(91)
            return solve(*args, **kwargs)
        monkeypatch.setattr(solvers, "solve_matrix_system", fail_once)
        matrix.scale(2.)
        assert owner.solve(matrix, rhs, x).converged
        assert len(calls) == 2 and owner.last_evidence.rebuilt
        np.testing.assert_allclose(x.array, [.5, 1.], atol=1e-12)


def test_zero_rhs_does_not_poison_a_later_nonzero_solve():
    with system(np.eye(2)) as (matrix, rhs, x), _AdaptiveLinearCorrection(policy()) as owner:
        rhs.set(0.)
        assert owner.solve(matrix, rhs, x).converged
        np.testing.assert_array_equal(x.array, 0.)
        rhs.set(3.)
        matrix.scale(2.)
        assert owner.solve(matrix, rhs, x).converged
        np.testing.assert_allclose(x.array, 1.5, atol=1e-12)


@pytest.mark.parametrize("fault", ["false_convergence", "nan"])
def test_false_convergence_is_not_accepted_or_reused(monkeypatch, fault):
    with system(np.eye(3)) as (matrix, rhs, x), _AdaptiveLinearCorrection(policy()) as owner:
        assert owner.solve(matrix, rhs, x).converged
        calls = []
        def incorrect(*args, **kwargs):
            calls.append(1)
            x.set(np.nan if fault == "nan" else 0.)
            return solvers.LinearSolveInfo(converged_reason=2, iterations=1, residual_norm=0.)
        monkeypatch.setattr(solvers, "solve_matrix_system", incorrect)
        info = owner.solve(matrix, rhs, x)
        assert not info.converged
        assert owner.last_evidence.rebuilt
        assert len(calls) == 2
        assert not owner.can_reuse


def test_exception_drops_lease_and_context_releases_ksp(monkeypatch):
    owner = _AdaptiveLinearCorrection(policy())
    with system(np.eye(3)) as (matrix, rhs, x):
        with pytest.raises(RuntimeError, match="injected"):
            with owner:
                owner.solve(matrix, rhs, x)
                allocated = owner.ksp
                def failed(*args, **kwargs):
                    raise RuntimeError("injected")
                monkeypatch.setattr(solvers, "solve_matrix_system", failed)
                owner.solve(matrix, rhs, x)
        assert not allocated and not owner.can_reuse and owner.ksp is None


@pytest.mark.parametrize("changes", [{"ksp_type":"preonly"}, {"ksp_type":"cg"},
                                   {"pc_type":"gamg"}, {"rtol":None}, {"atol":None}])
def test_unsupported_policies_reject(changes):
    with pytest.raises(ValueError):
        _AdaptiveLinearCorrection(policy(**changes))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_budget(limit):
    with pytest.raises(ValueError):
        _AdaptiveLinearCorrection(policy(), reuse_iteration_limit=limit)
