# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Outer solver reuse changes allocation ownership, not accepted physics."""

from contextlib import nullcontext

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import solvers
from agentfem._affine_equilibrium import _AffineLinearWorkspace
from agentfem.mechanics._affine_normal_augmentation import AffinePlaneNormalAugmentation
from test_retained_material_trial import make_step


def test_workspace_rejects_changed_reduction_options_and_closed_owner():
    matrix = PETSc.Mat().createAIJ([2, 2], nnz=1, comm=MPI.COMM_SELF)
    matrix.setValues([0], [0], [[1.]])
    matrix.setValues([1], [1], [[1.]])
    matrix.assemble()
    other = matrix.copy()
    options = solvers.direct_solver()
    try:
        with _AffineLinearWorkspace(matrix, options) as workspace:
            workspace.validate(matrix, options)
            with pytest.raises(ValueError, match="unchanged"):
                workspace.validate(other, options)
            with pytest.raises(ValueError, match="unchanged"):
                workspace.validate(matrix, object())
            matrix.scale(2.)
            with pytest.raises(ValueError, match="unchanged"):
                workspace.validate(matrix, options)
        with pytest.raises(RuntimeError, match="closed"):
            workspace.validate(matrix, options)
        workspace.close()  # idempotent cleanup
    finally:
        other.destroy()
        matrix.destroy()


def test_outer_workspace_matches_fresh_equilibria_and_restart(monkeypatch, tmp_path):
    reference, candidate = make_step(), make_step()
    original_create = solvers.create_ksp
    created = []

    def record(*args, **kwargs):
        solver = original_create(*args, **kwargs)
        created.append(solver)
        return solver

    with monkeypatch.context() as patch:
        patch.setattr(solvers, "create_ksp", record)
        patch.setattr(AffinePlaneNormalAugmentation, "_linear_workspace_scope",
                      lambda *args: nullcontext(None))
        reference.solve()
    reference_count = len(created)
    assert all(not solver for solver in created)
    created.clear()
    with monkeypatch.context() as patch:
        patch.setattr(solvers, "create_ksp", record)
        candidate.solve(until=.5)
        saved = candidate.save_checkpoint(tmp_path / "workspace")
        resumed = make_step()
        resumed.load_checkpoint(saved)
        resumed.solve()
    assert 0 < len(created) < reference_count
    assert all(not solver for solver in created)
    for a, b in (
        (resumed.solution.x.array, reference.solution.x.array),
        (resumed.response.state.committed_state_vectors(),
         reference.response.state.committed_state_vectors()),
        (resumed.response.first_piola_stress.values, reference.response.first_piola_stress.values),
        (resumed.response.tangent.values, reference.response.tangent.values),
        (resumed.contact_provider.normal_multiplier_state.accepted_multiplier,
         reference.contact_provider.normal_multiplier_state.accepted_multiplier),
    ):
        np.testing.assert_array_equal(a, b)
    key = "normal_contact_work:interior_plane"
    assert resumed.accepted_history_recorders[key].checkpoint_state() == (
        reference.accepted_history_recorders[key].checkpoint_state()
    )
    assert [x.iterations for x in resumed.accepted_increments] == [
        x.iterations for x in reference.accepted_increments
    ]


def test_outer_workspace_releases_objects_and_rolls_back_on_exception(monkeypatch):
    step = make_step()
    step.solve(until=.25)
    before = step.solution.x.array.copy()
    state = step.response.state.committed_state_vectors().copy()
    workspaces = []
    original_scope = AffinePlaneNormalAugmentation._linear_workspace_scope
    original_solve = solvers.solve_matrix_system
    allocated = []

    def scope(*args):
        workspace = original_scope(*args)
        workspaces.append(workspace)
        return workspace

    def fail(*args, **kwargs):
        allocated.append((workspaces[-1].linear_solver, workspaces[-1].reduced_tangent))
        original_solve(*args, **kwargs)
        raise RuntimeError("injected shared workspace failure")

    monkeypatch.setattr(AffinePlaneNormalAugmentation, "_linear_workspace_scope", scope)
    monkeypatch.setattr(solvers, "solve_matrix_system", fail)
    with pytest.raises(RuntimeError, match="injected shared workspace failure"):
        step.solve(until=.5)
    assert allocated and all(not obj for pair in allocated for obj in pair)
    assert all(w.closed and w.linear_solver is None and w.reduced_tangent is None
               for w in workspaces)
    np.testing.assert_array_equal(step.solution.x.array, before)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), state)
    assert step.accepted_load_factor == .25
