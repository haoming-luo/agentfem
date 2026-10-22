# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Retained material response is exact, one-shot, local and failure-safe."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest

from agentfem import solvers


def make_step():
    make = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "examples/finite_j2_periodic_augmented_plane.py"
    ))["make_step"]
    return make(cells=4, increments=4, affine_end_face_support=True)


@pytest.fixture
def transaction():
    step = make_step()
    transaction = step.state_transaction
    transaction.initialize()
    transaction.solution.x.array[:] = .0001 * np.arange(len(transaction.solution.x.array))
    transaction.solution.x.scatter_forward()
    return transaction


def refresh(transaction, *, target=.25):
    return transaction.refresh_trial(start_factor=0., target_factor=target)


def reuse(transaction, *, target=.25):
    return transaction._reuse_trial_response(start_factor=0., target_factor=target)


def test_retained_response_is_one_shot_and_scope_local(transaction):
    refresh(transaction)
    assert not reuse(transaction)
    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        generation = transaction._response_linearization_generation
        assert reuse(transaction)
        assert transaction._response_linearization_generation == generation
        assert not reuse(transaction)
        refresh(transaction)
    assert not transaction._retain_trial_response
    assert transaction._retained_trial_digest is None
    assert not reuse(transaction)


@pytest.mark.parametrize("change", [
    "solution", "accepted_solution", "committed", "trial", "stress", "tangent",
    "energy", "component", "gradient", "mises", "geometry", "material", "time", "diagnostic",
])
def test_retained_response_rejects_changed_inputs_and_writable_outputs(transaction, change):
    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        response = transaction.response
        arrays = {
            "solution": transaction.solution.x.array,
            "accepted_solution": transaction.accepted_solution.x.array,
            "committed": response.state.committed["equivalent_plastic_strain"].function.x.array,
            "trial": response.state.trial["equivalent_plastic_strain"].function.x.array,
            "stress": response.first_piola_stress.function.x.array,
            "tangent": response.tangent.function.x.array,
            "energy": response.strain_energy_density.function.x.array,
            "component": response.stored_energy_density_components["ELENER"].function.x.array,
            "gradient": transaction.deformation_gradient.function.x.array,
            "mises": transaction.equivalent_stress.function.x.array,
            "geometry": response.domain.geometry.x.reshape(-1),
        }
        if change in arrays:
            arrays[change][0] += .001
        elif change == "material":
            transaction.material = replace(transaction.material, young=1100.)
        elif change == "diagnostic":
            transaction.last_plastic_points += 1
        assert not reuse(transaction, target=.3 if change == "time" else .25)
        assert transaction._retained_trial_digest is None


def test_changed_live_gradient_is_not_hidden_by_unchanged_displacement(transaction, monkeypatch):
    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        old, new = transaction._evaluate_gradients()
        new = new.copy()
        new[0, 0, 0] += .01
        monkeypatch.setattr(transaction, "_evaluate_gradients", lambda: (old, new))
        assert not reuse(transaction)


def test_replaced_update_is_not_bypassed_by_retention(transaction, monkeypatch):
    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        def fail(**kwargs):
            raise RuntimeError("injected replacement update")
        monkeypatch.setattr(transaction, "refresh_trial", fail)
        assert not reuse(transaction)
        valid, message = solvers._try_refresh_affine_trial_state(
            transaction, start_factor=0., target_factor=.25,
        )
        assert not valid
        assert "injected replacement update" in message


def test_same_update_function_bound_to_another_owner_invalidates_lease(transaction, monkeypatch):
    from copy import copy
    from types import MethodType

    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        # A bound method includes its owner, not just its function body.
        other = copy(transaction)
        replacement = MethodType(transaction.refresh_trial.__func__, other)
        monkeypatch.setattr(transaction, "refresh_trial", replacement)
        assert not reuse(transaction)


@pytest.mark.parametrize("action", ["commit", "rollback", "resume", "restore", "exception"])
def test_lifecycle_does_not_retain_a_trial_lease(transaction, action):
    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        if action == "commit":
            transaction.commit_increment(start_factor=0., target_factor=.25)
        elif action == "rollback":
            transaction.rollback_increment(accepted_factor=0.)
        elif action == "resume":
            transaction.prepare_resume()
        elif action == "restore":
            transaction.restore_runtime_state(transaction.snapshot_runtime_state())
        else:
            transaction.solution.x.array[:] = np.nan
            with pytest.raises(Exception):
                refresh(transaction)
        assert transaction._retained_trial_digest is None
        assert not reuse(transaction)


def test_unsupported_mixed_route_does_not_enable_reuse(transaction):
    transaction.pressure_evaluator = object()
    with transaction._trial_response_reuse_scope():
        assert not transaction._retain_trial_response
        assert not reuse(transaction)


def test_outer_equilibrium_reuse_matches_recomputation_and_restart(tmp_path, monkeypatch):
    baseline = make_step()
    candidate = make_step()
    monkeypatch.setattr(baseline.state_transaction, "_trial_response_reuse_scope", nullcontext)
    baseline.solve()
    candidate.solve(until=.5)
    path = candidate.save_checkpoint(tmp_path / "retained-material")
    resumed = make_step()
    resumed.load_checkpoint(path)
    resumed.solve()
    np.testing.assert_array_equal(resumed.solution.x.array, baseline.solution.x.array)
    np.testing.assert_array_equal(resumed.response.state.committed_state_vectors(),
                                  baseline.response.state.committed_state_vectors())
    np.testing.assert_array_equal(resumed.contact_provider.normal_multiplier_state.accepted_multiplier,
                                  baseline.contact_provider.normal_multiplier_state.accepted_multiplier)
    key = "normal_contact_work:interior_plane"
    assert resumed.accepted_history_recorders[key].checkpoint_state() == (
        baseline.accepted_history_recorders[key].checkpoint_state()
    )
    assert sum(x.checks["material_response_reuses"] for x in resumed.accepted_increments) > 0
    assert all(x.checks["material_response_reuses"] == 0 for x in baseline.accepted_increments)
    assert [x.iterations for x in resumed.accepted_increments] == [
        x.iterations for x in baseline.accepted_increments
    ]
    assert resumed.state_transaction._retained_trial_digest is None


def test_failed_fingerprint_falls_back_to_regular_refresh(transaction, monkeypatch):
    from agentfem.mechanics import _retained_trial_response

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable optional reuse evidence")

    with transaction._trial_response_reuse_scope():
        refresh(transaction)
        monkeypatch.setattr(_retained_trial_response, "retained_trial_digest", fail)
        assert not reuse(transaction)
        assert solvers._try_refresh_affine_trial_state(
            transaction, start_factor=0., target_factor=.25,
        )[0]
        assert transaction._retained_trial_digest is None
