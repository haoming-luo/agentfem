# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in current-operator Krylov corrections share the ordinary State owner."""
from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, models, solvers, steps, studies
from agentfem._adaptive_linear_correction import _AdaptiveLinearCorrection
from test_adaptive_linear_correction import system
from test_finite_strain_j2_periodic import _periodic_j2_contact_step
from test_finite_strain_j2_standard import _standard_patch
from periodic_cube_fixture import periodic_unit_cube


def policy():
    return solvers.LinearSolverOptions(
        ksp_type="gmres", pc_type="lu", factor_solver_type="mumps",
        rtol=1e-12, atol=1e-14, max_it=50, preconditioner_reuse="adaptive",
    )


def make(selected=None):
    factory = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                / "examples/finite_j2_periodic_augmented_plane.py"))["make_step"]
    return factory(cells=4, increments=8, penalty=40000., affine_end_face_support=True,
                   linear_solver=selected)


def test_policy_transport_is_explicit_and_default_summary_unchanged():
    assert "preconditioner_reuse" not in solvers.direct_solver().summary()
    selected = policy()
    assert selected.summary()["preconditioner_reuse"] == "adaptive"
    newton = solvers.newton(linear_solver=selected)
    assert newton.for_affine_reduction().linear_options == selected
    with pytest.raises(NotImplementedError, match="AFM-LINEAR-REUSE"):
        newton.for_snes()
    with pytest.raises(NotImplementedError, match="AFM-LINEAR-REUSE"):
        selected.petsc_options()


@pytest.mark.parametrize("changes", [
    {"preconditioner_reuse": "automatic"}, {"preconditioner_reuse": True},
    {"ksp_type": "preonly"}, {"pc_type": "jacobi", "factor_solver_type": None},
    {"rtol": None}, {"atol": None}, {"max_it": None},
])
def test_invalid_reuse_policy_rejected(changes):
    with pytest.raises(ValueError):
        replace(policy(), **changes)


def test_raw_ksp_cannot_discard_managed_policy_or_change_solution():
    with system(np.eye(2)) as (matrix, rhs, solution):
        with pytest.raises(NotImplementedError, match="AFM-LINEAR-REUSE"):
            solvers.create_ksp(matrix.comm, policy())
        unmanaged = solvers.create_ksp(matrix.comm)
        try:
            solution.set(7.)
            with pytest.raises(NotImplementedError, match="AFM-LINEAR-REUSE"):
                solvers.solve_matrix_system(matrix, rhs, solution, policy(), ksp=unmanaged)
            np.testing.assert_array_equal(solution.array, [7., 7.])
        finally:
            unmanaged.destroy()


def test_strong_j2_rejects_unsupported_policy_at_step_lowering():
    with pytest.raises(NotImplementedError, match="AFM-LINEAR-REUSE"):
        _standard_patch(incrementation=steps.fixed(4),
                        solver_options=solvers.newton(linear_solver=policy()))


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="two MPI ranks required")
def test_distributed_reuse_rejected_before_mutation():
    fixture = periodic_unit_cube(MPI.COMM_WORLD)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True),
                          mesh=fixture.domain)
    u = model.field(fields.displacement(fixture.domain))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.))
    constraint = model.constraint(fixture.constraint(u))
    options = dict(material=material, constraints=constraint,
                   solver_options=solvers.newton(linear_solver=policy()))
    report = models.step_capability(model, target=u, options=options)
    assert not report["supported"]
    assert any(row["message"].startswith("AFM-LINEAR-REUSE")
               for row in report["rejection_reasons"])
    before = u.value.x.array.copy()
    with pytest.raises(NotImplementedError, match="AFM-LINEAR-REUSE"):
        model.step(target=u, **options)
    np.testing.assert_array_equal(u.value.x.array, before)
    assert not model.steps


def test_affine_multiple_contact_operators_use_adaptive_owner_without_augmentation():
    _, _, reference = _periodic_j2_contact_step(pair_count=2)
    _, _, candidate = _periodic_j2_contact_step(pair_count=2, linear_solver=policy())
    reference.solve()
    candidate.solve()
    np.testing.assert_allclose(candidate.solution.x.array, reference.solution.x.array,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(candidate.response.state.committed_state_vectors(),
                               reference.response.state.committed_state_vectors(),
                               rtol=1e-7, atol=1e-9)
    assert all(row.checks["maximum_true_linear_residual_ratio"] <= 1.
               for row in candidate.accepted_increments)


def test_serial_mixed_pressure_route_retains_managed_linear_policy():
    from test_finite_strain_j2_mixed import _mixed_cube_step
    reference, *_ = _mixed_cube_step()
    candidate, model, unknown, _, material, constraint = _mixed_cube_step(linear_solver=policy())
    assert models.step_capability(model, target=unknown, options={
        "material": material, "constraints": constraint,
        "solver_options": solvers.newton(linear_solver=policy()),
    })["supported"]
    reference.solve()
    candidate.solve()
    np.testing.assert_allclose(candidate.solution.x.array, reference.solution.x.array, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(candidate.response.state.committed_state_vectors(),
                               reference.response.state.committed_state_vectors(), rtol=1e-7, atol=1e-9)
    assert all(row.checks["maximum_true_linear_residual_ratio"] <= 1.
               for row in candidate.accepted_increments)


def test_public_cyclic_contact_and_restart_match_direct_solver(tmp_path):
    reference = make()
    reference.solve()
    partial = make(policy())
    partial.solve(until=.5)
    saved = partial.save_checkpoint(tmp_path / "adaptive")
    resumed = make(policy())
    resumed.load_checkpoint(saved)
    resumed.solve()
    for a, b in (
        (resumed.solution.x.array, reference.solution.x.array),
        (resumed.response.state.committed_state_vectors(),
         reference.response.state.committed_state_vectors()),
        (resumed.response.first_piola_stress.values, reference.response.first_piola_stress.values),
        (resumed.response.tangent.values, reference.response.tangent.values),
        (resumed.contact_provider.normal_multiplier_state.accepted_multiplier,
         reference.contact_provider.normal_multiplier_state.accepted_multiplier),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)
    assert len(resumed.accepted_increments) == len(reference.accepted_increments) == 8
    checks = [record.checks for record in resumed.accepted_increments]
    assert sum(row["linear_preconditioner_reuses"] for row in checks) > 0
    assert all(row["maximum_true_linear_residual_ratio"] <= 1. for row in checks)
    for pair in zip(resumed.accepted_history_recorders["normal_contact_work:interior_plane"].records,
                    reference.accepted_history_recorders["normal_contact_work:interior_plane"].records,
                    strict=True):
        assert pair[0]["work"] == pytest.approx(pair[1]["work"], rel=1e-7, abs=1e-10)


def test_adaptive_failure_releases_factor_and_restores_accepted_state(monkeypatch):
    step = make(policy())
    step.solve(until=.25)
    before = step.solution.x.array.copy()
    state = step.response.state.committed_state_vectors().copy()
    original = _AdaptiveLinearCorrection.solve
    owners = []

    def fail(owner, *args):
        owners.append(owner)
        original(owner, *args)
        raise RuntimeError("injected adaptive correction failure")

    monkeypatch.setattr(_AdaptiveLinearCorrection, "solve", fail)
    with pytest.raises(RuntimeError, match="injected adaptive correction failure"):
        step.solve(until=.5)
    assert owners and all(owner.closed and owner.ksp is None for owner in owners)
    np.testing.assert_array_equal(step.solution.x.array, before)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), state)
    assert step.accepted_load_factor == .25


def test_failed_fresh_correction_cutback_creates_a_new_factor_owner(monkeypatch):
    step = make(policy())
    step.incrementation = steps.automatic(initial=.25, minimum=.01, maximum=.25)
    original = _AdaptiveLinearCorrection.solve
    owners = []
    rejected = False

    def fail_first(owner, *args):
        nonlocal rejected
        if not owners or owner is not owners[-1]:
            owners.append(owner)
        info = original(owner, *args)
        if not rejected:
            rejected = True
            return replace(info, converged_reason=-3)
        return info

    monkeypatch.setattr(_AdaptiveLinearCorrection, "solve", fail_first)
    step.solve(until=.25)
    assert rejected and not step.attempted_increments[0].converged
    assert step.accepted_load_factor == .25
    assert len(owners) >= 2 and owners[0] is not owners[1]
    assert all(owner.closed and owner.ksp is None for owner in owners)
    records = step.accepted_history_recorders["normal_contact_work:interior_plane"].records
    assert len(records) == len(step.accepted_increments) + 1
    assert all(row.checks["maximum_true_linear_residual_ratio"] <= 1.
               for row in step.accepted_increments)
