from __future__ import annotations

import json

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    checkpointing,
    constitutive,
    fields,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.step_providers import step_capability

from periodic_cube_fixture import periodic_unit_cube


def _integrate_material_path(material, deformation_gradient, increments):
    state = material.state_schema.initial_state()
    old_gradient = np.eye(3)
    response = None
    for index in range(1, increments + 1):
        factor = index / increments
        new_gradient = np.eye(3) + factor * (
            deformation_gradient - np.eye(3)
        )
        response = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=new_gradient,
                time=factor,
                time_increment=1.0 / increments,
                properties=[],
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        state = response.state_new.copy()
        old_gradient = new_gradient
    return response


def _periodic_j2_step(
    *,
    stretch: float = 1.02,
    tangent_relative_step: float = 2.0e-6,
    relative_tolerance: float = 1.0e-8,
    checkpoint=None,
):
    fixture = periodic_unit_cube(MPI.COMM_SELF, stretch=stretch)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_periodic_restart",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
            tangent_relative_step=tangent_relative_step,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=relative_tolerance,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        checkpoint=checkpoint,
        progress=False,
    )
    return step


def test_public_finite_strain_j2_periodic_cube_matches_material_point(tmp_path):
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_periodic_cube",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    periodicity = model.constraint(fixture.constraint(displacement))
    reduction = periodicity.reduction()
    assert fixture.equations.summary()["equation_count"] == 48
    assert reduction.full_size == 81
    assert reduction.reduced_size == 21
    output = results.output_plan(
        tmp_path,
        field=results.field_output(
            "U",
            every=2,
            configuration="reference",
            backend="xdmf",
        ),
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="finite_strain_j2_periodic",
    )

    capability = step_capability(
        model,
        target=displacement,
        options={
            "material": material_record,
            "constraints": periodicity,
            "checkpoint": checkpointing.every(
                2,
                directory=tmp_path / "checkpoint_contract",
            ),
        },
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_affine_static"
    assert "checkpoint" in capability["provider"]["options"]["accepted"]

    step = model.step(
        target=displacement,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        output=output,
        progress=False,
    )
    result = step.solve_result()
    expected = _integrate_material_path(material, fixture.deformation_gradient, 4)
    expected_first_piola = (
        np.linalg.det(fixture.deformation_gradient)
        * expected.cauchy_stress
        @ np.linalg.inv(fixture.deformation_gradient).T
    )

    assert step.last_solve_info.converged
    assert len(step.last_solve_info.increments) == 4
    assert periodicity.mismatch() < 1.0e-10
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        step.response.first_piola_stress.owned_values,
        np.broadcast_to(
            expected_first_piola,
            step.response.first_piola_stress.owned_values.shape,
        ),
        rtol=5.0e-6,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        np.broadcast_to(
            expected.state_new,
            step.response.state.committed_state_vectors().shape,
        ),
        rtol=5.0e-6,
        atol=5.0e-8,
    )
    peeq = step.response.state.committed_state_vectors()[:, -1]
    assert np.ptp(peeq) < 1.0e-9
    assert np.mean(peeq) > 0.0

    assert {
        "Displacement",
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "FP",
        "PEEQ",
    } <= set(result.fields)
    assert result.fields["PEEQ"].location == "quadrature_points"
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(
        float(expected.state_new[-1]),
        rel=5.0e-6,
    )
    np.testing.assert_allclose(
        result.histories["homogenized_first_piola_stress"].values[-1],
        expected_first_piola,
        rtol=5.0e-6,
        atol=5.0e-6,
    )
    assert result.quantity("maximum_hill_mandel_relative_error") < 1.0e-8
    assert result.metadata["output_plan"]["status"] == "completed"


def test_public_finite_strain_j2_periodic_checkpoint_restart_is_equivalent(
    tmp_path,
):
    reference = _periodic_j2_step()
    reference.solve()
    expected_solution = reference.solution.x.array.copy()
    expected_state = reference.response.state.committed_state_vectors().copy()
    expected_first_piola = reference.response.first_piola_stress.values.copy()

    partial = _periodic_j2_step()
    partial.solve(until=0.5)
    assert partial.state_transaction.accepted_factor == pytest.approx(0.5)
    checkpoint = partial.save_checkpoint(
        tmp_path / "finite_strain_j2_periodic_restart"
    )
    assert checkpoint.is_file()

    restarted = _periodic_j2_step()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    np.testing.assert_allclose(
        restarted.solution.x.array,
        expected_solution,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        expected_state,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.first_piola_stress.values,
        expected_first_piola,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    assert [
        item.load_factor for item in restarted.last_solve_info.increments
    ] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert restarted.state_transaction.accepted_factor == pytest.approx(1.0)


def test_affine_checkpoint_refuses_mutated_constraint_identity(tmp_path):
    partial = _periodic_j2_step()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "constraint_identity")

    mutated = _periodic_j2_step(stretch=1.021)
    original_solution = mutated.solution.x.array.copy()
    original_state = mutated.response.state.committed_state_vectors().copy()
    with pytest.raises(ValueError, match="constraint equations differ"):
        mutated.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(mutated.solution.x.array, original_solution)
    np.testing.assert_array_equal(
        mutated.response.state.committed_state_vectors(),
        original_state,
    )
    assert mutated.accepted_load_factor == pytest.approx(0.0)
    assert mutated.state_transaction.accepted_factor == pytest.approx(0.0)


@pytest.mark.parametrize(
    "changed_options",
    (
        {"tangent_relative_step": 5.0e-6},
        {"relative_tolerance": 5.0e-8},
    ),
)
def test_affine_checkpoint_identity_covers_material_and_solver_numerics(
    tmp_path,
    changed_options,
):
    partial = _periodic_j2_step()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "numerical_identity")

    changed = _periodic_j2_step(**changed_options)
    with pytest.raises(ValueError, match="differ from the current analysis"):
        changed.load_checkpoint(checkpoint)


def test_failed_affine_checkpoint_load_restores_complete_runtime_state(tmp_path):
    source = _periodic_j2_step()
    source.solve(until=0.5)
    checkpoint = source.save_checkpoint(tmp_path / "atomic_restore")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["accepted_increments"][-1]["load_factor"] = 0.49
    checkpoint.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target = _periodic_j2_step()
    target.solve(until=0.25)
    transaction = target.state_transaction
    solution = target.solution.x.array.copy()
    accepted_solution = transaction.accepted_solution.x.array.copy()
    transaction_state = transaction.snapshot_runtime_state()
    accepted_load_factor = target.accepted_load_factor
    accepted_increments = list(target.accepted_increments)
    attempted_increments = list(target.attempted_increments)
    next_increment_size = target.next_increment_size
    execution_events = list(target.execution_events)
    last_solve_info = target.last_solve_info
    snapshots = list(target.snapshots)
    checkpoints = list(target.checkpoints)

    with pytest.raises(
        ValueError,
        match="coordinate and accepted history disagree",
    ):
        target.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(target.solution.x.array, solution)
    np.testing.assert_array_equal(
        transaction.accepted_solution.x.array,
        accepted_solution,
    )
    restored = transaction.snapshot_runtime_state()
    for name in (
        "first_piola_stress",
        "cauchy_stress",
        "tangent",
        "strain_energy_density",
        "deformation_gradient",
        "equivalent_stress",
    ):
        np.testing.assert_array_equal(restored[name], transaction_state[name])
    for name in transaction_state["committed_state"]:
        np.testing.assert_array_equal(
            restored["committed_state"][name],
            transaction_state["committed_state"][name],
        )
        np.testing.assert_array_equal(
            restored["trial_state"][name],
            transaction_state["trial_state"][name],
        )
    assert restored["accepted_factor"] == transaction_state["accepted_factor"]
    assert restored["last_plastic_points"] == transaction_state[
        "last_plastic_points"
    ]
    assert restored["last_maximum_plastic_increment"] == transaction_state[
        "last_maximum_plastic_increment"
    ]
    assert target.accepted_load_factor == accepted_load_factor
    assert target.accepted_increments == accepted_increments
    assert target.attempted_increments == attempted_increments
    assert target.next_increment_size == next_increment_size
    assert target.execution_events == execution_events
    assert target.last_solve_info is last_solve_info
    assert len(target.snapshots) == len(snapshots)
    assert all(
        restored_snapshot is original_snapshot
        for restored_snapshot, original_snapshot in zip(
            target.snapshots,
            snapshots,
            strict=True,
        )
    )
    assert target.checkpoints == checkpoints


class _FailingAcceptedObserver:
    """Fault injector implementing the accepted-observer rollback protocol."""

    def __init__(self):
        self.accepted = 0
        self.fail = True

    def snapshot_runtime_state(self):
        return {"accepted": self.accepted}

    def restore_runtime_state(self, state) -> None:
        self.accepted = int(state["accepted"])

    def reset(self, _snapshot) -> None:
        self.accepted = 0

    def accept(self, _snapshot) -> None:
        self.accepted += 1
        if self.fail:
            raise RuntimeError("injected accepted-observer failure")


def test_post_commit_observer_failure_restores_complete_accepted_boundary():
    step = _periodic_j2_step()
    observer = _FailingAcceptedObserver()
    step.accepted_observers = (observer,)
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()

    with pytest.raises(RuntimeError, match="accepted-observer failure"):
        step.solve()

    np.testing.assert_array_equal(step.solution.x.array, initial_solution)
    np.testing.assert_array_equal(
        step.state_transaction.accepted_solution.x.array,
        initial_solution,
    )
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(),
        initial_state,
    )
    np.testing.assert_array_equal(
        step.response.state.trial_state_vectors(),
        initial_state,
    )
    assert observer.accepted == 0
    assert step.accepted_load_factor == pytest.approx(0.0)
    assert step.state_transaction.accepted_factor == pytest.approx(0.0)
    assert step.accepted_increments == []
    assert step.attempted_increments == []

    observer.fail = False
    step.solve()
    assert step.accepted_load_factor == pytest.approx(1.0)
    assert step.state_transaction.accepted_factor == pytest.approx(1.0)
    assert observer.accepted == 4


def test_post_commit_snapshot_failure_restores_complete_accepted_boundary():
    reference = _periodic_j2_step()
    reference.solve(until=0.75)
    step = _periodic_j2_step()
    snapshot_field_factory = step.snapshot_field_factory

    def fail_after_initial_boundary():
        if np.max(np.abs(step.solution.x.array), initial=0.0) > 1.0e-14:
            raise RuntimeError("injected accepted-snapshot failure")
        return snapshot_field_factory()

    step.snapshot_field_factory = fail_after_initial_boundary
    with pytest.raises(RuntimeError, match="accepted-snapshot failure"):
        step.solve()

    np.testing.assert_allclose(
        step.solution.x.array,
        reference.solution.x.array,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    assert step.accepted_load_factor == pytest.approx(0.75)
    assert step.state_transaction.accepted_factor == pytest.approx(0.75)
    assert [item.load_factor for item in step.accepted_increments] == pytest.approx(
        [0.25, 0.5, 0.75]
    )

    step.snapshot_field_factory = snapshot_field_factory
    step.solve()
    assert step.accepted_load_factor == pytest.approx(1.0)


def test_post_commit_checkpoint_failure_restores_complete_accepted_boundary(
    tmp_path,
):
    policy = checkpointing.every(1, directory=tmp_path / "scheduled")
    step = _periodic_j2_step(checkpoint=policy)
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()
    save_checkpoint = step.save_checkpoint

    def fail_checkpoint(*_args, **_kwargs):
        raise OSError("injected checkpoint failure")

    step.save_checkpoint = fail_checkpoint
    with pytest.raises(OSError, match="checkpoint failure"):
        step.solve()

    np.testing.assert_array_equal(step.solution.x.array, initial_solution)
    np.testing.assert_array_equal(
        step.state_transaction.accepted_solution.x.array,
        initial_solution,
    )
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(),
        initial_state,
    )
    assert step.accepted_load_factor == pytest.approx(0.0)
    assert step.state_transaction.accepted_factor == pytest.approx(0.0)
    assert step.accepted_increments == []
    assert step.attempted_increments == []
    assert step.checkpoints == []

    step.save_checkpoint = save_checkpoint
    step.solve()
    assert step.accepted_load_factor == pytest.approx(1.0)
    assert len(step.checkpoints) == 4
