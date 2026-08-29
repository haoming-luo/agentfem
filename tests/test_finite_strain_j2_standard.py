"""Public standard-boundary finite-strain J2 workflow contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    amplitudes,
    checkpointing,
    constitutive,
    constraints,
    coordinates,
    fields,
    loads,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.step_providers import step_capability


def _standard_patch(
    *,
    incrementation,
    amplitude=None,
    output=None,
    checkpoint=None,
    progress=False,
    status_file=None,
):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="public_finite_strain_j2_standard_patch",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry", tag=3
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry", tag=4
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.02)
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    step = model.step(
        target=displacement,
        material=material_record,
        incrementation=incrementation,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        amplitude=amplitude,
        output=output,
        checkpoint=checkpoint,
        progress=progress,
        status_file=status_file,
    )
    return model, displacement, material, step


def test_public_standard_j2_provider_result_output_progress_and_checkpoint(tmp_path):
    output = results.output_plan(
        tmp_path / "output",
        field=results.field_output(
            "U",
            "F",
            "S",
            "MISES",
            "SENER",
            "PEEQ",
            every=1,
            configuration="reference",
        ),
        requests=(
            results.probe_history(
                "right_ux",
                at=(1.0, 0.0, 0.0),
                component=0,
            ),
        ),
        presentation=None,
        basename="standard_j2",
    )
    loading = amplitudes.tabular(
        (0.0, 0.5, 1.0),
        (0.0, 0.25, 1.0),
        name="quadratic_like_ramp",
    )
    model, displacement, material, step = _standard_patch(
        incrementation=steps.fixed(4),
        amplitude=loading,
        output=output,
        checkpoint=checkpointing.every(
            2,
            directory=tmp_path / "checkpoints",
        ),
        progress=True,
        status_file=tmp_path / "status.log",
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_strong_static"

    result = step.solve_result()

    assert step.last_solve_info.converged
    assert step.summary()["maturity"] == "experimental_global_mpi_restart"
    assert step.summary()["evidence_level"] == "internal_serial_mpi_restart_verified"
    assert len(step.snapshots) == 5
    assert len(step.checkpoints) == 2
    assert [item.coordinate_value for item in step.checkpoints] == pytest.approx(
        [0.5, 1.0]
    )
    np.testing.assert_allclose(
        result.histories["load_amplitude"].values,
        (0.0, 0.125, 0.25, 0.625, 1.0),
    )
    np.testing.assert_allclose(
        result.histories["right_ux"].values,
        (0.0, 0.0025, 0.005, 0.0125, 0.02),
        atol=2.0e-10,
    )
    assert {
        "Displacement",
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "ELENER",
        "HARDENER",
        "FP",
        "PEEQ",
        "RF",
    } <= set(result.fields)
    assert result.fields["PEEQ"].location == "quadrature_points"
    state = step.response.state.committed_state_vectors()
    assert np.max(state[:, -1]) > 0.0
    for plastic_gradient in state[:, :-1].reshape((-1, 3, 3)):
        assert np.linalg.det(plastic_gradient) == pytest.approx(
            1.0,
            abs=5.0e-10,
        )
    event_names = tuple(
        item["kind"] for item in result.metadata["execution"]["events"]
    )
    assert event_names.count("step_started") == 1
    assert event_names.count("increment_converged") == 4
    assert event_names.count("step_completed") == 1
    assert (tmp_path / "output" / "standard_j2.xdmf").is_file()
    assert (tmp_path / "output" / "standard_j2.h5").is_file()
    assert result.metadata["output_plan"]["status"] == "completed"
    status = (tmp_path / "status.log").read_text(encoding="utf-8")
    assert "CONVERGED" in status
    assert "COMPLETED" in status


def test_standard_j2_real_cutback_and_manual_restart_are_equivalent(tmp_path):
    automatic = steps.automatic(
        initial=1.0,
        minimum=1.0e-4,
        maximum=1.0,
        max_increments=100,
        max_cutbacks=12,
        cutback_factor=0.5,
        maximum_inelastic_increment=0.006,
    )
    _, _, _, cutback = _standard_patch(incrementation=automatic)
    cutback.solve()
    rejected = [item for item in cutback.attempted_increments if not item.converged]
    assert rejected
    assert "maximum equivalent plastic-strain increment" in (
        rejected[0].rejection_reason or ""
    )

    _, _, _, reference = _standard_patch(incrementation=steps.fixed(4))
    reference.solve()
    expected_u = reference.solution.x.array.copy()
    expected_state = reference.response.state.committed_state_vectors().copy()
    expected_p = reference.response.first_piola_stress.values.copy()

    _, _, _, partial = _standard_patch(incrementation=steps.fixed(4))
    partial.solve(until=0.5)
    assert partial.execution_events[-1].kind == "step_paused"
    saved = partial.save_checkpoint(tmp_path / "manual_restart")
    _, _, _, restarted = _standard_patch(incrementation=steps.fixed(4))
    restarted.load_checkpoint(saved)
    restarted.solve()
    resumed_events = tuple(item.kind for item in restarted.execution_events)
    assert "step_paused" in resumed_events
    assert "step_resumed" in resumed_events
    assert resumed_events.index("step_resumed") > resumed_events.index("step_paused")
    assert resumed_events[-1] == "step_completed"

    np.testing.assert_allclose(restarted.solution.x.array, expected_u, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        expected_state,
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        restarted.response.first_piola_stress.values,
        expected_p,
        rtol=2e-8,
        atol=2e-8,
    )


def test_public_standard_j2_consumes_registered_reference_body_force():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="finite_strain_j2_body_force",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="clamp", tag=1
    )
    model.fix(displacement, on=left, value=(0.0, 0.0, 0.0))
    material = constitutive.finite_strain_j2_logarithmic(
        young=1_000.0,
        poisson=0.3,
        yield_stress=1_000.0,
        hardening_modulus=10.0,
    )
    model.material(material)
    load = model.body_force(
        (0.0, 0.0, -1.0),
        target=displacement,
        name="dead_reference_body_force",
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()

    assert step.last_solve_info.converged
    assert np.all(np.isfinite(displacement.value.x.array))
    assert np.min(displacement.value.x.array) < 0.0
    reaction = result.fields["RF"].field.x.array.reshape((-1, 3)).sum(axis=0)
    np.testing.assert_allclose(reaction, (0.0, 0.0, 1.0), atol=2.0e-8)
    assert step.summary()["external_load"] == step.load_identity
    assert step.load_identity[0]["name"] == load.name


def test_public_standard_j2_lowers_registered_material_regions():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="regional_finite_strain_j2_standard_patch",
    )
    displacement = model.field(fields.displacement(domain))
    left_boundary = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left"
    )
    right_boundary = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right"
    )
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry"
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry"
    )
    model.fix(displacement, on=left_boundary, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right_boundary, component=0, value=0.01)
    regions = mesh.partition_cells(
        domain,
        soft=lambda x: x[0] <= 0.5,
        hard=lambda x: x[0] > 0.5,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=120_000.0,
            poisson=0.3,
            yield_stress=100.0,
            hardening_modulus=1_000.0,
        ),
        region=regions.soft,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=250.0,
            hardening_modulus=2_000.0,
        ),
        region=regions.hard,
    )

    capability = step_capability(model, target=displacement)
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_strong_static"
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        progress=False,
    )
    step.solve()

    assert isinstance(step.material, constitutive.QuadratureMaterialMap)
    assert step.last_solve_info.converged
    assert np.max(step.response.state.committed_state_vectors()[:, -1]) > 0.0


def test_standard_j2_remote_displacement_uses_the_shared_value_path():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="finite_strain_j2_remote_displacement",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry"
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry"
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    remote = model.remote_displacement(
        displacement,
        reference_point=coordinates.reference_point((1.0, 0.0, 0.0)),
        on=right,
        translation=(0.02, 0.0, 0.0),
    )
    reference = remote.reference_values.copy()
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.at(0.5, 1.0),
        amplitude=amplitudes.tabular(
            (0.0, 0.5, 1.0),
            (0.0, 0.25, 1.0),
        ),
        progress=False,
    )

    step.solve(until=0.5)

    np.testing.assert_allclose(remote.value.x.array, 0.25 * reference)
    assert step.summary()["accepted_load_factor"] == pytest.approx(0.5)
    assert step.value_path.summary()["field_values"] == 1


def test_standard_j2_constraint_capability_sees_nested_periodic_assets():
    model, displacement, material, _step = _standard_patch(
        incrementation=steps.fixed(1)
    )
    ordinary = tuple(model.constraints)
    periodic = constraints.PeriodicConstraintSpec(
        slave_marker=lambda x: np.isclose(x[0], 1.0),
        master_marker=lambda x: np.isclose(x[0], 0.0),
        map_slave_to_master=lambda x: x,
    )
    nested = constraints.ConstraintSet(
        dirichlet=list(ordinary),
        periodic=[periodic],
    )

    assets = constraints.constraint_assets((nested,))
    assert assets[-1] is periodic
    assert len(assets) == len(ordinary) + 1
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "constraints": nested},
    )
    assert not capability["supported"]


def test_standard_j2_rejects_follower_load_without_external_tangent():
    model, displacement, material, _step = _standard_patch(
        incrementation=steps.fixed(1)
    )
    right = mesh.boundary(
        displacement.value.function_space.mesh,
        lambda x: np.isclose(x[0], 1.0),
        name="right_pressure",
    )
    model.pressure(
        1.0,
        on=right,
        configuration="current",
        displacement=displacement,
    )

    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert not capability["supported"]

    with pytest.raises(NotImplementedError, match="No step provider accepted"):
        model.step(
            target=displacement,
            material=material,
            incrementation=steps.fixed(1),
            progress=False,
        )


def test_standard_j2_load_capability_sees_follower_inside_public_containers():
    model, displacement, material, _step = _standard_patch(
        incrementation=steps.fixed(1)
    )
    right = mesh.boundary(
        displacement.value.function_space.mesh,
        lambda x: np.isclose(x[0], 1.0),
        name="wrapped_right_pressure",
    )
    follower = loads.pressure(
        1.0,
        on=right,
        configuration="current",
        displacement=displacement,
    )
    driven = loads.with_amplitude(
        follower,
        amplitudes.ramp(),
        domain=displacement.value.function_space.mesh,
    )
    model.load(loads.LoadSet.create(driven))

    visible = loads.load_assets(model.loads)
    physical = loads.load_assets(model.loads, unwrap_amplitudes=True)
    assert visible == (driven,)
    assert physical == (follower,)
    assert physical[0].configuration == "current"

    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert not capability["supported"]


def test_standard_j2_rolls_back_if_accepted_state_finalization_fails(monkeypatch):
    _, _, _, step = _standard_patch(
        incrementation=steps.automatic(
            initial=0.5,
            minimum=0.05,
            maximum=0.5,
            max_increments=10,
        )
    )
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()

    def fail_checkpoint():
        raise OSError("intentional checkpoint failure")

    monkeypatch.setattr(step, "_write_scheduled_checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="intentional checkpoint failure"):
        step.solve()

    assert step.accepted_load_factor == pytest.approx(0.0)
    assert not step.accepted_increments
    assert step.next_increment_size is None
    np.testing.assert_allclose(step.solution.x.array, initial_solution)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), initial_state
    )
    assert step.execution_events[-1].kind == "step_failed"


def test_standard_j2_loading_update_failure_is_atomic(monkeypatch):
    _, _, _, step = _standard_patch(incrementation=steps.fixed(2))
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()
    original_update = step.value_path.update
    calls = 0

    def fail_second_update(factor):
        nonlocal calls
        calls += 1
        if calls == 2:
            constant, _reference = step.value_path.constants[-1]
            constant.value = 123.0
            raise ValueError("intentional amplitude update failure")
        original_update(factor)

    monkeypatch.setattr(step.value_path, "update", fail_second_update)
    with pytest.raises(RuntimeError, match="loading update failed"):
        step.solve()

    assert step.accepted_load_factor == pytest.approx(0.0)
    np.testing.assert_allclose(step.solution.x.array, initial_solution)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), initial_state
    )
    restored_loading = step._snapshot_loading_state()
    np.testing.assert_allclose(restored_loading["load_factor"], 0.0)
    for value in restored_loading["prescribed_values"]["constants"]:
        np.testing.assert_allclose(value, 0.0)
    assert step.execution_events[-1].kind == "step_failed"


def test_standard_j2_serial_checkpoint_restore_is_atomic(monkeypatch, tmp_path):
    _, _, _, source = _standard_patch(incrementation=steps.fixed(2))
    source.solve(until=0.5)
    saved = source.save_checkpoint(tmp_path / "valid")

    _, _, _, receiver = _standard_patch(incrementation=steps.fixed(2))
    initial_solution = receiver.solution.x.array.copy()
    initial_accepted = receiver.accepted_solution.x.array.copy()
    initial_state = receiver.state_transaction.snapshot_runtime_state()
    original_restore = receiver.response.restore
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("intentional state restore failure")
        original_restore(state)

    monkeypatch.setattr(receiver.response, "restore", fail_once)
    with pytest.raises(ValueError, match="intentional state restore failure"):
        receiver.load_checkpoint(saved)

    assert receiver.accepted_load_factor == pytest.approx(0.0)
    assert not receiver.accepted_increments
    assert not receiver.checkpoints
    np.testing.assert_allclose(receiver.solution.x.array, initial_solution)
    np.testing.assert_allclose(
        receiver.accepted_solution.x.array,
        initial_accepted,
    )
    restored = receiver.state_transaction.snapshot_runtime_state()
    np.testing.assert_allclose(
        restored["first_piola_stress"],
        initial_state["first_piola_stress"],
    )


def test_standard_j2_checkpoint_requires_an_accepted_boundary(tmp_path):
    _, _, _, step = _standard_patch(incrementation=steps.fixed(2))
    step.solve(until=0.5)

    step.solution.x.array[0] += 1.0e-3
    with pytest.raises(RuntimeError, match="U equals U_ACCEPTED"):
        step.save_checkpoint(tmp_path / "trial_serial")
    with pytest.raises(RuntimeError, match="U equals U_ACCEPTED"):
        step.save_checkpoint(tmp_path / "trial_portable", portable=True)

    step.solution.x.array[:] = step.accepted_solution.x.array
    step.solution.x.scatter_forward()
    step.state_transaction.accepted_factor = 0.25
    with pytest.raises(RuntimeError, match="fully accepted material state"):
        step.save_checkpoint(tmp_path / "mismatched_coordinate")


def test_standard_j2_portable_checkpoint_restore_is_atomic(monkeypatch, tmp_path):
    _, _, _, source = _standard_patch(incrementation=steps.fixed(2))
    source.solve(until=0.5)
    manifest = source.save_checkpoint(tmp_path / "portable", portable=True)

    _, _, _, receiver = _standard_patch(incrementation=steps.fixed(2))
    initial_solution = receiver.solution.x.array.copy()
    initial_state = receiver.state_transaction.snapshot_runtime_state()

    def fail_quadrature_load(*_args, **_kwargs):
        raise ValueError("intentional quadrature restore failure")

    monkeypatch.setattr(receiver.response.state, "load", fail_quadrature_load)
    with pytest.raises(RuntimeError, match="quadrature restore failure"):
        receiver.load_checkpoint(manifest)

    assert receiver.accepted_load_factor == pytest.approx(0.0)
    assert not receiver.accepted_increments
    assert not receiver.checkpoints
    np.testing.assert_allclose(receiver.solution.x.array, initial_solution)
    restored = receiver.state_transaction.snapshot_runtime_state()
    np.testing.assert_allclose(
        restored["first_piola_stress"],
        initial_state["first_piola_stress"],
    )


def test_standard_j2_rejects_legacy_experimental_checkpoint_identity(tmp_path):
    _, _, _, step = _standard_patch(incrementation=steps.fixed(1))
    portable = tmp_path / "legacy.checkpoint.json"
    portable.write_text(
        json.dumps(
            {
                "schema": (
                    "agentfem.finite-strain-j2-experimental-checkpoint.v2"
                )
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Legacy experimental"):
        step.load_checkpoint(portable)

    partition_bound = tmp_path / "legacy.npz"
    np.savez(
        partition_bound,
        schema="agentfem.finite-strain-j2-experimental-checkpoint.v1",
    )
    with pytest.raises(ValueError, match="Legacy experimental"):
        step.load_checkpoint(partition_bound)
