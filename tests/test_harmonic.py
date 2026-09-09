from __future__ import annotations

import json

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    checkpointing,
    fields,
    mesh,
    models,
    operators,
    procedures,
    results,
    solvers,
    studies,
)
from agentfem.constitutive import isotropic_elastic
from agentfem.diagnostics import SolveEventRecorder
from agentfem.mechanics import harmonic as harmonic_mechanics
from agentfem.mechanics.harmonic import _physical_cycle_vector_amplitudes
from agentfem.provenance import content_fingerprint
from agentfem.solvers import SolveEvent


def _rayleigh_bar(*, cells: int = 12, comm=MPI.COMM_SELF):
    length = 2.0
    traction = 3.0
    density = 1.0
    young = 1000.0
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, 1.0, 1.0),
        (cells, 1, 1),
        comm=comm,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="rayleigh_harmonic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        isotropic_elastic(
            young=young,
            poisson=0.0,
            density=density,
        )
    )
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    loaded_end = mesh.face(domain, axis="x", value=length)
    model.traction((traction, 0.0, 0.0), on=loaded_end)
    stiffness = model.stiffness(displacement)
    mass = model.mass(displacement)
    damping = operators.rayleigh_damping(
        mass,
        stiffness,
        mass_coefficient=0.4,
        stiffness_coefficient=2.0e-3,
    )
    force = model.external_force(displacement)
    return model, displacement, loaded_end, stiffness, mass, damping, force


def _tip_phasor(step, loaded_end):
    return results.average(
        step.solution_real[0], measure=loaded_end.measure
    ) + 1j * results.average(step.solution_imaginary[0], measure=loaded_end.measure)


def _rayleigh_bar_exact(frequency: float) -> complex:
    omega = 2.0 * np.pi * frequency
    effective_young = 1000.0 * (1.0 + 1j * omega * 2.0e-3)
    wave_number = np.sqrt((omega**2 - 1j * omega * 0.4) / effective_young)
    return 3.0 * np.tan(wave_number * 2.0) / (effective_young * wave_number)


def test_direct_harmonic_operator_contract_is_explicit_and_inspectable():
    _model, _u, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)
    system = operators.direct_harmonic_system(
        stiffness,
        force,
        M=mass,
        C=damping,
    )

    assert system.K is stiffness
    assert system.M is mass
    assert system.C is damping
    assert system.F is force
    assert system.validate().is_valid
    assert system.summary()["dissipation_channels"] == {
        "material_loss": False,
        "viscous_damping": True,
    }
    assert "omega C" in system.equation


def test_generic_direct_harmonic_rayleigh_bar_matches_complex_wave_solution():
    model, displacement, loaded_end, stiffness, mass, damping, force = _rayleigh_bar()
    omega = 8.0
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        angular_frequency=omega,
    )
    result = step.solve_result()
    tip = results.average(
        step.solution_real[0], measure=loaded_end.measure
    ) + 1j * results.average(step.solution_imaginary[0], measure=loaded_end.measure)

    young = 1000.0
    density = 1.0
    alpha = 0.4
    beta = 2.0e-3
    effective_young = young * (1.0 + 1j * omega * beta)
    wave_number = np.sqrt(density * (omega**2 - 1j * omega * alpha) / effective_young)
    expected = 3.0 * np.tan(wave_number * 2.0) / (effective_young * wave_number)

    assert tip == pytest.approx(expected, rel=4.0e-3)
    assert result.quantity("material_dissipated_energy_per_cycle") == pytest.approx(0.0)
    assert result.quantity("viscous_dissipated_energy_per_cycle") > 0.0
    assert result.quantity("relative_cycle_energy_balance_error") < 1.0e-9
    assert result.quantity("relative_residual_norm") < 1.0e-9
    assert result.metadata["step"]["system"]["damping"]["family"] == (
        "rayleigh_damping"
    )


def test_generic_harmonic_prepared_problem_reuses_allocation_across_frequency():
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=4)
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequency=0.5,
    )
    step.solve()
    first = step.complex_dofs.copy()
    step.set_frequency(frequency=1.25)
    step.solve()

    assert not np.allclose(first, step.complex_dofs)
    execution = step.summary()["backend_execution"]
    assert execution["problem_allocation_count"] == 1
    assert execution["matrix_allocation_count"] == 1
    assert execution["solve_count"] == 2
    assert execution["ksp_object_reused"] is True
    assert execution["factorization_reuse_claimed"] is False


def test_harmonic_vector_amplitude_is_the_exact_physical_cycle_maximum():
    amplitudes = _physical_cycle_vector_amplitudes(
        np.asarray((1.0, 0.0, 3.0, 0.0)),
        np.asarray((0.0, 1.0, 4.0, 0.0)),
        block_size=2,
    )

    # Circular motion has unit physical norm at every phase.  The old
    # coefficient norm would incorrectly report sqrt(2).
    assert amplitudes[0] == pytest.approx(1.0)
    # Collinear real/imaginary coefficients recover the scalar phasor norm.
    assert amplitudes[1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "mutation", ("system", "coefficient", "bcs", "phase", "solver")
)
def test_prepared_direct_harmonic_step_rejects_configuration_drift(mutation):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequency=0.5,
    )
    step.solve()
    previous = step.complex_dofs.copy()

    if mutation == "system":
        step.system = operators.direct_harmonic_system(
            step.system.K,
            step.system.F,
            M=step.system.M,
            C=step.system.C,
        )
    elif mutation == "coefficient":
        constants = tuple(step.system.F.expression.constants())
        assert constants
        constants[0].value[...] *= 2.0
    elif mutation == "bcs":
        step.bcs = ()
    elif mutation == "phase":
        step.load_phase = 0.25
    else:
        step.solver_options = solvers.direct_solver()

    with pytest.raises(RuntimeError, match="backend was prepared"):
        step.solve()
    assert step.complex_dofs == pytest.approx(previous)


def test_direct_harmonic_energy_evidence_rejects_nonfinite_channels(monkeypatch):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequency=0.5,
    )
    step.solve()
    monkeypatch.setattr(
        harmonic_mechanics,
        "_quadratic_integral",
        lambda _operator, _value: np.nan,
    )

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        step.energy_evidence()


def test_prepared_harmonic_summary_and_energy_reject_stale_configuration():
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequency=0.5,
    )
    step.solve()
    step.load_phase = 0.25

    with pytest.raises(RuntimeError, match="backend was prepared"):
        step.summary()
    with pytest.raises(RuntimeError, match="backend was prepared"):
        step.energy_evidence()


def test_nested_and_monolithic_harmonic_layouts_match_with_nonzero_load_phase():
    frequency = 0.875
    phase = 0.37
    _nested_model, nested_u, nested_end, nested_k, nested_m, nested_c, nested_f = (
        _rayleigh_bar(cells=8)
    )
    nested = _nested_model.step(
        target=nested_u,
        K=nested_k,
        M=nested_m,
        C=nested_c,
        F=nested_f,
        frequency=frequency,
        load_phase=phase,
    )
    nested.solve()

    direct_model, direct_u, direct_end, direct_k, direct_m, direct_c, direct_f = (
        _rayleigh_bar(cells=8)
    )
    direct = direct_model.step(
        target=direct_u,
        K=direct_k,
        M=direct_m,
        C=direct_c,
        F=direct_f,
        frequency=frequency,
        load_phase=phase,
        solver_options=solvers.direct_solver(),
    )
    direct.solve()

    assert _tip_phasor(direct, direct_end) == pytest.approx(
        _tip_phasor(nested, nested_end), rel=2.0e-11, abs=1.0e-13
    )
    assert direct.energy_evidence()["input_energy_per_cycle"] == pytest.approx(
        nested.energy_evidence()["input_energy_per_cycle"],
        rel=2.0e-10,
        abs=1.0e-13,
    )
    nested_backend = nested.summary()["backend_execution"]
    direct_backend = direct.summary()["backend_execution"]
    assert nested_backend["matrix_layout"] == "nested"
    assert nested_backend["component_residuals_available"] is True
    assert direct_backend["matrix_layout"] == "monolithic"
    assert direct_backend["component_residuals_available"] is False
    assert direct_backend["component_residuals_unavailable_reason"]


@pytest.mark.parametrize(
    ("options", "message"),
    (
        (solvers.LinearSolverOptions(pc_type="cholesky"), "cholesky"),
        (solvers.LinearSolverOptions(pc_type="icc"), "icc"),
        (
            solvers.LinearSolverOptions(ksp_type="cg", pc_type="gamg"),
            "ksp_type='cg'",
        ),
        (
            solvers.LinearSolverOptions(ksp_type="minres", pc_type="gamg"),
            "ksp_type='minres'",
        ),
    ),
)
def test_generic_harmonic_rejects_spd_only_solver_policies(options, message):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)

    with pytest.raises(ValueError, match=message):
        model.step(
            target=displacement,
            K=stiffness,
            M=mass,
            C=damping,
            F=force,
            frequency=0.5,
            solver_options=options,
        )


def _rayleigh_sweep(*, execution_order="forward", comm=MPI.COMM_SELF, **step_options):
    model, displacement, loaded_end, stiffness, mass, damping, force = _rayleigh_bar(
        comm=comm
    )
    response = results.harmonic_response(
        "tip_x",
        lambda point: _tip_phasor(point, loaded_end),
        unit="m",
        description="Mean loaded-end axial displacement phasor.",
    )
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequencies=(1.25, 0.5, 0.875),
        responses=(response,),
        execution_order=execution_order,
        **step_options,
    )
    return model, step


def test_direct_harmonic_sweep_is_canonical_bounded_and_matches_reference():
    model, step = _rayleigh_sweep()

    assert model.steps == [step]
    assert step.procedure.algorithm == "real_block_complex_harmonic_sweep"
    result = step.solve_result()

    axis = np.asarray(result.histories["tip_x_AMPLITUDE"].abscissa)
    actual = np.asarray(result.histories["tip_x_REAL"].values) + 1j * np.asarray(
        result.histories["tip_x_IMAG"].values
    )
    expected = np.asarray([_rayleigh_bar_exact(value) for value in axis])
    assert axis == pytest.approx((0.5, 0.875, 1.25))
    assert actual == pytest.approx(expected, rel=4.0e-3)
    assert (
        np.max(result.histories["relative_cycle_energy_balance_error"].values) < 1.0e-9
    )
    assert np.max(result.histories["relative_residual_norm"].values) < 1.0e-9
    assert np.all(result.histories["linear_converged"].values == 1.0)
    assert result.quantity("peak_tip_x_frequency") in axis
    assert "maximum_displacement_vector_amplitude" in result.histories
    execution = step.point_step.summary()["backend_execution"]
    assert execution["problem_allocation_count"] == 1
    assert execution["matrix_allocation_count"] == 1
    assert execution["solve_count"] == 3
    assert len(step.records) == 3


def test_direct_harmonic_sweep_forward_reverse_and_chunk_resume_are_equivalent():
    _forward_model, forward = _rayleigh_sweep(execution_order="forward")
    forward_result = forward.solve_result()
    _reverse_model, reverse = _rayleigh_sweep(execution_order="reverse")
    reverse.solve(max_points=1)
    assert not reverse.completed
    with pytest.raises(RuntimeError, match="partial harmonic sweep"):
        reverse.canonical_records()
    with pytest.raises(TypeError, match="integer"):
        reverse.solve(max_points=1.5)
    reverse_result = reverse.solve_result()

    for name in (
        "tip_x_REAL",
        "tip_x_IMAG",
        "tip_x_AMPLITUDE",
        "relative_residual_norm",
        "relative_cycle_energy_balance_error",
    ):
        assert reverse_result.histories[name].values == pytest.approx(
            forward_result.histories[name].values,
            rel=1.0e-11,
            abs=1.0e-13,
        )


def test_direct_harmonic_sweep_json_checkpoint_resumes_across_execution_order(
    tmp_path,
):
    _partial_model, partial = _rayleigh_sweep(execution_order="forward")
    partial.solve(max_points=1)
    checkpoint = partial.save_checkpoint(tmp_path / "rayleigh-sweep")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert payload["schema"] == checkpointing.HARMONIC_SWEEP_CHECKPOINT_SCHEMA
    assert payload["portable"] is True
    assert payload["completed_indices"] == [0]
    assert payload["field_state"] == "not_stored_scalar_ledger_only"
    assert payload["records"][0]["responses"]["tip_x"] == {
        "real": pytest.approx(partial.records[0]["responses"]["tip_x"].real),
        "imaginary": pytest.approx(partial.records[0]["responses"]["tip_x"].imag),
    }
    assert not tuple(tmp_path.glob("*.npz"))
    assert not tuple(tmp_path.glob("*.tmp"))

    _resumed_model, resumed = _rayleigh_sweep(execution_order="reverse")
    restored = resumed.load_checkpoint(checkpoint)
    assert tuple(restored["records"]) == (0,)
    assert resumed.last_live_field_frequency is None
    resumed_result = resumed.solve_result()

    _reference_model, reference = _rayleigh_sweep(execution_order="forward")
    reference_result = reference.solve_result()
    for name in (
        "tip_x_REAL",
        "tip_x_IMAG",
        "tip_x_AMPLITUDE",
        "relative_residual_norm",
        "relative_cycle_energy_balance_error",
    ):
        assert resumed_result.histories[name].values == pytest.approx(
            reference_result.histories[name].values,
            rel=1.0e-11,
            abs=1.0e-13,
        )
    assert resumed_result.metadata["live_field_state"] == (
        "available_for_last_executed_frequency"
    )
    assert resumed_result.metadata["execution"]["event_count"] > 0
    assert any(
        item.metadata["role"] == "restart_source"
        for item in resumed_result.checkpoints.values()
    )


def test_direct_harmonic_sweep_checkpoint_identity_is_fail_closed(tmp_path):
    _model, source = _rayleigh_sweep()
    source.solve(max_points=1)
    checkpoint = source.save_checkpoint(tmp_path / "identity")

    _changed_model, changed = _rayleigh_sweep()
    changed.scientific_assets = {
        **changed.scientific_assets,
        "materials": ("deliberately_changed_material_identity",),
    }
    with pytest.raises(ValueError, match="scientific identity differs"):
        changed.load_checkpoint(checkpoint)
    assert changed.records == {}

    # Raw executable K/M/C/F forms are a complete identity even when callers
    # do not add higher-level model summaries. Checkpointing therefore remains
    # available without manufacturing duplicate declarations.
    _undeclared_model, undeclared = _rayleigh_sweep()
    undeclared.scientific_assets = None
    undeclared.solve(max_points=1)
    undeclared_checkpoint = undeclared.save_checkpoint(tmp_path / "undeclared")
    assert undeclared_checkpoint.exists()


def test_harmonic_checkpoint_binds_executed_operator_and_constraint_dofs(tmp_path):
    _model, source = _rayleigh_sweep()
    source.solve(max_points=1)
    checkpoint = source.save_checkpoint(tmp_path / "executable-identity")

    _operator_model, changed_operator = _rayleigh_sweep()
    system = changed_operator.point_step.system
    changed_operator.point_step.system = operators.direct_harmonic_system(
        2.0 * system.K,
        system.F,
        M=system.M,
        C=system.C,
        K_loss=system.loss,
    )
    with pytest.raises(ValueError, match="scientific identity differs"):
        changed_operator.load_checkpoint(checkpoint)
    assert changed_operator.records == {}

    _constraint_model, changed_constraints = _rayleigh_sweep()
    changed_constraints.point_step.bcs = ()
    with pytest.raises(ValueError, match="scientific identity differs"):
        changed_constraints.load_checkpoint(checkpoint)
    assert changed_constraints.records == {}


def test_harmonic_sweep_rejects_operator_mutation_after_first_point():
    _model, step = _rayleigh_sweep()
    step.solve(max_points=1)
    system = step.point_step.system
    step.point_step.system = operators.direct_harmonic_system(
        system.K,
        2.0 * system.F,
        M=system.M,
        C=system.C,
        K_loss=system.loss,
    )

    with pytest.raises(RuntimeError, match="operators.*changed"):
        step.solve(max_points=1)
    assert len(step.records) == 1


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (
            lambda payload: payload.__setitem__(
                "schema", "agentfem.harmonic-sweep-checkpoint.v1"
            ),
            "Version 1 ledgers",
        ),
        (
            lambda payload: payload.__setitem__(
                "software", {"name": "AgentFEM", "version": "999.0"}
            ),
            "software identity differs",
        ),
    ),
)
def test_harmonic_checkpoint_rejects_unmigrated_schema_and_software(
    tmp_path, change, message
):
    _model, source = _rayleigh_sweep()
    source.solve(max_points=1)
    checkpoint = source.save_checkpoint(tmp_path / "versioned-source")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload.pop("payload_fingerprint")
    change(payload)
    payload["payload_fingerprint"] = content_fingerprint(payload)
    altered = tmp_path / f"altered-{message.split()[0]}.checkpoint.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    _target_model, target = _rayleigh_sweep()
    target.failure = {"sentinel": True}
    before_events = tuple(target.execution_events)
    with pytest.raises(ValueError, match=message):
        target.load_checkpoint(altered)
    assert target.records == {}
    assert target.failure == {"sentinel": True}
    assert tuple(target.execution_events) == before_events


def test_harmonic_checkpoint_late_event_failure_is_atomic(tmp_path):
    _model, source = _rayleigh_sweep()
    source.solve(max_points=1)
    checkpoint = source.save_checkpoint(tmp_path / "event-source")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["execution_events"] = [{"step_name": "missing_kind"}]
    payload.pop("payload_fingerprint")
    payload["payload_fingerprint"] = content_fingerprint(payload)
    altered = tmp_path / "invalid-event.checkpoint.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    _target_model, target = _rayleigh_sweep()
    target.records[2] = {"sentinel": True}
    target.failure = {"sentinel": True}
    target.execution_events.append(
        SolveEvent("sweep_started", target.name, increment=7)
    )
    before_records = dict(target.records)
    before_events = tuple(target.execution_events)
    before_checkpoints = tuple(target.checkpoints)

    with pytest.raises(KeyError, match="kind"):
        target.load_checkpoint(altered)

    assert target.records == before_records
    assert target.failure == {"sentinel": True}
    assert tuple(target.execution_events) == before_events
    assert tuple(target.checkpoints) == before_checkpoints
    assert target._frozen_executable_identity is None
    assert target._checkpoint_field_identity_record is None


def test_direct_harmonic_sweep_checkpoint_policy_progress_and_trace(tmp_path, capsys):
    status = tmp_path / "sweep.status"
    _model, step = _rayleigh_sweep(
        checkpoint=checkpointing.every(
            2,
            directory=tmp_path / "checkpoints",
            keep_last=1,
        ),
        progress=True,
        status_file=status,
    )
    result = step.solve_result()

    assert len(step.checkpoints) == 1
    assert len(result.checkpoints) == 1
    assert step.checkpoints[0].metadata["role"] == "scheduled_checkpoint"
    assert step.checkpoints[0].path.exists()
    assert "SWEEP POINT COORDINATE VALUE UNIT" in status.read_text(encoding="utf-8")
    console = capsys.readouterr().out
    assert "[SWEEP 1] STARTED" in console
    assert "frequency=" in console
    assert "COMPLETED" in console

    final_checkpoint = step.checkpoints[0].path
    payload = json.loads(final_checkpoint.read_text(encoding="utf-8"))
    assert payload["execution_events"][-1]["kind"] == "sweep_completed"
    assert result.metadata["step"]["checkpoint_policy"] == {
        **step.execution_context.policy.checkpoint.summary(),
        "requested_portable": False,
        "effective_portable": True,
        "portable": True,
        "portability_mode": "scalar_evidence_ledger_no_field_state",
    }

    _restored_model, restored = _rayleigh_sweep(progress=False)
    restored.load_checkpoint(final_checkpoint)
    restored_result = restored.solve_result()
    assert restored_result.metadata["live_field_state"] == (
        "not_restored_scalar_checkpoint_only"
    )
    assert restored_result.metadata["execution"]["events"][-1]["kind"] == (
        "sweep_completed"
    )

    _bad_model, bad = _rayleigh_sweep(checkpoint="not-a-policy", progress=False)
    with pytest.raises(TypeError, match="checkpointing.every"):
        bad.solve(max_points=1)
    assert bad.records == {}


def test_direct_harmonic_sweep_checkpoint_identity_is_mpi_partition_neutral(
    tmp_path,
):
    comm = MPI.COMM_WORLD
    root_path = comm.bcast(
        str(tmp_path / "mpi-sweep") if comm.rank == 0 else None,
        root=0,
    )
    _source_model, source = _rayleigh_sweep(comm=comm)
    source.solve(max_points=1)
    checkpoint = source.save_checkpoint(root_path)

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["rank_count_at_write"] == comm.size
    assert payload["portable"] is True
    _resumed_model, resumed = _rayleigh_sweep(
        execution_order="reverse",
        comm=comm,
    )
    resumed.load_checkpoint(checkpoint)
    assert tuple(resumed.records) == (0,)
    resumed.solve()
    assert resumed.completed


def test_harmonic_sweep_event_trace_is_bounded_and_retains_failure():
    recorder = SolveEventRecorder(max_events=16)
    for index in range(100):
        recorder.emit(
            SolveEvent(
                "sweep_point",
                "large_sweep",
                increment=index + 1,
                total_increments=100,
                coordinate_name="frequency",
                coordinate_value=float(index),
                coordinate_unit="Hz",
                display=False,
            )
        )
    recorder.emit(
        SolveEvent(
            "sweep_failed",
            "large_sweep",
            increment=100,
            total_increments=100,
            message="synthetic failure",
        )
    )

    assert len(recorder.events) == 16
    assert recorder.dropped_events == 85
    assert recorder.events[-1].kind == "sweep_failed"


def test_direct_harmonic_sweep_rejects_axis_procedure_and_output_mismatches(tmp_path):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar()
    with pytest.raises(ValueError, match="procedure disagree"):
        model.step(
            target=displacement,
            K=stiffness,
            M=mass,
            C=damping,
            F=force,
            frequencies=(0.5, 1.0),
            procedure=procedures.direct_harmonic(),
        )
    assert model.steps == []

    _model, step = _rayleigh_sweep()
    step.execution_context = step.execution_context.__class__(
        model=step.execution_context.model,
        target=step.execution_context.target,
        material=step.execution_context.material,
        policy=step.execution_context.policy.__class__(output=tmp_path / "fields.xdmf"),
    )
    with pytest.raises(NotImplementedError, match="snapshot frequencies"):
        step.solve_result()


@pytest.mark.parametrize(
    "frequencies",
    [(), (1.0, 1.0), (np.nan,), (-1.0,)],
)
def test_direct_harmonic_sweep_rejects_invalid_frequency_axes(frequencies):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)
    with pytest.raises((TypeError, ValueError)):
        model.step(
            target=displacement,
            K=stiffness,
            M=mass,
            C=damping,
            F=force,
            frequencies=frequencies,
        )
