# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json
from pathlib import Path
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import fields, mesh, operators
from agentfem.benchmarks.thermoelastic_coupling import _ThermoelasticPrototype


def test_joint_restart_matches_continuous_and_rejects_corrupt_history(tmp_path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path / "joint"), root=0)
    with _ThermoelasticPrototype(inward_heat_flux=2, dilation_rate=0.0001) as original:
        original.advance()
        manifest = original.save_checkpoint(path, total_steps=3)
        original.advance()
        expected = original.advance()
        with _ThermoelasticPrototype(
            inward_heat_flux=2, dilation_rate=0.0001
        ) as resumed:
            resumed.load_checkpoint(path, total_steps=3)
            assert resumed.completed_steps == 1
            resumed.advance()
            actual = resumed.advance()
            for key in (
                "quadratic_energy_change",
                "prescribed_motion_path_work",
                "boundary_heat_input",
            ):
                assert actual[key] == pytest.approx(expected[key], abs=1e-11)
            np.testing.assert_allclose(
                resumed.u.value.x.array, original.u.value.x.array, atol=1e-12
            )
            np.testing.assert_allclose(
                resumed.theta.value.x.array, original.theta.value.x.array, atol=1e-10
            )
            snapshots = [
                resumed.u.value.x.array.copy(),
                resumed.theta.value.x.array.copy(),
            ]
            if comm.rank == 0:
                data = json.loads(Path(manifest).read_text())
                data["completed_steps"] = 2
                Path(manifest).write_text(json.dumps(data))
            comm.barrier()
            with pytest.raises(ValueError, match="AFM-COUPLING-005"):
                resumed.load_checkpoint(path, total_steps=3)
            assert resumed.completed_steps == 3
            for field, snapshot in zip(
                (resumed.u.value, resumed.theta.value), snapshots
            ):
                np.testing.assert_array_equal(field.x.array, snapshot)


def test_checkpoint_refuses_trial_and_changed_physics(tmp_path):
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "joint"), root=0)
    with _ThermoelasticPrototype() as case:

        def save_trial(_):
            case.save_checkpoint(path, total_steps=3)

        with pytest.raises(RuntimeError, match="AFM-COUPLING-004"):
            case.advance(acceptance_check=save_trial)
        assert case.completed_steps == 0
        case.advance()
        case.save_checkpoint(path, total_steps=3)
    with _ThermoelasticPrototype(inward_heat_flux=1) as other:
        with pytest.raises(ValueError, match="time inputs differs"):
            other.load_checkpoint(path, total_steps=3)
        assert other.completed_steps == 0
        np.testing.assert_array_equal(other.theta.value.x.array, 0)


@pytest.mark.parametrize("rate", [-0.2, 0.2])
def test_prescribed_temperature_uses_owned_residual_heat_and_block_reference(rate):
    with _ThermoelasticPrototype(temperature_rate=rate) as case:
        for _ in range(3):
            record = case.advance()
            assert record["prescribed_temperature_heat_input"] * rate > 0
            assert record["linearized_heat_balance_absolute"] < 2e-8
            assert record["quadratic_balance_absolute"] < 1e-10
            assert record["temperature_reference_error"] < 2e-10
            assert record["displacement_reference_error"] < 2e-12


def test_prescribed_temperature_manufactured_uniform_heating():
    # C * T_dot = Q, alpha=0: uniform heating, zero boundary reaction.
    with _ThermoelasticPrototype(alpha=0, temperature_rate=0.1) as case:
        for _ in range(3):
            record = case.advance()
            np.testing.assert_allclose(
                case.theta.value.x.array, 0.1 * record["time"], atol=1e-12
            )
            assert abs(record["prescribed_temperature_heat_input"]) < 1e-10


def test_corrupt_checkpoint_field_payload_leaves_both_live_fields_unchanged(tmp_path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path / "payload"), root=0)
    with _ThermoelasticPrototype(temperature_rate=0.2) as case:
        case.advance()
        manifest = case.save_checkpoint(path, total_steps=3)
        case.advance()
        snapshots = [case.u.value.x.array.copy(), case.theta.value.x.array.copy()]
        if comm.rank == 0:
            data = json.loads(Path(manifest).read_text())
            shard = Path(manifest).parent / data["shards"][0]["path"]
            shard.write_bytes(b"corrupt")
        comm.barrier()
        with pytest.raises((ValueError, RuntimeError)):
            case.load_checkpoint(path, total_steps=3)
        assert case.completed_steps == 2
        assert float(case.prescribed_temperature.value) == pytest.approx(0.04)
        for field, snapshot in zip((case.u.value, case.theta.value), snapshots):
            np.testing.assert_array_equal(field.x.array, snapshot)


@pytest.mark.parametrize("alpha,relaxation", [(0.0, 1.0), (0.002, 1.0), (0.01, 0.25)])
def test_uniform_cube_matches_independent_closed_form_and_block_system(
    alpha, relaxation
):
    with _ThermoelasticPrototype(alpha=alpha) as case:
        for _ in range(3):
            record = case.advance(relaxation=relaxation)
            if alpha == 0:
                assert record["outer_iterations"] == 1
            bulk = 1000.0 / (3.0 * (1.0 - 2.0 * 0.25))
            expected = (
                10 * record["time"] / (case.capacity + 9 * bulk * alpha**2 * case.t0)
            )
            np.testing.assert_allclose(case.theta.value.x.array, expected, atol=2e-10)
            coordinates = case.u.space.tabulate_dof_coordinates()
            np.testing.assert_allclose(
                case.u.value.x.array.reshape(-1, 3),
                alpha * expected * coordinates,
                atol=2e-12,
            )
            assert record["temperature_reference_error"] < 2e-10
            assert record["displacement_reference_error"] < 2e-12
            assert record["linearized_heat_balance_absolute"] < 2e-8
            assert record["quadratic_balance_absolute"] < 1e-10
        evidence = case.result().metadata
        assert set(evidence["matrix_assemblies"].values()) == {1}
        assert evidence["solve_counts"]["thermal"] >= 3


@pytest.mark.parametrize("cells,dt", [(2, 0.1), (3, 0.05)])
def test_spatial_heat_feedback_matches_block_solution(cells, dt):
    with _ThermoelasticPrototype(cells=cells, dt=dt, nonuniform=True) as case:
        for _ in range(2):
            record = case.advance(relaxation=0.7)
            assert record["temperature_reference_error"] < 2e-10
            assert record["displacement_reference_error"] < 2e-12
            assert record["linearized_heat_balance_absolute"] < 2e-8
            assert record["quadratic_balance_absolute"] < 1e-10
            assert record["conduction_term"] > 0


@pytest.mark.parametrize("flux", [-2.0, 3.0])
def test_inward_boundary_heat_has_signed_physical_input_and_separate_identity(flux):
    with _ThermoelasticPrototype(inward_heat_flux=flux) as case:
        for _ in range(2):
            record = case.advance()
            # Unit cube has six unit-area faces; positive means heat entering.
            assert record["boundary_heat_input"] == pytest.approx(6 * flux * case.dt)
            assert record["volume_heat_input"] == pytest.approx(10 * case.dt)
            assert record["linearized_heat_balance_absolute"] < 2e-8
            assert record["quadratic_balance_absolute"] < 1e-10
            assert record["temperature_reference_error"] < 2e-10
        assert case.result().metadata["inputs"]["inward_heat_flux"] == flux


@pytest.mark.parametrize("rate", [-0.0001, 0.0002])
def test_prescribed_dilation_matches_closed_form_reaction_and_path_work(rate):
    with _ThermoelasticPrototype(dilation_rate=rate) as case:
        previous_reaction = 0.0
        for _ in range(3):
            record = case.advance()
            time = record["time"]
            theta = (10 - 3 * case.beta * case.t0 * rate) * time / case.capacity
            reaction = 3 * (2000 * rate * time - case.beta * theta)
            np.testing.assert_allclose(case.theta.value.x.array, theta, atol=2e-10)
            np.testing.assert_allclose(
                case.u.value.x.array.reshape(-1, 3),
                rate * time * case.u.space.tabulate_dof_coordinates(),
                atol=2e-12,
            )
            assert record["dilation_reaction"] == pytest.approx(reaction, abs=1e-9)
            assert record["prescribed_motion_path_work"] == pytest.approx(
                0.5 * (previous_reaction + reaction) * rate * case.dt, abs=1e-12
            )
            assert record["quadratic_balance_absolute"] < 1e-10
            assert record["linearized_heat_balance_absolute"] < 2e-8
            assert record["temperature_reference_error"] < 2e-10
            previous_reaction = reaction


def test_rejected_motion_restores_boundary_control_and_accepted_path():
    with _ThermoelasticPrototype(dilation_rate=0.0002) as case:
        case.advance()
        accepted = float(case.prescribed_dilation.value)

        def reject(_):
            raise RuntimeError("reject motion")

        with pytest.raises(RuntimeError, match="reject motion"):
            case.advance(acceptance_check=reject)
        assert float(case.prescribed_dilation.value) == accepted
        assert len(case.history) == case.completed_steps == 1
        assert case.advance()["quadratic_balance_absolute"] < 1e-10


def test_tiny_relaxation_cannot_fake_convergence_and_retry_preserves_time():
    with _ThermoelasticPrototype() as case:
        with pytest.raises(RuntimeError, match="No accepted"):
            case.result()
        with pytest.raises(RuntimeError, match="AFM-COUPLING-001"):
            case.advance(relaxation=1e-12, max_iterations=2)
        assert case.completed_steps == 0 and not case.history
        for function in (
            case.theta.value,
            case.u.value,
            case.theta_old,
            case.u_old,
            case.reference,
            case.reference_old,
        ):
            np.testing.assert_array_equal(function.x.array, 0.0)
        record = case.advance()
        assert record["temperature_reference_error"] < 2e-10


def test_one_rank_rejection_rolls_back_both_participants_and_reference():
    with _ThermoelasticPrototype(nonuniform=True) as case:
        case.advance()
        functions = (
            case.theta.value,
            case.u.value,
            case.theta_old,
            case.u_old,
            case.reference,
            case.reference_old,
        )
        snapshots = [f.x.array.copy() for f in functions]

        def reject(_record):
            if case.domain.comm.rank == 0:
                raise RuntimeError("injected acceptance failure")

        with pytest.raises(RuntimeError, match="injected acceptance failure"):
            case.advance(acceptance_check=reject)
        assert case.completed_steps == 1 and len(case.history) == 1
        for function, snapshot in zip(functions, snapshots):
            np.testing.assert_array_equal(function.x.array, snapshot)
        assert case.advance()["temperature_reference_error"] < 2e-10


def test_feedback_sign_and_explicit_scope():
    domain = mesh.cuboid((0.0,) * 3, (1.0,) * 3, (2,) * 3, comm=MPI.COMM_WORLD)
    u = fields.displacement(domain)
    u.value.interpolate(lambda x: 0.001 * x)
    t = fields.temperature(domain)
    operator = operators.thermoelastic_heat_source(
        t, u.value, coupling_coefficient=20.0, reference_temperature=300.0, dt=0.1
    )
    expression = ufl.replace(operator.expression, {t.test: ufl.as_ufl(1.0)})
    value = domain.comm.allreduce(fem.assemble_scalar(fem.form(expression)), op=MPI.SUM)
    assert value == pytest.approx(-180.0)
    assert operator.metadata["energy_semantics"] == "reversible_thermoelastic_exchange"
    with pytest.raises(ValueError, match="AFM-THERMO-001"):
        operators.thermoelastic_heat_source(
            t, u.value, coupling_coefficient=1.0, reference_temperature=300.0, dt=0.0
        )
    other = fields.displacement(
        mesh.cuboid((0.0,) * 3, (1.0,) * 3, (2,) * 3, comm=MPI.COMM_WORLD)
    )
    with pytest.raises(ValueError, match="AFM-THERMO-003"):
        operators.thermoelastic_heat_source(
            t,
            other.value,
            coupling_coefficient=1.0,
            reference_temperature=300.0,
            dt=1.0,
        )
    with pytest.raises(ValueError, match="AFM-THERMO-004"):
        operators.thermoelastic_heat_source(
            t,
            u.value,
            coupling_coefficient=1.0,
            reference_temperature=300.0,
            dt=1.0,
            measure=ufl.ds,
        )


def test_relaxation_stabilizes_strong_coupling_without_changing_solution():
    with _ThermoelasticPrototype(alpha=0.01) as case:
        with pytest.raises(RuntimeError, match="AFM-COUPLING-001"):
            case.advance(max_iterations=12)
        assert case.completed_steps == 0
        record = case.advance(relaxation=0.25)
        assert record["temperature_reference_error"] < 2e-10


def test_feedback_does_not_guess_two_dimensional_reduction():
    domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="AFM-THERMO-002"):
        operators.thermoelastic_heat_source(
            fields.temperature(domain),
            fields.displacement(domain).value,
            coupling_coefficient=1.0,
            reference_temperature=300.0,
            dt=1.0,
        )


def test_rank_local_iteration_policy_is_rejected_before_collective_solves():
    if MPI.COMM_WORLD.size == 1:
        pytest.skip("requires distinct ranks")
    with _ThermoelasticPrototype() as case:
        with pytest.raises(ValueError, match="AFM-COUPLING-003"):
            case.advance(max_iterations=100 + MPI.COMM_WORLD.rank)
        assert case.completed_steps == 0
        assert case.heat.solve_count == 0
