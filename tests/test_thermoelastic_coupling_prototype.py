# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import fields, mesh, operators
from agentfem.benchmarks.thermoelastic_coupling import _ThermoelasticPrototype


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
