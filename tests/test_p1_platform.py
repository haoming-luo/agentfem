from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    amplitudes,
    benchmarks,
    constitutive,
    diagnostics,
    fields,
    mesh,
    models,
    procedures,
    results,
    solvers,
    steps,
    studies,
    time,
)


def test_solution_procedure_separates_problem_from_algorithm():
    study = studies.implicit_dynamics(
        physics="solid_mechanics",
        dimension=3,
        method="generalized_alpha",
    )
    procedure = procedures.for_step(
        analysis=study.analysis,
        method=study.preferred_procedure,
    )

    assert study.analysis == "second_order_dynamics"
    assert procedure.family == "standard"
    assert procedure.algorithm == "generalized_alpha"
    assert procedure.stateful
    assert not procedure.explicit


def test_j2_algorithmic_tangent_matches_return_map_derivative():
    material = constitutive.J2LinearIsotropicHardening(
        young=200.0e3,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2.0e3,
    )
    strain = np.array(
        [
            [0.004, 0.0004, 0.0],
            [0.0004, -0.002, 0.0],
            [0.0, 0.0, -0.002],
        ]
    )
    direction = np.array(
        [
            [0.7, -0.2, 0.1],
            [-0.2, -0.4, 0.05],
            [0.1, 0.05, -0.3],
        ]
    )
    update = material.update(strain)
    analytical = np.einsum(
        "ijkl,kl->ij",
        update.algorithmic_tangent,
        direction,
    )
    step = 1.0e-7
    numerical = (
        material.update(strain + step * direction).stress
        - material.update(strain - step * direction).stress
    ) / (2.0 * step)

    assert not update.elastic
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-7, atol=1.0e-6)


def test_implicit_power_law_creep_tangent_matches_backward_euler_update():
    material = constitutive.isotropic_power_law(
        young=200.0e3,
        poisson=0.3,
        density=1.0,
        coefficient=2.0e-6,
        stress_exponent=3.0,
        reference_stress=100.0,
    )
    strain = np.array(
        [
            [0.002, 0.0002, 0.0],
            [0.0002, -0.001, 0.0],
            [0.0, 0.0, -0.001],
        ]
    )
    direction = np.array(
        [
            [0.7, -0.2, 0.1],
            [-0.2, -0.4, 0.05],
            [0.1, 0.05, -0.3],
        ]
    )
    update = material.update(strain, time_start=0.0, time_end=10.0)
    analytical = np.einsum(
        "ijkl,kl->ij",
        update.algorithmic_tangent,
        direction,
    )
    perturbation = 1.0e-7
    numerical = (
        material.update(
            strain + perturbation * direction,
            time_start=0.0,
            time_end=10.0,
        ).stress
        - material.update(
            strain - perturbation * direction,
            time_start=0.0,
            time_end=10.0,
        ).stress
    ) / (2.0 * perturbation)

    assert update.converged
    assert update.equivalent_increment > 0.0
    assert update.local_iterations > 0
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-7, atol=5.0e-4)


def test_creep_quadrature_state_uses_shared_atomic_transaction():
    domain = dolfinx_mesh.create_box(
        MPI.COMM_SELF,
        [np.zeros(3), np.ones(3)],
        [1, 1, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    state = constitutive.CreepQuadratureState.create(domain, degree=2)
    material = constitutive.isotropic_power_law(
        young=200.0e3,
        poisson=0.3,
        density=1.0,
        coefficient=1.0e-6,
        stress_exponent=3.0,
        reference_stress=100.0,
    )
    point_count = len(state.equivalent_creep_strain.values)
    strain = np.zeros((point_count, 3, 3))
    strain[:, 0, 0] = 0.002
    strain[:, 1, 1] = -0.001
    strain[:, 2, 2] = -0.001

    update = state.update(
        strain,
        material,
        time_start=0.0,
        time_end=10.0,
    )
    assert update["creeping_points"] == point_count
    assert update["maximum_creep_increment"] > 0.0
    assert np.all(state.equivalent_creep_strain.values == 0.0)
    assert np.all(state.trial_equivalent_creep_strain.values > 0.0)

    state.rollback()
    assert np.all(state.trial_equivalent_creep_strain.values == 0.0)
    state.update(strain, material, time_start=0.0, time_end=10.0)
    state.commit()
    assert np.all(state.equivalent_creep_strain.values > 0.0)
    assert state.transaction.schema == (
        "agentfem.power-law-creep-small-strain-state"
    )


def test_global_implicit_creep_relaxation_uses_public_step_and_fields():
    step, _ = _creep_relaxation_patch()
    simulation = step.solve_result()
    final_stress = results.average(
        step.state.stress.function[0, 0],
        measure=step.state.measure,
    )
    analytical = step.material.creep.relaxation_stress(
        initial_stress=step.material.young * 0.002,
        young=step.material.young,
        time=step.duration,
    )
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.implicit_creep_relaxation"
    )
    observables = {
        "mean_axial_stress": final_stress,
        "mean_equivalent_creep_strain": results.average(
            step.state.equivalent_creep_strain.function,
            measure=step.state.measure,
        ),
        "creep_dissipation": simulation.histories["creep_dissipation"].latest,
    }

    assert step.procedure.algorithm == "backward_euler_newton"
    assert step.last_solve_info.completed_step
    assert final_stress == pytest.approx(analytical, rel=2.5e-2)
    assert all(golden.verify(observables).values())
    assert {"S", "CE", "CEEQ", "MISES", "RF"} <= set(simulation.fields)
    assert simulation.quantity("maximum_equivalent_creep_strain") > 0.0
    assert simulation.histories["creep_dissipation"].latest > 0.0
    assert np.all(
        np.diff(simulation.histories["creep_dissipation"].values) >= -1.0e-12
    )


def test_global_creep_matches_official_abaqus_constant_stress_case():
    step = _creep_external_abaqus_constant_stress_patch()
    step.solve()
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.creep_abaqus_constant_stress"
    )
    observables = {
        "mean_axial_stress": results.average(
            step.state.stress.function[0, 0],
            measure=step.state.measure,
        ),
        "mean_equivalent_creep_strain": results.average(
            step.state.equivalent_creep_strain.function,
            measure=step.state.measure,
        ),
    }

    assert all(golden.verify(observables).values())
    assert len(step.accepted_increments) == 100


def test_global_implicit_creep_checkpoint_restart_matches_full_path(tmp_path):
    reference, _ = _creep_relaxation_patch()
    reference.solve()
    expected_u = reference.solution.x.array.copy()
    expected_ce = reference.state.creep_strain.values.copy()
    expected_ceeq = reference.state.equivalent_creep_strain.values.copy()
    expected_stress = reference.state.stress.values.copy()

    partial, _ = _creep_relaxation_patch()
    partial.solve(until=5.0)
    checkpoint = partial.save_checkpoint(tmp_path / "creep_restart.npz")
    restarted, _ = _creep_relaxation_patch()
    restarted.load_checkpoint(checkpoint)
    np.testing.assert_allclose(
        restarted.state.stress.values,
        partial.state.stress.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    restarted.solve()

    assert restarted.last_solve_info.completed_step
    np.testing.assert_allclose(restarted.solution.x.array, expected_u, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(restarted.state.creep_strain.values, expected_ce, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(restarted.state.equivalent_creep_strain.values, expected_ceeq, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(restarted.state.stress.values, expected_stress, rtol=2e-8, atol=2e-8)
    assert [item.end_time for item in restarted.last_solve_info.increments] == pytest.approx(
        np.arange(1.0, 11.0)
    )


def test_global_implicit_creep_state_limit_forces_real_cutback():
    control = steps.automatic(
        initial=1.0,
        minimum=1.0e-4,
        maximum=1.0,
        max_increments=100,
        max_cutbacks=10,
        cutback_factor=0.5,
        maximum_inelastic_increment=2.0e-5,
    )
    step, _ = _creep_relaxation_patch(incrementation=control)
    step.solve()

    rejected = [item for item in step.attempted_increments if not item.converged]
    assert step.last_solve_info.completed_step
    assert rejected
    assert "maximum equivalent creep-strain increment" in rejected[0].rejection_reason
    assert max(
        item.maximum_creep_increment for item in step.accepted_increments
    ) <= control.maximum_inelastic_increment * (1.0 + 1.0e-10)


def test_global_j2_checkpoint_restart_matches_uninterrupted_path(tmp_path):
    reference, reference_u = _j2_patch()
    reference.solve()
    expected_displacement = reference_u.value.x.array.copy()
    expected_peeq = reference.state.equivalent_plastic_strain.values.copy()

    partial, _ = _j2_patch()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "j2_restart.npz")
    assert checkpoint.with_suffix(".npz.checkpoint.json").is_file()
    assert partial.checkpoints[0].portable is False
    assert partial.checkpoints[0].schema == "agentfem.j2-step-checkpoint.v4"
    assert partial.execution_events[-1].kind == "step_paused"

    restarted, restarted_u = _j2_patch()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    assert reference.last_solve_info.converged
    assert restarted.last_solve_info.converged
    assert np.max(expected_peeq) > 0.0
    np.testing.assert_allclose(
        restarted_u.value.x.array,
        expected_displacement,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.state.equivalent_plastic_strain.values,
        expected_peeq,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    assert [
        item.load_factor for item in restarted.last_solve_info.increments
    ] == pytest.approx([0.25, 0.5, 0.75, 1.0])

    incompatible, _ = _j2_uniaxial_patch(
        amplitude=amplitudes.tabular((0.0, 1.0), (0.0, 0.5))
    )
    with pytest.raises(ValueError, match="amplitude differs"):
        incompatible.load_checkpoint(checkpoint)

    incompatible_material, _ = _j2_uniaxial_patch()
    incompatible_material.material = constitutive.J2LinearIsotropicHardening(
        young=200.0e3,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=3.0e3,
    )
    with pytest.raises(ValueError, match="material, procedure, increment control"):
        incompatible_material.load_checkpoint(checkpoint)


def test_global_j2_uniaxial_path_matches_versioned_analytical_golden():
    step, _ = _j2_uniaxial_patch()
    simulation = step.solve_result()
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.j2_global_restart"
    )
    actual = {
        "mean_axial_stress": results.average(
            step.state.stress.function[0, 0],
            measure=step.state.measure,
        ),
        "mean_equivalent_plastic_strain": results.average(
            step.state.equivalent_plastic_strain.function,
            measure=step.state.measure,
        ),
    }

    assert all(golden.verify(actual).values())
    assert simulation.quantity("internal_energy") > 0.0
    assert simulation.quantity("plastic_dissipation") > 0.0
    assert {"S", "PE", "PEEQ", "MISES", "RF"} <= set(simulation.fields)
    assert simulation.fields["S"].location == "quadrature_points"
    assert simulation.fields["S"].processing["postprocessed"] is False
    assert simulation.fields["MISES"].processing["derived_from"] == ("S",)
    presentation_fields = step.state.output_fields()
    assert tuple(item.name for item in presentation_fields) == (
        "S",
        "PE",
        "PEEQ",
        "MISES",
    )
    assert np.all(presentation_fields[-1].x.array >= 0.0)
    assert len(simulation.histories["newton_iterations"].values) == 4
    assert {
        "internal_energy",
        "external_work",
        "energy_balance_error",
        "generalized_reaction",
    } <= set(simulation.histories)
    final_internal = simulation.histories["internal_energy"].latest
    final_balance = simulation.histories["energy_balance_error"].latest
    assert abs(final_balance) / final_internal < 0.03
    assert np.all(np.diff(simulation.histories["plastic_dissipation"].values) >= 0.0)
    assert simulation.metadata["execution"]["event_count"] > 4


def test_global_j2_matches_published_abaqus_rate_independent_plasticity_case():
    """Reproduce the homogeneous uniaxial state in the Abaqus verification case."""

    step = _j2_external_abaqus_uniaxial_patch()
    step.solve()
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.j2_abaqus_rate_independent"
    )
    actual = {
        "mean_axial_stress": results.average(
            step.state.stress.function[0, 0],
            measure=step.state.measure,
        ),
        "mean_equivalent_plastic_strain": results.average(
            step.state.equivalent_plastic_strain.function,
            measure=step.state.measure,
        ),
    }

    assert all(golden.verify(actual).values())


def test_global_j2_multielement_patch_matches_the_uniaxial_golden():
    """Verify constitutive state and global assembly beyond one element."""

    step, _ = _j2_uniaxial_patch(cells=(4, 2, 2))
    simulation = step.solve_result()
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.j2_global_restart"
    )
    actual = {
        "mean_axial_stress": results.average(
            step.state.stress.function[0, 0],
            measure=step.state.measure,
        ),
        "mean_equivalent_plastic_strain": results.average(
            step.state.equivalent_plastic_strain.function,
            measure=step.state.measure,
        ),
    }

    assert all(golden.verify(actual).values())
    peeq = step.state.equivalent_plastic_strain.values.reshape(-1)
    assert peeq.size > 10
    assert np.max(peeq) - np.min(peeq) < 2.0e-10
    assert simulation.quantity("plastic_integration_points") == peeq.size
    final_internal = simulation.histories["internal_energy"].latest
    final_balance = simulation.histories["energy_balance_error"].latest
    assert abs(final_balance) / final_internal < 0.03


def test_global_j2_nonuniform_bending_path_localizes_plastic_state():
    """Exercise assembly and state recovery on a nonuniform structural path."""

    step, _ = _j2_nonuniform_bending_patch()
    simulation = step.solve_result()
    peeq = step.state.equivalent_plastic_strain.values.reshape(-1)

    assert step.last_solve_info.completed_step
    assert np.max(peeq) > 1.0e-4
    assert np.min(peeq) == pytest.approx(0.0, abs=1.0e-14)
    assert np.ptp(peeq) > 1.0e-4
    assert 0 < simulation.quantity("plastic_integration_points") < peeq.size
    assert simulation.histories["external_work"].latest > 0.0
    assert simulation.histories["internal_energy"].latest > 0.0
    relative_balance = abs(simulation.histories["energy_balance_error"].latest) / (
        simulation.histories["internal_energy"].latest
    )
    assert relative_balance < 0.08


def test_global_j2_proportionally_applies_prescribed_displacement():
    step, displacement = _j2_displacement_patch()
    step.solve(until=0.5)

    assert step.last_solve_info.converged
    assert step.summary()["loading"]["prescribed_values"] == 4
    assert np.max(displacement.value.x.array) == pytest.approx(0.0025)
    assert np.max(step.state.equivalent_plastic_strain.values) > 0.0

    step.solve()
    assert step.last_solve_info.converged
    assert np.max(displacement.value.x.array) == pytest.approx(0.005)


def test_global_j2_real_state_limit_forces_cutback_and_restart_equivalence(tmp_path):
    control = steps.automatic(
        initial=1.0,
        minimum=1.0e-3,
        maximum=1.0,
        max_cutbacks=8,
        cutback_factor=0.5,
        maximum_inelastic_increment=5.0e-4,
    )
    reference, _ = _j2_uniaxial_patch(incrementation=control)
    reference.solve(until=0.5)
    reference.solve()
    expected_u = reference.solution.x.array.copy()
    expected_peeq = reference.state.equivalent_plastic_strain.values.copy()

    partial, _ = _j2_uniaxial_patch(incrementation=control)
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "j2_cutback_restart.npz")
    restarted, _ = _j2_uniaxial_patch(incrementation=control)
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    rejected = [
        item
        for item in reference.last_solve_info.attempts
        if item.rejection_reason is not None
    ]
    assert reference.last_solve_info.completed_step
    assert restarted.last_solve_info.completed_step
    assert rejected
    assert all(not item.converged for item in rejected)
    assert any(
        event.kind == "increment_cutback"
        and event.message
        for event in reference.execution_events
    )
    np.testing.assert_allclose(restarted.solution.x.array, expected_u)
    np.testing.assert_allclose(
        restarted.state.equivalent_plastic_strain.values,
        expected_peeq,
    )
    np.testing.assert_allclose(
        [item.external_work for item in restarted.energy_history],
        [item.external_work for item in reference.energy_history],
    )


def test_global_j2_consumes_a_nonmonotone_cyclic_amplitude():
    history = amplitudes.tabular(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (0.0, 1.0, 0.0, -1.0, 0.0),
        name="fully_reversed_displacement",
    )
    step, _ = _j2_uniaxial_patch(
        amplitude=history,
        incrementation=steps.fixed(4),
    )
    result = step.solve_result()

    np.testing.assert_allclose(
        result.histories["load_amplitude"].values,
        [1.0, 0.0, -1.0, 0.0],
    )
    assert step.last_solve_info.completed_step
    assert np.max(step.state.equivalent_plastic_strain.values) > 0.005
    assert np.all(step.state.equivalent_plastic_strain.values >= 0.0)
    assert step.summary()["loading"]["amplitude"]["name"] == (
        "fully_reversed_displacement"
    )


def test_implicit_dynamics_provider_runs_newmark_and_records_procedure(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    study = studies.implicit_dynamics(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
        method="newmark",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=2.0e5,
            poisson=0.3,
            density=1.0e3,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((10.0, 0.0), on=right)

    step = model.step(
        target=displacement,
        dt=1.0e-3,
        steps=3,
        progress=False,
        status_file=tmp_path / "implicit.sta",
    )
    simulation = step.solve_result(output=tmp_path / "implicit.xdmf")

    assert step.procedure.algorithm == "newmark"
    assert np.max(np.abs(step.state.u.value.x.array)) > 0.0
    time_events = [
        event for event in step.execution_events
        if event.kind == "time_increment"
    ]
    assert len(time_events) == 3
    assert simulation.histories["accepted_increment"].latest == 3.0
    assert simulation.histories["time_increment"].latest == pytest.approx(1.0e-3)
    assert simulation.artifacts["fields_xdmf"].is_file()
    assert simulation.artifacts["fields_hdf5"].is_file()
    assert {"Displacement", "Velocity", "Acceleration"}.issubset(
        simulation.fields
    )
    assert simulation.metadata["execution"]["event_count"] == 5
    assert not (tmp_path / "implicit.sta").exists()


def test_thermoelastic_material_arrhenius_creep_and_free_expansion():
    material = constitutive.thermoelastic(
        young=200.0e9,
        poisson=0.3,
        density=7800.0,
        thermal_expansion=12.0e-6,
        conductivity=45.0,
        specific_heat=500.0,
        reference_temperature=300.0,
    )
    creep = constitutive.ArrheniusPowerLawCreep(
        coefficient=1.0e-10,
        stress_exponent=4.0,
        activation_energy=180.0e3,
        reference_temperature=800.0,
        reference_stress=100.0,
    )
    assert material.volumetric_heat_capacity == 3.9e6
    assert creep.temperature_factor(900.0) > 1.0
    assert creep.temperature_factor(700.0) < 1.0

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    temperature = fields.temperature(domain, value=400.0)
    model.material(material)
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    bottom = mesh.boundary(
        domain,
        lambda x: np.isclose(x[1], 0.0),
        name="bottom",
        tag=2,
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=bottom, component=1, value=0.0)
    thermal_force = model.thermal_expansion(displacement, temperature)
    thermoelastic_step = model.step(
        target=displacement,
        K=model.stiffness(displacement),
        F=thermal_force,
    )
    thermoelastic_result = thermoelastic_step.solve_result()
    assert "S" not in thermoelastic_result.fields

    coordinates = displacement.space.tabulate_dof_coordinates()
    values = displacement.value.x.array.reshape((-1, 2))
    right = np.isclose(coordinates[:, 0], 1.0)
    expected = material.thermal_expansion * 100.0
    np.testing.assert_allclose(values[right, 0], expected, rtol=2.0e-9)


def test_generalized_alpha_parameters_have_expected_limits():
    undamped = time.generalized_alpha(spectral_radius=1.0)
    dissipative = time.generalized_alpha(spectral_radius=0.0)

    assert undamped.alpha_m == undamped.alpha_f == 0.5
    assert undamped.gamma == 0.5
    assert dissipative.alpha_m == -1.0
    assert dissipative.alpha_f == 0.0
    assert dissipative.beta == 1.0


def test_linear_reaction_field_and_operator_energy_balance():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.isotropic_elastic(
            young=2.0e5,
            poisson=0.3,
            density=1.0e3,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((10.0, 0.0), on=right)
    stiffness = model.stiffness(displacement)
    step = model.step(
        target=displacement,
        K=stiffness,
        F=model.external_force(displacement),
    )
    step.solve()
    reactions = step.problem.reaction_field()
    total_x = np.sum(reactions.x.array.reshape((-1, 2))[:, 0])
    resultant = results.reaction_resultant(step.problem)
    velocity = fields.displacement(domain).value
    energy = diagnostics.mechanical_energy(
        mass=model.mass(displacement, material),
        stiffness=stiffness,
        displacement=displacement.value,
        velocity=velocity,
    )
    static_energy = diagnostics.linear_static_energy(
        stiffness=stiffness,
        force=step.problem.system.F,
        displacement=displacement.value,
    )

    np.testing.assert_allclose(total_x, -2.0, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(resultant, [-2.0, 0.0], rtol=1.0e-10, atol=1.0e-10)
    assert energy.kinetic == 0.0
    assert energy.strain > 0.0
    assert energy.total == energy.strain
    assert static_energy.external_work == pytest.approx(static_energy.strain_energy)
    assert static_energy.balance_error == pytest.approx(0.0, abs=1.0e-13)


def test_linear_static_result_includes_nonzero_prescribed_motion_work():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=2.0e5,
            poisson=0.3,
            density=1.0,
        )
    )
    left = mesh.face(domain, axis="x", value=0.0, name="left", tag=1)
    right = mesh.face(domain, axis="x", value=1.0, name="right", tag=2)
    model.fix(displacement, on=left, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.01)

    simulation = model.step(target=displacement).solve_result()

    assert simulation.quantity("natural_load_work") == pytest.approx(0.0, abs=1e-13)
    assert simulation.quantity("prescribed_motion_work") > 0.0
    assert simulation.quantity("external_work") == pytest.approx(
        simulation.quantity("strain_energy"), rel=2.0e-11, abs=1.0e-12
    )
    assert simulation.quantity("energy_balance_error") == pytest.approx(
        0.0, abs=1.0e-10
    )
    assert simulation.metadata["static_work"]["reaction_scope"] == (
        "strong Dirichlet constraints"
    )


def _j2_patch():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=3,
    )
    model = models.create(study=study, mesh=domain, name="j2_patch")
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=200.0e3,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2.0e3,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((250.0, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def _creep_relaxation_patch(
    *,
    incrementation=None,
    arrhenius_temperature_range=None,
    duration: float = 10.0,
):
    """Homogeneous 3D constant-strain relaxation with free contraction."""

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.creep_solid(),
        mesh=domain,
        name="power_law_creep_relaxation",
    )
    displacement = model.field(fields.displacement(domain))
    material_factory = (
        constitutive.isotropic_power_law
        if arrhenius_temperature_range is None
        else constitutive.isotropic_arrhenius_power_law
    )
    material_options = {}
    if arrhenius_temperature_range is not None:
        material_options.update(
            activation_energy=120.0e3,
            reference_temperature=800.0,
        )
    material = model.material(
        material_factory(
            young=200.0e3,
            poisson=0.3,
            density=1.0,
            coefficient=1.0e-6,
            stress_exponent=3.0,
            reference_stress=100.0,
            **material_options,
        )
    )
    temperature = None
    if arrhenius_temperature_range is not None:
        lower, upper = map(float, arrhenius_temperature_range)
        temperature = model.field(fields.temperature(domain, value=lower))
        temperature.value.interpolate(
            lambda x: lower + (upper - lower) * x[0]
        )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="left", tag=1),
        component=0,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0, name="y_symmetry", tag=2),
        component=1,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0, name="z_symmetry", tag=3),
        component=2,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0, name="right", tag=4),
        component=0,
        value=0.002,
    )
    step = model.step(
        target=displacement,
        material=material,
        duration=duration,
        incrementation=(steps.fixed(10) if incrementation is None else incrementation),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
        temperature=temperature,
    )
    return step, displacement


def test_global_arrhenius_creep_consumes_nonuniform_temperature_field(tmp_path):
    step, _ = _creep_relaxation_patch(
        arrhenius_temperature_range=(800.0, 900.0),
        duration=1.0,
        incrementation=steps.fixed(2),
    )

    simulation = step.solve_result()

    assert step.last_solve_info.completed_step
    assert "TEMP" in simulation.fields
    minimum = simulation.quantity("minimum_creep_temperature")
    maximum = simulation.quantity("maximum_creep_temperature")
    assert 800.0 < minimum < 850.0 < maximum < 900.0
    assert minimum + maximum == pytest.approx(1700.0)
    assert all(
        item.minimum_temperature == pytest.approx(minimum)
        and item.maximum_temperature == pytest.approx(maximum)
        for item in step.accepted_increments
    )
    assert simulation.histories["minimum_creep_temperature"].latest == pytest.approx(
        minimum
    )
    assert simulation.histories["maximum_creep_temperature"].latest == pytest.approx(
        maximum
    )
    assert np.ptp(step.state.equivalent_creep_strain.values) > 0.0

    checkpoint = step.save_checkpoint(tmp_path / "arrhenius_creep.npz")
    incompatible, _ = _creep_relaxation_patch(
        arrhenius_temperature_range=(810.0, 910.0),
        duration=1.0,
        incrementation=steps.fixed(2),
    )
    with pytest.raises(ValueError, match="temperature.*differs"):
        incompatible.load_checkpoint(checkpoint)


def _creep_external_abaqus_constant_stress_patch():
    """3D counterpart of the official Abaqus constant-stress creep case."""

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.creep_solid(),
        mesh=domain,
        name="abaqus_constant_stress_creep",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.isotropic_power_law(
            young=20.0e6,
            poisson=0.3,
            density=1.0,
            coefficient=2.5e-27,
            stress_exponent=5.0,
            time_exponent=-0.2,
            reference_stress=1.0,
            reference_time=1.0,
            name="Abaqus time-hardening creep data",
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="left", tag=1),
        component=0,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0, name="y_symmetry", tag=2),
        component=1,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0, name="z_symmetry", tag=3),
        component=2,
        value=0.0,
    )
    model.traction(
        (20_000.0, 0.0, 0.0),
        on=mesh.face(domain, axis="x", value=1.0, name="loaded", tag=4),
    )
    return model.step(
        target=displacement,
        material=material,
        duration=100_000.0,
        incrementation=steps.fixed(100),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-8,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )


def _j2_displacement_patch():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=3,
    )
    model = models.create(study=study, mesh=domain, name="j2_displacement_patch")
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=200.0e3,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2.0e3,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.005)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def _j2_uniaxial_patch(*, amplitude=None, incrementation=None, cells=(1, 1, 1)):
    """Homogeneous bar with free Poisson contraction and removed rigid modes."""

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, *cells)
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=3,
    )
    model = models.create(study=study, mesh=domain, name="j2_uniaxial_golden")
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=200.0e3,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2.0e3,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    y_symmetry = mesh.boundary(
        domain,
        lambda x: np.isclose(x[1], 0.0),
        name="y_symmetry",
        tag=3,
    )
    z_symmetry = mesh.boundary(
        domain,
        lambda x: np.isclose(x[2], 0.0),
        name="z_symmetry",
        tag=4,
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.005)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=(steps.fixed(4) if incrementation is None else incrementation),
        amplitude=amplitude,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def _j2_nonuniform_bending_patch():
    """Slender 3D cantilever driven into a mixed elastic/plastic state."""

    domain = dolfinx_mesh.create_box(
        MPI.COMM_SELF,
        [np.zeros(3), np.array((4.0, 1.0, 0.5))],
        [8, 2, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="j2_nonuniform_bending",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=200.0e3,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2.0e3,
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="clamped", tag=1),
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=4.0, name="driven", tag=2),
        component=1,
        value=0.02,
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(8),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def _j2_external_abaqus_uniaxial_patch():
    """Abaqus rate-independent Mises verification data at its second point."""

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="abaqus_rate_independent_j2",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=200.0e3,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=(220.0 - 200.0) / 0.0009,
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="left", tag=1),
        component=0,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0, name="y_symmetry", tag=2),
        component=1,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0, name="z_symmetry", tag=3),
        component=2,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0, name="right", tag=4),
        component=0,
        value=0.002,
    )
    return model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
