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
    assert partial.checkpoints[0].schema == "agentfem.j2-step-checkpoint.v3"
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
    assert {"S", "PE", "PEEQ", "RF"} <= set(simulation.fields)
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
    step.run()

    assert step.procedure.algorithm == "newmark"
    assert np.max(np.abs(step.state.u.value.x.array)) > 0.0
    simulation = step.solve_result()
    time_events = [
        event for event in step.execution_events
        if event.kind == "time_increment"
    ]
    assert len(time_events) == 3
    assert simulation.histories["accepted_increment"].latest == 3.0
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
    model.step(
        target=displacement,
        K=model.stiffness(displacement),
        F=thermal_force,
    ).solve()

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
    velocity = fields.displacement(domain).value
    energy = diagnostics.mechanical_energy(
        mass=model.mass(displacement, material),
        stiffness=stiffness,
        displacement=displacement.value,
        velocity=velocity,
    )

    np.testing.assert_allclose(total_x, -2.0, rtol=1.0e-10, atol=1.0e-10)
    assert energy.kinetic == 0.0
    assert energy.strain > 0.0
    assert energy.total == energy.strain


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


def _j2_uniaxial_patch(*, amplitude=None, incrementation=None):
    """Homogeneous bar with free Poisson contraction and removed rigid modes."""

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
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
