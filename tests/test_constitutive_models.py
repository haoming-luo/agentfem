from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    amplitudes,
    benchmarks,
    constitutive,
    fields,
    mesh,
    models,
    problems,
    results,
    steps,
    studies,
)
from agentfem.constitutive import creep, fatigue, hyperelasticity, plasticity
from agentfem.materials.properties import (
    ElasticAnisotropic2DProperties,
    ElasticIsotropicProperties,
)


def test_shared_quadrature_transaction_commits_and_rolls_back_atomically():
    class State:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        @property
        def values(self):
            return self._values.copy()

        def assign(self, values):
            self._values[:] = values

    committed_a = State([0.0, 0.0])
    committed_b = State([1.0])
    trial_a = State([2.0, 3.0])
    trial_b = State([4.0])
    transaction = constitutive.QuadratureTransaction(
        committed={"a": committed_a, "b": committed_b},
        trial={"a": trial_a, "b": trial_b},
        schema="agentfem.test-state",
    )

    transaction.rollback()
    np.testing.assert_allclose(trial_a.values, [0.0, 0.0])
    np.testing.assert_allclose(trial_b.values, [1.0])
    trial_a.assign([5.0, 6.0])
    trial_b.assign([7.0])
    transaction.commit()
    snapshot = transaction.snapshot()

    np.testing.assert_allclose(committed_a.values, [5.0, 6.0])
    np.testing.assert_allclose(committed_b.values, [7.0])
    committed_a.assign([99.0, 99.0])
    transaction.restore(snapshot)
    np.testing.assert_allclose(committed_a.values, [5.0, 6.0])
    np.testing.assert_allclose(trial_a.values, [5.0, 6.0])


def test_integration_point_recovery_is_weighted_and_traceable():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    source = constitutive.QuadratureField.create(
        domain,
        name="PEEQ",
        degree=3,
    )
    values = np.arange(source.values.size, dtype=float).reshape(source.values.shape)
    source.assign(values)

    recovered = results.recover_integration_point_field(
        source,
        name="PEEQ_CELL",
    )
    expected = np.einsum(
        "p,cp->c",
        source.weights / np.sum(source.weights),
        values.reshape((-1, len(source.points))),
    )

    np.testing.assert_allclose(recovered.field.x.array, expected)
    assert recovered.location == "cells"
    assert recovered.processing["source_position"] == "integration_points"
    assert recovered.processing["method"] == "quadrature_weighted_cell_average"
    assert recovered.processing["material_boundary_averaging"] is False


def test_linear_materials_reject_nonphysical_parameters():
    with pytest.raises(ValueError, match="young"):
        ElasticIsotropicProperties("bad", 0.0, 1.0, 0.3)
    with pytest.raises(ValueError, match="poisson"):
        ElasticIsotropicProperties("bad", 1.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="positive definite"):
        ElasticAnisotropic2DProperties(
            "bad",
            np.diag([1.0, 1.0, -1.0]),
            1.0,
        )


def test_neo_hookean_nominal_stress_is_energy_derivative():
    material = hyperelasticity.neo_hookean(young=2.0e6, poisson=0.3)
    stretches = np.array([1.2, 0.9, 1.05])
    analytical = hyperelasticity.principal_nominal_stress(stretches, material)
    step = 1.0e-7
    numerical = np.empty(3)
    for index in range(3):
        plus = stretches.copy()
        minus = stretches.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            hyperelasticity.principal_energy_density(plus, material)
            - hyperelasticity.principal_energy_density(minus, material)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-7)
    np.testing.assert_allclose(
        hyperelasticity.principal_nominal_stress([1.0, 1.0, 1.0], material),
        0.0,
        atol=1.0e-12,
    )
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.neo_hookean_release"
    )
    actual = {
        "principal_nominal_stress": analytical,
        "strain_energy_density": hyperelasticity.principal_energy_density(
            stretches,
            material,
        ),
        "deformation_jacobian": float(np.prod(stretches)),
    }
    assert all(golden.verify(actual).values())
    assert material.as_dict()["maturity"] == "fem_form_available"


def test_neo_hookean_model_step_solves_a_displacement_controlled_patch():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
    )
    model = models.create(study=study, mesh=domain, name="hyperelastic_patch")
    displacement = model.field(fields.displacement(domain, degree=1))
    model.material(
        hyperelasticity.neo_hookean(young=2.0e6, poisson=0.3)
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
    model.fix(displacement, on=right, component=0, value=0.05)

    problem = model.step(
        target=displacement,
        material=model.materials[0].item,
    )
    solved = problem.solve_result()

    stats = results.dof_statistics(solved.field("Displacement"))
    assert problem.last_solve_info.converged
    assert problem.last_solve_info.as_dict()["kind"] == "nonlinear_load_path"
    assert all(
        item.checks["minimum_quadrature_J"] > 0.0
        for item in problem.last_solve_info.increments
    )
    assert problem.snapshots[0].load_factor == 0.0
    assert problem.snapshots[-1].load_factor == 1.0
    assert any(item.kind == "step_completed" for item in problem.execution_events)
    assert stats["maximum"] >= 0.05 - 1.0e-12
    assert solved.metadata["solve"]["converged"] is True
    assert any(
        item.name == "neo_hookean_finite_strain_static"
        for item in models.step_providers()
    )


def test_plane_stress_neo_hookean_standard_step_uses_condensed_membrane_energy():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_stress",
            nonlinear=True,
        ),
        mesh=domain,
        name="plane_stress_hyperelastic_patch",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        hyperelasticity.neo_hookean_plane_stress(
            young=2.0e6,
            poisson=0.49,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left"),
    )
    model.prescribe(
        displacement,
        0.02,
        component=0,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right"),
    )
    step = model.step(target=displacement, material=material, progress=False)
    step.solve()

    assert step.last_solve_info.converged
    assert min(
        item.checks["minimum_quadrature_J"]
        for item in step.last_solve_info.increments
    ) > 0.0
    assert float(np.max(displacement.value.x.array)) == pytest.approx(0.02)


def test_neo_hookean_standard_path_rolls_back_and_cuts_back_failed_attempt(monkeypatch):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_strain",
            nonlinear=True,
        ),
        mesh=domain,
        name="hyperelastic_forced_cutback",
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    material = model.material(
        hyperelasticity.neo_hookean(young=2.0e6, poisson=0.3)
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left"),
    )
    model.prescribe(
        displacement,
        0.05,
        component=0,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right"),
    )
    original_solve = problems.solve_nonlinear_problem
    calls = 0

    def fail_first_attempt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            solution = args[1]
            solution.x.array[:] = 1.0e30
            solution.x.scatter_forward()
            raise RuntimeError("forced first-attempt failure")
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(problems, "solve_nonlinear_problem", fail_first_attempt)
    step = model.step(target=displacement, material=material, progress=False)
    step.solve()

    assert step.last_solve_info.converged
    assert any(item.kind == "increment_cutback" for item in step.execution_events)
    assert step.last_solve_info.attempts[0].converged is False
    assert np.max(np.abs(displacement.value.x.array)) < 1.0


def test_neo_hookean_standard_path_scales_natural_loads_by_increment():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_strain",
            nonlinear=True,
        ),
        mesh=domain,
        name="hyperelastic_traction_path",
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    material = model.material(
        hyperelasticity.neo_hookean(young=2.0e6, poisson=0.3)
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left"),
    )
    model.traction(
        (1.0e4, 0.0),
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right"),
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(3),
        progress=False,
    )
    step.solve()

    assert step.last_solve_info.converged
    assert [item.load_factor for item in step.last_solve_info.increments] == [
        pytest.approx(1.0 / 3.0),
        pytest.approx(2.0 / 3.0),
        pytest.approx(1.0),
    ]
    assert float(np.max(displacement.value.x.array)) > 0.0


def test_neo_hookean_natural_load_amplitude_follows_normalized_step_time():
    visited = []
    history = amplitudes.Amplitude(
        name="traction_ramp",
        kind="test_ramp",
        value=lambda factor: visited.append(float(factor)) or float(factor),
    )
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_strain",
            nonlinear=True,
        ),
        mesh=domain,
        name="hyperelastic_amplitude_path",
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    material = model.material(
        hyperelasticity.neo_hookean(young=2.0e6, poisson=0.3)
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left"),
    )
    driven = model.traction(
        (1.0e4, 0.0),
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right"),
        amplitude=history,
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(3),
        progress=False,
    )
    step.solve()

    assert step.last_solve_info.converged
    assert visited[-4:] == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert float(driven.scale.value) == pytest.approx(1.0)
    assert float(np.max(displacement.value.x.array)) > 0.0


def test_step_provider_registry_is_extensible_and_deterministic():
    registry = models.StepProviderRegistry()
    fallback = models.StepProvider(
        name="fallback",
        analyses=("custom",),
        accepts=lambda model, request: True,
        lower=lambda model, request: "fallback",
        priority=0,
    )
    preferred = models.StepProvider(
        name="preferred",
        analyses=("custom",),
        accepts=lambda model, request: request.options["enabled"],
        lower=lambda model, request: "preferred",
        priority=10,
    )
    registry.register(fallback)
    registry.register(preferred)
    request = __import__(
        "agentfem.step_providers",
        fromlist=["StepRequest"],
    ).StepRequest("custom", None, {"enabled": True})

    assert registry.lower(object(), request) == "preferred"
    assert [item.name for item in registry.providers()] == [
        "preferred",
        "fallback",
    ]
    with pytest.raises(TypeError):
        request.options["enabled"] = False


def test_model_pressure_and_symmetry_keep_engineering_semantics_visible():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
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

    symmetry = model.symmetry(displacement, on=left, normal_axis="x")
    pressure = model.pressure(
        2.0,
        on=right,
        configuration="current",
        displacement=displacement,
    )

    assert len(symmetry.bcs) == 1
    assert symmetry.summary()["dirichlet"][0]["name"] == "symmetry_x_component_0"
    assert pressure.summary()["configuration"] == "current"
    assert pressure.value.ufl_shape == (2,)


def test_j2_radial_return_lands_on_the_hardened_yield_surface():
    material = plasticity.J2LinearIsotropicHardening(
        young=200.0e3,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2.0e3,
    )
    strain = np.diag([0.004, -0.002, -0.002])
    update = material.update(strain)

    assert not update.elastic
    assert update.plastic_multiplier_increment > 0.0
    np.testing.assert_allclose(np.trace(update.state.plastic_strain), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        plasticity.von_mises(update.stress),
        material.current_yield_stress(update.state.equivalent_plastic_strain),
        rtol=1.0e-12,
    )


def test_uniaxial_plasticity_matches_bilinear_closed_form():
    material = plasticity.J2LinearIsotropicHardening(
        young=200.0e3,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2.0e3,
    )
    total_strain = 0.005
    stress, state = plasticity.update_uniaxial(total_strain, material)
    expected_increment = (
        material.young * total_strain - material.yield_stress
    ) / (material.young + material.hardening_modulus)
    expected_stress = material.yield_stress + material.hardening_modulus * expected_increment

    np.testing.assert_allclose(state.equivalent_plastic_strain, expected_increment)
    np.testing.assert_allclose(stress, expected_stress)


def test_power_law_creep_matches_constant_stress_and_relaxation_solutions():
    law = creep.PowerLawCreep(
        coefficient=1.6e-16,
        stress_exponent=5.0,
        time_exponent=-0.2,
        reference_stress=1.0,
        reference_time=1.0,
    )
    stress = 100.0
    duration = 1000.0
    expected = 1.6e-16 / 0.8 * stress**5 * duration**0.8
    np.testing.assert_allclose(
        law.constant_stress_strain(stress, duration),
        expected,
        rtol=1.0e-14,
    )
    relaxed = law.relaxation_stress(
        initial_stress=stress,
        young=138.0e3,
        time=duration,
    )
    assert 0.0 < relaxed < stress


def test_arrhenius_material_update_accelerates_with_temperature():
    material = constitutive.isotropic_arrhenius_power_law(
        young=200.0e3,
        poisson=0.3,
        density=1.0,
        coefficient=1.0e-8,
        stress_exponent=3.0,
        activation_energy=200.0e3,
        reference_temperature=800.0,
        reference_stress=100.0,
    )
    strain = np.diag((0.002, -0.001, -0.001))

    reference = material.update(
        strain, time_start=0.0, time_end=1.0, temperature=800.0,
    )
    hot = material.update(
        strain, time_start=0.0, time_end=1.0, temperature=900.0,
    )

    assert hot.equivalent_increment > reference.equivalent_increment > 0.0
    assert material.as_dict()["temperature_dependence"]["model"] == (
        "arrhenius_mises_power_law_creep"
    )


def test_mises_creep_tensor_increment_is_deviatoric_and_has_requested_equivalent():
    law = creep.PowerLawCreep(
        coefficient=1.0e-8,
        stress_exponent=3.0,
        reference_stress=100.0,
    )
    stress = np.diag([120.0, 0.0, 0.0])
    increment = law.tensor_increment(stress, 0.0, 10.0)

    np.testing.assert_allclose(np.trace(increment), 0.0, atol=1.0e-16)
    equivalent = np.sqrt(2.0 / 3.0 * np.tensordot(increment, increment))
    np.testing.assert_allclose(
        equivalent,
        law.constant_stress_increment(
            plasticity.von_mises(stress),
            0.0,
            10.0,
        ),
    )


def test_creep_history_driver_uses_exact_interval_integrals():
    law = creep.PowerLawCreep(
        coefficient=2.0e-8,
        stress_exponent=3.0,
        time_exponent=0.5,
        reference_stress=100.0,
    )
    history = creep.integrate_stress_history(
        law,
        times=(0.0, 2.0, 5.0),
        interval_stresses=(100.0, 50.0),
    )
    expected = (
        law.constant_stress_increment(100.0, 0.0, 2.0)
        + law.constant_stress_increment(50.0, 2.0, 5.0)
    )

    assert history.final_equivalent_strain == pytest.approx(expected)
    assert history.creep_strain is None
    assert history.as_dict()["tensor_history"] is False


def test_kachanov_rabotnov_exact_update_is_subdivision_invariant():
    law = creep.KachanovRabotnovCreep(
        creep_coefficient=2.0e-5,
        creep_exponent=4.0,
        damage_coefficient=1.0e-4,
        damage_exponent=5.0,
        damage_power=3.0,
        reference_stress=100.0,
        failure_damage=0.99,
    )
    one = law.update(80.0, 100.0)
    split = creep.CreepDamageState()
    for _ in range(10):
        split = law.update(80.0, 10.0, split).state

    np.testing.assert_allclose(split.damage, one.state.damage, rtol=2.0e-13)
    np.testing.assert_allclose(
        split.equivalent_creep_strain,
        one.state.equivalent_creep_strain,
        rtol=2.0e-13,
    )
    assert law.rupture_time(120.0) < law.rupture_time(80.0)
    assert law.as_dict()["maturity"] == "material_point_verified"


def test_kachanov_rabotnov_tensor_flow_is_deviatoric_and_damage_accumulates():
    law = creep.KachanovRabotnovCreep(
        creep_coefficient=1.0e-5,
        creep_exponent=3.0,
        damage_coefficient=2.0e-5,
        damage_exponent=4.0,
        damage_power=2.0,
        reference_stress=100.0,
    )
    update = law.update(np.diag([120.0, 0.0, 0.0]), 20.0)

    assert update.state.damage > 0.0
    assert update.equivalent_increment > 0.0
    np.testing.assert_allclose(np.trace(update.state.creep_strain), 0.0, atol=1.0e-16)
    equivalent = np.sqrt(
        2.0 / 3.0 * np.tensordot(update.state.creep_strain, update.state.creep_strain)
    )
    np.testing.assert_allclose(equivalent, update.state.equivalent_creep_strain)


def test_modified_theta_projection_recovers_synthetic_creep_curve():
    reference = creep.ModifiedThetaProjection(
        initial_strain=0.001,
        primary_strain=0.012,
        tertiary_strain=0.0015,
        rate=0.02,
    )
    times = np.linspace(0.0, 100.0, 41)
    fitted = creep.ModifiedThetaProjection.fit(
        times,
        reference.strain(times),
        rate_bounds=(0.019, 0.021),
        candidates=401,
    )

    np.testing.assert_allclose(fitted.strain(times), reference.strain(times), rtol=2.0e-4)
    assert fitted.fit_rmse < 1.0e-6
    target = float(reference.strain(70.0))
    assert reference.time_to_strain(target, maximum_time=100.0) == pytest.approx(70.0)
    assert reference.as_dict()["maturity"] == "curve_projection_verified"


def test_sinh_creep_has_associative_mises_tensor_increment():
    law = creep.SinhCreep(
        coefficient=2.0e-6,
        stress_scale=80.0,
        exponent=2.0,
    )
    stress = np.diag([120.0, 0.0, 0.0])
    increment = law.tensor_increment(stress, 50.0)
    equivalent = np.sqrt(2.0 / 3.0 * np.tensordot(increment, increment))

    np.testing.assert_allclose(
        equivalent,
        law.equivalent_rate(plasticity.von_mises(stress)) * 50.0,
    )
    assert law.equivalent_rate(120.0) > law.equivalent_rate(80.0)


def test_j2_uniaxial_complete_loading_unloading_reverse_path():
    material = plasticity.J2LinearIsotropicHardening(
        young=200.0e3,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2.0e3,
    )
    state = plasticity.UniaxialPlasticState()
    stresses = []
    for strain in (0.0, 0.0005, 0.003, 0.005, 0.002, 0.0, -0.003):
        stress, state = plasticity.update_uniaxial(strain, material, state)
        stresses.append(stress)

    assert stresses[1] == pytest.approx(material.young * 0.0005)
    assert state.equivalent_plastic_strain > 0.0
    assert stresses[4] < stresses[3]
    assert stresses[-1] < 0.0


def test_basquin_and_miner_block_loading():
    curve = fatigue.BasquinCurve(
        fatigue_strength_coefficient=1000.0,
        fatigue_strength_exponent=-0.1,
    )
    target_cycles = 1.0e6
    amplitude = curve.stress_amplitude(target_cycles)
    np.testing.assert_allclose(curve.cycles_to_failure(amplitude), target_cycles)

    blocks = (
        fatigue.FatigueBlock(amplitude, 2.5e5, "first"),
        fatigue.FatigueBlock(amplitude, 2.5e5, "second"),
    )
    np.testing.assert_allclose(fatigue.miner_damage(blocks, curve), 0.5)
    np.testing.assert_allclose(fatigue.life_scale_factor(blocks, curve), 2.0)
    assert constitutive.PowerLawCreep is creep.PowerLawCreep


def test_tabulated_sn_curve_uses_log_log_interpolation_and_explicit_bounds():
    curve = fatigue.TabulatedSNCurve(
        stress_amplitudes=(500.0, 300.0, 200.0),
        cycles=(1.0e4, 1.0e5, 1.0e6),
    )
    midpoint = np.sqrt(300.0 * 200.0)
    np.testing.assert_allclose(curve.cycles_to_failure(midpoint), np.sqrt(1.0e5 * 1.0e6))
    with pytest.raises(ValueError, match="outside"):
        curve.cycles_to_failure(100.0)


def test_rainflow_history_to_miner_damage_and_goodman_correction():
    curve = fatigue.BasquinCurve(
        fatigue_strength_coefficient=1000.0,
        fatigue_strength_exponent=-0.1,
    )
    history = [0.0, 100.0, 0.0, 100.0, 0.0]
    cycles = fatigue.rainflow_cycles(history)

    assert sum(cycle.count for cycle in cycles) == pytest.approx(2.0)
    assert all(cycle.stress_amplitude == pytest.approx(50.0) for cycle in cycles)
    expected = 2.0 / curve.cycles_to_failure(50.0)
    assert fatigue.damage_from_history(history, curve) == pytest.approx(expected)
    assert fatigue.goodman_amplitude(50.0, 100.0, 500.0) == pytest.approx(62.5)


def test_fatigue_assessment_consumes_named_simulation_history():
    curve = fatigue.BasquinCurve(
        fatigue_strength_coefficient=1000.0,
        fatigue_strength_exponent=-0.1,
    )
    simulation = results.SimulationResult("bracket")
    simulation.add_history(
        "hotspot_stress",
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 100.0, 0.0, 100.0, 0.0],
        unit="MPa",
    )
    assessment = fatigue.assess_result_history(
        simulation,
        "hotspot_stress",
        curve,
    )

    assert assessment.source == "bracket:hotspot_stress"
    assert assessment.damage == pytest.approx(
        fatigue.damage_from_history(
            simulation.histories["hotspot_stress"].values,
            curve,
        )
    )
    assert assessment.repeated_history_life == pytest.approx(
        1.0 / assessment.damage
    )


def test_capability_and_benchmark_catalogs_state_the_actual_maturity():
    assert constitutive.capability("neo_hookean").maturity == "fem_integrated"
    assert (
        constitutive.capability("j2_plasticity").maturity
        == "fem_integrated"
    )
    assert constitutive.capability("stress_life_fatigue").maturity == "postprocessor"
    assert benchmarks.list_benchmarks(capability="j2_plasticity")[0].level == (
        "material_point"
    )
    assert any(
        item.level == "finite_element"
        for item in benchmarks.list_benchmarks(capability="j2_plasticity")
    )
