from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import benchmarks, constitutive, fields, mesh, models, results, studies
from agentfem.constitutive import creep, fatigue, hyperelasticity, plasticity
from agentfem.materials.properties import (
    ElasticAnisotropic2DProperties,
    ElasticIsotropicProperties,
)


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
    assert stats["maximum"] >= 0.05 - 1.0e-12
    assert solved.metadata["solve"]["converged"] is True
    assert any(
        item.name == "neo_hookean_finite_strain_static"
        for item in models.step_providers()
    )


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
        == "material_point_verified"
    )
    assert constitutive.capability("stress_life_fatigue").maturity == "postprocessor"
    assert benchmarks.list_benchmarks(capability="j2_plasticity")[0].level == (
        "material_point"
    )
