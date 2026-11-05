# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent analytical references, not ASTM or material certification."""

import numpy as np
import pytest

from agentfem import (
    amplitudes, assessments, constitutive, fields, mesh, models, results,
    solvers, steps, studies,
)


def _uniaxial_step(material, *, displacement, duration, count, amplitude=None):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(study=studies.creep_solid(), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(material)
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1),
                  component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4),
              component=0, value=displacement)
    return model.step(target=u, duration=duration, amplitude=amplitude,
                      incrementation=steps.fixed(count), progress=False,
                      solver_options=solvers.newton(relative_tolerance=1.e-10,
                                                    absolute_tolerance=1.e-9))


@pytest.mark.parametrize("exponent", [1., 3.])
def test_global_norton_relaxation_and_dwell_damage_refine_to_closed_form(exponent, record_property):
    # Free lateral contraction: d sigma/dt = -E A (sigma/s_ref)^n.
    # Use the continuous closed form, not AgentFEM's backward-Euler recurrence.
    young, coefficient, reference, initial, end = 200_000., 1.e-4, 100., 200., 2.
    material = constitutive.isotropic_power_law(
        young=young, poisson=.3, density=1., coefficient=coefficient,
        reference_stress=reference, stress_exponent=exponent)
    rate_constant = young * coefficient / reference ** exponent
    def exact(time):
        if exponent == 1.:
            return initial * np.exp(-rate_constant * time)
        return (initial ** (1. - exponent)
                + (exponent - 1.) * rate_constant * time) ** (1. / (1. - exponent))
    # Synthetic rupture law t_r(sigma)=1000*(initial/sigma)^n gives an
    # independent integral: Dc=(initial-sigma_end)/(1000*k*initial^n).
    exact_damage = (initial - exact(end)) / (1000. * rate_constant * initial ** exponent)
    stress_errors, damage_errors, integrated_damage_errors = [], [], []
    for count in (16, 32, 64):
        step = _uniaxial_step(material, displacement=.001, duration=end, count=count)
        step.solve()
        times = np.linspace(0., end, count + 1)
        stress = np.array([initial] + [f.generalized_reaction / .001 for f in step.energy_history])
        assert np.all(np.diff(stress) < 0.)
        assert stress[-1] > exact(end)  # expected BE bias for this relaxation
        stress_errors.append(np.max(np.abs(stress - exact(times))))
        np.testing.assert_allclose(step.state.stress.values[:, 1:, :], 0., atol=1.e-7)
        np.testing.assert_allclose(step.state.creep_strain.values[:, 0, 0],
                                   (initial - stress[-1]) / young, atol=1.e-10)
        history = results.SimulationResult("analytic_relaxation")
        history.add_history("axial_stress", times, stress, unit="MPa")
        history.add_history("temperature", times, np.full(count + 1, 800.), unit="K")
        blocks = assessments.creep_blocks_from_result(
            history, stress_history="axial_stress", temperature_history="temperature",
            dwells=[assessments.DwellInterval(a, b) for a, b in zip(times[:-1], times[1:], strict=True)],
            rupture_time=lambda sigma, temperature: 1000. * (initial / sigma) ** exponent,
            rupture_source="synthetic analytical verification; seconds and MPa",
        )
        damage = assessments.creep_time_fraction(blocks).damage
        assert damage > exact_damage  # maxima are conservative for this monotone law
        damage_errors.append(damage - exact_damage)
        integrated = assessments.creep_damage_from_result(
            history, stress_history="axial_stress", temperature_history="temperature",
            dwells=[assessments.DwellInterval(0., end)],
            rupture_time=lambda sigma, temperature: 1000. * (initial / sigma) ** exponent,
            rupture_source="synthetic analytical verification; seconds and MPa",
            stress_unit="MPa", time_unit="s", stress_measure="signed_axial",
            maximum_source_interval=end / count * (1. + 1.e-12),
        )
        assert exact_damage < integrated.damage < damage
        integrated_damage_errors.append(integrated.damage - exact_damage)
    for errors in (stress_errors, damage_errors, integrated_damage_errors):
        assert 1.7 < errors[0] / errors[1] < 2.2
        assert 1.7 < errors[1] / errors[2] < 2.2
    assert stress_errors[-1] / initial < .01
    assert damage_errors[-1] / exact_damage < .05
    record_property("stress_max_abs_errors_MPa_16_32_64", stress_errors)
    record_property("time_fraction_errors_16_32_64", damage_errors)
    record_property("integrated_time_fraction_errors_16_32_64", integrated_damage_errors)
    record_property("continuous_final_stress_MPa", float(exact(end)))
    record_property("continuous_time_fraction", float(exact_damage))


def test_global_perfect_plastic_cycle_area_and_strain_life_independent_reference(record_property):
    # Classical uniaxial rectangular elastic/perfect-plastic loop:
    # closed-loop work = 4*sigma_y*(eps_a - sigma_y/E).
    young, yield_stress, amplitude = 200_000., 200., .003
    law = constitutive.CreepPlasticityMaterial(
        creep=constitutive.isotropic_power_law(
            young=young, poisson=.3, density=1., coefficient=0., reference_stress=100., stress_exponent=1.),
        plastic=constitutive.J2LinearIsotropicHardening(
            young=young, poisson=.3, yield_stress=yield_stress, hardening_modulus=0.),
    )
    step = _uniaxial_step(law, displacement=amplitude, duration=3., count=72,
                          amplitude=amplitudes.tabular([0., 1., 2., 3.], [0., 1., -1., 1.]))
    step.solve()
    strain = amplitude * np.interp(np.linspace(0., 3., 73), [0., 1., 2., 3.], [0., 1., -1., 1.])
    stress = np.array([0.] + [f.generalized_reaction / amplitude for f in step.energy_history])
    # First ramp initializes the loop; the interval [1,3] is exactly one cycle.
    loop_strain, loop_stress = strain[24:], stress[24:]
    work = np.sum(.5 * (loop_stress[1:] + loop_stress[:-1]) * np.diff(loop_strain))
    assert work == pytest.approx(4. * yield_stress * (amplitude - yield_stress / young), rel=1.e-8)
    np.testing.assert_allclose(stress[[24, 48, 72]], [200., -200., 200.], atol=1.e-6)
    assert step.state.equivalent_creep_strain.global_max() == 0.
    # Equal exponents admit independent analytical inversion: N=.5*(eps_a/C)^(-2).
    curve = constitutive.CoffinMansonBasquinCurve(
        young=young, fatigue_strength_coefficient=1000., fatigue_strength_exponent=-.5,
        fatigue_ductility_coefficient=.5, fatigue_ductility_exponent=-.5,
        calibrated_cycles=(1., 1.e8), stress_unit="MPa", source="synthetic verification")
    life = constitutive.assess_strain_history(
        loop_strain, curve, strain_measure="mechanical_total",
        source="prescribed uniform axial mechanical strain", mean_stress_correction="none")
    expected_cycles = .5 * (amplitude / (1000. / young + .5)) ** -2
    assert sum(c.count for c in life.cycles) == pytest.approx(1.)
    assert life.damage == pytest.approx(1. / expected_cycles, rel=1.e-12)
    record_property("closed_cycle_work_MPa", float(work))
    record_property("strain_life_damage", float(life.damage))
    record_property("reference_cycles_to_failure", float(expected_cycles))
