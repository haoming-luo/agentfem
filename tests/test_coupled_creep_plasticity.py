# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from agentfem import constitutive
from agentfem.constitutive._creep_plasticity import CreepPlasticityMaterial
from agentfem.constitutive.plasticity import deviatoric, von_mises


def _material(chaboche=False, coefficient=1.e-5, yield_stress=200., sinh=False):
    plastic = (constitutive.ChabocheCombinedHardening(
        young=200_000., poisson=0.3, yield_stress=yield_stress,
        backstress_moduli=(12_000., 3_000.), dynamic_recovery=(100., 8.),
        isotropic_saturation=60., isotropic_rate=10.,
    ) if chaboche else constitutive.J2LinearIsotropicHardening(
        young=200_000., poisson=0.3, yield_stress=yield_stress, hardening_modulus=2_000.,
    ))
    creep = (constitutive.isotropic_sinh(
        young=200_000., poisson=0.3, density=1., coefficient=coefficient,
        stress_scale=100., exponent=2.,
    ) if sinh else constitutive.isotropic_power_law(
        young=200_000., poisson=0.3, density=1., coefficient=coefficient,
        reference_stress=100., stress_exponent=3.,
    ))
    return CreepPlasticityMaterial(creep=creep, plastic=plastic)


@pytest.mark.parametrize("chaboche", [False, True])
@pytest.mark.parametrize("sinh", [False, True])
def test_combined_nonproportional_history_equations_and_tangent(chaboche, sinh):
    law = _material(chaboche=chaboche, sinh=sinh)
    state = law.initial_state()
    path = [np.diag([0.004, -0.002, -0.002]),
            np.array([[0.002, 0.003, 0.], [0.003, -0.001, 0.], [0., 0., -0.001]]),
            np.diag([-0.003, 0.001, 0.002]),
            np.diag([-0.003, 0.001, 0.002])]
    direction = np.array([[0.7, 0.1, -0.05], [0.1, -0.2, 0.03], [-0.05, 0.03, 0.4]])
    for i, strain in enumerate(path):
        old_plastic = state.plastic.plastic_strain.copy()
        old_creep = state.creep.creep_strain.copy()
        arguments = dict(state=state, time_start=i * 2., time_end=(i + 1) * 2.)
        update = law.update(strain, **arguments)
        delta = update.state.creep.creep_strain - state.creep.creep_strain
        q = von_mises(update.stress)
        rate = law.creep.creep.equivalent_rate(q, (i + 1) * 2.)
        np.testing.assert_allclose(delta, 2 * rate * 1.5 * deviatoric(update.stress) / q, atol=1.e-12)
        # Re-evaluate the old-state plastic return at the simultaneous creep
        # strain: neither constitutive branch is left at a different stress.
        plastic = law.plastic.update(strain - update.state.creep.creep_strain, state.plastic)
        np.testing.assert_allclose(update.stress, plastic.stress, atol=5.e-8)
        eps = 1.e-7
        numerical = (law.update(strain + eps * direction, **arguments).stress
                     - law.update(strain - eps * direction, **arguments).stress) / (2 * eps)
        exact = np.einsum("ijkl,kl->ij", update.algorithmic_tangent, direction)
        np.testing.assert_allclose(exact, numerical, rtol=3.e-6, atol=0.01)
        assert update.creep_work >= 0.
        assert abs(update.plastic_energy.balance_residual) < 1.e-8
        np.testing.assert_array_equal(state.plastic.plastic_strain, old_plastic)
        np.testing.assert_array_equal(state.creep.creep_strain, old_creep)
        state = update.state


@pytest.mark.parametrize("chaboche", [False, True])
def test_zero_creep_reduces_exactly_to_existing_plasticity(chaboche):
    law = _material(chaboche=chaboche, coefficient=0.)
    strain = np.diag([0.004, -0.002, -0.002])
    update = law.update(strain, time_start=0., time_end=1.)
    expected = law.plastic.update(strain)
    np.testing.assert_allclose(update.stress, expected.stress, atol=1.e-8)
    np.testing.assert_allclose(update.algorithmic_tangent, expected.algorithmic_tangent, atol=1.e-7)
    assert update.creep_increment == 0.


def test_inactive_plasticity_reduces_to_existing_creep():
    law = _material(yield_stress=1.e8)
    strain = np.diag([0.004, -0.002, -0.002])
    update = law.update(strain, time_start=0., time_end=1.)
    expected = law.creep.update(strain, time_start=0., time_end=1.)
    np.testing.assert_allclose(update.stress, expected.stress, rtol=1.e-8, atol=1.e-8)
    np.testing.assert_allclose(update.algorithmic_tangent, expected.algorithmic_tangent, rtol=1.e-8)
    assert update.plastic_increment == 0.


def test_failure_does_not_mutate_old_state():
    law = _material(chaboche=True)
    state = law.initial_state()
    with pytest.raises(RuntimeError, match="did not converge"):
        law.update(np.diag([0.004, -0.002, -0.002]), time_start=0., time_end=1.,
                   state=state, maximum_iterations=1)
    np.testing.assert_array_equal(state.creep.creep_strain, 0.)
    np.testing.assert_array_equal(state.plastic.plastic_strain, 0.)
    np.testing.assert_array_equal(state.plastic.backstresses, 0.)


def test_combined_isotropic_solution_matches_independent_scalar_equations():
    # Coaxial J2 with linear hardening reduces to two scalar equations. Solve
    # the endpoint equivalent stress independently with scipy's bracket root.
    from scipy.optimize import brentq

    law = _material(coefficient=1.e-7)
    strain = np.diag([0.004, -0.002, -0.002])
    qtrial = 3 * law.plastic.shear_modulus * 0.004
    shear3 = 3 * law.plastic.shear_modulus
    hardening = law.plastic.hardening_modulus
    yield_stress = law.plastic.yield_stress
    expected_q = brentq(lambda q: q + shear3 * (
        law.creep.creep.equivalent_rate(q, 1.) + max(0., (q - yield_stress) / hardening)
    ) - qtrial, 0., qtrial, xtol=1.e-12)
    update = law.update(strain, time_start=0., time_end=1.)
    assert update.plastic_increment > 0.
    assert von_mises(update.stress) == pytest.approx(expected_q, rel=1.e-10)
    assert update.plastic_increment == pytest.approx((expected_q - yield_stress) / hardening, rel=1.e-9)


def _structure(*, chaboche=True, incrementation=None, thermal=False):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    from agentfem import amplitudes, fields, histories, mesh, models, solvers, steps, studies

    domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(study=studies.creep_solid(), mesh=domain, name="coupled_hold")
    u = model.field(fields.displacement(domain))
    law = _material(chaboche=chaboche, coefficient=1.e-6)
    temperature = None
    if thermal:
        properties = constitutive.thermoelastic(
            young=200_000., poisson=0.3, density=1., thermal_expansion=1.e-5,
            conductivity=1., specific_heat=1., reference_temperature=800.)
        law = CreepPlasticityMaterial(
            creep=constitutive.isotropic_arrhenius_power_law(
                elastic=properties, coefficient=1.e-6, stress_exponent=3., reference_stress=100.,
                activation_energy=120_000., reference_temperature=800.), plastic=law.plastic)
        field = model.field(fields.temperature(domain, value=800.))
        temperature = histories.temperature(field)
        temperature.record(0.)
        field.value.x.array[:] = 900.
        field.value.x.scatter_forward()
        temperature.record(4.)
    selected = model.material(law)
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1),
                  component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=0.004)
    amplitude = amplitudes.tabular([0., 1., 2., 3., 4.], [0., 1., 1., -0.5, -0.5])
    return model.step(target=u, material=selected, duration=4., amplitude=amplitude,
                      incrementation=steps.fixed(16) if incrementation is None else incrementation,
                      progress=False, temperature=temperature, solver_options=solvers.newton(
                          relative_tolerance=1.e-9, absolute_tolerance=1.e-9,
                          maximum_iterations=25, line_search="backtracking"))


@pytest.mark.parametrize("chaboche", [False, True])
@pytest.mark.parametrize("portable", [False, True])
def test_coupled_global_hold_reverse_restart_and_energy(tmp_path, chaboche, portable):
    full = _structure(chaboche=chaboche)
    result = full.solve_result()
    assert result.fields["PE"].location == "quadrature_points"
    assert result.fields["CE"].location == "quadrature_points"
    assert result.fields["PEEQ_CELL"].location == "cells"
    assert full.state.plastic_fields["equivalent_plastic_strain"].global_max() > 0.
    assert full.state.equivalent_creep_strain.global_max() > 0.
    for frame in full.energy_history:
        assert frame.plastic_work == pytest.approx(
            frame.plastic_stored_energy + frame.plastic_dissipation + frame.plastic_backward_euler_dissipation,
            rel=1.e-8, abs=1.e-10)
        assert frame.plastic_dissipation >= 0.
    paused = _structure(chaboche=chaboche)
    paused.solve(until=2.)
    checkpoint = paused.save_checkpoint(tmp_path / "joint.npz", portable=portable)
    resumed = _structure(chaboche=chaboche)
    resumed.load_checkpoint(checkpoint)
    resumed.solve_result()
    for key, expected in full.state.snapshot().items():
        np.testing.assert_allclose(resumed.state.snapshot()[key], expected, atol=1.e-9, rtol=1.e-8)
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, atol=1.e-10)
    assert resumed.energy_history[-1].internal_energy == pytest.approx(full.energy_history[-1].internal_energy, rel=1.e-8)


def test_coupled_global_cycle_matches_independent_uniaxial_return_equations():
    from scipy.optimize import brentq

    step = _structure(chaboche=False)
    step.solve_result()
    p, ce, pe = 0., 0., 0.
    reference_stress = []
    rate = step.material.creep.creep.equivalent_rate
    for time in np.arange(0.25, 4.01, 0.25):
        strain = 0.004 * np.interp(time, [0., 1., 2., 3., 4.], [0., 1., 1., -0.5, -0.5])
        trial = 200_000. * (strain - ce - pe)
        sign = np.sign(trial)
        radius = 200. + 2_000. * p
        q = brentq(lambda q, time=time, radius=radius, trial=trial: q / 200_000. + 0.25 * rate(q, time)
                   + max(0., (q - radius) / 2_000.) - abs(trial) / 200_000.,
                   0., abs(trial), xtol=1.e-11)
        dp = max(0., (q - radius) / 2_000.)
        ce += sign * 0.25 * rate(q, time)
        pe += sign * dp
        p += dp
        reference_stress.append(sign * q)
    # Unit area and displacement target 0.004: generalized reaction = .004 RF.
    np.testing.assert_allclose([f.generalized_reaction / 0.004 for f in step.energy_history],
                               reference_stress, rtol=2.e-7, atol=1.e-6)
    np.testing.assert_allclose(step.state.plastic_fields["equivalent_plastic_strain"].values, p, atol=1.e-10)
    np.testing.assert_allclose(step.state.creep_strain.values[:, 0, 0], ce, atol=1.e-10)


def test_coupled_quadrature_late_point_failure_is_atomic(monkeypatch):
    step = _structure()
    state = step.state
    strain = np.broadcast_to(np.diag([0.004, -0.002, -0.002]), state.creep_strain.values.shape)
    state.update(strain, step.material, time_start=0., time_end=0.25)
    state.commit()
    before = state.snapshot()
    before_trial = {key: value.values.copy() for key, value in state.transaction.trial.items()}
    before_stress = state.stress.values.copy()
    original = CreepPlasticityMaterial.update
    count = 0
    def fail_late(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("injected late point failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CreepPlasticityMaterial, "update", fail_late)
    with pytest.raises(RuntimeError, match="local point 2"):
        state.update(-strain, step.material, time_start=0.25, time_end=0.5)
    for key in before:
        np.testing.assert_array_equal(state.snapshot()[key], before[key])
        np.testing.assert_array_equal(state.transaction.trial[key].values, before_trial[key])
    np.testing.assert_array_equal(state.stress.values, before_stress)


def test_coupled_free_thermal_expansion_and_arrhenius_rate():
    thermal = constitutive.thermoelastic(young=200_000., poisson=0.3, density=1.,
                                        thermal_expansion=1.e-5, reference_temperature=800.,
                                        conductivity=1., specific_heat=1.)
    creep = constitutive.isotropic_arrhenius_power_law(
        elastic=thermal, coefficient=1.e-6, stress_exponent=3., reference_stress=100.,
        activation_energy=120_000., reference_temperature=800.,
    )
    coupled = CreepPlasticityMaterial(creep=creep, plastic=_material().plastic)
    free = coupled.update(0.001 * np.eye(3), time_start=0., time_end=1., temperature=900.)
    np.testing.assert_allclose(free.stress, 0., atol=1.e-9)
    assert free.creep_increment == 0.
    assert free.plastic_increment == 0.
    strain = np.diag([0.004, -0.002, -0.002])
    hot = coupled.update(strain + 0.001 * np.eye(3), time_start=0., time_end=1., temperature=900.)
    warm = coupled.update(strain, time_start=0., time_end=1., temperature=800.)
    assert hot.creep_increment > warm.creep_increment


def test_coupled_global_inelastic_limit_cuts_back_plasticity():
    from agentfem import steps

    limit = 5.e-4
    control = steps.automatic(initial=0.25, minimum=1.e-5, maximum=0.25,
                              maximum_inelastic_increment=limit, max_increments=150,
                              max_cutbacks=15)
    step = _structure(incrementation=control)
    step.solve()
    rejected = [entry for entry in step.attempted_increments if not entry.converged]
    assert rejected
    assert any(entry.maximum_plastic_increment > limit for entry in rejected)
    assert all(max(entry.maximum_plastic_increment, entry.maximum_creep_increment) <= limit
               for entry in step.accepted_increments)
    assert step.last_solve_info.completed_step


@pytest.mark.parametrize("portable", [False, True])
def test_coupled_checkpoint_late_corruption_preserves_both_histories(tmp_path, portable):
    import json

    source = _structure()
    source.solve(until=2.)
    checkpoint = source.save_checkpoint(tmp_path / "joint.npz", portable=portable)
    if portable:
        payload = json.loads(checkpoint.read_text())
        payload["energy_history"] = [{"broken": True}]
        checkpoint.write_text(json.dumps(payload))
    else:
        with np.load(checkpoint, allow_pickle=False) as data:
            payload = {key: data[key].copy() for key in data.files}
        payload["energy_history"] = json.dumps([{"broken": True}])
        np.savez(checkpoint, **payload)
    receiver = _structure()
    receiver.solve(until=1.)
    old = receiver.state.snapshot()
    displacement = receiver.solution.x.array.copy()
    with pytest.raises((KeyError, ValueError, TypeError)):
        receiver.load_checkpoint(checkpoint)
    assert receiver.accepted_time == 1.
    np.testing.assert_array_equal(receiver.solution.x.array, displacement)
    for key, values in old.items():
        np.testing.assert_array_equal(receiver.state.snapshot()[key], values)
    assert receiver.checkpoint_capabilities().rank_count_portability == "unsupported"


def test_coupled_global_thermal_history_restart(tmp_path):
    full = _structure(thermal=True)
    result = full.solve_result()
    assert result.histories["maximum_creep_temperature"].latest == pytest.approx(900.)
    assert full.temperature.active_time == 4.
    partial = _structure(thermal=True)
    partial.solve(until=2.)
    checkpoint = partial.save_checkpoint(tmp_path / "thermal.npz", portable=True)
    resumed = _structure(thermal=True)
    resumed.load_checkpoint(checkpoint)
    assert resumed.temperature.active_time == 2.
    resumed.solve()
    for key, values in full.state.snapshot().items():
        np.testing.assert_allclose(resumed.state.snapshot()[key], values, atol=2.e-9, rtol=2.e-8)
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, atol=1.e-10)


def _linear_creep_ramp_hold(increments, cells=1):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    from agentfem import amplitudes, fields, mesh, models, solvers, steps, studies

    domain = dmesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(study=studies.creep_solid(), mesh=domain)
    u = model.field(fields.displacement(domain))
    material = CreepPlasticityMaterial(
        creep=constitutive.isotropic_power_law(
            young=200_000., poisson=.3, density=1., coefficient=.001,
            reference_stress=100., stress_exponent=1.),
        plastic=constitutive.J2LinearIsotropicHardening(
            young=200_000., poisson=.3, yield_stress=200., hardening_modulus=2_000.),
    )
    model.material(material)
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1),
                  component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=.004)
    return model.step(target=u, duration=2., amplitude=amplitudes.tabular([0., 1., 2.], [0., 1., 1.]),
                      incrementation=steps.fixed(increments), progress=False,
                      solver_options=solvers.newton(relative_tolerance=1.e-10, absolute_tolerance=1.e-10))


def test_global_joint_linear_creep_time_refinement_against_continuous_solution():
    # Independent continuous uniaxial equations: elastic-creep until yield,
    # simultaneous linear isotropic plasticity/creep during the remaining ramp,
    # then elastic-creep relaxation at fixed strain. Not a same-integrator reference.
    young, hardening, rate, strain_rate, yield_stress = 200_000., 2_000., 1.e-5, .004, 200.
    yield_time = -np.log1p(-rate * yield_stress / strain_rate) / (young * rate)
    tangent = young * hardening / (young + hardening)
    peak = strain_rate / rate + (yield_stress - strain_rate / rate) * np.exp(
        -tangent * rate * (1. - yield_time))
    final = peak * np.exp(-young * rate)
    errors = []
    for increments in (16, 32, 64):
        step = _linear_creep_ramp_hold(increments)
        step.solve()
        stress = step.energy_history[-1].generalized_reaction / .004
        errors.append(abs(stress - final))
        assert step.state.plastic_fields["equivalent_plastic_strain"].global_max() > 0.
        assert step.state.equivalent_creep_strain.global_max() > 0.
    assert 1.7 < errors[0] / errors[1] < 2.3
    assert 1.7 < errors[1] / errors[2] < 2.3
    assert errors[2] / final < .07


def test_joint_homogeneous_patch_is_mesh_independent_not_spatial_convergence():
    coarse = _linear_creep_ramp_hold(16)
    fine = _linear_creep_ramp_hold(16, cells=2)
    coarse.solve()
    fine.solve()
    np.testing.assert_allclose(
        [f.generalized_reaction for f in fine.energy_history],
        [f.generalized_reaction for f in coarse.energy_history], rtol=1.e-8, atol=1.e-9)
    for key, values in fine.state.snapshot().items():
        np.testing.assert_allclose(values, np.broadcast_to(coarse.state.snapshot()[key][0], values.shape),
                                   rtol=1.e-8, atol=1.e-9)


@pytest.mark.parametrize("scope", ["axisymmetric", "regional", "static", "dynamic"])
def test_joint_provider_inspection_and_execution_reject_unsupported_scope(scope):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    from agentfem import fields, models, studies
    from agentfem.step_providers import step_capability

    if scope == "axisymmetric":
        domain = dmesh.create_unit_square(MPI.COMM_SELF, 1, 1)
        study = studies.creep_solid(dimension=2, assumption="axisymmetric")
    else:
        domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
        study = (studies.static_solid(dimension=3) if scope == "static" else
                 studies.dynamic_solid(dimension=3) if scope == "dynamic" else studies.creep_solid())
    model = models.create(study=study, mesh=domain)
    u = model.field(fields.displacement(domain))
    # Region meaning is not evaluated: this composition must reject regional
    # dispatch before lowering any quadrature/operator implementation.
    model.material(_material(), region=object() if scope == "regional" else None)
    options = {} if scope in ("static", "dynamic") else dict(duration=1.)
    capability = step_capability(model, target=u, options=options)
    assert not capability["supported"]
    assert not capability["ready"]
    if scope not in ("static", "dynamic"):
        assert capability["rejection_reasons"]
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        model.step(target=u, **options)
    assert not model.steps
