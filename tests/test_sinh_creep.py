# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent local and structural checks of the shared sinh corrector."""

import numpy as np
import pytest

from agentfem import constitutive, fields, mesh, models, solvers, steps, studies
from agentfem.constitutive.plasticity import von_mises


def material(**options):
    return constitutive.isotropic_sinh(
        young=200_000., poisson=0.3, density=1.,
        coefficient=options.pop("coefficient", 1.e-5),
        stress_scale=options.pop("stress_scale", 100.), **options,
    )


@pytest.mark.parametrize("exponent", [1., 3., 5.5])
def test_sinh_scalar_batch_history_and_fixed_old_state_tangent(exponent):
    law = material(exponent=exponent)
    rng = np.random.default_rng(32)
    strains = rng.normal(size=(12, 3, 3)) * 0.001
    strains = (strains + strains.swapaxes(1, 2)) / 2
    strains[0] = 0.
    old = [constitutive.ImplicitCreepState() for _ in strains]
    for start, strain_batch in enumerate((strains, -0.8 * strains, 0.5 * strains)):
        scalar = [law.update(e, time_start=start, time_end=start + 1., state=s)
                  for e, s in zip(strain_batch, old, strict=True)]
        batch = law.update_many(
            strain_batch, time_start=start, time_end=start + 1.,
            creep_strain=np.array([s.creep_strain for s in old]),
            equivalent_creep_strain=[s.equivalent_creep_strain for s in old],
        )
        np.testing.assert_allclose(batch.stress, [s.stress for s in scalar], atol=1.e-10)
        np.testing.assert_allclose(batch.algorithmic_tangent,
                                   [s.algorithmic_tangent for s in scalar], atol=1.e-7)
        np.testing.assert_allclose(batch.creep_strain, [s.state.creep_strain for s in scalar], atol=1.e-13)
        np.testing.assert_allclose(batch.equivalent_creep_strain,
                                   [s.state.equivalent_creep_strain for s in scalar], atol=1.e-13)
        direction = rng.normal(size=(3, 3))
        direction = (direction + direction.T) / 2
        for index in (1, 5):
            eps = 1.e-7
            arguments = dict(time_start=start, time_end=start + 1., state=old[index])
            numerical = (law.update(strain_batch[index] + eps * direction, **arguments).stress
                         - law.update(strain_batch[index] - eps * direction, **arguments).stress) / (2 * eps)
            exact = np.einsum("ijkl,kl->ij", scalar[index].algorithmic_tangent, direction)
            np.testing.assert_allclose(exact, numerical, rtol=2.e-6, atol=0.003)
        old = [s.state for s in scalar]


def test_sinh_high_trial_stress_does_not_overflow_or_false_converge():
    # q_trial / scale > 1000; directly evaluating sinh at the trial overflows.
    law = material(stress_scale=1., exponent=3.)
    strain = np.diag([0.01, -0.005, -0.005])
    update = law.update(strain, time_start=0., time_end=1.)
    q = von_mises(update.stress)
    assert np.isfinite(update.stress).all()
    assert np.isfinite(update.algorithmic_tangent).all()
    assert update.equivalent_increment == pytest.approx(law.creep.equivalent_rate(q), rel=1.e-8)
    batch = law.update_many(strain[None], time_start=0., time_end=1.,
                            creep_strain=np.zeros((1, 3, 3)), equivalent_creep_strain=[0.])
    np.testing.assert_allclose(batch.stress[0], update.stress, atol=1.e-10)


def test_sinh_zero_rate_and_unsupported_parameters():
    law = material(coefficient=0.)
    strain = np.diag([0.002, -0.001, -0.001])
    update = law.update(strain, time_start=0., time_end=10.)
    assert update.equivalent_increment == 0.
    np.testing.assert_allclose(update.stress, np.einsum("ijkl,kl->ij", law.elastic_tangent(), strain))
    with pytest.raises(ValueError, match="exponent >= 1"):
        material(exponent=0.5)
    with pytest.raises(RuntimeError, match="did not converge"):
        material(exponent=3.).update(strain, time_start=0., time_end=10., maximum_iterations=1)


def test_sinh_public_rate_avoids_intermediate_overflow_without_clipping():
    law = constitutive.SinhCreep(coefficient=1.e-300, stress_scale=1., exponent=1.)
    # sinh(800) itself overflows, but coefficient*sinh(800) is representable.
    expected = np.exp(np.log(1.e-300) + 800. - np.log(2.))
    assert law.equivalent_rate(800.) == pytest.approx(expected, rel=1.e-13)
    assert constitutive.SinhCreep(0., 1.).equivalent_rate(1.e6) == 0.
    with pytest.raises(OverflowError, match="representable"):
        constitutive.SinhCreep(1., 1.).equivalent_rate(1000.)
    with pytest.raises(ValueError, match="finite"):
        law.equivalent_rate(np.inf)


@pytest.mark.parametrize("family", ["sinh", "power"])
def test_linear_creep_zero_stress_tangent_includes_rate_derivative(family):
    law = (material(exponent=1.) if family == "sinh" else
           constitutive.isotropic_power_law(young=200_000., poisson=0.3, density=1.,
                                            coefficient=1.e-5, reference_stress=100., stress_exponent=1.))
    strain = np.zeros((3, 3))
    direction = np.diag([1., -0.5, -0.5])
    delta = 1.e-7
    response = law.update(strain, time_start=0., time_end=10.)
    numerical = (law.update(delta * direction, time_start=0., time_end=10.).stress
                 - law.update(-delta * direction, time_start=0., time_end=10.).stress) / (2 * delta)
    np.testing.assert_allclose(np.einsum("ijkl,kl->ij", response.algorithmic_tangent, direction),
                               numerical, rtol=1.e-8)
    batch = law.update_many(strain[None], time_start=0., time_end=10.,
                            creep_strain=strain[None], equivalent_creep_strain=[0.])
    np.testing.assert_allclose(batch.algorithmic_tangent[0], response.algorithmic_tangent)


def test_sinh_relaxation_converges_to_independent_continuous_solution():
    # Fixed deviatoric strain: dq/dt = -3 G A sinh(q/s), n=1.
    law = material(coefficient=1.e-4)
    strain = np.diag([0.001, -0.0005, -0.0005])
    q0 = 3 * law.shear_modulus * 0.001
    end = 2.
    reference = 2 * law.creep.stress_scale * np.arctanh(
        np.tanh(q0 / (2 * law.creep.stress_scale))
        * np.exp(-3 * law.shear_modulus * law.creep.coefficient * end / law.creep.stress_scale))
    errors = []
    for count in (20, 40, 80):
        state = constitutive.ImplicitCreepState()
        for i in range(count):
            update = law.update(strain, time_start=i * end / count,
                                time_end=(i + 1) * end / count, state=state)
            state = update.state
        errors.append(abs(von_mises(update.stress) - reference))
    assert 1.8 < errors[0] / errors[1] < 2.1
    assert 1.8 < errors[1] / errors[2] < 2.1


def _structure(count=8, coefficient=1.e-5):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(study=studies.creep_solid(), mesh=domain, name="sinh_hold")
    u = model.field(fields.displacement(domain))
    selected = model.material(material(coefficient=coefficient, exponent=2.))
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1),
                  component=component, value=0.)
    model.traction((100., 0., 0.), on=mesh.face(domain, axis="x", value=1., tag=4))
    return model.step(target=u, material=selected, duration=2.,
                      incrementation=steps.fixed(count), progress=False,
                      solver_options=solvers.newton(relative_tolerance=1.e-10,
                                                    absolute_tolerance=1.e-9))


def test_sinh_public_step_constant_stress_and_restart(tmp_path):
    full = _structure()
    full.solve_result()
    expected = full.material.creep.equivalent_rate(100.) * 2.
    np.testing.assert_allclose(full.state.equivalent_creep_strain.values, expected, rtol=1.e-8)
    np.testing.assert_allclose(full.state.stress.values[:, 0, 0], 100., atol=1.e-7)
    partial = _structure()
    partial.solve(until=1.)
    checkpoint = partial.save_checkpoint(tmp_path / "hold.npz")
    resumed = _structure()
    resumed.load_checkpoint(checkpoint)
    resumed.solve_result()
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, atol=1.e-10)
    np.testing.assert_allclose(resumed.state.creep_strain.values, full.state.creep_strain.values, atol=1.e-10)
    incompatible = _structure(coefficient=2.e-5)
    before = incompatible.state.creep_strain.values.copy()
    with pytest.raises((ValueError, RuntimeError)):
        incompatible.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(incompatible.state.creep_strain.values, before)


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("damage", ["ledger", "state"])
def test_creep_checkpoint_late_failure_restores_complete_live_step(tmp_path, portable, damage):
    import json

    source = _structure()
    source.solve(until=1.)
    checkpoint = source.save_checkpoint(tmp_path / "source.npz", portable=portable)
    if portable:
        payload = json.loads(checkpoint.read_text())
        if damage == "ledger":
            payload["energy_history"] = [{"broken": True}]
        else:
            payload["quadrature_state"]["sha256"] = "0" * 64
        checkpoint.write_text(json.dumps(payload))
    else:
        with np.load(checkpoint, allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
        if damage == "ledger":
            payload["energy_history"] = json.dumps([{"broken": True}])
        else:
            payload["equivalent_creep_strain"] = np.array([np.nan])
        np.savez(checkpoint, **payload)
    receiving = _structure()
    receiving.solve(until=0.5)
    before_u = receiving.solution.x.array.copy()
    before_state = receiving.state.snapshot()
    before_stress = receiving.state.stress.values.copy()
    before_tangent = receiving.state.tangent.values.copy()
    before_energy = list(receiving.energy_history)
    before_info = receiving.last_solve_info
    with pytest.raises((ValueError, RuntimeError, KeyError, TypeError)):
        receiving.load_checkpoint(checkpoint)
    assert receiving.accepted_time == 0.5
    assert receiving.last_solve_info is before_info
    assert receiving.energy_history == before_energy
    np.testing.assert_array_equal(receiving.solution.x.array, before_u)
    np.testing.assert_array_equal(receiving.state.stress.values, before_stress)
    np.testing.assert_array_equal(receiving.state.tangent.values, before_tangent)
    for key, values in before_state.items():
        np.testing.assert_array_equal(receiving.state.snapshot()[key], values)
        np.testing.assert_array_equal(receiving.state.transaction.trial[key].values, values)
    # Rejection must not poison the following ordinary continuation.
    receiving.solve()
    reference = _structure()
    reference.solve()
    np.testing.assert_allclose(receiving.solution.x.array, reference.solution.x.array, atol=1.e-10)
