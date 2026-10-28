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
