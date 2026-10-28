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
