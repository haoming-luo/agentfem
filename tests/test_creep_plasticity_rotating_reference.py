# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent continuous rotating-stress reference, not alloy calibration."""

import numpy as np
import pytest

from agentfem import constitutive


def _reference(time, *, thermal, chaboche=False, order=64):
    # Prescribed deviatoric stress rotates between two orthonormal tensors.
    # q=Y+H*p with p=beta*t ensures continuing plastic loading. Both creep
    # and plastic flow directions follow the CURRENT stress, not an old axis.
    young, poisson, yield_stress, hardening = 200_000., .3, 200., 2_000.
    beta, coefficient, reference_stress = .002, 1.e-5, 100.
    shear = young / (2. * (1. + poisson))
    first = np.diag([1., -1., 0.]) / np.sqrt(2.)
    second = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]) / np.sqrt(2.)

    def direction(t):
        return (np.cos(np.pi * t)[..., None, None] * first
                + np.sin(np.pi * t)[..., None, None] * second)

    def temperature(t):
        return 800. + (50. * np.sin(np.pi * t)**2 if thermal else 0. * t)

    def backstress(t):
        # Exact convolution of d alpha_i/dt = (2/3) C_i beta n_p
        # - gamma_i beta alpha_i, with zero initial backstress. This is
        # the continuous equation, not the discrete return-map recurrence.
        branches = []
        for modulus, recovery in zip((12_000., 3_000.), (100., 8.), strict=True):
            decay = recovery * beta
            convolution = (np.exp(1.j * np.pi * t) - np.exp(-decay * t)) / (decay + 1.j * np.pi)
            branches.append(np.sqrt(2. / 3.) * modulus * beta * (
                convolution.real[..., None, None] * first
                + convolution.imag[..., None, None] * second))
        return np.stack(branches, axis=-3)

    def prescribed_stress(t):
        radius = (yield_stress + 60. * (-np.expm1(-10. * beta * t))
                  if chaboche else yield_stress + hardening * beta * t)
        stress = np.sqrt(2. / 3.) * radius[..., None, None] * direction(t)
        return stress + np.sum(backstress(t), axis=-3) if chaboche else stress

    nodes, weights = np.polynomial.legendre.leggauss(order)
    times, weights = .5 * time * (nodes + 1.), .5 * time * weights
    flow_direction = np.sqrt(1.5) * direction(times)
    stresses = prescribed_stress(times)
    q = np.sqrt(1.5 * np.sum(stresses * stresses, axis=(-2, -1)))
    rates = coefficient * (q / reference_stress)**3
    if thermal:
        rates *= np.exp(-120_000. / 8.31446261815324 * (1. / temperature(times) - 1. / 800.))
    plastic = np.einsum("q,qij->ij", weights * beta, flow_direction)
    creep_direction = 1.5 * stresses / q[:, None, None]
    creep = np.einsum("q,qij->ij", weights * rates, creep_direction)
    stress = prescribed_stress(np.asarray(time))
    kelvin = float(temperature(np.asarray(time)))
    strain = stress / (2. * shear) + plastic + creep
    if thermal:
        strain += 1.e-5 * (kelvin - 800.) * np.eye(3)
    result = dict(strain=strain, stress=stress, plastic=plastic, creep=creep,
                  peeq=beta * time, ceeq=float(np.dot(weights, rates)), temperature=kelvin)
    if chaboche:
        result["backstress"] = backstress(np.asarray(time))
    return result


def _material(thermal, chaboche):
    plastic = (constitutive.ChabocheCombinedHardening(
        young=200_000., poisson=.3, yield_stress=200.,
        backstress_moduli=(12_000., 3_000.), dynamic_recovery=(100., 8.),
        isotropic_saturation=60., isotropic_rate=10.) if chaboche else
        constitutive.J2LinearIsotropicHardening(
            young=200_000., poisson=.3, yield_stress=200., hardening_modulus=2_000.))
    if thermal:
        elastic = constitutive.thermoelastic(
            young=200_000., poisson=.3, density=1., thermal_expansion=1.e-5,
            conductivity=1., specific_heat=1., reference_temperature=800.)
        creep = constitutive.isotropic_arrhenius_power_law(
            elastic=elastic, coefficient=1.e-5, reference_stress=100., stress_exponent=3.,
            activation_energy=120_000., reference_temperature=800.)
    else:
        creep = constitutive.isotropic_power_law(
            young=200_000., poisson=.3, density=1., coefficient=1.e-5,
            reference_stress=100., stress_exponent=3.)
    return constitutive.CreepPlasticityMaterial(creep=creep, plastic=plastic)


@pytest.mark.parametrize("thermal", [False, True])
@pytest.mark.parametrize("chaboche", [False, True])
def test_rotating_continuous_flow_refines_without_discrete_update_oracle(thermal, chaboche, record_property):
    # Preload to yield at t=0 is elastic. Its work is outside this test.
    # Reference integration uses only prescribed continuous rates, never
    # AgentFEM stress updates or their backward-Euler recurrence.
    for time in (.125, .5, .875, 1.):
        coarse = _reference(time, thermal=thermal, chaboche=chaboche)
        fine = _reference(time, thermal=thermal, chaboche=chaboche, order=128)
        for key in ("strain", "plastic", "creep", "ceeq"):
            np.testing.assert_allclose(coarse[key], fine[key], rtol=1.e-12, atol=1.e-15)
    errors = {key: [] for key in ("stress", "plastic", "creep", "peeq", "ceeq")}
    if chaboche:
        errors["backstress"] = []
        end = _reference(1., thermal=thermal, chaboche=True)
        shifted = end["stress"] - end["backstress"].sum(axis=0)
        # This path deliberately separates current creep and plastic axes.
        assert np.linalg.norm(end["stress"] / np.linalg.norm(end["stress"])
                              - shifted / np.linalg.norm(shifted)) > .01
    for count in (32, 64, 128):
        material = _material(thermal, chaboche)
        state = material.initial_state()
        maximum = {key: 0. for key in errors}
        for index in range(1, count + 1):
            time = index / count
            reference = _reference(time, thermal=thermal, chaboche=chaboche)
            update = material.update(reference["strain"], state=state,
                time_start=(index - 1) / count, time_end=time,
                temperature=reference["temperature"] if thermal else None)
            actual = dict(stress=update.stress, plastic=update.state.plastic.plastic_strain,
                          creep=update.state.creep.creep_strain,
                          peeq=update.state.plastic.equivalent_plastic_strain,
                          ceeq=update.state.creep.equivalent_creep_strain)
            if chaboche:
                actual["backstress"] = update.state.plastic.backstresses
            for key in errors:
                maximum[key] = max(maximum[key], float(np.linalg.norm(actual[key] - reference[key])))
            assert update.creep_work >= 0.
            assert update.plastic_energy.modeled_irreversible_dissipation >= 0.
            state = update.state
        for key in errors:
            errors[key].append(maximum[key])
    for key, values in errors.items():
        assert 1.7 < values[0] / values[1] < 2.3, (key, values)
        assert 1.7 < values[1] / values[2] < 2.3, (key, values)
        record_property(f"maximum_{key}_errors_32_64_128", values)
    assert errors["stress"][-1] / 200. < .01
