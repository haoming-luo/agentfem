from __future__ import annotations

import numpy as np
import pytest

from agentfem.constitutive import (
    ArrheniusShift,
    GeneralizedMaxwell,
    MaxwellState,
    WLFShift,
    fit_relaxation_prony,
    standard_linear_solid,
)


def test_standard_linear_solid_has_correct_time_and_frequency_limits():
    material = standard_linear_solid(
        equilibrium_modulus=2.0,
        relaxing_modulus=8.0,
        relaxation_time=3.0,
    )

    np.testing.assert_allclose(
        material.relaxation_modulus([0.0, 1.0e6]),
        [10.0, 2.0],
        atol=1.0e-10,
    )
    assert material.storage_modulus(0.0) == pytest.approx(2.0)
    assert material.loss_modulus(0.0) == pytest.approx(0.0)
    assert material.storage_modulus(1.0e9) == pytest.approx(10.0)


def test_generalized_maxwell_exact_update_commits_and_restores_state():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    state = MaxwellState.zero(1)
    snapshot = state.snapshot()
    update = material.update(state, np.asarray(0.1), 0.5)

    expected_tangent = 2.0 + 8.0 * (1.0 - np.exp(-0.5)) / 0.5
    assert update.algorithmic_modulus == pytest.approx(expected_tangent)
    assert float(update.stress) == pytest.approx(expected_tangent * 0.1)
    expected_dissipation = 1.6**2 / 8.0 * (
        0.5 - 2.0 * (1.0 - np.exp(-0.5)) + 0.5 * (1.0 - np.exp(-1.0))
    )
    assert update.dissipated_energy_increment == pytest.approx(
        expected_dissipation
    )
    assert update.dissipated_energy_increment > 0.0
    assert float(state.strain) == 0.0
    state.commit(update)
    assert float(state.strain) == pytest.approx(0.1)
    state.restore(snapshot)
    assert float(state.strain) == 0.0
    assert state.dissipated_energy == 0.0


def test_prony_factory_and_temperature_shift_contracts():
    wlf = WLFShift(reference_temperature=293.15, c1=17.44, c2=51.6)
    arrhenius = ArrheniusShift(
        activation_energy=50.0e3,
        reference_temperature=293.15,
    )
    material = GeneralizedMaxwell.from_prony(
        10.0,
        [0.2, 0.3],
        [1.0, 100.0],
        shift=wlf,
    )

    assert material.equilibrium_modulus == pytest.approx(5.0)
    np.testing.assert_allclose(material.prony_ratios, [0.2, 0.3])
    assert float(wlf.factor(293.15)) == pytest.approx(1.0)
    assert float(arrhenius.factor(293.15)) == pytest.approx(1.0)
    assert float(wlf.factor(313.15)) < 1.0
    assert float(arrhenius.factor(313.15)) < 1.0


def test_temperature_shift_rejects_singular_or_nonfinite_inputs():
    wlf = WLFShift(reference_temperature=293.15, c1=17.44, c2=51.6)
    arrhenius = ArrheniusShift(
        activation_energy=50.0e3,
        reference_temperature=293.15,
    )

    with pytest.raises(ValueError, match="singularity"):
        wlf.factor(293.15 - 51.6)
    with pytest.raises(ValueError, match="finite"):
        arrhenius.factor(np.nan)


def test_fixed_spectrum_prony_fit_recovers_positive_reference_model():
    reference = GeneralizedMaxwell(3.0, [5.0, 2.0], [0.2, 20.0])
    time = np.concatenate(([0.0], np.logspace(-3, 3, 120)))
    measured = reference.relaxation_modulus(time)
    fit = fit_relaxation_prony(time, measured, [0.2, 20.0])

    assert fit.relative_root_mean_square_error < 1.0e-12
    assert fit.model.equilibrium_modulus == pytest.approx(3.0)
    np.testing.assert_allclose(fit.model.branch_moduli, [5.0, 2.0])
