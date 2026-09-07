from __future__ import annotations

import numpy as np
import pytest

from agentfem import procedures, studies
from agentfem.constitutive import (
    ArrheniusShift,
    GeneralizedMaxwell,
    IsotropicGeneralizedMaxwell,
    MaxwellState,
    WLFShift,
    fit_relaxation_prony,
    isotropic_generalized_maxwell,
    standard_linear_solid,
)


def test_viscoelastic_study_resolves_its_declared_procedure():
    study = studies.viscoelastic_solid(dimension=3)
    procedure = procedures.for_step(
        analysis=study.analysis,
        method=study.preferred_procedure,
        stateful=True,
    )

    assert study.physics == "solid_mechanics"
    assert study.is_transient
    assert procedure.algorithm == "exact_generalized_maxwell_equilibrium"
    assert procedure.stateful


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
    expected_dissipation = (
        1.6**2 / 8.0 * (0.5 - 2.0 * (1.0 - np.exp(-0.5)) + 0.5 * (1.0 - np.exp(-1.0)))
    )
    assert update.dissipated_energy_increment == pytest.approx(expected_dissipation)
    assert update.dissipated_energy_increment > 0.0
    assert float(state.strain) == 0.0
    state.commit(update)
    assert float(state.strain) == pytest.approx(0.1)
    state.restore(snapshot)
    assert float(state.strain) == 0.0
    assert state.dissipated_energy == 0.0


def test_generalized_maxwell_small_increment_retains_instantaneous_tangent():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    state = MaxwellState.zero(1)
    update = material.update(state, np.asarray(1.0e-12), 1.0e-16)

    assert update.algorithmic_modulus == pytest.approx(
        material.instantaneous_modulus,
        rel=1.0e-15,
    )
    assert np.isfinite(update.dissipated_energy_increment)
    assert update.dissipated_energy_increment >= 0.0


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


def test_maxwell_state_rejects_invalid_branch_count_snapshot_and_commit():
    with pytest.raises(ValueError, match="positive integer"):
        MaxwellState.zero(1.5)
    with pytest.raises(ValueError, match="positive integer"):
        MaxwellState.zero(True)

    state = MaxwellState.zero(1)
    invalid_snapshot = state.snapshot()
    invalid_snapshot["dissipated_energy"] = np.nan
    with pytest.raises(ValueError, match="finite state"):
        state.restore(invalid_snapshot)

    invalid_update = GeneralizedMaxwell(2.0, [8.0], [1.0]).update(
        state,
        np.asarray(0.1),
        0.5,
    )
    invalid_update = type(invalid_update)(
        strain=invalid_update.strain,
        overstress=invalid_update.overstress,
        stress=invalid_update.stress,
        algorithmic_modulus=invalid_update.algorithmic_modulus,
        dissipated_energy_increment=np.nan,
        mechanical_work_increment=invalid_update.mechanical_work_increment,
    )
    with pytest.raises(ValueError, match="nonnegative dissipation"):
        state.commit(invalid_update)


def test_maxwell_update_rejects_nonfinite_time_increment_and_shifted_times():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    state = MaxwellState.zero(1)
    with pytest.raises(ValueError, match="finite and positive"):
        material.update(state, np.asarray(0.1), np.nan)

    shifted = GeneralizedMaxwell(
        2.0,
        [8.0],
        [np.finfo(float).max],
        shift=WLFShift(reference_temperature=300.0, c1=1.0, c2=100.0),
    )
    with pytest.raises(ValueError, match="Shifted relaxation times"):
        shifted.shifted_relaxation_times(201.0)


def test_fixed_spectrum_prony_fit_recovers_positive_reference_model():
    reference = GeneralizedMaxwell(3.0, [5.0, 2.0], [0.2, 20.0])
    time = np.concatenate(([0.0], np.logspace(-3, 3, 120)))
    measured = reference.relaxation_modulus(time)
    fit = fit_relaxation_prony(time, measured, [0.2, 20.0])

    assert fit.relative_root_mean_square_error < 1.0e-12
    assert fit.model.equilibrium_modulus == pytest.approx(3.0)
    np.testing.assert_allclose(fit.model.branch_moduli, [5.0, 2.0])


def test_generalized_maxwell_history_uses_common_state_procedure_and_result():
    material = GeneralizedMaxwell(2.0, [8.0, 3.0], [0.2, 5.0])
    time = np.linspace(0.0, 2.0, 21)
    strain = np.minimum(time, 0.5) * 0.02

    step = material.history(time, strain, temperature=293.15)
    result = step.solve_result()

    assert step.procedure.algorithm == "exact_generalized_maxwell_update"
    assert not step.procedure.requires_global_solve
    assert result.metadata["procedure"] == step.procedure.summary()
    assert result.metadata["state"]["accepted_increments"] == 20
    assert result.metadata["state"]["restartable"]
    assert result.metadata["scope"]["level"] == "material_point"
    assert not result.metadata["scope"]["global_fem_provider"]
    assert set(result.histories) >= {
        "strain",
        "stress",
        "branch_overstress",
        "stored_energy",
        "dissipated_energy",
        "mechanical_work",
        "energy_balance_error",
        "temperature",
    }
    assert np.all(np.diff(result.histories["dissipated_energy"].values) >= 0.0)
    assert result.quantity("maximum_absolute_energy_balance_error") < 1.0e-14
    assert result.scientific_input_manifest()["complete"]


def test_generalized_maxwell_history_restart_matches_uninterrupted_path():
    material = GeneralizedMaxwell(2.0, [8.0, 3.0], [0.2, 5.0])
    time = np.linspace(0.0, 2.0, 21)
    strain = np.minimum(time, 0.5) * 0.02
    complete = material.history(time, strain).solve()

    first = material.history(time[:11], strain[:11]).solve()
    restarted = material.history(
        time[10:],
        strain[10:],
        initial_state=first.final_state,
    ).solve()

    np.testing.assert_allclose(restarted.stress, complete.stress[10:])
    np.testing.assert_allclose(
        restarted.branch_overstress,
        complete.branch_overstress[10:],
    )
    assert restarted.final_state.dissipated_energy == pytest.approx(
        complete.final_state.dissipated_energy
    )
    np.testing.assert_allclose(
        restarted.final_state.overstress,
        complete.final_state.overstress,
    )


def test_instantaneous_state_relaxes_to_the_closed_form_and_closes_energy():
    material = GeneralizedMaxwell(2.0, [8.0, 3.0], [0.2, 5.0])
    time = np.linspace(0.0, 10.0, 101)
    strain = np.full(time.size, 0.02)
    initial = material.initial_state(0.02, condition="instantaneous")

    response = material.history(time, strain, initial_state=initial).solve()

    np.testing.assert_allclose(
        response.stress,
        0.02 * material.relaxation_modulus(time),
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(response.mechanical_work, response.mechanical_work[0])
    assert np.max(np.abs(response.energy_balance_error)) < 1.0e-16
    assert response.dissipated_energy[-1] > 0.0


def test_generalized_maxwell_initial_state_has_explicit_history_semantics():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    equilibrated = material.initial_state([0.1, 0.2])
    instantaneous = material.initial_state([0.1, 0.2], condition="instantaneous")

    np.testing.assert_allclose(equilibrated.overstress, 0.0)
    np.testing.assert_allclose(instantaneous.overstress, [[0.8, 1.6]])
    with pytest.raises(ValueError, match="condition"):
        material.initial_state(0.0, condition="unknown")


def test_generalized_maxwell_history_rejects_incompatible_restart_and_time():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        material.history([0.0, 0.0], [0.0, 0.1]).solve()

    state = MaxwellState.zero(1)
    state.strain[...] = 0.2
    with pytest.raises(ValueError, match="first strain sample"):
        material.history([0.0, 1.0], [0.0, 0.1], initial_state=state).solve()


def test_isotropic_tensor_prony_factory_preserves_instantaneous_elasticity():
    material = isotropic_generalized_maxwell(
        instantaneous_young_modulus=1200.0,
        instantaneous_poisson_ratio=0.3,
        shear_relaxation_ratios=[0.2, 0.3],
        bulk_relaxation_ratios=[0.1, 0.2],
        relaxation_times=[0.5, 5.0],
    )

    assert material.instantaneous_shear_modulus == pytest.approx(
        1200.0 / (2.0 * 1.3)
    )
    assert material.instantaneous_bulk_modulus == pytest.approx(
        1200.0 / (3.0 * 0.4)
    )
    bulk, shear = material.relaxation_moduli([0.0, 1.0e8])
    assert bulk[0] == pytest.approx(material.instantaneous_bulk_modulus)
    assert shear[0] == pytest.approx(material.instantaneous_shear_modulus)
    assert bulk[-1] == pytest.approx(material.equilibrium_bulk_modulus)
    assert shear[-1] == pytest.approx(material.equilibrium_shear_modulus)


def test_isotropic_tensor_maxwell_relaxation_and_energy_are_exact():
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1000.0,
        instantaneous_poisson_ratio=0.25,
        shear_relaxation_ratios=[0.4],
        bulk_relaxation_ratios=[0.2],
        relaxation_times=[2.0],
    )
    strain = np.diag([0.02, -0.005, 0.003])
    state = material.initial_state(strain, condition="instantaneous")
    unpacked = material.state_schema.unpack(state)
    initial_stress = (
        material.instantaneous_bulk_modulus * np.trace(strain) * np.eye(3)
        + 2.0
        * material.instantaneous_shear_modulus
        * (strain - np.trace(strain) * np.eye(3) / 3.0)
    )

    update = material.update(state, strain, 1.25)
    bulk, shear = material.relaxation_moduli(1.25)
    expected = (
        bulk * np.trace(strain) * np.eye(3)
        + 2.0 * shear * (strain - np.trace(strain) * np.eye(3) / 3.0)
    )

    np.testing.assert_allclose(
        material.equilibrium_bulk_modulus * np.trace(strain) * np.eye(3)
        + 2.0
        * material.equilibrium_shear_modulus
        * (strain - np.trace(strain) * np.eye(3) / 3.0)
        + np.sum(unpacked["shear_overstress"], axis=0)
        + np.sum(unpacked["bulk_overstress"]) * np.eye(3),
        initial_stress,
    )
    np.testing.assert_allclose(update.stress, expected, rtol=2.0e-14, atol=1.0e-13)
    assert update.mechanical_work_increment == pytest.approx(0.0, abs=1.0e-14)
    assert update.dissipated_energy_increment > 0.0
    previous_energy = (
        material.equilibrium_shear_modulus
        * np.sum((strain - np.trace(strain) * np.eye(3) / 3.0) ** 2)
        + 0.5 * material.equilibrium_bulk_modulus * np.trace(strain) ** 2
        + sum(
            np.sum(value**2) / (4.0 * modulus)
            for value, modulus in zip(
                unpacked["shear_overstress"],
                material.shear_branch_moduli,
                strict=True,
            )
            if modulus > 0.0
        )
        + sum(
            value**2 / (2.0 * modulus)
            for value, modulus in zip(
                unpacked["bulk_overstress"],
                material.bulk_branch_moduli,
                strict=True,
            )
            if modulus > 0.0
        )
    )
    assert previous_energy == pytest.approx(
        update.stored_energy_density + update.dissipated_energy_increment,
        rel=2.0e-13,
        abs=1.0e-13,
    )


def test_isotropic_tensor_maxwell_algorithmic_tangent_matches_fixed_state_difference():
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1500.0,
        instantaneous_poisson_ratio=0.32,
        shear_relaxation_ratios=[0.25, 0.15],
        bulk_relaxation_ratios=[0.05, 0.1],
        relaxation_times=[0.1, 8.0],
    )
    old_strain = np.asarray(
        [[0.01, 0.002, 0.0], [0.002, -0.003, 0.001], [0.0, 0.001, 0.004]]
    )
    state = material.initial_state(old_strain, condition="instantaneous")
    new_strain = old_strain + np.asarray(
        [[0.003, -0.001, 0.0005], [-0.001, 0.001, 0.0], [0.0005, 0.0, -0.002]]
    )
    update = material.update(state, new_strain, 0.7)
    direction = np.asarray(
        [[0.7, -0.2, 0.1], [-0.2, -0.4, 0.3], [0.1, 0.3, 0.2]]
    )
    step = 1.0e-7
    plus = material.update(state, new_strain + step * direction, 0.7).stress
    minus = material.update(state, new_strain - step * direction, 0.7).stress
    numerical = (plus - minus) / (2.0 * step)
    analytical = np.einsum("ijkl,kl->ij", update.consistent_tangent, direction)
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-8, atol=2.0e-7)
