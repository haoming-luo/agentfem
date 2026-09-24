from __future__ import annotations

import numpy as np
import pytest

from agentfem import campaigns, constitutive, datasets


def _uniaxial_path(values, *, name="cyclic_strain"):
    strains = np.zeros((len(values), 3, 3), dtype=float)
    strains[:, 0, 0] = values
    return constitutive.material_strain_path(
        np.arange(len(values), dtype=float),
        strains,
        name=name,
        coordinate_name="load_coordinate",
    )


def _j2():
    return constitutive.J2LinearIsotropicHardening(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )


def _chaboche():
    return constitutive.chaboche(
        young=210_000.0,
        poisson=0.3,
        yield_stress=220.0,
        backstresses=((25_000.0, 120.0), (8_000.0, 20.0)),
        isotropic_saturation=80.0,
        isotropic_rate=12.0,
    )


def test_material_loading_path_refinement_preserves_every_physical_knot():
    path = _uniaxial_path((0.0, 0.004, -0.002, 0.003))
    refined = path.refine(3)
    nested = refined.refine(2)

    for coordinate, strain in zip(path.coordinate, path.strain, strict=True):
        selected = np.flatnonzero(refined.coordinate == coordinate)
        assert selected.size == 1
        np.testing.assert_array_equal(refined.strain[selected[0]], strain)
    for coordinate in refined.coordinate:
        assert np.count_nonzero(nested.coordinate == coordinate) == 1
    assert path.fingerprint.startswith("sha256:")
    assert path.fingerprint != refined.fingerprint
    assert path.summary()["interpolation"] == "piecewise_linear_exact_knots"
    renamed = constitutive.material_strain_path(
        path.coordinate,
        path.strain,
        name="cosmetic_name_does_not_change_science",
        coordinate_name=path.coordinate_name,
    )
    assert renamed.fingerprint == path.fingerprint
    with pytest.raises(ValueError, match="read-only"):
        path.strain[0, 0, 0] = 1.0


def test_tabulated_isotropic_hardening_integrates_piecewise_linear_storage():
    hardening = constitutive.TabulatedIsotropicHardening(
        equivalent_plastic_strain=(0.0, 0.1, 0.2),
        yield_stress=(100.0, 120.0, 130.0),
    )

    assert hardening.value(0.05) == pytest.approx(110.0)
    assert hardening.value(0.3) == pytest.approx(130.0)
    assert hardening.hardening_storage(0.15) == pytest.approx(2.125)
    with pytest.raises(ValueError, match="hardening, not softening"):
        constitutive.TabulatedIsotropicHardening(
            equivalent_plastic_strain=(0.0, 0.1),
            yield_stress=(100.0, 90.0),
        )


def test_mixed_material_control_recovers_uniaxial_stress_state():
    material = constitutive.J2LinearIsotropicHardening(
        young=210_000.0,
        poisson=0.3,
        yield_stress=1.0e9,
    )
    strain = np.zeros((2, 3, 3))
    strain[1, 0, 0] = 1.0e-3
    stress = np.zeros_like(strain)
    strain_control = np.zeros((3, 3), dtype=bool)
    strain_control[0, 0] = True
    path = constitutive.material_mixed_path(
        (0.0, 1.0),
        strain=strain,
        stress=stress,
        strain_control=strain_control,
    )

    response = material.history(path).solve()

    assert response.stress[-1, 0, 0] == pytest.approx(210.0)
    assert response.stress[-1, 1, 1] == pytest.approx(0.0, abs=1.0e-10)
    assert response.stress[-1, 2, 2] == pytest.approx(0.0, abs=1.0e-10)
    assert response.strain[-1, 1, 1] == pytest.approx(-0.3e-3)
    assert response.strain[-1, 2, 2] == pytest.approx(-0.3e-3)
    assert response.to_result().metadata["material_history"]["control"] == "mixed"


def test_material_path_never_guesses_how_stress_targets_are_controlled():
    strain = np.zeros((2, 3, 3))
    stress = np.zeros_like(strain)
    with pytest.raises(ValueError, match="explicit strain_control"):
        constitutive.MaterialLoadingPath(
            (0.0, 1.0),
            strain,
            stress=stress,
        )

    stress[0, 0, 0] = 1.0
    strain_control = np.zeros((3, 3), dtype=bool)
    path = constitutive.material_mixed_path(
        (0.0, 1.0),
        strain=strain,
        stress=stress,
        strain_control=strain_control,
    )
    with pytest.raises(ValueError, match="start at zero stress"):
        _j2().history(path).solve()


def test_j2_response_only_matches_consistent_update_without_a_tangent():
    material = _j2()
    strain = np.diag((0.004, 0.0, 0.0))

    response_only = material.update(strain, linearization="none")
    consistent = material.update(strain, linearization="consistent")

    np.testing.assert_allclose(response_only.stress, consistent.stress)
    np.testing.assert_allclose(
        response_only.state.plastic_strain,
        consistent.state.plastic_strain,
    )
    assert response_only.state.equivalent_plastic_strain == pytest.approx(
        consistent.state.equivalent_plastic_strain
    )
    assert response_only.algorithmic_tangent is None
    assert consistent.algorithmic_tangent.shape == (3, 3, 3, 3)
    with pytest.raises(ValueError, match="linearization"):
        material.update(strain, linearization="elastic")


def test_chaboche_response_only_does_not_evaluate_algorithmic_tangent(monkeypatch):
    material = _chaboche()
    strain = np.diag((0.006, 0.0, 0.0))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("response-only integration requested a tangent")

    monkeypatch.setattr(
        constitutive.ChabocheCombinedHardening,
        "_algorithmic_tangent",
        fail_if_called,
    )
    update = material.update(strain, linearization="none")

    assert not update.elastic
    assert update.algorithmic_tangent is None
    energy = update.energy_increment
    assert energy.reference_yield_dissipation > 0.0
    assert energy.dynamic_recovery_dissipation > 0.0
    assert energy.backward_euler_dissipation >= 0.0
    assert energy.modeled_irreversible_dissipation > (
        energy.reference_yield_dissipation
    )
    assert abs(energy.balance_residual) < 1.0e-10 * energy.plastic_work
    with pytest.raises(AssertionError, match="requested a tangent"):
        material.update(strain, linearization="consistent")


def test_material_histories_enter_campaign_and_scientific_dataset_directly(tmp_path):
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("hardening_modulus", 500.0, 1_000.0, unit="MPa")
    )
    path = _uniaxial_path((0.0, 0.004, -0.002, 0.003))

    def evaluate(parameters):
        material = constitutive.J2LinearIsotropicHardening(
            young=210_000.0,
            poisson=0.3,
            yield_stress=250.0,
            hardening_modulus=parameters["hardening_modulus"],
        )
        return material.history(path).solve_result()

    campaign = campaigns.create(
        name="j2_material_trajectories",
        parameter_space=space,
        outputs=(
            datasets.Quantity(
                "stress",
                shape=(4, 3, 3),
                kind="history",
            ),
            datasets.Quantity(
                "equivalent_plastic_strain",
                shape=(4,),
                kind="history",
            ),
        ),
        evaluate=evaluate,
        scientific_inputs={"loading_path": path},
    )
    sampling = campaigns.explicit(
        space,
        ({"hardening_modulus": 500.0}, {"hardening_modulus": 1_000.0}),
    )

    report = campaign.run(sampling, output_directory=tmp_path)
    accepted = report.require_dataset(minimum_samples=2)

    assert accepted.y_matrix().shape == (2, 40)
    assert set(accepted.metadata["history_axes"]) == {
        "stress",
        "equivalent_plastic_strain",
    }
    assert report.scientific_inputs["complete"] is True
    for sample in accepted.samples:
        summary = sample.provenance["simulation_result"]
        assert summary["metadata"]["material_history"]["path_fingerprint"] == (
            path.fingerprint
        )
        assert summary["metadata"]["scope"]["level"] == "material_point"


def test_j2_history_uses_common_result_and_unambiguous_energy_names():
    material = _j2()
    path = _uniaxial_path((0.0, 0.004, -0.001, 0.002))

    step = material.history(path)
    result = step.solve_result()

    assert step.procedure.requires_global_solve is False
    assert result.metadata["material_history"]["linearization"] == "none"
    assert result.metadata["energy"]["plastic_work"] == ("signed_work_not_dissipation")
    assert result.metadata["energy"]["modeled_irreversible_dissipation"] == (
        "available"
    )
    assert "reference_yield_dissipation" in result.histories
    assert "plastic_dissipation" not in result.histories
    assert "algorithmic_tangent" not in result.histories
    assert result.histories["stress"].value_shape == (3, 3)
    assert result.quantity("accepted_increment_count") == path.segment_count
    assert result.scientific_input_manifest()["complete"] is True


def test_chaboche_history_exposes_state_but_fails_closed_on_full_dissipation():
    material = _chaboche()
    path = _uniaxial_path((0.0, 0.006, -0.004, 0.005))

    result = material.history(path).solve_result()

    assert "backstress_components" in result.histories
    assert "total_backstress" in result.histories
    assert result.metadata["energy"]["dynamic_recovery_dissipation"] == ("available")
    assert result.metadata["energy"]["modeled_irreversible_dissipation"] == (
        "available"
    )
    assert result.metadata["energy"]["complete_discrete_plastic_energy_balance"] is True
    assert result.histories["dynamic_recovery_dissipation"].latest > 0.0
    assert result.histories["backward_euler_dissipation"].latest >= 0.0
    assert abs(result.histories["plastic_energy_balance_residual"].latest) < 1.0e-9
    assert "plastic_dissipation" not in result.histories


def test_chaboche_history_restart_and_linearization_preserve_accepted_state():
    material = _chaboche()
    path = _uniaxial_path((0.0, 0.006, -0.004, 0.005, -0.002))
    response_only = material.history(path).solve()
    consistent = material.history(path, linearization="consistent").solve()

    np.testing.assert_allclose(response_only.stress, consistent.stress)
    np.testing.assert_allclose(
        response_only.final_state.plastic_strain,
        consistent.final_state.plastic_strain,
    )
    np.testing.assert_allclose(
        response_only.final_state.backstresses,
        consistent.final_state.backstresses,
    )
    first = material.history(
        constitutive.material_strain_path(
            path.coordinate[:3],
            path.strain[:3],
            coordinate_name=path.coordinate_name,
        )
    ).solve()
    second = material.history(
        constitutive.material_strain_path(
            path.coordinate[2:],
            path.strain[2:],
            coordinate_name=path.coordinate_name,
        ),
        initial_state=first.final_state,
    ).solve()
    np.testing.assert_allclose(second.stress, response_only.stress[2:])
    np.testing.assert_allclose(
        second.final_state.backstresses,
        response_only.final_state.backstresses,
    )


def test_material_history_rejects_ambiguous_initial_and_temperature_states():
    material = _j2()
    nonzero_start = _uniaxial_path((0.001, 0.002))
    with pytest.raises(ValueError, match="start at zero strain"):
        material.history(nonzero_start).solve()

    temperature_path = constitutive.material_strain_path(
        (0.0, 1.0),
        np.zeros((2, 3, 3)),
        temperature=(293.15, 300.0),
    )
    with pytest.raises(ValueError, match="does not yet accept temperature"):
        material.history(temperature_path).solve()
