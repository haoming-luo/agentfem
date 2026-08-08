from types import SimpleNamespace

import pytest

from agentfem import benchmarks
from agentfem.benchmarks import dynamic_fracture as fracture_benchmarks


def test_v1_prestrained_wave_arrival_converges_to_acoustic_tensor():
    coarse = benchmarks.finite_strain_wave_arrival(
        prestrain=0.0, cells=40,
    )
    fine = benchmarks.finite_strain_wave_arrival(
        prestrain=0.0, cells=80,
    )
    prestrained = benchmarks.finite_strain_wave_arrival(
        prestrain=0.1, cells=80,
    )

    assert fine.relative_error < coarse.relative_error
    assert fine.relative_error < 0.02
    assert prestrained.relative_error < 0.02
    assert prestrained.predicted_reference_speed != pytest.approx(
        fine.predicted_reference_speed
    )
    assert fine.maximum_relative_energy_error < coarse.maximum_relative_energy_error
    assert fine.maximum_relative_energy_error < 2.0e-3


def test_v2_cohesive_dissipation_is_exact_and_energy_error_converges():
    coarse = benchmarks.cohesive_energy_balance(dt=2.0e-3)
    fine = benchmarks.cohesive_energy_balance(dt=5.0e-4)

    assert coarse.maximum_damage == pytest.approx(1.0)
    assert fine.maximum_damage == pytest.approx(1.0)
    assert coarse.fracture_dissipation == pytest.approx(
        coarse.declared_fracture_energy, abs=1.0e-12
    )
    assert fine.fracture_dissipation == pytest.approx(
        fine.declared_fracture_energy, abs=1.0e-12
    )
    assert fine.final_relative_energy_error < coarse.final_relative_energy_error
    assert fine.final_relative_energy_error < 1.0e-5


def test_v3_classical_crack_remains_sub_rayleigh_under_refinement_and_damping():
    coarse = benchmarks.classical_cohesive_crack(cells=40)
    fine = benchmarks.classical_cohesive_crack(cells=60)
    smaller_dt = benchmarks.classical_cohesive_crack(
        cells=60, time_step_scale=0.4,
    )
    damped = benchmarks.classical_cohesive_crack(cells=40, damping=0.5)

    for result in (coarse, fine, smaller_dt, damped):
        assert result.propagated_length > 0.15
        assert 0.0 < result.speed_ratio < 0.8
        assert result.maximum_simultaneous_failed_fraction < 0.1
        assert result.final_relative_energy_error < 5.0e-4
    assert abs(fine.speed_ratio - coarse.speed_ratio) < 0.05
    assert abs(smaller_dt.speed_ratio - fine.speed_ratio) < 0.02
    assert abs(damped.speed_ratio - coarse.speed_ratio) < 0.02
    assert damped.numerical_damping_dissipation > 0.0


def test_v4_case_smoke_exposes_plane_stress_preload_impact_and_bulk_modes():
    result = benchmarks.prestressed_weak_interface_separation(
        label="v4_smoke",
        cells=30,
        total_time=0.004,
        axial_strain=0.12,
        strength=150.0,
        fracture_energy=1.0,
        initial_stiffness=1.0e5,
        impact_displacement=0.002,
        impact_rise_time=0.002,
        retain_trace=True,
    )
    assert result.preload_energy_jump == pytest.approx(0.0)
    assert result.transverse_cells == 2
    assert 0.0 < result.preload_ligament_traction_ratio < 1.0
    assert result.pressure_wave_speed > result.shear_wave_speed
    assert result.impact_displacement == pytest.approx(0.002)
    assert result.crack_speed_fit_length == pytest.approx(0.3)
    assert result.summary()["loading"] == (
        "homogeneous_prestrain_then_remote_impact"
    )
    assert result.trace is not None
    assert result.trace.time.size == result.trace.damage.shape[0]
    assert result.trace.path_coordinate.size == 30
    assert result.trace.summary()["metadata"]["facet_reduction"]["damage"] == (
        "quadrature_maximum"
    )
    stages = result.performance["stages"]
    assert stages["bulk_residual_assembly"]["calls"] > 0
    assert stages["cohesive_force_assembly"]["calls"] > 0
    # The accepted residual is reused by the reaction/work ledger. Only the
    # initial state needs one additional assembly; history cadence must not
    # double the nonlinear bulk work.
    assert stages["bulk_residual_assembly"]["calls"] == (
        stages["residual_assembly"]["calls"] + 1
    )
    assert stages["cohesive_force_assembly"]["calls"] == (
        stages["residual_assembly"]["calls"] + 1
    )


def test_plane_stress_matches_thin_three_dimensional_affine_patch():
    regular = benchmarks.plane_stress_thin_3d_crosscheck(
        axial_stretch=1.12,
        reference_thickness=0.02,
        cells=(2, 2, 1),
    )
    thinner = benchmarks.plane_stress_thin_3d_crosscheck(
        axial_stretch=1.12,
        reference_thickness=0.005,
        cells=(3, 2, 2),
    )

    for result in (regular, thinner):
        assert result.accepted
        assert result.jacobian > 0.0
        assert result.maximum_relative_stress_error < 1.0e-10
        assert result.relative_energy_error < 1.0e-10
        assert result.traction_free_stress_ratio < 1.0e-10
        assert result.lateral_stretch == pytest.approx(result.thickness_stretch)
    assert thinner.thin_3d_first_piola[1, 1] == pytest.approx(
        regular.thin_3d_first_piola[1, 1],
        rel=1.0e-12,
    )


def test_v4_strip_accepts_independent_even_transverse_resolution():
    result = benchmarks.prestressed_weak_interface_separation(
        label="v4_2d_mesh_smoke",
        cells=30,
        transverse_cells=4,
        total_time=0.002,
        axial_strain=0.12,
        strength=150.0,
        fracture_energy=1.0,
        initial_stiffness=1.0e5,
    )
    assert result.cells == 30
    assert result.transverse_cells == 4

    with pytest.raises(ValueError, match="even integer"):
        benchmarks.prestressed_weak_interface_separation(
            cells=30,
            transverse_cells=3,
        )


def test_v4_convergence_contract_validates_controls_before_long_runs():
    with pytest.raises(ValueError, match="spatial_speed_tolerance"):
        benchmarks.jmps_weak_interface_convergence_v4(
            spatial_speed_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="history_every"):
        benchmarks.jmps_weak_interface_convergence_v4(history_every=0)


def test_v4_refinement_separates_mechanism_from_speed_convergence(monkeypatch):
    speeds = iter((10.0, 9.3, 8.4, 9.98))

    def synthetic_case(**kwargs):
        return SimpleNamespace(
            label=kwargs["label"],
            maximum_fitted_speed=next(speeds),
            regime="supershear",
            maximum_simultaneous_failed_fraction=0.02,
            final_relative_energy_error=1.0e-4,
        )

    monkeypatch.setattr(
        fracture_benchmarks,
        "prestressed_weak_interface_separation",
        synthetic_case,
    )
    study = fracture_benchmarks.jmps_weak_interface_convergence_v4()
    assert study.mechanism_preserved
    assert not study.speed_converged
    assert not study.accepted
    assert study.fine_spatial_speed_change > study.spatial_speed_change
    assert any(
        "not yet asymptotically converged" in message
        for message in study.acceptance_failures
    )
