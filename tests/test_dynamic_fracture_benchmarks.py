import pytest

from agentfem import benchmarks


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
    )
    assert result.preload_energy_jump == pytest.approx(0.0)
    assert 0.0 < result.preload_ligament_traction_ratio < 1.0
    assert result.pressure_wave_speed > result.shear_wave_speed
    assert result.impact_displacement == pytest.approx(0.002)
    assert result.summary()["loading"] == (
        "homogeneous_prestrain_then_remote_impact"
    )
