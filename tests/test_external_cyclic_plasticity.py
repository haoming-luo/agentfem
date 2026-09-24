"""External material-point comparisons for nonlinear combined hardening."""

from __future__ import annotations

import pytest

from agentfem import benchmarks


def test_abaqus_ofhc_copper_cyclic_observables_are_reproduced():
    assessment, results = benchmarks.abaqus_ofhc_copper_cyclic_benchmark()

    assert assessment.accepted
    assert assessment.symmetric_final_peeq == pytest.approx(
        assessment.symmetric_reference_peeq,
        rel=0.01,
    )
    assert assessment.tension_torsion_maximum_normal_stress == pytest.approx(
        assessment.tension_torsion_reference_stress,
        rel=0.01,
    )
    assert assessment.fourth_to_final_cycle_change < 0.01
    assert results["symmetric"].metadata["material_history"]["control"] == "mixed"
    assert (
        results["symmetric"].quantity("maximum_stress_control_residual")
        < 1.0e-8
    )
    assert (
        results["tension_torsion"].quantity("maximum_stress_control_residual")
        < 1.0e-8
    )
    assert (
        results["tension_torsion"].metadata["external_benchmark"]["accepted"]
        is True
    )
    assert (
        results["tension_torsion"]
        .metadata["external_benchmark"]
        ["acceptance_tolerances_are"]
        == "agentfem_defined"
    )


def test_abaqus_ofhc_benchmark_rejects_underresolved_cycles():
    with pytest.raises(ValueError, match="at least 8"):
        benchmarks.abaqus_ofhc_copper_cyclic_benchmark(points_per_cycle=4)


def test_abaqus_316_unsymmetric_path_reproduces_published_ratcheting_trend():
    assessment, results = benchmarks.abaqus_316_steel_ratcheting_path_comparison(
        cycle_count=12,
        refinement=2,
    )

    assert assessment.accepted
    assert assessment.maximum_stress_control_residual < 1.0e-7
    assert assessment.one_backstress_accumulation > 0.0
    assert 0.0 < assessment.two_to_one_accumulation_ratio < 1.0
    assert results[1].metadata["external_benchmark"]["evidence_level"] == (
        "published_path_and_qualitative_trend"
    )
    assert results[2].metadata["external_benchmark"]["reference_curve"] == (
        "figure_only_no_numerical_table"
    )


def test_abaqus_316_ratcheting_path_rejects_one_cycle():
    with pytest.raises(ValueError, match="at least two"):
        benchmarks.abaqus_316_steel_ratcheting_path_comparison(cycle_count=1)


def test_axisymmetric_chaboche_global_path_matches_material_point():
    assessment, result = benchmarks.axisymmetric_chaboche_ratcheting_crosscheck(
        cycle_count=2,
        refinement=1,
    )

    assert assessment.accepted
    assert assessment.maximum_relative_peak_strain_error < 2.0e-3
    assert assessment.final_residual_norm < 1.0e-7
    assert assessment.as_dict()["external_specimen_golden"] is False
    assert "global_peak_axial_strain" in result.histories
    assert "material_point_peak_axial_strain" in result.histories
    assert result.metadata["structural_crosscheck"]["accepted"] is True
