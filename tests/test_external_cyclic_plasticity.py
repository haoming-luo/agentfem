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
    assert results["symmetric"].quantity(
        "maximum_stress_control_residual"
    ) < 1.0e-8
    assert results["tension_torsion"].quantity(
        "maximum_stress_control_residual"
    ) < 1.0e-8
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
