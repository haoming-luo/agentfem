from __future__ import annotations

import numpy as np

from agentfem import benchmarks


def test_assembled_dcb_curve_is_structural_evidence_not_relabelled_oracle():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
        source=(
            "AgentFEM nondimensional DCB smoke fixture; geometry family informed "
            "by NASA/TP-2016-219211"
        ),
    )
    curve = benchmarks.dcb_finite_element_curve(
        specification,
        crack_length=(1.5, 2.0, 2.5),
        load=1.0e-3,
        specimen_length=6.0,
        elements_along=12,
        elements_per_arm=1,
    )

    compliance = np.asarray([point.compliance for point in curve.points])
    residual = np.asarray([point.residual_norm for point in curve.points])
    assert np.all(np.diff(compliance) > 0.0)
    assert np.all(residual < 1.0e-8)
    assert np.all(curve.energy_release.total_energy_release_rate > 0.0)
    assert "assembled DCB" in curve.source
    assert curve.summary()["schema"] == "agentfem.dcb-finite-element-curve.v1"


def test_dcb_finite_element_curve_rejects_off_grid_crack_tip_before_solving():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
    )
    try:
        benchmarks.dcb_finite_element_curve(
            specification,
            crack_length=(1.4, 2.0, 2.5),
            load=1.0e-3,
            specimen_length=6.0,
            elements_along=12,
            elements_per_arm=1,
        )
    except ValueError as exc:
        assert "align" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("An off-grid crack tip must be rejected.")
