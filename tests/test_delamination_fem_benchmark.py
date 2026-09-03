from __future__ import annotations

import numpy as np

from agentfem import benchmarks


def _synthetic_dcb_curve(specification, *, element_size, bias, residual=1.0e-10):
    crack = np.asarray((1.5, 2.0, 2.5), dtype=float)
    compliance = benchmarks.dcb_beam_compliance(specification, crack) * (1.0 + bias)
    load = np.full(crack.shape, 1.0e-3)
    points = tuple(
        benchmarks.DCBFiniteElementPoint(
            crack_length=float(a),
            effective_crack_length=float(a),
            load=float(p),
            opening=float(p * c),
            compliance=float(c),
            element_size=float(element_size),
            elements_per_arm=max(1, int(round(1.0 / element_size))),
            newton_iterations=1,
            residual_norm=float(residual),
        )
        for a, p, c in zip(crack, load, compliance)
    )
    energy = benchmarks.compliance_energy_release_curve(
        specification,
        crack_length=crack,
        load=load,
        compliance=compliance,
        source=f"synthetic assembled level h={element_size}",
    )
    return benchmarks.DCBFiniteElementCurve(
        specification=specification,
        points=points,
        energy_release=energy,
        source=f"synthetic assembled level h={element_size}",
    )


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


def test_dcb_compliance_certificate_is_scoped_and_accepts_refining_evidence():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
        source="declared thin-beam verification fixture",
    )
    curves = tuple(
        _synthetic_dcb_curve(specification, element_size=h, bias=0.2 * h**2)
        for h in (1.0, 0.5, 0.25)
    )
    certificate = benchmarks.certify_dcb_compliance_convergence(
        specification,
        curves,
        reference_relative_tolerance=0.02,
        refinement_relative_tolerance=0.05,
    )

    assert certificate.accepted
    assert certificate.asymptotic_trend
    assert certificate.reference_errors_nonincreasing
    assert certificate.observed_order is not None
    assert certificate.observed_order > 1.5
    assert len(set(certificate.curve_identity_sha256)) == 3
    summary = certificate.summary()
    assert summary["scope"] == "precracked elastic structural compliance"
    assert "crack propagation" in summary["excludes"]


def test_dcb_compliance_certificate_fails_closed_on_residual_or_mesh_order():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
        source="declared thin-beam verification fixture",
    )
    bad_residual = (
        _synthetic_dcb_curve(specification, element_size=1.0, bias=0.2),
        _synthetic_dcb_curve(specification, element_size=0.5, bias=0.05),
        _synthetic_dcb_curve(
            specification, element_size=0.25, bias=0.0125, residual=1.0e-4
        ),
    )
    certificate = benchmarks.certify_dcb_compliance_convergence(
        specification,
        bad_residual,
        reference_relative_tolerance=0.02,
        refinement_relative_tolerance=0.05,
    )
    assert not certificate.accepted

    wrong_order = (bad_residual[0], bad_residual[2], bad_residual[1])
    try:
        benchmarks.certify_dcb_compliance_convergence(
            specification,
            wrong_order,
            reference_relative_tolerance=1.0,
            refinement_relative_tolerance=1.0,
        )
    except ValueError as exc:
        assert "coarse to fine" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("A non-refining DCB sequence must be rejected.")


def test_dcb_compliance_does_not_confuse_beam_model_error_with_mesh_convergence():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
        source="declared thin-beam verification fixture",
    )
    curves = tuple(
        _synthetic_dcb_curve(specification, element_size=h, bias=bias)
        for h, bias in zip((1.0, 0.5, 0.25), (0.0100, 0.0110, 0.0105))
    )
    certificate = benchmarks.certify_dcb_compliance_convergence(
        specification,
        curves,
        reference_relative_tolerance=0.02,
        refinement_relative_tolerance=0.002,
    )

    assert certificate.accepted
    assert certificate.asymptotic_trend
    assert not certificate.reference_errors_nonincreasing
