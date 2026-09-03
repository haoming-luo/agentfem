from __future__ import annotations

from dataclasses import replace

import numpy as np

from agentfem import benchmarks, solvers


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

    incompatible = (bad_residual[0], bad_residual[1], replace(
        bad_residual[2], interface_stiffness=10.0
    ))
    try:
        benchmarks.certify_dcb_compliance_convergence(
            specification,
            incompatible,
            reference_relative_tolerance=1.0,
            refinement_relative_tolerance=1.0,
        )
    except ValueError as exc:
        assert "discrete model contract" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Mixed DCB discretization contracts must be rejected.")


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


def _synthetic_dcb_propagation_curve(
    specification, *, element_size, peak_reaction, damaged_length, energy_error
):
    law = {
        "family": "cohesive_traction_separation",
        "mode": "normal",
        "strength": 1.0,
        "fracture_energy": 0.1,
    }
    points = []
    for index, opening in enumerate((0.0, 0.1, 0.2)):
        fraction = index / 2.0
        points.append(
            benchmarks.DCBCohesivePropagationPoint(
                increment=index,
                opening=opening,
                reaction=peak_reaction * fraction,
                bulk_strain_energy=0.01 * fraction,
                cohesive_stored_energy=0.005 * fraction,
                cohesive_dissipation=0.02 * fraction,
                external_work=0.035 * fraction,
                energy_balance_error=energy_error * fraction,
                relative_energy_balance_error=energy_error * fraction,
                maximum_damage=0.99 * fraction,
                damaged_length=1.0 + (damaged_length - 1.0) * fraction,
                failed_length=1.0,
                process_zone_length=max(damaged_length - 1.0, 0.0),
                newton_iterations=2,
                residual_norm=1.0e-11,
            )
        )
    return benchmarks.DCBCohesivePropagationCurve(
        specification=specification,
        points=tuple(points),
        element_size=element_size,
        process_zone_elements=4.0,
        law=law,
        source=f"synthetic cohesive propagation h={element_size}",
    )


def test_dcb_propagation_certificate_requires_growth_energy_and_refinement():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
        source="source-identified DCB propagation fixture",
    )
    curves = tuple(
        _synthetic_dcb_propagation_curve(
            specification,
            element_size=h,
            peak_reaction=peak,
            damaged_length=length,
            energy_error=error,
        )
        for h, peak, length, error in (
            (1.0, 1.10, 2.10, 0.020),
            (0.5, 1.04, 2.04, 0.010),
            (0.25, 1.02, 2.02, 0.005),
        )
    )
    certificate = benchmarks.certify_dcb_cohesive_propagation(
        specification,
        curves,
        refinement_relative_tolerance=0.05,
        energy_relative_tolerance=0.03,
        required_process_zone_elements=3.0,
    )

    assert certificate.accepted
    assert certificate.propagation_observed
    assert certificate.maximum_relative_energy_errors[-1] == 0.005
    assert certificate.summary()["scope"] == "Mode-I DCB cohesive propagation"

    no_growth = tuple(
        _synthetic_dcb_propagation_curve(
            specification,
            element_size=h,
            peak_reaction=1.0,
            damaged_length=1.0,
            energy_error=0.0,
        )
        for h in (1.0, 0.5, 0.25)
    )
    rejected = benchmarks.certify_dcb_cohesive_propagation(
        specification,
        no_growth,
        refinement_relative_tolerance=0.05,
    )
    assert not rejected.accepted
    assert not rejected.propagation_observed


def test_assembled_dcb_cohesive_path_commits_damage_and_closes_energy():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=0.25,
        elastic_modulus=1000.0,
        source=(
            "AgentFEM nondimensional propagation smoke fixture; specimen family "
            "informed by NASA/TP-2016-219211"
        ),
    )
    curve = benchmarks.dcb_cohesive_propagation_curve(
        specification,
        precrack_length=2.0,
        specimen_length=6.0,
        opening=np.linspace(0.0, 0.30, 31),
        strength=0.02,
        fracture_energy=0.001,
        initial_stiffness=100.0,
        elements_along=24,
        elements_per_arm=1,
    )

    assert curve.points[-1].damaged_length > curve.points[0].damaged_length
    assert curve.points[-1].maximum_damage > 0.95
    assert curve.points[-1].cohesive_dissipation > 0.0
    assert max(point.residual_norm for point in curve.points) < 1.0e-8
    assert max(point.relative_energy_balance_error for point in curve.points[1:]) < 0.08
    assert curve.summary()["evidence_scope"].startswith("displacement-controlled")


def test_dcb_continuation_cuts_back_without_changing_requested_output_points():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=1.0,
        elastic_modulus=1000.0,
        source="AgentFEM cutback regression fixture",
    )
    options = solvers.newton(
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
        maximum_iterations=8,
        line_search=None,
        linear_solver=solvers.direct_solver(),
        error_if_not_converged=False,
    )
    requested = np.asarray((0.0, 0.3, 0.8))
    curve = benchmarks.dcb_cohesive_propagation_curve(
        specification,
        precrack_length=2.0,
        specimen_length=6.0,
        opening=requested,
        strength=2.0,
        fracture_energy=0.1,
        initial_stiffness=1.0e4,
        elements_along=12,
        elements_per_arm=2,
        solver_options=options,
    )

    assert np.allclose([point.opening for point in curve.points], requested)
    assert curve.points[-1].cutbacks >= 1
    assert curve.points[-1].accepted_subincrements >= 2
    assert curve.points[-1].residual_norm < 1.0e-8


def test_assembled_dcb_propagation_has_three_level_convergence_evidence():
    specification = benchmarks.delamination_benchmark_spec(
        "dcb",
        width=1.0,
        arm_thickness=0.25,
        elastic_modulus=1000.0,
        source=(
            "AgentFEM nondimensional DCB convergence fixture; specimen family "
            "informed by NASA/TP-2016-219211"
        ),
    )
    study = benchmarks.dcb_cohesive_propagation_convergence(
        specification,
        precrack_length=2.0,
        specimen_length=6.0,
        opening=np.linspace(0.0, 0.30, 31),
        strength=0.02,
        fracture_energy=0.001,
        initial_stiffness=100.0,
        mesh_levels=((12, 1), (24, 2), (48, 4)),
        refinement_relative_tolerance=0.25,
        energy_relative_tolerance=0.08,
        required_process_zone_elements=3.0,
    )

    certificate = study.certificate
    assert certificate.accepted
    assert certificate.propagation_observed
    assert certificate.peak_reaction_changes[-1] < 0.10
    assert certificate.damaged_length_changes[-1] < 0.06
    assert certificate.maximum_relative_energy_errors[-1] < 0.05
    assert all(
        right < left
        for left, right in zip(
            certificate.maximum_relative_energy_errors[:-1],
            certificate.maximum_relative_energy_errors[1:],
        )
    )
