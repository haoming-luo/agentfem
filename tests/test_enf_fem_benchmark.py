from __future__ import annotations

from dataclasses import replace

import numpy as np

from agentfem import benchmarks


def _specification():
    return benchmarks.delamination_benchmark_spec(
        "enf",
        width=1.0,
        arm_thickness=0.25,
        elastic_modulus=1000.0,
        half_span=3.0,
        source=(
            "AgentFEM nondimensional ENF verification fixture; specimen family "
            "informed by NASA/TP-2016-219211"
        ),
    )


def test_assembled_enf_curve_has_mode_ii_compliance_and_residual_evidence():
    curve = benchmarks.enf_finite_element_curve(
        _specification(),
        crack_length=(1.0, 1.5, 2.0),
        control_displacement=1.0e-4,
        elements_along=24,
        elements_per_arm=2,
    )

    compliance = np.asarray([point.compliance for point in curve.points])
    assert np.all(np.diff(compliance) > 0.0)
    assert max(point.residual_norm for point in curve.points) < 1.0e-8
    assert np.all(curve.energy_release.total_energy_release_rate > 0.0)
    assert np.allclose(curve.energy_release.mode_i_energy_release_rate, 0.0)
    np.testing.assert_allclose(
        curve.energy_release.mode_ii_energy_release_rate,
        curve.energy_release.total_energy_release_rate,
    )
    assert curve.summary()["discretization"]["loading"].startswith("three-point")


def test_enf_curve_rejects_off_grid_cracks_and_odd_midspan_mesh():
    specification = _specification()
    try:
        benchmarks.enf_finite_element_curve(
            specification,
            crack_length=(1.1, 1.5, 2.0),
            control_displacement=1.0e-4,
            elements_along=24,
            elements_per_arm=1,
        )
    except ValueError as exc:
        assert "align" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Off-grid ENF cracks must be rejected.")

    try:
        benchmarks.enf_finite_element_curve(
            specification,
            crack_length=(1.0, 1.5, 2.0),
            control_displacement=1.0e-4,
            elements_along=25,
            elements_per_arm=1,
        )
    except ValueError as exc:
        assert "even" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("An ENF mesh without a mid-span node must be rejected.")


def test_assembled_enf_compliance_converges_toward_mode_ii_beam_oracle():
    study = benchmarks.enf_finite_element_convergence(
        _specification(),
        crack_length=(1.0, 1.5, 2.0),
        control_displacement=1.0e-4,
        mesh_levels=((12, 1), (24, 2), (48, 4)),
        reference_relative_tolerance=0.03,
        refinement_relative_tolerance=0.15,
    )

    certificate = study.certificate
    assert certificate.accepted
    assert certificate.asymptotic_trend
    assert certificate.relative_errors_to_reference[-1] < 0.02
    assert certificate.successive_relative_changes[-1] < 0.12
    assert certificate.observed_order is not None
    assert certificate.observed_order > 1.0
    assert all(
        np.allclose(curve.energy_release.mode_i_energy_release_rate, 0.0)
        for curve in study.curves
    )

    incompatible = (*study.curves[:-1], replace(
        study.curves[-1], interface_stiffness=2.0 * study.curves[-1].interface_stiffness
    ))
    try:
        benchmarks.certify_enf_compliance_convergence(
            _specification(),
            incompatible,
            reference_relative_tolerance=1.0,
            refinement_relative_tolerance=1.0,
        )
    except ValueError as exc:
        assert "discrete model contract" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Mixed ENF discretization contracts must be rejected.")
