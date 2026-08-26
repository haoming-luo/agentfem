from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from agentfem import fracture


def _annular_samples(actual, auxiliary, *, inner=0.05, outer=0.4, order=48):
    radial_points, radial_weights = np.polynomial.legendre.leggauss(order)
    angle_points, angle_weights = np.polynomial.legendre.leggauss(order)
    radius = inner + 0.5 * (radial_points + 1.0) * (outer - inner)
    theta = pi * angle_points
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    points = np.column_stack(
        (
            actual.tip.point[0] + rr.ravel() * np.cos(tt.ravel()),
            actual.tip.point[1] + rr.ravel() * np.sin(tt.ravel()),
        )
    )
    q_gradient = -np.column_stack((np.cos(tt.ravel()), np.sin(tt.ravel()))) / (
        outer - inner
    )
    weights = (
        0.5
        * (outer - inner)
        * pi
        * np.outer(radial_weights, angle_weights)
        * rr
    ).ravel()
    return fracture.InteractionIntegralSamples2D(
        actual_stress=actual.stress(points),
        actual_displacement_gradient=actual.displacement_gradient(points),
        auxiliary_stress=auxiliary.stress(points),
        auxiliary_displacement_gradient=auxiliary.displacement_gradient(points),
        q_gradient=q_gradient,
        weights=weights,
    )


def test_interaction_integral_reduces_quadrature_samples_without_solver_assumptions():
    actual_stress = np.array([[[2.0, 3.0], [3.0, 4.0]]])
    zero = np.zeros((1, 2, 2))
    auxiliary_gradient = np.array([[[5.0, 0.0], [7.0, 0.0]]])
    samples = fracture.InteractionIntegralSamples2D(
        actual_stress=actual_stress,
        actual_displacement_gradient=zero,
        auxiliary_stress=zero,
        auxiliary_displacement_gradient=auxiliary_gradient,
        q_gradient=np.array([[0.0, 2.0]]),
        weights=np.array([3.0]),
    )

    assert samples.number_of_samples == 1
    assert fracture.interaction_integral(samples) == pytest.approx(258.0)


def test_interaction_integral_samples_fail_closed_on_shape_and_weight_errors():
    base = dict(
        actual_stress=np.zeros((1, 2, 2)),
        actual_displacement_gradient=np.zeros((1, 2, 2)),
        auxiliary_stress=np.zeros((1, 2, 2)),
        auxiliary_displacement_gradient=np.zeros((1, 2, 2)),
        q_gradient=np.zeros((1, 2)),
        weights=np.ones(1),
    )
    with pytest.raises(ValueError, match="actual_stress"):
        fracture.InteractionIntegralSamples2D(
            **{**base, "actual_stress": np.zeros((1, 2))}
        )
    with pytest.raises(ValueError, match="positive"):
        fracture.InteractionIntegralSamples2D(**{**base, "weights": [0.0]})


def test_interaction_integral_report_matches_independent_infinite_plate_oracle():
    cracks = fracture.crack_set(
        fracture.segment("main", center=(0.0, 0.0), length=2.0, angle=0.0)
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=210.0e9,
        poisson_ratio=0.3,
        assumption="plane_strain",
    )
    reference = fracture.infinite_plate_stress_intensity(
        crack=cracks,
        tip_id="main:end",
        stress=fracture.remote_stress(yy=80.0e6, xy=20.0e6),
        material=material,
    )
    expected_i = 2.0 * reference.k_i / material.effective_modulus
    expected_ii = 2.0 * reference.k_ii / material.effective_modulus
    report = fracture.interaction_integral_report(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2, 0.3),
        mode_i_integrals=(expected_i * 0.999, expected_i, expected_i * 1.001),
        mode_ii_integrals=(expected_ii * 0.999, expected_ii, expected_ii * 1.001),
        material=material,
    )
    verification = fracture.verify_stress_intensity(report, reference)

    assert report.k_i == pytest.approx(80.0e6 * sqrt(pi))
    assert report.k_ii == pytest.approx(20.0e6 * sqrt(pi))
    assert report.j_integral == pytest.approx(
        (report.k_i**2 + report.k_ii**2) / material.effective_modulus,
        rel=2.0e-6,
    )
    assert verification.status == "accepted"


def test_plane_stress_and_plane_strain_use_distinct_interaction_normalization():
    cracks = fracture.crack_set(
        fracture.segment("main", center=(0.0, 0.0), length=2.0, angle=0.0)
    )
    plane_stress = fracture.linear_elastic_fracture_material(
        young_modulus=1000.0, poisson_ratio=0.25, assumption="plane_stress"
    )
    plane_strain = fracture.linear_elastic_fracture_material(
        young_modulus=1000.0, poisson_ratio=0.25, assumption="plane_strain"
    )
    common = dict(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2),
        mode_i_integrals=(2.0, 2.0),
        mode_ii_integrals=(0.0, 0.0),
    )

    stress_report = fracture.interaction_integral_report(
        **common, material=plane_stress
    )
    strain_report = fracture.interaction_integral_report(
        **common, material=plane_strain
    )

    assert stress_report.k_i == pytest.approx(1000.0)
    assert strain_report.k_i == pytest.approx(1000.0 / (1.0 - 0.25**2))


def test_williams_field_stress_and_gradient_follow_local_tip_coordinates():
    cracks = fracture.crack_set(
        fracture.segment("main", center=(0.0, 0.0), length=2.0, angle=0.0)
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1000.0, poisson_ratio=0.2, assumption="plane_stress"
    )
    field = fracture.WilliamsField2D(
        cracks.tip("main:end"), material, k_i=4.0, k_ii=3.0
    )
    scale = 1.0 / sqrt(2.0 * pi * 2.0)

    stress = field.stress([[3.0, 0.0]])[0]
    assert stress[0, 0] == pytest.approx(4.0 * scale)
    assert stress[1, 1] == pytest.approx(4.0 * scale)
    assert stress[0, 1] == pytest.approx(3.0 * scale)

    point = np.array([2.2, 0.4])
    step = 1.0e-6
    numerical = np.column_stack(
        [
            (
                field.displacement(point + step * direction)[0]
                - field.displacement(point - step * direction)[0]
            )
            / (2.0 * step)
            for direction in np.eye(2)
        ]
    )
    np.testing.assert_allclose(
        field.displacement_gradient(point)[0], numerical, rtol=2.0e-8, atol=1.0e-10
    )


@pytest.mark.parametrize("assumption", ("plane_stress", "plane_strain"))
def test_williams_fields_recover_mixed_mode_sif_through_domain_integral(assumption):
    cracks = fracture.crack_set(
        fracture.segment("main", center=(0.0, 0.0), length=2.0, angle=0.0)
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1200.0, poisson_ratio=0.23, assumption=assumption
    )
    tip = cracks.tip("main:end")
    actual = fracture.WilliamsField2D(tip, material, k_i=8.0, k_ii=-3.0)
    mode_i = fracture.WilliamsField2D(tip, material, k_i=1.0)
    mode_ii = fracture.WilliamsField2D(tip, material, k_ii=1.0)

    integral_i = fracture.interaction_integral(_annular_samples(actual, mode_i))
    integral_ii = fracture.interaction_integral(_annular_samples(actual, mode_ii))

    assert 0.5 * material.effective_modulus * integral_i == pytest.approx(
        8.0, rel=2.0e-5
    )
    assert 0.5 * material.effective_modulus * integral_ii == pytest.approx(
        -3.0, rel=2.0e-5
    )
