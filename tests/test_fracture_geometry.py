from __future__ import annotations

from math import pi, sqrt

import pytest

from agentfem import fracture


def test_straight_cracks_have_stable_tip_identity_and_fingerprint():
    first = fracture.segment(
        "main",
        center=(0.0, 0.0),
        length=2.0,
        angle=0.0,
    )
    second = fracture.segment(
        "secondary",
        start=(-0.5, 1.0),
        end=(0.5, 1.0),
    )
    cracks = fracture.crack_set(first, second, name="two_cracks")

    assert first.start == pytest.approx((-1.0, 0.0))
    assert first.end == pytest.approx((1.0, 0.0))
    assert [tip.tip_id for tip in cracks.tips] == [
        "main:start",
        "main:end",
        "secondary:start",
        "secondary:end",
    ]
    assert cracks.tip("main:end").extension_direction == pytest.approx((1.0, 0.0))
    for tip in cracks.tips:
        ex, ey = tip.extension_direction
        nx, ny = tip.normal
        assert ex * ny - ey * nx == pytest.approx(1.0)
    assert cracks.fingerprint == fracture.crack_set(
        first, second, name="two_cracks"
    ).fingerprint


def test_segment_center_form_uses_radians_and_rejects_ambiguous_input():
    vertical = fracture.segment(
        "vertical",
        center=(1.0, 2.0),
        length=4.0,
        angle=0.5 * pi,
    )
    assert vertical.start == pytest.approx((1.0, 0.0))
    assert vertical.end == pytest.approx((1.0, 4.0))

    with pytest.raises(ValueError, match="exactly one"):
        fracture.segment(
            "ambiguous",
            start=(0.0, 0.0),
            end=(1.0, 0.0),
            center=(0.5, 0.0),
            length=1.0,
            angle=0.0,
        )


@pytest.mark.parametrize(
    "second",
    (
        fracture.segment("crossing", start=(0.0, -1.0), end=(0.0, 1.0)),
        fracture.segment("touching", start=(1.0, 0.0), end=(2.0, 1.0)),
        fracture.segment("overlap", start=(0.5, 0.0), end=(1.5, 0.0)),
    ),
)
def test_crack_set_fails_closed_for_intersection_touch_or_overlap(second):
    first = fracture.segment("first", start=(-1.0, 0.0), end=(1.0, 0.0))
    with pytest.raises(
        fracture.UnsupportedCrackGeometryError,
        match="AFM-FRACTURE-GEOMETRY-001",
    ):
        fracture.crack_set(first, second)


def test_admissible_tip_radius_respects_other_cracks_and_domain_boundary():
    cracks = fracture.crack_set(
        fracture.segment("main", start=(-1.0, 0.0), end=(1.0, 0.0)),
        fracture.segment("upper", start=(-1.0, 1.0), end=(1.0, 1.0)),
    )

    selected = cracks.admissible_tip_radius(
        "main:end",
        bounds=(-2.0, 2.0, -2.0, 2.0),
    )

    assert selected == pytest.approx(0.45)


def test_stress_intensity_report_keeps_every_ring_and_path_status():
    cracks = fracture.crack_set(
        fracture.segment("main", start=(-1.0, 0.0), end=(1.0, 0.0))
    )
    accepted = fracture.stress_intensity_report(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2, 0.3),
        k_i=(10.0, 10.1, 9.9),
        k_ii=(2.0, 2.02, 1.98),
        j_integral=(4.0, 4.02, 3.98),
        relative_path_tolerance=0.02,
    )
    uncertain = fracture.stress_intensity_report(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2),
        k_i=(10.0, 14.0),
        k_ii=(2.0, 1.0),
        j_integral=(4.0, 6.0),
        relative_path_tolerance=0.02,
    )

    assert accepted.status == "accepted"
    assert accepted.summary()["K_I_by_radius"] == (10.0, 10.1, 9.9)
    assert uncertain.status == "uncertain"


def test_stress_intensity_report_rejects_j_only_path_dependence():
    cracks = fracture.crack_set(
        fracture.segment("main", start=(-1.0, 0.0), end=(1.0, 0.0))
    )
    report = fracture.stress_intensity_report(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2),
        k_i=(10.0, 10.0),
        k_ii=(0.0, 0.0),
        j_integral=(1.0, 2.0),
        relative_path_tolerance=0.02,
    )

    assert report.status == "uncertain"


def test_infinite_plate_reference_resolves_remote_stress_in_each_tip_frame():
    cracks = fracture.crack_set(
        fracture.segment("main", center=(0.0, 0.0), length=2.0, angle=0.0)
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=210.0e9,
        poisson_ratio=0.3,
        assumption="plane_stress",
    )
    loading = fracture.remote_stress(yy=100.0e6, xy=25.0e6)

    references = [
        fracture.infinite_plate_stress_intensity(
            crack=cracks,
            tip_id=tip.tip_id,
            stress=loading,
            material=material,
        )
        for tip in cracks.tips
    ]

    for reference in references:
        assert reference.k_i == pytest.approx(100.0e6 * sqrt(pi))
        assert reference.k_ii == pytest.approx(25.0e6 * sqrt(pi))
        assert reference.j_integral == pytest.approx(
            (reference.k_i**2 + reference.k_ii**2) / 210.0e9
        )


def test_inclined_crack_reference_and_plane_strain_effective_modulus():
    cracks = fracture.crack_set(
        fracture.segment("inclined", center=(0.0, 0.0), length=4.0, angle=0.25 * pi)
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=70.0e9,
        poisson_ratio=0.25,
        assumption="plane-strain",
    )
    reference = fracture.infinite_plate_stress_intensity(
        crack=cracks,
        tip_id="inclined:end",
        stress=fracture.remote_stress(yy=12.0e6),
        material=material,
    )
    expected_component = 6.0e6 * sqrt(2.0 * pi)

    assert reference.k_i == pytest.approx(expected_component)
    assert reference.k_ii == pytest.approx(expected_component)
    assert material.effective_modulus == pytest.approx(70.0e9 / (1.0 - 0.25**2))


def test_stress_intensity_verification_requires_accuracy_and_path_independence():
    cracks = fracture.crack_set(
        fracture.segment("main", center=(0.0, 0.0), length=2.0, angle=0.0)
    )
    reference = fracture.infinite_plate_stress_intensity(
        crack=cracks,
        tip_id="main:end",
        stress=fracture.remote_stress(yy=10.0),
        material=fracture.linear_elastic_fracture_material(
            young_modulus=1000.0,
            poisson_ratio=0.2,
            assumption="plane_stress",
        ),
    )
    accepted = fracture.stress_intensity_report(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2, 0.3),
        k_i=(reference.k_i * 0.999, reference.k_i, reference.k_i * 1.001),
        k_ii=(0.0, 0.0, 0.0),
        j_integral=(
            reference.j_integral * 0.999,
            reference.j_integral,
            reference.j_integral * 1.001,
        ),
    )
    inaccurate = fracture.stress_intensity_report(
        crack=cracks,
        tip_id="main:end",
        integration_radii=(0.1, 0.2),
        k_i=(reference.k_i * 1.2, reference.k_i * 1.2),
        k_ii=(0.0, 0.0),
        j_integral=(reference.j_integral * 1.44,) * 2,
    )

    assert fracture.verify_stress_intensity(accepted, reference).status == "accepted"
    assert fracture.verify_stress_intensity(inaccurate, reference).status == "failed"
