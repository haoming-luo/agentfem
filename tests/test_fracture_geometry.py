from __future__ import annotations

from math import pi

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
