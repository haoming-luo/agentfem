# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from agentfem import constitutive, materials


def _strengths():
    return constitutive.composite_strengths_2d(
        xt=1500.0,
        xc=1000.0,
        yt=50.0,
        yc=200.0,
        s12=100.0,
        name="carbon_epoxy_strengths",
    )


def test_maximum_stress_is_sign_aware_and_reports_exact_load_factor():
    assessment = constitutive.assess_ply_failure(
        [-500.0, 25.0, -20.0],
        _strengths(),
        criterion="maximum_stress",
    )

    assert assessment.indices == pytest.approx(
        {
            "fiber_tension": 0.0,
            "fiber_compression": 0.5,
            "matrix_tension": 0.5,
            "matrix_compression": 0.0,
            "in_plane_shear": 0.2,
        }
    )
    assert assessment.governing_mode in {"fiber_compression", "matrix_tension"}
    assert assessment.maximum_index == pytest.approx(0.5)
    assert assessment.load_factor_to_first_failure == pytest.approx(2.0)
    assert assessment.accepted is True


def test_hashin_modes_and_proportional_failure_factor_are_consistent():
    strengths = _strengths()
    stress = np.array([750.0, 25.0, 50.0])
    assessment = constitutive.assess_ply_failure(stress, strengths)

    assert assessment.indices["fiber_tension"] == pytest.approx(0.5)
    assert assessment.indices["matrix_tension"] == pytest.approx(0.5)
    assert assessment.indices["fiber_compression"] == 0.0
    assert assessment.indices["matrix_compression"] == 0.0
    assert assessment.load_factor_to_first_failure == pytest.approx(np.sqrt(2.0))
    at_failure = constitutive.assess_ply_failure(
        assessment.load_factor_to_first_failure * stress,
        strengths,
    )
    assert at_failure.maximum_index == pytest.approx(1.0, rel=1.0e-12)


def test_hashin_matrix_compression_uses_nonlinear_proportional_root():
    strengths = constitutive.composite_strengths_2d(
        xt=1500.0,
        xc=1000.0,
        yt=50.0,
        yc=200.0,
        s12=80.0,
    )
    stress = np.array([0.0, -100.0, 50.0])
    assessment = constitutive.assess_ply_failure(stress, strengths)

    assert assessment.governing_mode == "matrix_compression"
    assert assessment.indices["matrix_compression"] == pytest.approx(0.5)
    at_failure = constitutive.assess_ply_failure(
        assessment.load_factor_to_first_failure * stress,
        strengths,
    )
    assert at_failure.maximum_index == pytest.approx(1.0, rel=1.0e-12)
    assert assessment.load_factor_to_first_failure != pytest.approx(np.sqrt(2.0))


def test_ply_failure_assessment_is_extensible_and_fail_closed():
    class UserCriterion:
        name = "user_linear_interaction"

        def evaluate(self, material_stress, strengths):
            return {
                "combined": (
                    abs(material_stress[0]) / strengths.longitudinal_tension
                    + abs(material_stress[1]) / strengths.transverse_tension
                )
            }

    custom = constitutive.assess_ply_failure(
        [150.0, 5.0, 0.0],
        _strengths(),
        criterion=UserCriterion(),
    )
    assert custom.criterion == "user_linear_interaction"
    assert custom.maximum_index == pytest.approx(0.2)
    assert custom.load_factor_to_first_failure == pytest.approx(5.0)
    assert custom.as_dict()["interpretation"] == (
        "initiation_screening_not_damage_evolution"
    )

    with pytest.raises(ValueError, match="sigma_11"):
        constitutive.assess_ply_failure([1.0, 2.0], _strengths())
    with pytest.raises(ValueError, match="Unknown ply failure criterion"):
        constitutive.assess_ply_failure([1.0, 2.0, 3.0], _strengths(), criterion="puck")
    with pytest.raises(ValueError, match="positive and finite"):
        constitutive.composite_strengths_2d(xt=0, xc=1, yt=1, yc=1, s12=1)


def test_laminate_failure_rotates_every_section_point_to_material_axes():
    lamina = constitutive.orthotropic_plane_stress_2d(
        ex=135.0e9,
        ey=10.0e9,
        nuxy=0.3,
        gxy=5.0e9,
        density=1600.0,
    )
    section = materials.laminate(
        [
            materials.ply(lamina, 0.125e-3, angle=0.0, name="ply_0", integration_points=1),
            materials.ply(lamina, 0.125e-3, angle=90.0, name="ply_90", integration_points=1),
        ],
        name="cross_ply",
    )
    response = section.evaluate((1.0e-3, 0.0, 0.0))
    assessment = constitutive.assess_laminate_failure(
        section,
        response,
        _strengths(),
        criterion="maximum_stress_2d",
    )

    assert len(assessment.points) == 2
    point_0 = assessment.by_id("ply_0:section-point-1")
    point_90 = assessment.by_id("ply_90:section-point-1")
    np.testing.assert_allclose(point_0.material_stress, response.section_points[0].stress)
    assert point_90.material_stress[0] == pytest.approx(response.section_points[1].stress[1])
    assert point_90.material_stress[1] == pytest.approx(response.section_points[1].stress[0])
    assert assessment.governing_point_id in {
        "ply_0:section-point-1",
        "ply_90:section-point-1",
    }
    assert assessment.as_dict()["interpretation"] == (
        "first_ply_initiation_not_progressive_damage"
    )

    with pytest.raises(ValueError, match="match ply names exactly"):
        constitutive.assess_laminate_failure(
            section,
            response,
            {"ply_0": _strengths()},
        )
