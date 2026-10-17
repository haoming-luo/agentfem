"""Structural contact symmetry gate separate from the Hertz asymptotic claim."""

from __future__ import annotations

import math

import pytest

from examples.finite_j2_hertz_contact import run_case


@pytest.mark.parametrize("grading", [0.0, 2.0])
def test_full_and_quarter_indentation_agree_with_contact_on_symmetry_dofs(grading):
    full = run_case(cells=4, penalty=1.0e5, indentation=0.02, grading=grading)
    quarter = run_case(
        cells=4, penalty=1.0e5, indentation=0.02, quarter=True,
        grading=grading,
    )
    assert full["status"] == quarter["status"] == "completed"
    assert quarter["cell_count"] * 4 == full["cell_count"]
    assert quarter["active_quadrature_points"] * 4 == full["active_quadrature_points"]
    assert quarter["active_reference_area"] == pytest.approx(
        full["active_reference_area"], rel=1.0e-12,
    )
    assert quarter["maximum_active_quadrature_radius"] == pytest.approx(
        full["maximum_active_quadrature_radius"], rel=1.0e-12,
    )
    assert quarter["full_contact_force"] == pytest.approx(
        full["full_contact_force"], rel=1.0e-9,
    )
    assert quarter["contact_potential"] * 4 == pytest.approx(
        full["contact_potential"], rel=1.0e-9,
    )
    assert quarter["hertz_force_sampled_on_reference_trace"] == pytest.approx(
        full["hertz_force_sampled_on_reference_trace"], rel=1.0e-12,
    )
    assert quarter["nominal_pressure_l1_difference_over_hertz_force"] == pytest.approx(
        full["nominal_pressure_l1_difference_over_hertz_force"], rel=1.0e-9,
    )


def test_hertz_reference_is_labeled_as_limit_not_exact_solution():
    result = run_case(cells=4, penalty=1.0e5, indentation=0.02)
    assert result["full_contact_force"] > 0.0
    assert result["hertz_half_space_force"] > 0.0
    assert result["hertz_half_space_contact_radius"] > 0.0
    assert result["hertz_half_space_contact_area"] > 0.0
    assert result["hertz_half_space_peak_pressure"] > 0.0
    assert result["peak_quadrature_pressure"] > 0.0
    assert result["total_case_wall_seconds"] > 0.0
    assert result["accepted_increment_seconds"] > 0.0
    assert all(
        value >= 0.0
        for value in result["accepted_solver_phase_seconds"].values()
    )
    assert "half_space" in result["reference_scope"]
    assert "finite_penalty" in result["model_scope"]


def test_benchmark_keeps_contact_radius_while_reducing_indent_to_radius_ratio():
    reference = run_case(
        cells=2, penalty=1.0e5, indentation=0.02, radius=2.0,
        quarter=True, increments=2,
    )
    smaller_strain = run_case(
        cells=2, penalty=1.0e5, indentation=0.002, radius=20.0,
        quarter=True, increments=2,
    )
    assert smaller_strain["status"] == "completed"
    assert smaller_strain["hertz_half_space_contact_radius"] == pytest.approx(
        reference["hertz_half_space_contact_radius"],
    )
    assert smaller_strain["hertz_half_space_force"] == pytest.approx(
        reference["hertz_half_space_force"] / 10.0,
    )
    assert smaller_strain["plastic_dissipation"] == 0.0


@pytest.mark.parametrize("contact_degree,actual_degree", [(None, 4), (7, 7)])
def test_quadratic_displacement_hertz_screen_is_a_separate_mesh_diagnostic(
    contact_degree, actual_degree,
):
    result = run_case(
        cells=2, penalty=1.0e5, indentation=0.002, radius=20.0,
        quarter=True, increments=2,
        displacement_degree=2,
        contact_quadrature_degree=contact_degree,
    )
    assert result["status"] == "completed"
    assert result["displacement_degree"] == 2
    assert result["bulk_quadrature_degree"] == 4
    assert result["contact_quadrature_degree"] == actual_degree
    assert f"degree_{actual_degree}" in result["contact_quadrature_rule"]
    assert result["full_contact_force"] > 0.0
    assert result["plastic_dissipation"] == 0.0
    assert result["hertz_force_sampled_on_reference_trace"] >= 0.0
    assert math.isfinite(result["hertz_trace_sampling_force_ratio"])
    assert result["nominal_pressure_l1_difference_over_hertz_force"] >= 0.0


def test_quadratic_hertz_screen_rejects_low_explicit_bulk_rule():
    with pytest.raises(NotImplementedError, match="AFM-J2-QUADRATURE"):
        run_case(
            cells=2, penalty=1.0e5, indentation=0.002, radius=20.0,
            quarter=True, increments=2, displacement_degree=2,
            bulk_quadrature_degree=2,
        )


@pytest.mark.parametrize("invalid", [0, 3])
def test_benchmark_rejects_unsupported_displacement_order(invalid):
    with pytest.raises(ValueError, match="displacement degree"):
        run_case(cells=2, penalty=1.0e5, indentation=0.02,
                 displacement_degree=invalid)


def test_explicit_iterative_solver_policy_preserves_contact_solution():
    options = dict(cells=4, penalty=1.0e5, indentation=0.02, quarter=True)
    direct = run_case(**options)
    iterative = run_case(**options, linear_policy="gmres_gamg")
    assert direct["linear_policy"] == "direct"
    assert iterative["linear_policy"] == "gmres_gamg"
    assert iterative["linear_iterations"] >= iterative["linear_solve_calls"]
    assert all(reason > 0 for reason in iterative["linear_converged_reasons"])
    for name in (
        "full_contact_force", "active_reference_area", "maximum_penetration",
        "contact_potential", "stored_energy", "tool_work",
    ):
        assert iterative[name] == pytest.approx(direct[name], rel=1.0e-8, abs=1.0e-10)
