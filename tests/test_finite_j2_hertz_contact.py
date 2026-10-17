"""Structural contact symmetry gate separate from the Hertz asymptotic claim."""

from __future__ import annotations

import pytest

from examples.finite_j2_hertz_contact import run_case


def test_full_and_quarter_indentation_agree_with_contact_on_symmetry_dofs():
    full = run_case(cells=4, penalty=1.0e5, indentation=0.02)
    quarter = run_case(
        cells=4, penalty=1.0e5, indentation=0.02, quarter=True,
    )
    assert full["status"] == quarter["status"] == "completed"
    assert quarter["cell_count"] * 4 == full["cell_count"]
    assert quarter["active_quadrature_points"] * 4 == full["active_quadrature_points"]
    assert quarter["full_contact_force"] == pytest.approx(
        full["full_contact_force"], rel=1.0e-9,
    )
    assert quarter["contact_potential"] * 4 == pytest.approx(
        full["contact_potential"], rel=1.0e-9,
    )


def test_hertz_reference_is_labeled_as_limit_not_exact_solution():
    result = run_case(cells=4, penalty=1.0e5, indentation=0.02)
    assert result["full_contact_force"] > 0.0
    assert result["hertz_half_space_force"] > 0.0
    assert "half_space" in result["reference_scope"]
    assert "finite_penalty" in result["model_scope"]
