"""Exact affine contact-boundary oracle; not a public Step acceptance test."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_contact_boundary_quadrature.py"
clipped_affine_gap_integral = runpy.run_path(str(_SCRIPT))["clipped_affine_gap_integral"]


@pytest.mark.parametrize("width", (0.05, 0.2, 0.5))
def test_clipped_affine_triangle_matches_independent_exact_potential(width):
    gaps = np.asarray((-width, 1 - width, -width))
    value = clipped_affine_gap_integral(gaps)
    assert value["active_reference_area"] == pytest.approx(width - width**2 / 2)
    assert value["potential"] == pytest.approx(width**3 / 6 - width**4 / 24)
    assert np.all(np.linalg.eigvalsh(value["nodal_tangent"]) >= -1e-14)


def test_clipped_affine_triangle_residual_and_tangent_are_potential_derivatives():
    gaps = np.asarray((-0.2, 0.8, -0.2))
    direction = np.asarray((0.3, -0.4, 0.7))
    value = clipped_affine_gap_integral(gaps, penalty=17.0)
    h = 1e-6
    plus = clipped_affine_gap_integral(gaps + h * direction, penalty=17.0)
    minus = clipped_affine_gap_integral(gaps - h * direction, penalty=17.0)
    energy_derivative = (plus["potential"] - minus["potential"]) / (2 * h)
    residual_derivative = (plus["nodal_residual"] - minus["nodal_residual"]) / (2 * h)
    assert energy_derivative == pytest.approx(value["nodal_residual"] @ direction, rel=1e-9)
    np.testing.assert_allclose(
        residual_derivative, value["nodal_tangent"] @ direction,
        rtol=1e-9, atol=1e-10,
    )


def test_clipped_affine_triangle_open_and_full_contact_limits():
    open_result = clipped_affine_gap_integral((0.2, 0.3, 0.4))
    assert open_result["active_reference_area"] == 0.0
    assert open_result["potential"] == 0.0
    assert not np.any(open_result["nodal_residual"])
    full = clipped_affine_gap_integral((-0.1, -0.1, -0.1), penalty=2.0)
    assert full["active_reference_area"] == pytest.approx(0.5)
    assert full["potential"] == pytest.approx(0.005)


@pytest.mark.parametrize("gaps", ((0.0, 0.0), (1.0, 2.0, float("nan"))))
def test_clipped_affine_triangle_rejects_invalid_inputs(gaps):
    with pytest.raises(ValueError, match="three finite"):
        clipped_affine_gap_integral(gaps)
