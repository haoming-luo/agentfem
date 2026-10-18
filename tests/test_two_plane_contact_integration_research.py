"""Research-only two-feature area plus line derivative oracle."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_two_plane_contact_integration.py"
two_plane_triangle_contact = runpy.run_path(str(_SCRIPT))["two_plane_triangle_contact"]


@pytest.mark.parametrize("width, center", ((0.05, 0.3), (0.2, 0.5), (0.1, 0.8)))
def test_two_plane_integral_and_full_feature_tangent(width, center):
    k = 7.0
    gaps = np.asarray((
        (-center - width, 1 - center - width, -center - width),
        (center - width, center - 1 - width, center - width),
    ))
    normals = np.asarray(((1.0, 0, 0), (-1.0, 0, 0)))
    reference = two_plane_triangle_contact(gaps, normals, penalty=k)
    assert reference["potential"] == pytest.approx(
        k * width**3 * (1 - center) / 3, rel=1e-12, abs=0.0,
    )
    assert np.sum(reference["nodal_residual"][:, 0]) == pytest.approx(
        k * width**3 / 3,
    )
    assert np.sum(reference["nodal_tangent"][0::3, 0::3]) == pytest.approx(
        0.0, abs=1e-12,
    )
    assert np.sum(reference["feature_tangent"][0::3, 0::3]) < 0.0
    direction = np.random.default_rng(71).normal(size=(3, 3))
    h = 1e-7
    changes = np.einsum("ai,fi->fa", direction, normals)
    plus = two_plane_triangle_contact(gaps + h * changes, normals, penalty=k)
    minus = two_plane_triangle_contact(gaps - h * changes, normals, penalty=k)
    energy_derivative = (plus["potential"] - minus["potential"]) / (2 * h)
    force_derivative = (
        plus["nodal_residual"] - minus["nodal_residual"]
    ) / (2 * h)
    assert energy_derivative == pytest.approx(
        np.sum(reference["nodal_residual"] * direction), rel=1e-7,
    )
    np.testing.assert_allclose(
        force_derivative.reshape(-1),
        reference["nodal_tangent"] @ direction.reshape(-1),
        rtol=2e-7, atol=1e-9,
    )
    np.testing.assert_allclose(
        reference["nodal_tangent"], reference["nodal_tangent"].T,
        atol=1e-12,
    )


def test_two_plane_oracle_rejects_coincident_feature_partition():
    with pytest.raises(ValueError, match="unique feature"):
        two_plane_triangle_contact(
            np.zeros((2, 3)), ((1, 0, 0), (-1, 0, 0)),
        )


def test_two_plane_oracle_is_invariant_to_feature_and_vertex_order():
    gaps = np.asarray(((-0.42, 0.61, -0.37), (0.31, -0.28, 0.19)))
    normals = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    reference = two_plane_triangle_contact(gaps, normals, penalty=8.0)
    exchanged = two_plane_triangle_contact(gaps[::-1], normals[::-1], penalty=8.0)
    assert exchanged["potential"] == pytest.approx(reference["potential"], rel=1e-14)
    np.testing.assert_allclose(exchanged["nodal_residual"], reference["nodal_residual"], atol=1e-14)
    np.testing.assert_allclose(exchanged["nodal_tangent"], reference["nodal_tangent"], atol=1e-14)
    permutation = np.asarray((2, 0, 1))
    reordered = two_plane_triangle_contact(gaps[:, permutation], normals, penalty=8.0)
    dofs = np.arange(9).reshape(3, 3)[permutation].reshape(-1)
    assert reordered["potential"] == pytest.approx(reference["potential"], rel=1e-14)
    np.testing.assert_allclose(
        reordered["nodal_residual"], reference["nodal_residual"][permutation], atol=1e-14,
    )
    np.testing.assert_allclose(
        reordered["nodal_tangent"], reference["nodal_tangent"][np.ix_(dofs, dofs)],
        atol=1e-14,
    )


def test_two_plane_oracle_rejects_nonunique_boundary_switch():
    with pytest.raises(ValueError, match="two-sided tangent"):
        two_plane_triangle_contact(
            ((-0.2, 0.8, -0.2), (-0.2, -0.2, -0.2)),
            ((1.0, 0, 0), (-1.0, 0, 0)),
        )


def test_two_plane_oracle_oblique_active_switch_matches_energy_and_force_differences():
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    normals = np.asarray(((0.8, 0.6, 0.0), (-0.8, 0.6, 0.0)))
    gaps = np.column_stack((
        0.8 * (vertices[:, 0] - 0.45) + 0.6 * vertices[:, 1] - 0.2,
        -0.8 * (vertices[:, 0] - 0.45) + 0.6 * vertices[:, 1] - 0.2,
    )).T
    reference = two_plane_triangle_contact(gaps, normals, penalty=13.0)
    assert reference["potential"] > 0.0
    assert np.linalg.norm(reference["feature_tangent"]) > 0.0
    direction = np.random.default_rng(2026).normal(size=(3, 3))
    change = np.einsum("ai,fi->fa", direction, normals)
    for h in (1e-5, 1e-6):
        plus = two_plane_triangle_contact(gaps + h * change, normals, penalty=13.0)
        minus = two_plane_triangle_contact(gaps - h * change, normals, penalty=13.0)
        assert (plus["potential"] - minus["potential"]) / (2 * h) == pytest.approx(
            np.sum(reference["nodal_residual"] * direction), rel=2e-8,
        )
        np.testing.assert_allclose(
            ((plus["nodal_residual"] - minus["nodal_residual"]) / (2 * h)).reshape(-1),
            reference["nodal_tangent"] @ direction.reshape(-1),
            rtol=1e-7, atol=1e-9,
        )
