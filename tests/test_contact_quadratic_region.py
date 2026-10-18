"""Independent parabolic feature-boundary check for the conic primitive."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

from agentfem.boundary_models.contact_quadratic_region import (
    QuadraticReferenceLevelSet,
    quadratic_level_set_boundary_quadrature,
    quadratic_region_triangle_quadrature,
)


_ORACLE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "tools"
        / "verify_quadratic_contact_feature_transition.py"
    )
)
_VERTICES = np.asarray(((1.0, 0.0), (2.0, 0.0), (2.0, 3.0)))


def _feature_level_set(current_vertices):
    x0, y0 = current_vertices[0]
    ux, uy = current_vertices[1] - current_vertices[0]
    vx, vy = current_vertices[2] - current_vertices[0]
    return QuadraticReferenceLevelSet(
        a=-0.5 * vx * vx,
        b0=vy - (x0 - 3.0) * vx,
        b1=-ux * vx,
        c0=y0 - 0.5 * ((x0 - 3.0) ** 2 + 1.0),
        c1=uy - (x0 - 3.0) * ux,
        c2=-0.5 * ux * ux,
    )


@pytest.mark.parametrize("perturbed", [False, True])
def test_parabolic_edge_vertex_region_integrals_match_independent_oracle(perturbed):
    displacement = (
        np.asarray(((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01)))
        if perturbed
        else np.zeros((3, 2))
    )
    level_set = _feature_level_set(_VERTICES + displacement)
    edge_shapes, edge_weights = quadratic_region_triangle_quadrature(
        level_set,
        negative=True,
        order=24,
    )
    vertex_shapes, vertex_weights = quadratic_region_triangle_quadrature(
        level_set,
        negative=False,
        order=24,
    )
    assert edge_weights.sum() + vertex_weights.sum() == pytest.approx(0.5)
    assert np.all(edge_weights > 0.0) and np.all(vertex_weights > 0.0)
    current = _VERTICES + displacement
    potential = 0.0
    residual = np.zeros((3, 2))
    bulk = np.zeros((3, 2, 3, 2))
    for shapes, unit_weights, vertex_branch in (
        (edge_shapes, edge_weights, False),
        (vertex_shapes, vertex_weights, True),
    ):
        points = shapes @ current
        difference = (
            points - np.asarray((3.0, 1.0))
            if vertex_branch
            else np.column_stack((np.zeros(len(points)), points[:, 1]))
        )
        hessian = np.eye(2) if vertex_branch else np.diag((0.0, 1.0))
        # The affine physical map has reference Jacobian determinant 3.
        weights = 3.0 * unit_weights
        potential += 0.5 * np.sum(weights * np.sum(difference**2, axis=1))
        residual += np.einsum("q,qa,qi->ai", weights, shapes, difference)
        bulk += np.einsum("q,qa,qb,ij->aibj", weights, shapes, shapes, hessian)
    oracle_potential, oracle_residual, oracle_bulk, _entry = _ORACLE[
        "split_residual_and_bulk_tangent"
    ](displacement)
    np.testing.assert_allclose(potential, oracle_potential, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(residual, oracle_residual, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(bulk, oracle_bulk, rtol=2e-8, atol=2e-9)


def test_quadratic_region_refuses_an_identically_tied_feature_branch():
    with pytest.raises(ValueError, match="nonzero level set"):
        QuadraticReferenceLevelSet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("perturbed", [False, True])
def test_parabolic_moving_boundary_term_matches_independent_oracle(perturbed):
    displacement = (
        np.asarray(((0.012, -0.006), (-0.008, 0.014), (0.004, -0.01)))
        if perturbed
        else np.zeros((3, 2))
    )
    current = _VERTICES + displacement
    shapes, coarea_weights = quadratic_level_set_boundary_quadrature(
        _feature_level_set(current),
        order=24,
    )
    assert len(shapes) > 0
    assert np.all(coarea_weights > 0.0)
    current_x = (shapes @ current)[:, 0]
    difference = np.column_stack((3.0 - current_x, np.ones(len(shapes))))
    moving = -3.0 * np.einsum(
        "q,qa,qb,qi,qj->aibj",
        coarea_weights,
        shapes,
        shapes,
        difference,
        difference,
    )
    _potential, _residual, _bulk, entry = _ORACLE["split_residual_and_bulk_tangent"](
        displacement
    )
    expected = _ORACLE["moving_feature_tangent"](entry, displacement)
    np.testing.assert_allclose(moving, expected, rtol=2e-8, atol=2e-9)


def test_coarea_rule_transposes_a_vertical_reference_feature():
    # q=s-1/4 has a vertical segment of reference length 3/4.
    shapes, weights = quadratic_level_set_boundary_quadrature(
        QuadraticReferenceLevelSet(0.0, 0.0, 0.0, -0.25, 1.0, 0.0),
        order=12,
    )
    assert weights.sum() == pytest.approx(0.75, rel=1e-13)
    np.testing.assert_allclose(shapes[:, 1], 0.25, rtol=0.0, atol=1e-13)
    assert np.all(shapes[:, 2] >= 0.0)


def test_coarea_rule_refuses_a_two_orientation_singular_component():
    # q=s*(t-1/2) has a vertical component as well as an interior line;
    # silently retaining just one component would yield a false tangent.
    with pytest.raises(NotImplementedError, match="triangle edge"):
        quadratic_level_set_boundary_quadrature(
            QuadraticReferenceLevelSet(0.0, 0.0, 1.0, 0.0, -0.5, 0.0),
            order=12,
        )


def test_coarea_rule_refuses_an_isolated_singular_point():
    # (s-0.4)^2+(t-0.3)^2=0 has no regular curve. Returning an empty
    # quadrature would silently omit a distributional contribution.
    with pytest.raises(NotImplementedError, match="singular point"):
        quadratic_level_set_boundary_quadrature(
            QuadraticReferenceLevelSet(1.0, -0.6, 0.0, 0.25, -0.8, 1.0),
            order=12,
        )


def test_coarea_rule_certifies_a_uniformly_absent_boundary():
    shapes, weights = quadratic_level_set_boundary_quadrature(
        QuadraticReferenceLevelSet(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        order=12,
    )
    assert shapes.shape == (0, 3)
    assert weights.shape == (0,)


@pytest.mark.parametrize("factor", [1e-300, 1e-6, 1.0, 1e6, 1e300])
def test_coarea_weights_follow_level_set_scaling(factor):
    shapes, weights = quadratic_level_set_boundary_quadrature(
        QuadraticReferenceLevelSet(0.0, 0.0, 0.0, -0.25 * factor, factor, 0.0),
        order=12,
    )
    assert weights.sum() == pytest.approx(0.75 / factor, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(shapes[:, 1], 0.25, rtol=0.0, atol=1e-13)
    _, area_weights = quadratic_region_triangle_quadrature(
        QuadraticReferenceLevelSet(0.0, 0.0, 0.0, -0.25 * factor, factor, 0.0),
        negative=True,
        order=12,
    )
    assert area_weights.sum() == pytest.approx(0.21875, rel=1e-12, abs=0.0)


def test_vertical_moving_boundary_matches_repartitioned_area_derivative():
    def area(threshold):
        _, weights = quadratic_region_triangle_quadrature(
            QuadraticReferenceLevelSet(0.0, 0.0, 0.0, -threshold, 1.0, 0.0),
            negative=True,
            order=12,
        )
        return weights.sum()

    threshold = 0.25
    step = 1e-5
    numerical = (area(threshold + step) - area(threshold - step)) / (2 * step)
    _, boundary_weights = quadratic_level_set_boundary_quadrature(
        QuadraticReferenceLevelSet(0.0, 0.0, 0.0, -threshold, 1.0, 0.0),
        order=12,
    )
    assert numerical == pytest.approx(boundary_weights.sum(), rel=1e-11)
    assert numerical == pytest.approx(1.0 - threshold, rel=1e-11)


def test_circular_level_set_coarea_matches_analytical_disk_derivative():
    angle = 2.0 * np.pi * np.arange(3) / 3.0
    face = np.column_stack((1.2 * np.cos(angle), 1.2 * np.sin(angle), np.full(3, 0.95)))
    origin = face[0]
    u = face[1] - origin
    v = face[2] - origin
    level_set = QuadraticReferenceLevelSet(
        a=float(v @ v),
        b0=2.0 * float(origin @ v),
        b1=2.0 * float(u @ v),
        c0=float(origin @ origin) - 1.0,
        c1=2.0 * float(origin @ u),
        c2=float(u @ u),
    )
    shapes, coarea_weights = quadratic_level_set_boundary_quadrature(
        level_set,
        order=24,
    )
    assert len(shapes) > 0
    jacobian = np.linalg.norm(np.cross(u, v))
    # d/d(R²) area{rho²<R²} = pi when the disk is fully inside the face.
    assert jacobian * coarea_weights.sum() == pytest.approx(np.pi, rel=5e-8)
