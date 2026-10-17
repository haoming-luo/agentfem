from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.rigid import _closest_point_on_triangle
from agentfem.boundary_models.implicit_facet_contact import (
    linearize_piecewise_planar_contact,
)


VERTICES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
OUTWARD = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))


def _tool():
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=VERTICES, triangles=OUTWARD, name="tetra_tool",
    )


def _response(surface, query):
    trace = boundary_models.ContactTrace(
        point_ids=(7,), node_ids=((0, 1, 2),),
        shape_values=((0.2, 0.3, 0.5),), weights=(0.75,),
    )
    evaluation = trace.evaluate(np.tile(query, (3, 1)))
    record = boundary_models.ContactProjectionRecord(
        trace.point_ids, surface.project(evaluation.query_points),
    )
    response = boundary_models.frictionless_penalty_contact_law(125.0).evaluate(
        record.projection,
    )
    return linearize_piecewise_planar_contact(
        evaluation, record, response, surface,
    )


def test_convex_closed_tool_projection_and_sharp_branch_identity():
    surface = _tool()
    projection = surface.project(((0.1, 0.2, 0.3), (0.2, 0.1, 0.3), (0.5, 0.5, 0.5)))
    assert projection.all_valid
    np.testing.assert_allclose(projection.signed_gaps[:2], (-0.1, -0.1))
    np.testing.assert_array_equal(projection.entity_ids[:2], (2, 1))
    np.testing.assert_allclose(projection.normals[:2], ((-1, 0, 0), (0, -1, 0)))
    assert projection.signed_gaps[2] > 0.0
    tie = surface.project(((0.1, 0.1, 0.1),))
    assert tie.valid[0] and tie.entity_ids[0] == 0
    assert surface.summary()["execution_scope"] == "serial_reference_geometry"


def test_convex_support_plane_batch_boundary_preserves_point_order_and_branch():
    surface = _tool()
    samples = np.asarray(
        ((0.10, 0.20, 0.30), (0.20, 0.10, 0.30),
         (0.15, 0.16, 0.17), (0.10, 0.10, 0.10)),
        dtype=float,
    )
    queries = np.tile(samples, (1026, 1))[:4101]
    batched = surface.project(queries)
    singles = [surface.project((point,)) for point in samples]
    np.testing.assert_allclose(
        batched.signed_gaps,
        np.tile([item.signed_gaps[0] for item in singles], 1026)[:4101],
        rtol=0.0, atol=1.0e-15,
    )
    np.testing.assert_array_equal(
        batched.entity_ids,
        np.tile([item.entity_ids[0] for item in singles], 1026)[:4101],
    )


def test_convex_tool_rejects_open_or_inward_geometry():
    with pytest.raises(ValueError, match="closed"):
        boundary_models.convex_polyhedral_rigid_surface(
            vertices=VERTICES, triangles=OUTWARD[:-1],
        )
    with pytest.raises(ValueError, match="inward-oriented"):
        boundary_models.convex_polyhedral_rigid_surface(
            vertices=VERTICES, triangles=tuple(tuple(reversed(face)) for face in OUTWARD),
        )


def test_convex_exterior_bvh_preserves_exact_triangle_distance_and_cutoff():
    surface = _tool()
    rng = np.random.default_rng(771)
    candidates = rng.uniform(-0.5, 1.5, size=(150, 3))
    exterior = candidates[np.max(
        surface.facets.facet_normals @ candidates.T
        - surface._offsets[:, None], axis=0,
    ) >= 0.0]
    projection = surface.project(exterior)
    assert projection.all_valid
    triangles = surface.facets.vertices[surface.facets.triangles]
    for point, closest, gap in zip(
        exterior, projection.closest_points, projection.signed_gaps, strict=True,
    ):
        brute = min(
            np.linalg.norm(point - _closest_point_on_triangle(point, *triangle))
            for triangle in triangles
        )
        assert gap == pytest.approx(brute, abs=1e-13)
        assert np.linalg.norm(point - closest) == pytest.approx(brute, abs=1e-13)
    limited = surface.project(exterior, maximum_distance=0.15)
    np.testing.assert_array_equal(
        limited.valid, projection.signed_gaps <= 0.15 + 1e-14,
    )


@pytest.mark.parametrize("point", ((0.1, 0.2, 0.3), (0.2, 0.1, 0.3)))
def test_convex_sharp_contact_energy_gradient_and_newton_tangent(point):
    surface = _tool()
    query = np.asarray(point, dtype=float)
    direction = np.asarray((0.3, -0.2, 0.1))
    contribution = _response(surface, query)
    assert contribution.active_points == 1
    assert contribution.potential_energy > 0
    h = 1e-7
    plus = _response(surface, query + h * direction)
    minus = _response(surface, query - h * direction)
    expected = (
        plus.nodal_structural_residual - minus.nodal_structural_residual
    ) / (2 * h)
    np.testing.assert_allclose(
        contribution.tangent_action(np.tile(direction, (3, 1))),
        expected, rtol=1e-8, atol=1e-8,
    )
    energy_derivative = (plus.potential_energy - minus.potential_energy) / (2 * h)
    np.testing.assert_allclose(
        energy_derivative,
        np.sum(contribution.nodal_structural_residual * direction),
        rtol=1e-8, atol=1e-8,
    )


def test_convex_projection_is_objective_under_prescribed_rigid_motion():
    surface = _tool()
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.3, -0.2, 0.1),
        rotation=(0, 0, np.pi / 2),
    )
    reference = np.asarray((0.1, 0.2, 0.3))
    state = motion.state(1.0)
    query = np.asarray(state["reference_point"]) + state["rotation_matrix"] @ reference
    projection = surface.project((query,), motion=motion, factor=1.0)
    assert projection.valid[0]
    assert projection.signed_gaps[0] == pytest.approx(-0.1)
    np.testing.assert_allclose(
        projection.normals[0], state["rotation_matrix"] @ (-np.eye(3)[0]),
    )


def test_noncoplanar_feature_switch_keeps_potential_continuous():
    surface = _tool()
    midpoint = np.asarray((0.1, 0.1, 0.2))
    perturbation = np.asarray((1.0, -1.0, 0.0))
    h = 1.0e-7
    left = _response(surface, midpoint - h * perturbation)
    right = _response(surface, midpoint + h * perturbation)
    tied = _response(surface, midpoint)
    assert left.active_points == right.active_points == tied.active_points == 1
    np.testing.assert_allclose(
        left.potential_energy, right.potential_energy, rtol=1e-5,
    )
    assert tied.potential_energy == pytest.approx(left.potential_energy, rel=1e-5)
    assert not np.allclose(
        left.nodal_structural_residual, right.nodal_structural_residual,
    )
    # The tie has a deterministic limiting normal, not a false smooth normal
    # or a fabricated classical two-sided derivative.
    assert np.isfinite(tied.nodal_structural_residual).all()
