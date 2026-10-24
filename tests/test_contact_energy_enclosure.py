"""Bounded energy evidence for a closed tool; never a Newton contact law."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem.boundary_models.closed_polyhedron import closed_triangulated_rigid_solid
from agentfem.boundary_models.contact_energy_enclosure import (
    bound_closed_contact_penalty_energy,
)
from agentfem.boundary_models.rigid import prescribed_rigid_motion


def _reentrant_tool():
    polygon = np.asarray(((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), dtype=float)
    vertices = np.vstack(
        (
            np.column_stack((polygon, np.zeros(6))),
            np.column_stack((polygon, np.ones(6))),
        )
    )
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    faces = [tuple(reversed(face)) for face in top]
    faces += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        faces.extend(((first, second, second + 6), (first, second + 6, first + 6)))
    return closed_triangulated_rigid_solid(
        vertices=vertices, triangles=faces, reviewed_embedded=True
    )


_SLAVE = np.asarray(((0.9, 0.9, 0.5), (3.0, 0.9, 0.5), (0.9, 3.0, 0.5)))


def test_reentrant_face_energy_lower_bound_detects_contact_missed_by_q2():
    tool = _reentrant_tool()
    q2 = np.asarray(
        ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
    )
    projection = tool.project(q2 @ _SLAVE)
    assert projection.all_valid
    assert np.all(projection.signed_gaps > 0.0)

    result = bound_closed_contact_penalty_energy(
        _SLAVE,
        _SLAVE,
        tool,
        penalty=1.0,
        maximum_depth=6,
        maximum_cells=5_000,
        relative_tolerance=0.0,
    )
    # The independent extruded-L segment-distance refinement in
    # tools/verify_reentrant_contact_integral.py gives ~0.000349913.
    reference = 0.000349913
    assert result.lower_bound > 0.0
    assert result.lower_bound < reference < result.upper_bound
    assert result.reference_area == pytest.approx(2.205)
    assert result.status == "depth_budget_exhausted"
    assert not result.within_requested_tolerance
    assert result.nonunique_nearest_points > 0
    assert result.unresolved_signed_distance_points == 0
    assert result.leaf_count <= 5_000


def test_ambiguous_center_retains_scalar_distance_but_no_normal():
    result = bound_closed_contact_penalty_energy(
        _SLAVE,
        _SLAVE,
        _reentrant_tool(),
        penalty=1.0,
        maximum_depth=0,
        relative_tolerance=0.0,
    )
    assert result.nonunique_nearest_points == 1
    assert result.unresolved_signed_distance_points == 0
    assert result.lower_bound == 0.0
    assert result.upper_bound > 0.000349913
    assert result.status == "depth_budget_exhausted"


def test_smooth_constant_depth_and_strict_separation_are_enclosed():
    tool = _reentrant_tool()
    face = np.asarray(((0.1, 0.3, 0.4), (0.1, 0.4, 0.4), (0.1, 0.3, 0.5)))
    area = 0.005
    exact = 0.5 * 2.0 * 0.1**2 * area
    active = bound_closed_contact_penalty_energy(
        face,
        face,
        tool,
        penalty=2.0,
        maximum_depth=4,
        relative_tolerance=0.0,
    )
    assert active.lower_bound < exact < active.upper_bound
    assert active.lower_bound > 0.0
    far = bound_closed_contact_penalty_energy(
        face,
        face + (10.0, 0.0, 0.0),
        tool,
        penalty=2.0,
    )
    assert far.lower_bound == 0.0
    assert far.upper_bound <= np.finfo(float).tiny
    assert far.within_requested_tolerance


def test_rigid_tool_translation_preserves_same_energy_bounds():
    tool = _reentrant_tool()
    original = bound_closed_contact_penalty_energy(
        _SLAVE,
        _SLAVE,
        tool,
        penalty=3.0,
        maximum_depth=6,
        relative_tolerance=0.0,
    )
    offset = np.asarray((0.2, -0.1, 0.05))
    moved = bound_closed_contact_penalty_energy(
        _SLAVE,
        _SLAVE + offset,
        tool,
        penalty=3.0,
        motion=prescribed_rigid_motion(translation=offset),
        factor=1.0,
        maximum_depth=6,
        relative_tolerance=0.0,
    )
    assert moved.lower_bound == pytest.approx(original.lower_bound, abs=1e-12)
    # Roundoff can change whether a medial-axis centroid is classified as
    # ambiguous. Its conservative vertex fallback changes the *width*, not
    # the physical energy enclosed by either rigidly equivalent trial.
    physical_reference = 3.0 * 0.000349913
    assert original.lower_bound < physical_reference < original.upper_bound
    assert moved.lower_bound < physical_reference < moved.upper_bound
    assert moved.reference_area == original.reference_area
    assert moved.leaf_count == original.leaf_count


@pytest.mark.parametrize(
    "options",
    (
        {"penalty": 0.0},
        {"penalty": 1.0, "relative_tolerance": -1.0},
        {"penalty": 1.0, "maximum_cells": 0},
        {"penalty": 1.0, "maximum_depth": 17},
    ),
)
def test_invalid_enclosure_policy_rejected(options):
    with pytest.raises(ValueError):
        bound_closed_contact_penalty_energy(
            _SLAVE, _SLAVE, _reentrant_tool(), **options
        )


def test_smooth_face_with_all_competitors_bvh_excluded_retains_exact_operator():
    from agentfem.boundary_models.contact_smooth_face_research import (
        integrate_certified_smooth_closed_face,
    )

    face = np.asarray(((1.03125, 1.81875, 0.5),
                       (1.03125, 1.884375, 0.5),
                       (0.965625, 1.884375, 0.5)))
    tool = _reentrant_tool()

    def evaluate(current):
        return integrate_certified_smooth_closed_face(face, current, tool, penalty=7.0)

    result = evaluate(face)
    assert result.examined_master_facet_count == 0
    assert 0.0 < result.active_reference_fraction < 1.0
    # Exact integral of the clipped affine gap on the reference triangle:
    # active area fraction = (d / (d + positive_gap))**2 and mean g**2=d**2/6.
    depth = 1.0 - face[2, 0]
    fraction = (depth / (face[0, 0] - face[2, 0]))**2
    area = 0.5 * np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0]))
    assert result.response.potential == pytest.approx(7.0 * area * fraction * depth**2 / 12)
    direction = np.asarray(((0.3, -0.1, 0.2), (-0.2, 0.1, 0.1), (0.1, 0.2, -0.1)))
    h = 1e-6
    plus, minus = evaluate(face + h * direction), evaluate(face - h * direction)
    assert (plus.response.potential - minus.response.potential) / (2*h) == pytest.approx(
        np.sum(result.response.nodal_residual * direction), rel=1e-8, abs=1e-12,
    )
    np.testing.assert_allclose(
        (plus.response.nodal_residual - minus.response.nodal_residual) / (2*h),
        np.einsum('aibj,bj->ai', result.response.nodal_tangent, direction),
        rtol=1e-8, atol=1e-12,
    )
