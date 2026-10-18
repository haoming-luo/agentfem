from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import basix.ufl
from dolfinx import fem, mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import boundary_models
from agentfem import mesh as agent_mesh
from agentfem.boundary_models import closed_polyhedron as closed_module
from agentfem.boundary_models.closed_polyhedron import (
    closed_triangulated_rigid_solid,
)
from agentfem.boundary_models.contact_response import frictionless_penalty_contact_law
from agentfem.boundary_models.contact_state import ContactProjectionRecord
from agentfem.boundary_models.contact_separation_bound import (
    certify_current_triangle_separation,
)
from agentfem.boundary_models.contact_feature_path import AcceptedContactFeaturePath
from agentfem.boundary_models.contact_planar_branch_bound import (
    _coplanar_patch_covers_face,
    certify_closed_planar_branch,
    certify_reviewed_planar_branch,
)
from agentfem.boundary_models.contact_trace import ContactTrace
from agentfem.boundary_models.initial_gap_bound import (
    certify_initial_reviewed_planar_branch_gap,
)
from agentfem.boundary_models.implicit_contact import integrate_implicit_rigid_contact
from agentfem.boundary_models.implicit_closed_contact import (
    sampled_mixed_feature_slave_faces,
)
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_rigid_contact,
)


def _concave_tool():
    """An outward L-prism with one true reentrant vertical edge."""

    polygon = np.asarray(
        ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)),
        dtype=float,
    )
    vertices = np.vstack((
        np.column_stack((polygon, np.zeros(6))),
        np.column_stack((polygon, np.ones(6))),
    ))
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in top]
    triangles += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend(((first, second, second + 6),
                          (first, second + 6, first + 6)))
    return closed_triangulated_rigid_solid(
        vertices=vertices, triangles=triangles, reviewed_embedded=True,
        name="reviewed_L_prism",
    )


def _tiled_closed_cube(subdivisions):
    """Closed, consistently oriented box with many coplanar seam facets."""

    faces = (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
    )
    vertices, triangles, indices = [], [], {}

    def index(coordinates):
        key = tuple(int(round(subdivisions * value)) for value in coordinates)
        if key not in indices:
            indices[key] = len(vertices)
            vertices.append(np.asarray(key, dtype=float) / subdivisions)
        return indices[key]

    def face_index(origin, axis_u, axis_v, i, j):
        return index(origin + (i * axis_u + j * axis_v) / subdivisions)

    for origin, axis_u, axis_v in faces:
        origin = np.asarray(origin, dtype=float)
        axis_u = np.asarray(axis_u, dtype=float)
        axis_v = np.asarray(axis_v, dtype=float)
        for i in range(subdivisions):
            for j in range(subdivisions):
                first = face_index(origin, axis_u, axis_v, i, j)
                second = face_index(origin, axis_u, axis_v, i + 1, j)
                third = face_index(origin, axis_u, axis_v, i, j + 1)
                fourth = face_index(origin, axis_u, axis_v, i + 1, j + 1)
                triangles.extend(((first, second, third), (second, fourth, third)))
    return closed_triangulated_rigid_solid(
        vertices=np.asarray(vertices), triangles=np.asarray(triangles),
        reviewed_embedded=True, name="tiled_closed_cube",
    )


def test_current_face_separation_uses_whole_triangle_bound_and_current_rigid_pose():
    tool = _concave_tool()
    triangle = np.asarray((((3., 0.4, 0.5), (3., 0.8, 0.5),
                            (3., 0.4, 0.8)),))
    far = certify_current_triangle_separation(triangle, tool)
    assert far.certified
    assert far.minimum_gap_lower_bound > 0
    assert far.sampled_point_count == 0
    assert far.status == "strictly_separated_aabb"
    notch = certify_current_triangle_separation(
        np.asarray((((1.2, 1.6, 0.5), (1.35, 1.6, 0.5),
                     (1.2, 1.75, 0.5)),)), tool,
    )
    assert notch.certified
    assert notch.sampled_point_count >= 1
    assert notch.minimum_gap_lower_bound > 0
    motion = boundary_models.prescribed_rigid_motion(translation=(2.5, 0, 0))
    approached = certify_current_triangle_separation(
        triangle, tool, motion=motion, factor=1.0, maximum_depth=3,
    )
    assert not approached.certified
    assert approached.minimum_gap_lower_bound is None
    assert approached.status in {"contact_not_excluded", "depth_budget_exhausted"}


def test_closed_planar_branch_covers_tiled_master_seams_beyond_small_reference_budget():
    tool = _tiled_closed_cube(12)
    assert len(tool.facets.triangles) == 1728
    triangle = np.asarray((((1.02, 0.38, 0.38), (1.02, 0.45, 0.38),
                            (1.02, 0.38, 0.45)),))
    bound = certify_closed_planar_branch(triangle, tool)
    assert bound.certified, bound.status
    np.testing.assert_allclose(bound.plane_normals, ((1.0, 0.0, 0.0),))


def test_reviewed_open_patch_proves_whole_face_across_coplanar_seam():
    surface = boundary_models.triangulated_rigid_surface(
        vertices=((1., 0., 0.), (1., 1., 0.), (1., 0., 1.), (1., 1., 1.)),
        triangles=((0, 1, 2), (1, 3, 2)), name="reviewed_open_patch",
    )
    search = boundary_models.triangle_surface_bvh(surface)
    crossing = np.asarray((((0.98, 0.42, 0.42), (0.98, 0.58, 0.42),
                            (0.98, 0.42, 0.58)),))
    proven = certify_reviewed_planar_branch(
        crossing, surface, projector=search,
    )
    assert proven.certified, proven.status
    np.testing.assert_allclose(proven.plane_normals, ((1., 0., 0.),))
    np.testing.assert_allclose(
        (crossing[0] - proven.plane_points[0]) @ proven.plane_normals[0],
        -0.02,
    )
    beyond = crossing + np.asarray((0., 0.6, 0.))
    refused = certify_reviewed_planar_branch(
        beyond, surface, projector=search,
    )
    assert not refused.certified
    assert refused.status in {
        "projection_can_leave_master_patch", "another_master_feature_can_compete",
    }

    admitted = certify_initial_reviewed_planar_branch_gap(
        crossing, surface, projector=search, penetration_limit=0.03,
    )
    assert admitted.maximum_penetration_bound == pytest.approx(0.02, abs=1e-12)
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        certify_initial_reviewed_planar_branch_gap(
            crossing, surface, projector=search, penetration_limit=0.01,
        )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-INITIAL-UNCERTIFIED"):
        certify_initial_reviewed_planar_branch_gap(
            beyond, surface, projector=search, penetration_limit=0.03,
        )


def test_reviewed_open_sheet_refinement_proves_only_the_same_nearest_plane():
    surface = boundary_models.triangulated_rigid_surface(
        vertices=(
            (1., 0., 0.), (1., 0., 2.), (1., 2., 2.), (1., 2., 0.),
            (0.7, 1.9, 0.8), (0.8, 1.9, 0.8), (0.7, 1.9, 0.9),
        ),
        triangles=((0, 1, 2), (0, 2, 3), (4, 5, 6)),
        name="open_patch_with_nonplanar_competitor",
    )
    search = boundary_models.triangle_surface_bvh(surface)
    broad = np.asarray((((0.95, 0.4, 0.8), (0.95, 1.4, 0.8),
                         (0.95, 0.4, 1.1)),))
    coarse = certify_reviewed_planar_branch(
        broad, surface, projector=search, maximum_refinement_levels=0,
    )
    refined = certify_reviewed_planar_branch(
        broad, surface, projector=search,
    )
    assert not coarse.certified
    assert coarse.status == "another_master_feature_can_compete"
    assert refined.certified, refined.status
    assert refined.minimum_competitor_clearance > 0
    samples = np.asarray((
        broad[0, 0], broad[0, 1], broad[0, 2],
        broad[0].mean(axis=0), 0.2 * broad[0, 0] + 0.3 * broad[0, 1]
        + 0.5 * broad[0, 2],
    ))
    np.testing.assert_allclose(
        search.project(samples).signed_gaps,
        (samples - refined.plane_points[0]) @ refined.plane_normals[0],
        atol=1e-12,
    )
    near_competitor = broad.copy()
    near_competitor[:, :, 1] += 0.3
    rejected = certify_reviewed_planar_branch(
        near_competitor, surface, projector=search,
    )
    assert not rejected.certified
    assert rejected.status == "another_master_feature_can_compete"


def test_current_face_separation_refuses_hidden_and_budget_limited_contact():
    tool = _concave_tool()
    # All three standard degree-two points are in the free reentrant notch;
    # the corner of this affine face nevertheless penetrates the L-solid.
    crossing = np.asarray((((0.9, 0.9, 0.5), (3., 0.9, 0.5),
                             (0.9, 3., 0.5)),))
    barycentric = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                              (1 / 6, 2 / 3, 1 / 6),
                              (1 / 6, 1 / 6, 2 / 3)))
    sampled = barycentric @ crossing[0]
    assert np.all(tool.project(sampled).signed_gaps > 0)
    assert tool.project(crossing[0, :1]).signed_gaps[0] < 0
    result = certify_current_triangle_separation(crossing, tool)
    assert not result.certified
    assert result.minimum_gap_lower_bound is None
    budget = certify_current_triangle_separation(
        np.asarray((((1.2, 1.2, 0.5), (1.7, 1.2, 0.5),
                     (1.2, 1.6, 0.5)),)),
        tool, maximum_samples=1,
    )
    assert not budget.certified
    assert budget.status in {"sample_budget_exhausted", "depth_budget_exhausted"}


def test_closed_smooth_face_uses_facet_normal_without_near_gap_cancellation():
    tool = _concave_tool()
    points = np.asarray(((1.0 - 1.0e-6, 1.3, 0.5),
                         (1.0 + 1.0e-6, 1.3, 0.5)))
    projection = tool.project(points)
    assert projection.all_valid
    np.testing.assert_allclose(projection.signed_gaps, (-1e-6, 1e-6), atol=1e-14)
    np.testing.assert_array_equal(
        projection.normals, np.asarray(((1., 0., 0.), (1., 0., 0.))),
    )


def test_closed_solid_planar_branch_needs_full_face_and_competitor_clearance():
    tool = _concave_tool()
    cell = np.asarray((((0.95, 1.25, 0.45), (0.95, 1.35, 0.45),
                        (0.95, 1.25, 0.55)),))
    bound = certify_closed_planar_branch(cell, tool)
    assert bound.certified, bound.status
    np.testing.assert_allclose(bound.plane_normals[0], (1., 0., 0.))
    assert bound.minimum_competitor_clearance > 0
    projection = tool.project(cell[0])
    assert projection.all_valid
    assert np.all(projection.signed_gaps < 0)
    np.testing.assert_allclose(
        projection.signed_gaps,
        (cell[0] - bound.plane_points[0]) @ bound.plane_normals[0],
    )
    shifted = boundary_models.prescribed_rigid_motion(translation=(0.2, 0, 0))
    moved = certify_closed_planar_branch(
        cell + np.asarray((0.2, 0., 0.)), tool,
        motion=shifted, factor=1.0,
    )
    assert moved.certified, moved.status
    np.testing.assert_allclose(moved.plane_normals, bound.plane_normals)
    np.testing.assert_allclose(moved.plane_points, bound.plane_points + (0.2, 0., 0.))
    spin = boundary_models.prescribed_rigid_motion(
        translation=(0., 0., 0.), rotation=(0.0, 0.0, 0.2),
        reference_point=(0., 0., 0.),
    )
    rotation = spin.state(1.0)["rotation_matrix"]
    rotated = certify_closed_planar_branch(
        cell @ rotation.T, tool, motion=spin, factor=1.0,
    )
    assert rotated.certified, rotated.status
    np.testing.assert_allclose(rotated.plane_normals, bound.plane_normals @ rotation.T)
    np.testing.assert_allclose(rotated.plane_points, bound.plane_points @ rotation.T)

    # Disconnected slave patches may use different smooth master planes. The
    # proof remains whole-face and does not admit a face crossing their edge.
    lower_wall = np.asarray((((1.25, 0.05, 0.45), (1.35, 0.05, 0.45),
                               (1.25, 0.05, 0.55)),))
    two_patches = certify_closed_planar_branch(
        np.concatenate((cell, lower_wall)), tool,
    )
    assert two_patches.certified, two_patches.status
    np.testing.assert_allclose(two_patches.plane_normals[0], (1., 0., 0.))
    np.testing.assert_allclose(two_patches.plane_normals[1], (0., -1., 0.))

    # The slave face crosses the diagonal triangulation seam of one flat L
    # wall. This is not a physical feature switch; the coplanar union covers it.
    across_flat_seam = np.asarray((((0.95, 1.45, 0.50),
                                    (0.95, 1.55, 0.50),
                                    (0.95, 1.50, 0.60)),))
    seam = certify_closed_planar_branch(across_flat_seam, tool)
    assert seam.certified, seam.status
    np.testing.assert_allclose(seam.plane_normals[0], (1., 0., 0.))

    broad_face = np.asarray((((0.95, 1.25, 0.45), (0.95, 1.45, 0.45),
                              (0.95, 1.25, 0.55)),))
    coarse_only = certify_closed_planar_branch(
        broad_face, tool, maximum_refinement_levels=0,
    )
    assert not coarse_only.certified
    assert coarse_only.status == "another_master_feature_can_compete"
    broad = certify_closed_planar_branch(broad_face, tool)
    assert broad.certified, broad.status
    assert broad.minimum_competitor_clearance > 0
    np.testing.assert_allclose(broad.plane_normals[0], (1., 0., 0.))
    rotated_broad = certify_closed_planar_branch(
        broad_face @ rotation.T, tool, motion=spin, factor=1.0,
    )
    assert rotated_broad.certified, rotated_broad.status
    np.testing.assert_allclose(
        rotated_broad.plane_normals, broad.plane_normals @ rotation.T,
    )
    exhausted = certify_closed_planar_branch(
        broad_face, tool, _refinement_budget=[0],
    )
    assert not exhausted.certified
    assert exhausted.status == "geometry_budget_exceeded"
    barycentric = np.asarray([
        (1 - r - s, r, s)
        for r in np.linspace(0.0, 1.0, 17)
        for s in np.linspace(0.0, 1.0 - r, 9)
    ])
    sampled = barycentric @ broad_face[0]
    np.testing.assert_allclose(
        tool.project(sampled).signed_gaps,
        (sampled - broad.plane_points[0]) @ broad.plane_normals[0],
        atol=1e-12,
    )
    for invalid_depth in (-1, 4, True):
        with pytest.raises(ValueError, match="refinement depth"):
            certify_closed_planar_branch(
                broad_face, tool, maximum_refinement_levels=invalid_depth,
            )

    # The face near the reentrant edge sees both wall and edge/other face.
    crossing = np.asarray((((0.95, 1.01, 0.10), (0.95, 1.35, 0.10),
                             (0.95, 1.01, 0.40)),))
    hazard = certify_closed_planar_branch(crossing, tool)
    assert not hazard.certified
    assert hazard.status in {
        "another_master_feature_can_compete",
        "projection_can_leave_master_patch",
        "ambiguous_center_projection",
    }
    assert not certify_closed_planar_branch(
        cell, tool, maximum_master_facets=1,
    ).certified


def test_coplanar_patch_coverage_uses_union_not_sum_of_facet_areas():
    face = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    left = np.asarray(((0., 0., 0.), (0.5, 0., 0.), (0., 1., 0.)))
    right = np.asarray(((0.5, 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    assert _coplanar_patch_covers_face(
        face, np.asarray((left, right)), (0, 1), np.asarray((0., 0., 1.)),
    )
    # The two masters still have the same *sum* of area as the slave, but
    # overlap on the left and leave an uncovered strip on the right.
    shifted = np.asarray(((0.4, 0., 0.), (0.9, 0., 0.), (0., 1., 0.)))
    assert not _coplanar_patch_covers_face(
        face, np.asarray((left, shifted)), (0, 1), np.asarray((0., 0., 1.)),
    )

    # A flat tool can be triangulated much more finely than a slave face.
    # Coverage does not depend on master-facet ordering or only one seam.
    tiled = []
    for i in range(4):
        for j in range(4):
            x, y = i / 4, j / 4
            a = (x, y, 0.)
            b = (x + 0.25, y, 0.)
            c = (x + 0.25, y + 0.25, 0.)
            d = (x, y + 0.25, 0.)
            tiled.extend(((a, b, c), (a, c, d)))
    tiled = np.asarray(tiled)
    broad = np.asarray(((0.1, 0.1, 0.), (0.9, 0.1, 0.), (0.1, 0.9, 0.)))
    normal = np.asarray((0., 0., 1.))
    shuffled = tiled[np.random.default_rng(73).permutation(len(tiled))]
    assert _coplanar_patch_covers_face(
        broad, shuffled, range(len(shuffled)), normal,
    )
    assert not _coplanar_patch_covers_face(
        broad, tiled, tuple(index for index in range(len(tiled)) if index != 10),
        normal,
    )


def test_closed_smooth_branch_budget_admits_a_modest_refined_trace():
    triangles = []
    for i in range(16):
        for j in range(16):
            y = 1.25 + 0.1 * i / 16
            z = 0.45 + 0.1 * j / 16
            dy = dz = 0.1 / 16
            a = (0.95, y, z)
            b = (0.95, y + dy, z)
            c = (0.95, y + dy, z + dz)
            d = (0.95, y, z + dz)
            triangles.extend(((a, b, c), (a, c, d)))
    trace = np.asarray(triangles)
    assert len(trace) == 512
    assert certify_closed_planar_branch(trace, _concave_tool()).certified
    limited = certify_closed_planar_branch(
        trace, _concave_tool(), maximum_slave_faces=256,
    )
    assert not limited.certified
    assert limited.status == "geometry_budget_exceeded"


def _reentrant_vertex_tool():
    """Seven unit voxels form a solid with the positive octant removed."""

    occupied = {
        (i, j, k)
        for i in range(2) for j in range(2) for k in range(2)
        if (i, j, k) != (1, 1, 1)
    }
    vertices = []
    vertex_ids = {}
    triangles = []
    positive_axes = ((1, 2), (2, 0), (0, 1))
    for cell in sorted(occupied):
        for axis in range(3):
            u, v = positive_axes[axis]
            for side in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += side
                if tuple(neighbor) in occupied:
                    continue
                corners = []
                for bit_u, bit_v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    position = list(cell)
                    position[axis] += int(side > 0)
                    position[u] += bit_u
                    position[v] += bit_v
                    key = tuple(position)
                    if key not in vertex_ids:
                        vertex_ids[key] = len(vertices)
                        vertices.append(np.asarray(key, dtype=float) - 1.0)
                    corners.append(vertex_ids[key])
                if side < 0:
                    corners.reverse()
                triangles.extend(((corners[0], corners[1], corners[2]),
                                  (corners[0], corners[2], corners[3])))
    return closed_triangulated_rigid_solid(
        vertices=vertices, triangles=triangles, reviewed_embedded=True,
        name="reviewed_reentrant_vertex",
    )


def _contribution(surface, query):
    trace = ContactTrace(
        point_ids=(7,), node_ids=((0, 1, 2),),
        shape_values=((0.2, 0.3, 0.5),), weights=(0.75,),
    )
    evaluation = trace.evaluate(np.tile(query, (3, 1)))
    projection = surface.project(evaluation.query_points)
    record = ContactProjectionRecord(trace.point_ids, projection)
    response = frictionless_penalty_contact_law(125.0).evaluate(projection)
    return integrate_implicit_rigid_contact(
        evaluation, record, response,
        surface.gap_differential(projection).hessian,
    )


def test_sampled_feature_transition_diagnostic_distinguishes_reentrant_edge():
    surface = _concave_tool()
    trace = ContactTrace(
        point_ids=(11, 12, 13), node_ids=((0, 1, 2),) * 3,
        shape_values=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        weights=(1 / 3, 1 / 3, 1 / 3),
    )

    def count(points, *, motion=None, factor=0.0):
        reference_points = np.asarray(points, dtype=float)
        current_points = reference_points
        if motion is not None:
            pose = motion.state(factor)
            current_points = (
                np.asarray(pose["reference_point"])
                + (reference_points - np.asarray(motion.reference_point))
                @ np.asarray(pose["rotation_matrix"]).T
            )
        evaluation = trace.evaluate(current_points)
        record = ContactProjectionRecord(
            trace.point_ids,
            surface.project(evaluation.query_points, motion=motion, factor=factor),
        )
        response = frictionless_penalty_contact_law(10.0).evaluate(record.projection)
        assert np.all(response.active)
        return sampled_mixed_feature_slave_faces(
            evaluation, record, response, surface, motion=motion, factor=factor,
        )

    assert count(((0.95, 1.2, 0.5), (0.95, 1.5, 0.5), (0.95, 1.8, 0.5))) == 0
    mixed = ((0.9, 0.9, 0.5), (0.95, 1.5, 0.5), (0.95, 1.8, 0.5))
    assert count(mixed) == 1
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.2, -0.1, 0), rotation=(0, 0, 0.3),
        reference_point=(0, 0, 0),
    )
    assert count(mixed, motion=motion, factor=0.5) == 1


def test_closed_concave_surface_classifies_inside_notch_and_reentrant_edge():
    tool = _concave_tool()
    assert tool.summary()["self_intersection_certificate"] is False
    points = np.asarray(((1.4, 1.5, 0.5), (0.9, 0.9, 0.5),
                         (0.95, 1.5, 0.5)))
    projection = tool.project(points)
    assert projection.all_valid, projection.status_codes
    np.testing.assert_allclose(
        projection.signed_gaps,
        (0.4, -np.sqrt(0.02), -0.05), atol=1e-12,
    )
    np.testing.assert_allclose(
        projection.closest_points[1], (1.0, 1.0, 0.5), atol=1e-12,
    )
    np.testing.assert_allclose(
        projection.normals[1], (1 / np.sqrt(2), 1 / np.sqrt(2), 0),
        atol=1e-12,
    )
    medial_axis = tool.project(((1.5, 1.5, 0.5),))
    assert not medial_axis.valid[0]
    assert medial_axis.status_codes[0] == "nonunique_nearest_feature"


def test_closed_solid_certified_exterior_aabb_skips_only_unneeded_winding(
    monkeypatch,
):
    tool = _concave_tool()
    outside_box = np.asarray(((3.0, 1.5, 0.5),))
    reference = tool.project(outside_box)
    assert reference.all_valid and reference.signed_gaps[0] > 0.0

    def forbidden_winding(*_args):
        raise AssertionError("Direct winding was not needed outside the tool AABB")

    monkeypatch.setattr(closed_module, "_solid_winding_number", forbidden_winding)
    screened = tool.project(outside_box)
    np.testing.assert_array_equal(screened.signed_gaps, reference.signed_gaps)
    np.testing.assert_array_equal(screened.normals, reference.normals)
    motion = boundary_models.prescribed_rigid_motion(
        translation=(10.0, 0.0, 0.0), reference_point=(0.0, 0.0, 0.0),
    )
    moved = tool.project(((13.0, 1.5, 0.5),), motion=motion, factor=1.0)
    np.testing.assert_array_equal(moved.signed_gaps, reference.signed_gaps)
    with pytest.raises(AssertionError, match="not needed"):
        tool.project(((1.4, 1.5, 0.5),))  # In the AABB, outside the L solid.


@pytest.mark.parametrize("fixture", (_concave_tool, _reentrant_vertex_tool))
def test_opt_in_pseudonormal_matches_reference_signed_feature_queries(fixture):
    reference = fixture()
    fast = closed_module.ClosedTriangulatedRigidSolid(
        facets=reference.facets, reviewed_embedded=True,
        name=reference.name, sign_method="angle_weighted_pseudonormal",
    )
    assert fast.geometry_fingerprint != reference.geometry_fingerprint
    assert not fast.summary()["query_winding_audit"]
    assert reference.summary()["query_winding_audit"]
    rng = np.random.default_rng(20261018)
    points = rng.uniform(-0.35, 2.3, (2000, 3))
    if fixture is _concave_tool:
        points = np.vstack((points, (
            (0.9, 0.9, 0.5), (1.4, 1.5, 0.5),
            (0.9, 0.9, 0.2), (0.9, 1.5, 0.5),
        )))
    else:
        points = np.vstack((points, (
            (-0.1, -0.2, -0.15), (0.1, 0.2, 0.15),
        )))
    expected = reference.project(points)
    actual = fast.project(points)
    np.testing.assert_array_equal(actual.valid, expected.valid)
    np.testing.assert_array_equal(actual.status_codes, expected.status_codes)
    np.testing.assert_allclose(actual.signed_gaps, expected.signed_gaps)
    np.testing.assert_allclose(actual.normals, expected.normals)
    np.testing.assert_allclose(actual.closest_points, expected.closest_points)


def test_opt_in_pseudonormal_matches_winding_near_reentrant_edge():
    reference = _concave_tool()
    fast = closed_module.ClosedTriangulatedRigidSolid(
        facets=reference.facets, reviewed_embedded=True,
        name=reference.name, sign_method="angle_weighted_pseudonormal",
    )
    rng = np.random.default_rng(183)
    count = 2000
    radii = 10.0 ** rng.uniform(-8.0, -1.0, count)
    angles = rng.uniform(-np.pi, np.pi, count)
    points = np.column_stack((
        1.0 + radii * np.cos(angles),
        1.0 + radii * np.sin(angles),
        rng.uniform(0.05, 0.95, count),
    ))
    expected = reference.project(points)
    actual = fast.project(points)
    np.testing.assert_array_equal(actual.valid, expected.valid)
    np.testing.assert_array_equal(actual.status_codes, expected.status_codes)
    np.testing.assert_allclose(actual.signed_gaps, expected.signed_gaps, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual.normals, expected.normals, rtol=0.0, atol=1e-12)


def test_opt_in_pseudonormal_uses_current_nearest_feature_without_winding(
    monkeypatch,
):
    reference = _concave_tool()
    fast = closed_module.ClosedTriangulatedRigidSolid(
        facets=reference.facets, reviewed_embedded=True,
        name=reference.name, sign_method="angle_weighted_pseudonormal",
    )
    queries = np.asarray(((1.4, 1.5, 0.5), (0.9, 0.9, 0.5)))
    expected = reference.project(queries)

    def forbidden_winding(*_args):
        raise AssertionError("An unambiguous pseudonormal needs no winding")

    monkeypatch.setattr(closed_module, "_solid_winding_number", forbidden_winding)
    actual = fast.project(queries)
    np.testing.assert_allclose(actual.signed_gaps, expected.signed_gaps)
    np.testing.assert_allclose(actual.normals, expected.normals)
    with pytest.raises(ValueError, match="sign_method"):
        closed_module.ClosedTriangulatedRigidSolid(
            facets=reference.facets, reviewed_embedded=True,
            sign_method="unreviewed_fast",
        )


def test_coplanar_triangulation_seam_is_not_a_physical_contact_edge():
    tool = _concave_tool()
    above = tool.project(((0.5, 0.5, 1.2),))
    assert above.all_valid
    np.testing.assert_allclose(above.signed_gaps, (0.2,), atol=1e-12)
    np.testing.assert_allclose(
        tool.gap_differential(above).hessian, np.zeros((1, 3, 3)),
        atol=1e-12,
    )
    on_seam = tool.project(((0.5, 0.5, 1.0),))
    assert on_seam.all_valid, on_seam.status_codes
    np.testing.assert_allclose(on_seam.signed_gaps, (0.0,), atol=1e-12)


def test_subdivided_physical_edge_stays_an_edge_not_a_vertex():
    original = _concave_tool()
    midpoint = np.asarray(((1.0, 1.0, 0.5),))
    vertices = np.vstack((original.facets.vertices, midpoint))
    triangles = []
    split = 0
    for triangle in original.facets.triangles:
        oriented = tuple(map(int, triangle))
        if {3, 9}.issubset(oriented):
            for shift in range(3):
                cyclic = oriented[shift:] + oriented[:shift]
                if {cyclic[0], cyclic[1]} == {3, 9}:
                    a, b, c = cyclic
                    triangles.extend(((a, 12, c), (12, b, c)))
                    split += 1
                    break
        else:
            triangles.append(oriented)
    assert split == 2
    tool = closed_triangulated_rigid_solid(
        vertices=vertices, triangles=triangles, reviewed_embedded=True,
    )
    selected = tool.project(((0.9, 0.9, 0.5),))
    assert selected.all_valid
    np.testing.assert_allclose(selected.closest_points[0], midpoint[0])
    hessian = tool.gap_differential(selected).hessian[0]
    assert abs(hessian[2, 2]) < 1.0e-12
    assert np.linalg.norm(hessian) > 0.0


def test_concave_edge_penalty_is_work_conjugate_and_tangent_is_consistent():
    tool = _concave_tool()
    query = np.asarray((0.9, 0.9, 0.5))
    direction = np.asarray((0.07, -0.03, 0.04))
    selected = _contribution(tool, query)
    assert selected.active_points == 1
    # Independent local oracle: inside this concave edge branch the signed
    # penalty potential is k*w*((x-1)^2+(y-1)^2)/2, with no z curvature.
    radial = query - np.asarray((1.0, 1.0, query[2]))
    expected_force = 125.0 * 0.75 * radial
    np.testing.assert_allclose(
        np.sum(selected.nodal_structural_residual, axis=0),
        expected_force, atol=1e-12,
    )
    np.testing.assert_allclose(
        selected.potential_energy,
        0.5 * 125.0 * 0.75 * np.dot(radial, radial), atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sum(selected.tangent_action(np.tile(direction, (3, 1))), axis=0),
        125.0 * 0.75 * np.asarray((direction[0], direction[1], 0.0)),
        atol=1e-12,
    )
    h = 1e-6
    plus = _contribution(tool, query + h * direction)
    minus = _contribution(tool, query - h * direction)
    numerical_force = (
        plus.nodal_structural_residual - minus.nodal_structural_residual
    ) / (2 * h)
    np.testing.assert_allclose(
        selected.tangent_action(np.tile(direction, (3, 1))),
        numerical_force, rtol=1e-7, atol=1e-8,
    )
    numerical_energy = (plus.potential_energy - minus.potential_energy) / (2 * h)
    np.testing.assert_allclose(
        numerical_energy, np.sum(selected.nodal_structural_residual * direction),
        rtol=1e-7, atol=1e-8,
    )


def test_reentrant_vertex_penalty_matches_three_dimensional_quadratic_oracle():
    tool = _reentrant_vertex_tool()
    point = np.asarray((-0.1, -0.2, -0.15))
    projection = tool.project((point,))
    assert projection.all_valid, projection.status_codes
    np.testing.assert_allclose(projection.closest_points[0], np.zeros(3))
    np.testing.assert_allclose(
        projection.signed_gaps[0], -np.linalg.norm(point), atol=1e-12,
    )
    selected = _contribution(tool, point)
    direction = np.asarray((0.03, -0.02, 0.05))
    np.testing.assert_allclose(
        np.sum(selected.nodal_structural_residual, axis=0),
        125.0 * 0.75 * point, atol=1e-11,
    )
    np.testing.assert_allclose(
        np.sum(selected.tangent_action(np.tile(direction, (3, 1))), axis=0),
        125.0 * 0.75 * direction, atol=1e-10,
    )


def test_closed_solid_projection_and_edge_hessian_are_rigid_motion_objective():
    tool = _concave_tool()
    point = np.asarray((0.9, 0.9, 0.5))
    reference = tool.project((point,))
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.12, -0.07, 0.04),
        rotation=(0.08, -0.11, 0.05),
        reference_point=(0.2, 0.3, 0.4),
    )
    pose = motion.state(1.0)
    rotation = np.asarray(pose["rotation_matrix"])
    moved_point = (
        pose["reference_point"]
        + (point - np.asarray(motion.reference_point)) @ rotation.T
    )
    moved = tool.project((moved_point,), motion=motion, factor=1.0)
    assert moved.all_valid
    np.testing.assert_allclose(moved.signed_gaps, reference.signed_gaps)
    np.testing.assert_allclose(
        moved.normals[0], rotation @ reference.normals[0], atol=1e-12,
    )
    np.testing.assert_allclose(
        tool.gap_differential(moved, motion=motion, factor=1.0).hessian[0],
        rotation @ tool.gap_differential(reference).hessian[0] @ rotation.T,
        atol=1e-12,
    )


def test_closed_solid_requires_explicit_embedding_review_and_outward_orientation():
    tool = _concave_tool()
    with pytest.raises(ValueError, match="reviewed_embedded"):
        closed_triangulated_rigid_solid(
            vertices=tool.facets.vertices, triangles=tool.facets.triangles,
            reviewed_embedded=False,
        )
    with pytest.raises(ValueError, match="outward orientation"):
        closed_triangulated_rigid_solid(
            vertices=tool.facets.vertices,
            triangles=tool.facets.triangles[:, ::-1],
            reviewed_embedded=True,
        )


def test_point_touching_closed_shells_fail_vertex_manifold_review():
    vertices = np.asarray((
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    ), dtype=float)
    first = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    second = ((0, 4, 5), (0, 6, 4), (0, 5, 6), (4, 6, 5))
    with pytest.raises(ValueError, match="non-manifold vertex link"):
        closed_triangulated_rigid_solid(
            vertices=vertices, triangles=first + second,
            reviewed_embedded=True,
        )


def test_dolfinx_closed_tool_separation_is_certified_only_at_the_current_state():
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    original = _concave_tool()
    tool = closed_triangulated_rigid_solid(
        vertices=original.facets.vertices + np.asarray((5., 0., 0.)),
        triangles=original.facets.triangles,
        reviewed_embedded=True,
        name="translated_reentrant_tool",
    )
    boundary = agent_mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="loaded_face",
    )
    pair = boundary_models.rigid_contact_pair(
        boundary, boundary_models.rigid_body(tool), penalty=125.0,
    )
    operator = dolfinx_implicit_rigid_contact(pair, displacement)
    far = operator.current_point_diagnostics()
    assert far["active_point_count"] == 0
    assert far["whole_face_separation"]["certified"] is True
    assert far["whole_face_separation"]["minimum_gap_lower_bound"] > 0
    assert far["feature_transition_integral_certified"] is True
    path = AcceptedContactFeaturePath((operator,))
    path.reset(SimpleNamespace(index=0, load_factor=0.0))
    path.accept(SimpleNamespace(index=1, load_factor=0.5))
    assert next(iter(path.pairs.values()))["all_observed_integrals_certified"] is True
    displacement.x.array.reshape((-1, 3))[:, 0] = 5.1
    near = operator.current_point_diagnostics()
    assert near["feature_transition_integral_certified"] is False
    assert near["active_point_count"] > 0
    path.accept(SimpleNamespace(index=2, load_factor=1.0))
    assert next(iter(path.pairs.values()))["all_observed_integrals_certified"] is False


def test_dolfinx_reentrant_edge_assembles_current_residual_and_tangent():
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix.ufl.element(
            "Lagrange", domain.basix_cell(), 1, shape=(3,),
        ),
    )
    displacement = fem.Function(space)
    coordinates = space.tabulate_dof_coordinates().reshape((-1, 3))
    values = displacement.x.array.reshape((-1, 3))
    values[:, 1] = 0.85 - 0.9 * coordinates[:, 1]
    values[:, 2] = 0.85 - 0.9 * coordinates[:, 2]
    original = _concave_tool()
    vertices = np.column_stack((
        2.0 * original.facets.vertices[:, 2] - 1.0,
        original.facets.vertices[:, 0],
        original.facets.vertices[:, 1],
    ))
    tool = closed_triangulated_rigid_solid(
        vertices=vertices, triangles=original.facets.triangles,
        reviewed_embedded=True, name="reentrant_edge_tool",
    )
    boundary = agent_mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="loaded_face",
    )
    pair = boundary_models.rigid_contact_pair(
        boundary, boundary_models.rigid_body(tool), penalty=125.0,
    )
    operator = dolfinx_implicit_rigid_contact(pair, displacement)
    selected = operator.evaluate()
    assert selected.active_points == operator.adapter.trace.point_count
    assert selected.potential_energy > 0.0
    point_evidence = operator.current_point_diagnostics()
    assert point_evidence["sampled_mixed_feature_slave_faces"] >= 0
    assert point_evidence["feature_transition_integral_certified"] is False
    base = displacement.x.array.copy()
    direction = np.random.default_rng(84).normal(size=base.shape)
    h = 1.0e-7
    displacement.x.array[:] = base + h * direction
    plus = operator.evaluate()
    displacement.x.array[:] = base - h * direction
    minus = operator.evaluate()
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=1.0e-7, atol=1.0e-7,
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.01, 0.005, -0.004),
        rotation=(0.0, 0.0, 0.01), reference_point=(0.0, 0.0, 0.0),
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion, start_time=0.0, end_time=1.0,
    )
    moving_pair = boundary_models.rigid_contact_pair(
        boundary,
        boundary_models.rigid_body(tool, motion_schedule=schedule),
        penalty=125.0,
    )
    moving = dolfinx_implicit_rigid_contact(moving_pair, displacement)
    moving.update_motion(1.0)
    selected = moving.evaluate()
    assert selected.active_points == moving.adapter.trace.point_count
    assert selected.surface_generalized_moment is not None
    evaluation, record, response, _ = moving._current_point_response()
    reference = np.asarray(motion.state(1.0)["reference_point"])
    expected_moment = np.sum(np.cross(
        record.projection.closest_points - reference,
        response.surface_generalized_tractions * evaluation.trace.weights[:, None],
    ), axis=0)
    np.testing.assert_allclose(
        selected.surface_generalized_moment, expected_moment, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sum(selected.nodal_structural_residual, axis=0),
        -selected.contact_force_on_structure, atol=1e-12,
    )
    displacement.x.array[:] = base + h * direction
    plus = moving.evaluate()
    displacement.x.array[:] = base - h * direction
    minus = moving.evaluate()
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=1.0e-7, atol=1.0e-7,
    )
