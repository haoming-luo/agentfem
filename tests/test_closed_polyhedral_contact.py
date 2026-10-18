from __future__ import annotations

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
from agentfem.boundary_models.contact_trace import ContactTrace
from agentfem.boundary_models.implicit_contact import integrate_implicit_rigid_contact
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
