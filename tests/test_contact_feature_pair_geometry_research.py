"""Bounded finite-feature and closed-tool local contact research certificates."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem.boundary_models.contact_feature_pair_geometry_research import (
    _affine_feature_dominates_facet,
    certify_partial_closed_face,
    certify_two_facet_distance_competition,
    certify_wholly_interior_closed_face,
    integrate_certified_closed_two_feature_penalty,
    integrate_certified_partial_closed_two_feature_penalty,
    integrate_discovered_partial_closed_two_feature_penalty,
)
from agentfem.boundary_models.contact_feature_pair_research import (
    AffineDistanceFeature,
    integrate_two_affine_feature_distances,
)
from agentfem.boundary_models.closed_polyhedron import closed_triangulated_rigid_solid
from agentfem.boundary_models.contact_convex_partition import (
    _clip_polygon,
    _polygon_rule,
)
from agentfem.boundary_models.rigid import _closest_point_on_triangle


_SLAVE = np.asarray(((1.0, 1.1, 0.0), (2.0, 1.1, 0.0), (2.0, 3.0, 0.0)))
_MASTERS = np.asarray(
    (
        ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, -5.0, 0.0)),
        ((3.0, 1.0, 0.0), (5.0, 1.0, 0.0), (3.0, -1.0, 0.0)),
        ((10.0, 10.0, 0.0), (11.0, 10.0, 0.0), (10.0, 11.0, 0.0)),
    )
)


def test_finite_edge_vertex_pair_is_certified_over_whole_slave_face():
    proof = certify_two_facet_distance_competition(_SLAVE, _MASTERS, (0, 1))
    assert proof.feature_kinds == ("edge", "vertex")
    assert proof.minimum_competitor_clearance > 0.0
    assert proof.examined_facet_count == 3
    assert proof.box_exclusion_count == 1
    response = integrate_two_affine_feature_distances(
        _SLAVE,
        _SLAVE,
        proof.first,
        proof.second,
    )
    assert response.boundary_point_count > 0
    assert min(response.branch_areas) > 0.0
    # Independent pointwise nearest-triangle oracle checks the certificate
    # over a deterministic mesh; the certificate itself did not use samples.
    for numerator_s in range(21):
        for numerator_t in range(21 - numerator_s):
            s, t = numerator_s / 20, numerator_t / 20
            point = (1 - s - t) * _SLAVE[0] + s * _SLAVE[1] + t * _SLAVE[2]
            actual = min(
                np.linalg.norm(point - _closest_point_on_triangle(point, *triangle))
                for triangle in _MASTERS
            )
            certified = min(
                np.linalg.norm((point - feature.anchor) @ feature.normal_projector)
                for feature in (proof.first, proof.second)
            )
            assert certified == pytest.approx(actual, abs=2e-12)


def test_third_competitor_refuses_the_pair_certificate():
    interfering = _MASTERS.copy()
    interfering[2] = np.asarray(((1.5, 2.0, 0.0), (1.6, 2.0, 0.0), (1.5, 2.1, 0.0)))
    with pytest.raises(NotImplementedError, match="additional finite master"):
        certify_two_facet_distance_competition(_SLAVE, interfering, (0, 1))


def test_selected_facet_changing_feature_refuses_the_pair_certificate():
    crossing = _SLAVE.copy()
    crossing[0, 1] = -1.0
    with pytest.raises(NotImplementedError, match="changes closest feature"):
        certify_two_facet_distance_competition(crossing, _MASTERS, (0, 1))


def test_degenerate_slave_or_duplicate_candidates_are_rejected():
    with pytest.raises(ValueError, match="nondegenerate slave"):
        certify_two_facet_distance_competition(
            np.repeat(_SLAVE[:1], 3, axis=0), _MASTERS, (0, 1)
        )
    with pytest.raises(ValueError, match="bounded finite"):
        certify_two_facet_distance_competition(_SLAVE, _MASTERS, (0, 0))


def test_unresolved_world_coordinate_offset_is_refused():
    translation = np.asarray((1e12, -1e12, 1e12))
    with pytest.raises(ValueError, match="world-coordinate precision"):
        certify_two_facet_distance_competition(
            _SLAVE + translation,
            _MASTERS + translation,
            (0, 1),
        )


def test_nonplanar_and_unordered_four_vertex_regions_are_refused():
    square = np.asarray(
        ((1.0, 1.1, 0.0), (2.0, 1.1, 0.0), (2.0, 2.0, 0.0), (1.0, 2.0, 0.0))
    )
    warped = square.copy()
    warped[3, 2] = 0.01
    with pytest.raises(ValueError, match="planar"):
        certify_two_facet_distance_competition(warped, _MASTERS, (0, 1))
    unordered = square[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="convex and ordered"):
        certify_two_facet_distance_competition(unordered, _MASTERS, (0, 1))


def test_affine_dominance_does_not_merge_a_nearby_distinct_edge():
    selected_master = np.asarray(((2.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)))
    feature = AffineDistanceFeature(
        np.asarray((1.0, 1.0, 0.0)),
        np.asarray(((0.0,), (0.0,), (1.0,))),
    )
    slave = np.asarray(
        ((0.895, 0.875, 0.02), (0.905, 0.875, 0.17), (0.895, 0.885, 0.16))
    )
    near_but_distinct = np.asarray(
        ((1.0, 1.0 - 1e-10, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 1.0))
    )
    assert not _affine_feature_dominates_facet(
        slave, near_but_distinct, feature, selected_master, 2e-12
    )


def _closed_tetrahedron():
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    )
    triangles = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    return closed_triangulated_rigid_solid(
        vertices=vertices,
        triangles=triangles,
        reviewed_embedded=True,
    )


def test_wholly_interior_closed_tetrahedron_has_physical_penalty_limit():
    tool = _closed_tetrahedron()
    slave = np.asarray(((0.27, 0.30, 2.0), (0.33, 0.30, 2.0), (0.30, 0.35, 2.0)))
    clearance = certify_wholly_interior_closed_face(slave, tool)
    assert clearance > 0.2
    pair = certify_two_facet_distance_competition(
        slave,
        tool._triangles,
        (1, 2),
    )
    assert pair.feature_kinds == ("face", "face")
    stiffness = 700.0
    certified = integrate_certified_closed_two_feature_penalty(
        slave,
        slave,
        tool,
        (1, 2),
        penalty=stiffness,
    )
    response = certified.response
    assert certified.geometry.feature_kinds == ("face", "face")
    assert certified.minimum_interior_clearance == pytest.approx(clearance)
    assert certified.tool_fingerprint == tool.geometry_fingerprint
    assert response.boundary_point_count > 0
    # Independent exact degree-two integration of min(x,y)^2 on the two
    # polygons cut by the *linear* x=y line (not the conic implementation).
    unit_triangle = [np.eye(3)[index] for index in range(3)]
    double_area = float(
        np.linalg.norm(np.cross(slave[1] - slave[0], slave[2] - slave[0]))
    )
    independent = 0.0
    for keep_x in (True, False):
        values = slave[:, 0] - slave[:, 1]
        polygon = _clip_polygon(unit_triangle, values if keep_x else -values)
        shapes, weights = _polygon_rule(polygon)
        if weights:
            points = np.asarray(shapes) @ slave
            selected = points[:, 0] if keep_x else points[:, 1]
            independent += 0.5 * stiffness * double_area * np.dot(weights, selected**2)
    assert response.potential == pytest.approx(independent, rel=2e-9)
    for s, t in ((0.2, 0.2), (0.7, 0.1), (0.1, 0.7)):
        point = (1 - s - t) * slave[0] + s * slave[1] + t * slave[2]
        projection = tool.project(point)
        assert projection.all_valid
        assert projection.signed_gaps[0] == pytest.approx(-min(point[0], point[1]))


def test_interior_ball_does_not_infer_sign_from_vertices_alone():
    tool = _closed_tetrahedron()
    # A large face reaches the boundary; a few inside points cannot prove
    # that the complete affine face remains inside the tetrahedron.
    nearly_boundary = np.asarray(((0.01, 0.3, 2.0), (0.7, 0.3, 2.0), (0.3, 0.7, 2.0)))
    with pytest.raises(NotImplementedError, match="interior ball"):
        certify_wholly_interior_closed_face(nearly_boundary, tool)
    with pytest.raises(NotImplementedError, match="interior ball"):
        integrate_certified_closed_two_feature_penalty(
            nearly_boundary,
            nearly_boundary,
            tool,
            (1, 2),
            penalty=700.0,
        )


def _closed_L_prism():
    polygon = np.asarray(((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), dtype=float)
    vertices = np.vstack(
        (
            np.column_stack((polygon, np.zeros(6))),
            np.column_stack((polygon, np.ones(6))),
        )
    )
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in top]
    triangles += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend(((first, second, second + 6), (first, second + 6, first + 6)))
    return closed_triangulated_rigid_solid(
        vertices=vertices,
        triangles=triangles,
        reviewed_embedded=True,
    )


def test_reentrant_edge_and_bottom_face_compete_in_wholly_inside_L_prism():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, 0.15), (0.905, 0.875, 0.17), (0.895, 0.885, 0.16))
    )
    assert certify_wholly_interior_closed_face(slave, tool) > 0.0
    pair = certify_two_facet_distance_competition(
        slave,
        tool._triangles,
        (0, 12),
    )
    assert pair.feature_kinds == ("face", "edge")
    assert pair.equivalent_feature_count + pair.affine_dominance_count > 0
    result = integrate_certified_closed_two_feature_penalty(
        slave,
        slave,
        tool,
        (0, 12),
        penalty=500.0,
    )
    assert result.response.boundary_point_count > 0
    assert min(result.response.branch_areas) > 0.0
    for numerator_s in range(11):
        for numerator_t in range(11 - numerator_s):
            s, t = numerator_s / 10, numerator_t / 10
            point = (1 - s - t) * slave[0] + s * slave[1] + t * slave[2]
            projection = tool.project(point)
            assert projection.all_valid
            assert projection.signed_gaps[0] < 0.0
            pair_distance = min(
                np.linalg.norm((point - feature.anchor) @ feature.normal_projector)
                for feature in (pair.first, pair.second)
            )
            assert -projection.signed_gaps[0] == pytest.approx(pair_distance, abs=2e-12)

    step = 1e-6
    numerical_gradient = np.zeros((3, 3))
    numerical_tangent = np.zeros((3, 3, 3, 3))
    for node in range(3):
        for component in range(3):
            direction = np.zeros((3, 3))
            direction[node, component] = step
            plus = integrate_certified_closed_two_feature_penalty(
                slave,
                slave + direction,
                tool,
                (0, 12),
                penalty=500.0,
            ).response
            minus = integrate_certified_closed_two_feature_penalty(
                slave,
                slave - direction,
                tool,
                (0, 12),
                penalty=500.0,
            ).response
            numerical_gradient[node, component] = (plus.potential - minus.potential) / (
                2 * step
            )
            numerical_tangent[:, :, node, component] = (
                plus.nodal_residual - minus.nodal_residual
            ) / (2 * step)
    np.testing.assert_allclose(
        result.response.nodal_residual,
        numerical_gradient,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.response.nodal_tangent,
        numerical_tangent,
        rtol=2e-5,
        atol=2e-7,
    )
    assert np.linalg.norm(result.response.moving_boundary_tangent) > 1e-4


def test_partially_active_reentrant_face_still_refuses_the_local_law():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    # The face crosses the bottom boundary. In its exterior portion the
    # finite reentrant edge ends at the bottom corner; it cannot be treated
    # as the same infinite edge over the full slave triangle.
    with pytest.raises(NotImplementedError, match="interior ball"):
        certify_wholly_interior_closed_face(slave, tool)
    with pytest.raises(NotImplementedError, match="changes closest feature"):
        certify_two_facet_distance_competition(slave, tool._triangles, (0, 12))
    with pytest.raises(NotImplementedError, match="interior ball"):
        integrate_certified_closed_two_feature_penalty(
            slave,
            slave,
            tool,
            (0, 12),
            penalty=500.0,
        )


def test_partially_active_reentrant_face_has_certified_local_penalty():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    active, normal = certify_partial_closed_face(slave, tool, 0)
    assert len(active) == 4
    np.testing.assert_allclose(normal, (0.0, 0.0, -1.0))
    result = integrate_certified_partial_closed_two_feature_penalty(
        slave, slave, tool, (0, 12), penalty=500.0
    )
    assert result.geometry.feature_kinds == ("face", "edge")
    assert result.active_reference_fraction > 0.0
    assert result.active_reference_fraction < 1.0
    assert result.response.boundary_point_count > 0
    assert np.linalg.norm(result.response.moving_boundary_tangent) > 1e-5
    for numerator_s in range(21):
        for numerator_t in range(21 - numerator_s):
            s, t = numerator_s / 20, numerator_t / 20
            point = (1 - s - t) * slave[0] + s * slave[1] + t * slave[2]
            projection = tool.project(point)
            assert projection.all_valid
            assert (projection.signed_gaps[0] < 0.0) == (point[2] > 0.0)

    step = 1e-6
    for node in range(3):
        for component in range(3):
            direction = np.zeros((3, 3))
            direction[node, component] = step
            plus = integrate_certified_partial_closed_two_feature_penalty(
                slave, slave + direction, tool, (0, 12), penalty=500.0
            ).response
            minus = integrate_certified_partial_closed_two_feature_penalty(
                slave, slave - direction, tool, (0, 12), penalty=500.0
            ).response
            assert result.response.nodal_residual[node, component] == pytest.approx(
                (plus.potential - minus.potential) / (2 * step),
                rel=3e-5,
                abs=3e-7,
            )
            np.testing.assert_allclose(
                result.response.nodal_tangent[:, :, node, component],
                (plus.nodal_residual - minus.nodal_residual) / (2 * step),
                rtol=3e-5,
                atol=3e-7,
            )


def test_partial_reentrant_face_matches_independent_piecewise_quadrature():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    response = integrate_certified_partial_closed_two_feature_penalty(
        slave, slave, tool, (0, 12), penalty=500.0
    ).response
    reference_area = 0.5 * np.linalg.norm(
        np.cross(slave[1] - slave[0], slave[2] - slave[0])
    )

    def reference(number):
        # Independent physical-distance oracle: for each fixed barycentric u,
        # split v at z=0 and at z**2=(x-1)**2+(y-1)**2. The production
        # integrator instead certifies/clips a two-feature surface partition.
        u_nodes, u_weights = np.polynomial.legendre.leggauss(number)
        v_nodes, v_weights = np.polynomial.legendre.leggauss(3)
        total = np.zeros(10)
        polynomial = np.polynomial.Polynomial
        u_symbol = polynomial([0, 1])
        z0_poly = slave[0, 2] + (slave[1, 2] - slave[0, 2]) * u_symbol
        dx_poly = slave[0, 0] - 1 + (slave[1, 0] - slave[0, 0]) * u_symbol
        y0_poly = slave[0, 1] - 1 + (slave[1, 1] - slave[0, 1]) * u_symbol
        dz = slave[2, 2] - slave[0, 2]
        dy = slave[2, 1] - slave[0, 1]
        changes = [0.0, 1.0]
        for v_poly in (polynomial([0]), 1 - u_symbol):
            z_poly = z0_poly + dz * v_poly
            y_poly = y0_poly + dy * v_poly
            for branch_poly in (z_poly, z_poly**2 - dx_poly**2 - y_poly**2):
                changes.extend(
                    float(root.real)
                    for root in branch_poly.roots()
                    if abs(root.imag) < 1e-12 and 0 < root.real < 1
                )
        for u_left, u_right in zip(sorted(changes)[:-1], sorted(changes)[1:]):
            u_points = u_left + (u_nodes + 1) * (u_right - u_left) / 2
            u_quadrature = u_weights * (u_right - u_left) / 2
            for u, u_weight in zip(u_points, u_quadrature):
                upper = 1 - u
                z0 = slave[0, 2] + u * (slave[1, 2] - slave[0, 2])
                dx = slave[0, 0] + u * (slave[1, 0] - slave[0, 0]) - 1
                y0 = slave[0, 1] + u * (slave[1, 1] - slave[0, 1]) - 1
                boundaries = [0.0, upper]
                zero = -z0 / dz
                if 0 < zero < upper:
                    boundaries.append(zero)
                roots = np.roots(
                    [dz**2 - dy**2, 2 * (z0 * dz - y0 * dy), z0**2 - dx**2 - y0**2]
                )
                boundaries.extend(
                    float(root.real)
                    for root in roots
                    if abs(root.imag) < 1e-12 and 0 < root.real < upper
                )
                for left, right in zip(sorted(boundaries)[:-1], sorted(boundaries)[1:]):
                    v = left + (v_nodes + 1) * (right - left) / 2
                    shape = np.column_stack((1 - u - v, np.full_like(v, u), v))
                    points = shape @ slave
                    if points[1, 2] <= 0:
                        continue
                    bottom_sq = points[:, 2] ** 2
                    edge_sq = (points[:, 0] - 1) ** 2 + (points[:, 1] - 1) ** 2
                    bottom = bottom_sq <= edge_sq
                    density = 250.0 * np.where(bottom, bottom_sq, edge_sq)
                    gradient = np.zeros_like(points)
                    gradient[bottom, 2] = 500.0 * points[bottom, 2]
                    gradient[~bottom, :2] = 500.0 * (points[~bottom, :2] - 1)
                    weighted = v_weights * (right - left) / 2
                    total[0] += u_weight * weighted @ density
                    total[1:] += u_weight * np.einsum(
                        "q,qi,qj->ij", weighted, shape, gradient
                    ).reshape(-1)
        return 2 * reference_area * total[0], 2 * reference_area * total[1:].reshape(
            3, 3
        )

    coarse_energy, coarse_force = reference(64)
    fine_energy, fine_force = reference(128)
    assert fine_energy == pytest.approx(coarse_energy, rel=1e-7)
    np.testing.assert_allclose(fine_force, coarse_force, rtol=1e-6, atol=1e-8)
    assert response.potential == pytest.approx(fine_energy, rel=1e-7)
    np.testing.assert_allclose(
        response.nodal_residual, fine_force, rtol=1e-6, atol=1e-8
    )


def test_partial_local_penalty_refuses_additional_crossing_facet():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (1.005, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    with pytest.raises(NotImplementedError, match="[Aa]nother tool facet|winding sign"):
        integrate_certified_partial_closed_two_feature_penalty(
            slave, slave, tool, (0, 12), penalty=500.0
        )


def test_partial_reentrant_tool_translation_work_matches_force():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    response = integrate_certified_partial_closed_two_feature_penalty(
        slave, slave, tool, (0, 12), penalty=500.0
    ).response
    np.testing.assert_allclose(
        response.force_on_structure,
        -response.force_on_tool,
    )
    np.testing.assert_allclose(
        response.force_on_structure,
        -np.sum(response.nodal_residual, axis=0),
    )
    np.testing.assert_allclose(
        response.moment_on_structure_about_origin,
        -response.rigid_rotation_reaction_about_origin,
    )
    step = 1e-6
    for component in range(3):
        translation = np.zeros(3)
        translation[component] = step
        energies = []
        for sign in (1.0, -1.0):
            moved = closed_triangulated_rigid_solid(
                vertices=tool.facets.vertices + sign * translation,
                triangles=tool.facets.triangles,
                reviewed_embedded=True,
            )
            energies.append(
                integrate_certified_partial_closed_two_feature_penalty(
                    slave, slave, moved, (0, 12), penalty=500.0
                ).response.potential
            )
        assert -(energies[0] - energies[1]) / (2 * step) == pytest.approx(
            response.rigid_translation_reaction[component],
            rel=3e-5,
            abs=3e-7,
        )

    # Re-certify every tilted trial; a shared rotated edge endpoint must not
    # become a spurious third physical branch through projector roundoff.
    for component in range(3):
        axis = np.eye(3)[component]
        skew = np.asarray(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        energies = []
        for sign in (1.0, -1.0):
            angle = sign * step
            rotation = (
                np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
            )
            moved = closed_triangulated_rigid_solid(
                vertices=tool.facets.vertices @ rotation.T,
                triangles=tool.facets.triangles,
                reviewed_embedded=True,
            )
            energies.append(
                integrate_certified_partial_closed_two_feature_penalty(
                    slave, slave, moved, (0, 12), penalty=500.0
                ).response.potential
            )
        assert -(energies[0] - energies[1]) / (2 * step) == pytest.approx(
            response.rigid_rotation_reaction_about_origin[component],
            rel=3e-5,
            abs=3e-7,
        )


def test_partial_contact_certificate_survives_perturbed_nearest_point_oracles():
    tool = _closed_L_prism()
    base = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    generator = np.random.default_rng(17044)
    for _ in range(32):
        slave = base + generator.uniform(-0.001, 0.001, size=(3, 3))
        certified = integrate_certified_partial_closed_two_feature_penalty(
            base, slave, tool, (0, 12), penalty=500.0, order=16
        )
        assert certified.geometry.feature_kinds == ("face", "edge")
        for numerator_s in range(8):
            for numerator_t in range(8 - numerator_s):
                s, t = numerator_s / 7, numerator_t / 7
                point = (1 - s - t) * slave[0] + s * slave[1] + t * slave[2]
                oracle = tool.project(point)
                assert oracle.all_valid
                assert (oracle.signed_gaps[0] < 0.0) == (point[2] > 0.0)
                if point[2] > 0.0:
                    selected_distance = min(
                        np.linalg.norm(
                            (point - feature.anchor) @ feature.normal_projector
                        )
                        for feature in (
                            certified.geometry.first,
                            certified.geometry.second,
                        )
                    )
                    assert -oracle.signed_gaps[0] == pytest.approx(
                        selected_distance, abs=2e-12
                    )


@pytest.mark.parametrize("scale", (1e-3, 1.0, 1e3))
def test_partial_contact_reference_area_law_scales_with_length(scale):
    tool = _closed_L_prism()
    base = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    original = integrate_certified_partial_closed_two_feature_penalty(
        base, base, tool, (0, 12), penalty=500.0
    ).response
    scaled = closed_triangulated_rigid_solid(
        vertices=scale * tool.facets.vertices,
        triangles=tool.facets.triangles,
        reviewed_embedded=True,
    )
    response = integrate_certified_partial_closed_two_feature_penalty(
        scale * base, scale * base, scaled, (0, 12), penalty=500.0
    ).response
    assert response.potential / scale**4 == pytest.approx(original.potential, rel=2e-8)
    np.testing.assert_allclose(
        response.nodal_residual / scale**3,
        original.nodal_residual,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        response.nodal_tangent / scale**2,
        original.nodal_tangent,
        rtol=2e-8,
        atol=2e-8,
    )


def test_partial_contact_finds_bounded_finite_features_without_user_facet_ids():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    discovered = integrate_discovered_partial_closed_two_feature_penalty(
        slave, slave, tool, penalty=500.0
    )
    expected = integrate_certified_partial_closed_two_feature_penalty(
        slave, slave, tool, (0, 12), penalty=500.0
    )
    assert discovered.geometry.feature_kinds == ("face", "edge")
    assert discovered.response.potential == pytest.approx(
        expected.response.potential, rel=1e-12
    )
    np.testing.assert_allclose(
        discovered.response.nodal_residual,
        expected.response.nodal_residual,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        discovered.response.nodal_tangent,
        expected.response.nodal_tangent,
        rtol=1e-12,
        atol=1e-12,
    )


def test_partial_contact_discovery_refuses_incomplete_facet_budget():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (0.905, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    with pytest.raises(ValueError, match="bounded 3D facets"):
        integrate_discovered_partial_closed_two_feature_penalty(
            slave, slave, tool, penalty=500.0, maximum_master_facets=8
        )


def test_partial_contact_discovery_does_not_fallback_on_a_third_tool_boundary():
    tool = _closed_L_prism()
    slave = np.asarray(
        ((0.895, 0.875, -0.02), (1.005, 0.875, 0.175), (0.895, 0.885, 0.16))
    )
    with pytest.raises(NotImplementedError, match="No complete two-feature"):
        integrate_discovered_partial_closed_two_feature_penalty(
            slave, slave, tool, penalty=500.0
        )
