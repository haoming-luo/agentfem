# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bound one smooth plane branch per affine slave face of a reviewed tool.

This conservative geometric precondition is intentionally stronger than
sampling a few nearest facets. If each affine slave triangle projects wholly
inside the union of a coplanar master patch and no noncoplanar master facet
can become closer, its signed gap is one affine plane function everywhere.
Distinct slave faces may have distinct certified planes. The fixed-reference
planar active-set clipper may then integrate each branch exactly. A failed
certificate says nothing about physical admissibility and cannot be treated
as an inactive contact law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .rigid import (
    PrescribedRigidMotion, TriangulatedRigidSurface, _closest_point_on_triangle,
    _triangle_barycentric_coordinates,
)
from .search import TriangleSurfaceBVH


def _clip_polygon_halfspace(polygon, values, *, keep_positive):
    """Clip one convex polygon by an affine barycentric halfspace."""

    if len(polygon) == 0:
        return np.empty((0, 3))
    signed = values if keep_positive else -values
    retained = []
    for index in range(len(polygon)):
        start = polygon[index - 1]
        end = polygon[index]
        a = float(signed[index - 1])
        b = float(signed[index])
        if (a >= 0.0) != (b >= 0.0):
            retained.append(start + a / (a - b) * (end - start))
        if b >= 0.0:
            retained.append(end)
    return np.asarray(retained, dtype=float).reshape(-1, 3)


def _planar_polygon_area(polygon, normal):
    if len(polygon) < 3:
        return 0.0
    vectors = polygon[1:] - polygon[0]
    return 0.5 * abs(float(np.sum(
        np.cross(vectors[:-1], vectors[1:]) @ normal,
    )))


def _quadratic_plane_distance_difference_lower_bound(
    cell, candidate_normal, candidate_offset, other_normal, other_offset,
):
    """Bernstein lower bound for other-plane squared distance minus candidate.

    Both signed plane gaps are affine on one P1 face, so their squared
    difference is exactly quadratic. Its six degree-two Bernstein
    coefficients bound the entire face; three vertex samples alone do not.
    The other *infinite plane* gives a lower bound on distance to its finite
    master triangle, hence a positive result safely excludes that triangle.
    """

    samples = np.vstack((
        cell,
        0.5 * (cell[0] + cell[1]),
        0.5 * (cell[1] + cell[2]),
        0.5 * (cell[2] + cell[0]),
    ))
    candidate = samples @ candidate_normal - candidate_offset
    other = samples @ other_normal - other_offset
    values = other**2 - candidate**2
    coefficients = np.asarray((
        *values[:3],
        2.0 * values[3] - 0.5 * (values[0] + values[1]),
        2.0 * values[4] - 0.5 * (values[1] + values[2]),
        2.0 * values[5] - 0.5 * (values[2] + values[0]),
    ))
    return float(np.min(coefficients)), candidate[:3], other[:3]


def _coplanar_patch_covers_face(
    feet, master, coplanar_slots, normal, *, maximum_pieces=256,
):
    """Subtract a union of coplanar master triangles from one slave triangle.

    The uncovered polygons form a disjoint decomposition even if master
    triangles overlap. Zero-area shared seams do not create uncovered contact
    area. Any positive-area remainder or exhausted piece budget fails closed.
    """

    face_area = _planar_polygon_area(feet, normal)
    if face_area <= 0.0:
        return False
    area_margin = 128.0 * np.finfo(float).eps * face_area
    uncovered = [feet]
    for slot in coplanar_slots:
        triangle = master[int(slot)]
        next_uncovered = []
        for polygon in uncovered:
            inside_remainder = polygon
            for corner in range(3):
                barycentric = np.asarray([
                    _triangle_barycentric_coordinates(point, triangle)[corner]
                    for point in inside_remainder
                ])
                outside = _clip_polygon_halfspace(
                    inside_remainder, barycentric, keep_positive=False,
                )
                if _planar_polygon_area(outside, normal) > area_margin:
                    next_uncovered.append(outside)
                inside_remainder = _clip_polygon_halfspace(
                    inside_remainder, barycentric, keep_positive=True,
                )
                if len(inside_remainder) < 3:
                    break
            if len(next_uncovered) > maximum_pieces:
                return False
        uncovered = next_uncovered
        if not uncovered:
            return True
    return not uncovered


@dataclass(frozen=True, eq=False)
class ClosedPlanarBranchBound:
    certified: bool
    status: str
    plane_points: np.ndarray | None
    plane_normals: np.ndarray | None
    slave_face_count: int
    examined_master_facet_count: int
    minimum_competitor_clearance: float | None


def certify_reviewed_planar_branch(
    triangles,
    surface: ClosedTriangulatedRigidSolid | TriangulatedRigidSurface,
    *,
    projector: TriangleSurfaceBVH | None = None,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    maximum_slave_faces: int = 1024,
    maximum_master_facets: int = 4096,
    maximum_refinement_levels: int = 1,
    _refinement_budget: list[int] | None = None,
) -> ClosedPlanarBranchBound:
    """Certify an exact affine gap branch on each complete current P1 face.

    For center ``c`` and enclosing radius ``r``, both distances to the
    candidate master plane and to any competing triangle are 1-Lipschitz.
    Thus ``d_other(c)-d_candidate(c)>2r`` excludes that competitor everywhere
    on the slave face. A fast barycentric check handles one containing master
    triangle; otherwise a bounded polygon subtraction proves that the
    projected slave triangle is covered by the coplanar master-triangle union.
    Coplanar, identically oriented facets share the same signed-gap function.
    If the radius test alone is inconclusive, a bounded subdivision may prove
    that every child still has this *same* plane. Different child planes never
    license an approximate feature-switch integral.
    """

    if isinstance(surface, ClosedTriangulatedRigidSolid):
        if projector is not None:
            raise ValueError("Closed-solid branch uses its own signed-distance projector.")
        facets = surface.facets
        query_projector = surface
        master = surface._triangles
        slots = surface._facet_slots
        broad_phase = surface._bvh
    elif isinstance(surface, TriangulatedRigidSurface):
        if not isinstance(projector, TriangleSurfaceBVH) or projector.surface is not surface:
            raise ValueError(
                "Open faceted branch requires the matching reviewed triangle BVH."
            )
        facets = surface
        query_projector = projector
        master = projector._triangle_points
        slots = {int(identity): slot for slot, identity in enumerate(surface.facet_ids)}
        broad_phase = projector
    else:
        raise TypeError("Planar branch bound requires reviewed triangle geometry.")
    cells = np.asarray(triangles, dtype=float)
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or not np.all(np.isfinite(cells)):
        raise ValueError("Planar branch bound requires finite triangles (n, 3, 3).")
    if motion is not None and not isinstance(motion, PrescribedRigidMotion):
        raise TypeError("Planar branch bound requires a prescribed rigid motion.")
    selected_factor = float(factor)
    if not np.isfinite(selected_factor) or not 0.0 <= selected_factor <= 1.0 + 1.0e-12:
        raise ValueError("Planar branch bound factor must lie in [0, 1].")
    if (
        type(maximum_slave_faces) is not int or maximum_slave_faces < 1
        or type(maximum_master_facets) is not int or maximum_master_facets < 1
        or type(maximum_refinement_levels) is not int
        or not 0 <= maximum_refinement_levels <= 3
    ):
        raise ValueError("Planar branch budgets or refinement depth are invalid.")
    refinement_budget = [4096] if _refinement_budget is None else _refinement_budget
    if (
        not isinstance(refinement_budget, list)
        or len(refinement_budget) != 1
        or type(refinement_budget[0]) is not int
        or refinement_budget[0] < 0
    ):
        raise ValueError("Planar branch refinement budget is invalid.")
    face_count = len(cells)
    master_count = len(master)

    def failed(status: str, examined: int = 0) -> ClosedPlanarBranchBound:
        return ClosedPlanarBranchBound(
            False, status, None, None, face_count, examined, None,
        )

    if not face_count:
        return failed("empty_slave_boundary")
    if face_count > maximum_slave_faces or master_count > maximum_master_facets:
        return failed("geometry_budget_exceeded")
    if motion is None:
        reference_cells = cells
        rotation = np.eye(3)
        reference = np.zeros(3)
        moved_reference = reference
    else:
        pose = motion.state(selected_factor)
        rotation = np.asarray(pose["rotation_matrix"], dtype=float)
        reference = np.asarray(motion.reference_point, dtype=float)
        moved_reference = np.asarray(pose["reference_point"], dtype=float)
        reference_cells = reference + (cells - moved_reference) @ rotation

    centers = np.mean(reference_cells, axis=1)
    radii = np.max(np.linalg.norm(
        reference_cells - centers[:, None, :], axis=2,
    ), axis=1)
    # Query in the tool's reference frame, then use the exact rigid map for
    # the certified world-space plane. This avoids mixing two pose conventions.
    projection = query_projector.project(centers)
    if not projection.all_valid:
        return failed("ambiguous_center_projection")
    scale = max(
        1.0, float(np.max(np.abs(reference_cells))),
        float(np.max(np.abs(master))),
    )
    length_margin = 128.0 * np.finfo(float).eps * scale
    angular_margin = 128.0 * np.finfo(float).eps
    squared_margin = 256.0 * np.finfo(float).eps * scale**2
    barycentric_margin = 1.0e-10
    normals = facets._facet_normals
    offsets = np.einsum("ij,ij->i", normals, master[:, 0, :])
    selected_normals = []
    selected_points = []
    examined = 0
    minimum_clearance = float("inf")
    for face_index, (cell, center, radius, identity, gap) in enumerate(zip(
        reference_cells, centers, radii, projection.entity_ids,
        projection.signed_gaps, strict=True,
    )):
        slot = slots[int(identity)]
        selected_triangle = master[slot]
        normal = normals[slot]
        point = selected_triangle[0]
        offset = float(np.dot(normal, point))
        plane_gap = float(np.dot(normal, center) - offset)
        if abs(plane_gap - gap) > length_margin:
            return failed("inconsistent_signed_gap", examined)
        # A slave triangle may cross a triangulation seam, but only if the
        # union of coplanar master facets covers its *whole* projection.
        feet = cell - (cell @ normal - offset)[:, None] * normal
        single_triangle = all(
            np.min(_triangle_barycentric_coordinates(foot, selected_triangle))
            > barycentric_margin for foot in feet
        )
        if not single_triangle:
            foot_lower = np.min(feet, axis=0) - length_margin
            foot_upper = np.max(feet, axis=0) + length_margin
            patch_slots = broad_phase.overlapping_facets(foot_lower, foot_upper)
            coplanar = (
                (np.linalg.norm(normals[patch_slots] - normal, axis=1)
                 <= angular_margin)
                & (np.abs(offsets[patch_slots] - offset) <= length_margin)
            )
            if not _coplanar_patch_covers_face(
                feet, master, patch_slots[coplanar], normal,
            ):
                return failed("projection_can_leave_master_patch", examined)
        candidate_distance = abs(plane_gap)
        # d_other(c)-d_plane(c)>2r is the exact whole-face exclusion test.
        # A BVH box is a lower bound on its triangle distance. Therefore a
        # node outside this radius cannot contain a competing closest facet;
        # exact triangle distances are still checked for every retained slot.
        search_radius = np.nextafter(
            candidate_distance + 2.0 * radius + length_margin, np.inf,
        )
        nearby_slots = broad_phase.facets_within_distance(center, search_radius)
        competing = (
            (np.linalg.norm(normals[nearby_slots] - normal, axis=1)
             > angular_margin)
            | (np.abs(offsets[nearby_slots] - offset) > length_margin)
        )
        # A skipped leaf has a strictly positive clearance at least equal to
        # the search margin; do not present the minimum of only the visited
        # leaves as though it were a global geometric minimum.
        face_clearance = (
            length_margin if len(nearby_slots) < master_count else float("inf")
        )
        uncertain_competitor = False
        for other_slot in nearby_slots[competing]:
            other = master[other_slot]
            examined += 1
            foot = _closest_point_on_triangle(center, *other)
            competitor_distance = float(np.linalg.norm(center - foot))
            clearance = competitor_distance - candidate_distance - 2.0 * radius
            if clearance > length_margin:
                face_clearance = min(face_clearance, clearance)
                continue
            lower_squared, candidate_vertices, other_vertices = (
                _quadratic_plane_distance_difference_lower_bound(
                    cell, normal, offset, normals[other_slot], offsets[other_slot],
                )
            )
            if lower_squared > squared_margin:
                # |g_other_plane| <= max at vertices because its gap is affine.
                # Divide the squared-distance bound by an upper bound on the
                # sum of plane distances to obtain a positive length bound.
                denominator = np.nextafter((
                    np.max(np.abs(candidate_vertices))
                    + np.max(np.abs(other_vertices))
                ), np.inf)
                face_clearance = min(
                    face_clearance,
                    (lower_squared - squared_margin) / denominator,
                )
                continue
            uncertain_competitor = True
            break
        if uncertain_competitor:
            if not maximum_refinement_levels:
                return failed("another_master_feature_can_compete", examined)
            if refinement_budget[0] < 4:
                return failed("geometry_budget_exceeded", examined)
            refinement_budget[0] -= 4
            a, b, c = cells[face_index]
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            children = np.asarray((
                (a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca),
            ))
            refined = certify_reviewed_planar_branch(
                children, surface, projector=projector, motion=motion,
                factor=selected_factor, maximum_slave_faces=maximum_slave_faces,
                maximum_master_facets=maximum_master_facets,
                maximum_refinement_levels=maximum_refinement_levels - 1,
                _refinement_budget=refinement_budget,
            )
            examined += refined.examined_master_facet_count
            if not refined.certified:
                return failed(
                    "geometry_budget_exceeded"
                    if refined.status == "geometry_budget_exceeded"
                    else "another_master_feature_can_compete",
                    examined,
                )
            world_normal = normal @ rotation.T
            world_point = moved_reference + (point - reference) @ rotation.T
            world_offset = float(world_normal @ world_point)
            if (
                np.any(np.linalg.norm(
                    refined.plane_normals - world_normal, axis=1,
                ) > angular_margin)
                or np.any(np.abs(
                    np.einsum("ij,ij->i", refined.plane_normals, refined.plane_points)
                    - world_offset,
                ) > length_margin)
            ):
                return failed("another_master_feature_can_compete", examined)
            face_clearance = (
                float("inf") if refined.minimum_competitor_clearance is None
                else refined.minimum_competitor_clearance
            )
        minimum_clearance = min(minimum_clearance, face_clearance)
        selected_points.append(point)
        selected_normals.append(normal)
    world_points = moved_reference + (np.asarray(selected_points) - reference) @ rotation.T
    world_normals = np.asarray(selected_normals) @ rotation.T
    world_points.setflags(write=False)
    world_normals.setflags(write=False)
    return ClosedPlanarBranchBound(
        True, "smooth_planar_branch_per_slave_face", world_points, world_normals,
        face_count, examined,
        None if minimum_clearance == float("inf") else minimum_clearance,
    )


def certify_closed_planar_branch(
    triangles,
    surface: ClosedTriangulatedRigidSolid,
    **options,
) -> ClosedPlanarBranchBound:
    """Backward-compatible closed-solid specialization of the reviewed proof."""

    if not isinstance(surface, ClosedTriangulatedRigidSolid):
        raise TypeError("Closed planar branch bound requires a reviewed closed solid.")
    return certify_reviewed_planar_branch(triangles, surface, **options)


__all__ = [
    "ClosedPlanarBranchBound", "certify_closed_planar_branch",
    "certify_reviewed_planar_branch",
]
