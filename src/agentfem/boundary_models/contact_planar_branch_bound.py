# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bound one smooth plane branch per affine slave face of a closed tool.

This conservative geometric precondition is intentionally stronger than
sampling a few nearest facets. If each affine slave triangle projects wholly
inside one master triangle and no noncoplanar master facet can become closer,
its signed gap is one affine plane function everywhere. Distinct slave faces
may have distinct certified planes. The fixed-reference planar active-set
clipper may then integrate each branch exactly. A failed
certificate says nothing about physical admissibility and cannot be treated
as an inactive contact law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .rigid import (
    PrescribedRigidMotion, _closest_point_on_triangle,
    _triangle_barycentric_coordinates,
)


@dataclass(frozen=True, eq=False)
class ClosedPlanarBranchBound:
    certified: bool
    status: str
    plane_points: np.ndarray | None
    plane_normals: np.ndarray | None
    slave_face_count: int
    examined_master_facet_count: int
    minimum_competitor_clearance: float | None


def certify_closed_planar_branch(
    triangles,
    surface: ClosedTriangulatedRigidSolid,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    maximum_slave_faces: int = 256,
    maximum_master_facets: int = 256,
) -> ClosedPlanarBranchBound:
    """Certify an exact affine gap branch on each complete current P1 face.

    For center ``c`` and enclosing radius ``r``, both distances to the
    candidate master triangle and to any competing triangle are 1-Lipschitz.
    Thus ``d_other(c)-d_candidate(c)>2r`` excludes that competitor everywhere
    on the slave face. The orthogonal projection of all three slave vertices
    must also lie *strictly* inside the candidate triangle. Barycentric
    affinity then keeps every interior projection there. Coplanar,
    identically oriented master triangles share the same signed-gap function
    and are not competitors. Numerical margins only make admission stricter.
    """

    if not isinstance(surface, ClosedTriangulatedRigidSolid):
        raise TypeError("Planar branch bound requires a reviewed closed solid.")
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
    ):
        raise ValueError("Planar branch budgets must be positive integers.")
    master = surface._triangles
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
    projection = surface.project(centers)
    if not projection.all_valid:
        return failed("ambiguous_center_projection")
    scale = max(
        1.0, float(np.max(np.abs(reference_cells))),
        float(np.max(np.abs(master))),
    )
    length_margin = 128.0 * np.finfo(float).eps * scale
    angular_margin = 1.0e-10
    barycentric_margin = 1.0e-10
    normals = surface.facets._facet_normals
    offsets = np.einsum("ij,ij->i", normals, master[:, 0, :])
    master_lower = np.min(master, axis=1)
    master_upper = np.max(master, axis=1)
    selected_normals = []
    selected_points = []
    examined = 0
    minimum_clearance = float("inf")
    for cell, center, radius, identity, gap in zip(
        reference_cells, centers, radii, projection.entity_ids,
        projection.signed_gaps, strict=True,
    ):
        slot = surface._facet_slots[int(identity)]
        selected_triangle = master[slot]
        normal = normals[slot]
        point = selected_triangle[0]
        offset = float(np.dot(normal, point))
        plane_gap = float(np.dot(normal, center) - offset)
        if abs(plane_gap - gap) > length_margin:
            return failed("inconsistent_signed_gap", examined)
        # A whole P1 slave triangle projects into this *one* master triangle.
        # This deliberately rejects coplanar triangulation-seam crossings.
        feet = cell - (cell @ normal - offset)[:, None] * normal
        if any(
            np.min(_triangle_barycentric_coordinates(foot, selected_triangle))
            <= barycentric_margin for foot in feet
        ):
            return failed("projection_can_leave_master_triangle", examined)
        candidate_distance = abs(plane_gap)
        competing = (
            (np.linalg.norm(normals - normal, axis=1) > angular_margin)
            | (np.abs(offsets - offset) > length_margin)
        )
        # The AABB contains each master triangle. Its distance is a lower
        # bound on the exact triangle distance, so a box already farther
        # than d(candidate)+2r cannot compete anywhere on this slave face.
        box_offset = np.maximum(
            np.maximum(master_lower - center, center - master_upper), 0.0,
        )
        box_clearance = (
            np.linalg.norm(box_offset, axis=1)
            - candidate_distance - 2.0 * radius
        )
        box_excluded = competing & (box_clearance > length_margin)
        if np.any(box_excluded):
            minimum_clearance = min(
                minimum_clearance, float(np.min(box_clearance[box_excluded])),
            )
        for other_slot in np.flatnonzero(competing & ~box_excluded):
            other = master[other_slot]
            examined += 1
            foot = _closest_point_on_triangle(center, *other)
            competitor_distance = float(np.linalg.norm(center - foot))
            clearance = competitor_distance - candidate_distance - 2.0 * radius
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance <= length_margin:
                return failed("another_master_feature_can_compete", examined)
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


__all__ = ["ClosedPlanarBranchBound", "certify_closed_planar_branch"]
