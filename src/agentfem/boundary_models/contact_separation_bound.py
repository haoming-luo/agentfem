# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Conservative zero-contact certificate on current affine slave triangles.

This is an *observation*, not a contact law. A reviewed closed solid has a
1-Lipschitz signed distance. For every point of a planar triangle, its gap is
at least the gap at the centroid minus the largest centroid-to-vertex
distance. Subdivision can prove strict separation; exhausting the budget,
an ambiguous projection, or a near-contact region cannot prove it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .rigid import PrescribedRigidMotion


@dataclass(frozen=True)
class ContactSeparationBound:
    certified: bool
    minimum_gap_lower_bound: float | None
    sampled_point_count: int
    maximum_refinement_depth: int
    status: str


def certify_current_triangle_separation(
    triangles,
    surface: ClosedTriangulatedRigidSolid,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    maximum_depth: int = 7,
    maximum_samples: int = 100_000,
) -> ContactSeparationBound:
    """Prove that a fixed set of *current* P1 faces has zero normal contact.

    A certificate applies only at this configuration and to these complete
    affine faces. It does not prove continuous-time nonintersection between
    accepted increments or validate a later active feature-switch integral.
    """

    if not isinstance(surface, ClosedTriangulatedRigidSolid):
        raise TypeError("Current face separation requires a reviewed closed solid.")
    cells = np.asarray(triangles, dtype=float)
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or not np.all(np.isfinite(cells)):
        raise ValueError("Current face separation requires finite triangles (n, 3, 3).")
    if motion is not None and not isinstance(motion, PrescribedRigidMotion):
        raise TypeError("Current face separation requires a prescribed rigid motion.")
    selected_factor = float(factor)
    if not np.isfinite(selected_factor) or not 0.0 <= selected_factor <= 1.0 + 1.0e-12:
        raise ValueError("Current face separation motion factor must be in [0, 1] within load-coordinate tolerance.")
    if (
        type(maximum_depth) is not int or maximum_depth < 0
        or type(maximum_samples) is not int or maximum_samples < 1
    ):
        raise ValueError("Current face separation budgets must be nonnegative/positive integers.")
    if not len(cells):
        return ContactSeparationBound(True, None, 0, 0, "empty_slave_boundary")

    lower_box, upper_box = surface._bvh.bounds
    scale = max(
        1.0, float(np.max(np.abs(cells))),
        float(np.max(np.abs(lower_box))),
        float(np.max(np.abs(upper_box))),
    )
    margin = 64.0 * np.finfo(float).eps * scale
    count = 0
    minimum_lower = float("inf")
    # This rigid inverse map is isometric. If a complete current slave-face
    # box and the complete tool box are strictly apart in tool coordinates,
    # the triangle cannot contact the tool; no winding/BVH point query is
    # needed. Otherwise only the unresolved faces enter adaptive subdivision.
    if motion is None:
        reference_cells = cells
    else:
        pose = motion.state(selected_factor)
        reference_cells = (
            np.asarray(motion.reference_point, dtype=float)
            + (cells - np.asarray(pose["reference_point"], dtype=float))
            @ np.asarray(pose["rotation_matrix"], dtype=float)
        )
    triangle_lower = np.min(reference_cells, axis=1)
    triangle_upper = np.max(reference_cells, axis=1)
    box_clearance = np.linalg.norm(np.maximum(
        np.maximum(lower_box - triangle_upper, triangle_lower - upper_box),
        0.0,
    ), axis=1)
    box_safe = box_clearance > margin
    if np.any(box_safe):
        minimum_lower = float(np.min(box_clearance[box_safe] - margin))
    cells = cells[~box_safe]
    if not len(cells):
        return ContactSeparationBound(
            True, minimum_lower, 0, 0, "strictly_separated_aabb",
        )
    for depth in range(maximum_depth + 1):
        if count + len(cells) > maximum_samples:
            return ContactSeparationBound(
                False, None, count, depth, "sample_budget_exhausted",
            )
        centers = np.mean(cells, axis=1)
        radii = np.max(np.linalg.norm(cells - centers[:, None, :], axis=2), axis=1)
        projection = surface.project(
            centers, motion=motion, factor=selected_factor,
        )
        count += len(cells)
        if not projection.all_valid:
            return ContactSeparationBound(
                False, None, count, depth, "ambiguous_projection",
            )
        gaps = np.asarray(projection.signed_gaps, dtype=float)
        if np.any(gaps <= margin):
            return ContactSeparationBound(
                False, None, count, depth, "contact_not_excluded",
            )
        lower = gaps - radii - margin
        safe = lower > 0.0
        if np.any(safe):
            minimum_lower = min(minimum_lower, float(np.min(lower[safe])))
        cells = cells[~safe]
        if not len(cells):
            return ContactSeparationBound(
                True, minimum_lower, count, depth, "strictly_separated",
            )
        if depth == maximum_depth:
            break
        a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        cells = np.concatenate((
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ))
    return ContactSeparationBound(
        False, None, count, maximum_depth, "depth_budget_exhausted",
    )


__all__ = ["ContactSeparationBound", "certify_current_triangle_separation"]
