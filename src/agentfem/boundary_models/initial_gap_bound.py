# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Conservative reference-triangle penetration bound for a true signed distance.

For a closed-solid signed-distance function ``g`` with Lipschitz constant one,
every point of a planar subtriangle lies within the maximum vertex distance
from its centroid ``c``. Consequently ``g(x) >= g(c) - radius``. Adaptive
subdivision tightens this *upper bound on penetration* without pretending
that a few quadrature points certify a whole contact face. The caller must
admit only a reviewed closed-surface projector that returns signed distance;
an open oriented triangle sheet is not such a projector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .convex_polyhedron import ConvexPolyhedralRigidSurface
from .rigid import RigidPlaneSurface, TriangulatedRigidSurface


def initial_gap_bound_method(
    surface, *, interpolation_degree: int, geometry_degree: int,
    facet_topology: str, contact_integration: str = "gauss",
) -> str | None:
    """Return only geometry/trace combinations with a face-wise proof."""

    if interpolation_degree != 1 or geometry_degree != 1:
        return None
    if isinstance(surface, RigidPlaneSurface):
        return "exact_plane_nodal"
    if (
        facet_topology == "triangle"
        and isinstance(surface, TriangulatedRigidSurface)
        and contact_integration == "reviewed_planar_branch_partition"
    ):
        return "certified_reviewed_planar_branch_nodal"
    if facet_topology == "triangle" and isinstance(
        surface, ConvexPolyhedralRigidSurface,
    ):
        return "convex_support_plane_or_adaptive_lipschitz"
    if facet_topology == "triangle" and isinstance(
        surface, ClosedTriangulatedRigidSolid,
    ):
        return "adaptive_closed_signed_distance_lipschitz"
    return None


def certify_initial_reviewed_planar_branch_gap(
    triangles, surface: TriangulatedRigidSurface, *, projector,
    penetration_limit: float,
) -> InitialGapBound:
    """Bound an open sheet's oriented initial gap on certified affine faces.

    This does not treat an open sheet as a closed signed-distance solid. The
    whole-face branch proof first establishes one oriented plane per slave
    face; only then is the affine gap's minimum taken at its three vertices.
    An uncovered rim, nearest-feature switch or exhausted proof budget fails
    closed instead of falling back to sampled quadrature gaps.
    """

    from .contact_planar_branch_bound import certify_reviewed_planar_branch

    cells = np.asarray(triangles, dtype=float)
    limit = float(penetration_limit)
    if not isinstance(surface, TriangulatedRigidSurface):
        raise TypeError("Open-sheet initial gap requires reviewed triangle geometry.")
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or not np.all(np.isfinite(cells)):
        raise ValueError("Open-sheet initial gap requires finite planar triangles.")
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("Initial penetration limit must be finite and nonnegative.")
    certificate = certify_reviewed_planar_branch(
        cells, surface, projector=projector, factor=0.0,
    )
    if not certificate.certified:
        raise NotImplementedError(
            "AFM-CONTACT-INITIAL-UNCERTIFIED: the open-sheet reference trace "
            f"has no complete oriented planar branch ({certificate.status})."
        )
    gaps = np.einsum(
        "fvi,fi->fv",
        cells - certificate.plane_points[:, None, :],
        certificate.plane_normals,
    )
    scale = max(1.0, float(np.max(np.abs(cells))), limit)
    margin = 128.0 * np.finfo(float).eps * scale
    bound = max(0.0, -float(np.min(gaps)) + margin)
    if bound > limit:
        raise ValueError(
            "AFM-CONTACT-INITIAL-OVERLAP: certified open-sheet reference "
            f"penetration bound {bound:.6g} exceeds the declared limit {limit:.6g}."
        )
    return InitialGapBound(bound, 3 * len(cells), 0)


@dataclass(frozen=True)
class InitialGapBound:
    maximum_penetration_bound: float
    sampled_point_count: int
    maximum_refinement_depth: int


def certify_initial_convex_triangle_gap(
    triangles,
    surface: ConvexPolyhedralRigidSurface,
    *,
    penetration_limit: float,
    maximum_depth: int = 9,
    maximum_samples: int = 500_000,
) -> InitialGapBound:
    """Use an exact dominant support plane when possible; otherwise subdivide.

    For a convex intersection of outward half-spaces, the maximum support-
    plane gap is the true signed distance inside the solid and a lower bound
    outside it. If one plane dominates at all three vertices of an affine
    triangle, it dominates everywhere on that triangle, so its minimum is a
    nodal minimum. A mixed-feature triangle keeps the conservative Lipschitz
    certificate rather than guessing its minimizer from a few samples.
    """

    if not isinstance(surface, ConvexPolyhedralRigidSurface):
        raise TypeError("Convex initial gap requires certified convex geometry.")
    cells = np.asarray(triangles, dtype=float)
    limit = float(penetration_limit)
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or not np.all(np.isfinite(cells)):
        raise ValueError("Initial gap certificate requires finite planar triangles (n, 3, 3).")
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("Initial penetration limit must be finite and nonnegative.")
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int) or maximum_depth < 0:
        raise ValueError("Initial gap certificate maximum_depth must be nonnegative.")
    if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples < 1:
        raise ValueError("Initial gap certificate maximum_samples must be positive.")
    if not len(cells):
        return InitialGapBound(0.0, 0, 0)
    normals = np.asarray(surface.facets.facet_normals, dtype=float)
    offsets = np.asarray(surface._offsets, dtype=float)
    scale = max(1.0, float(np.max(np.abs(cells))), float(np.max(np.abs(offsets))), limit)
    margin = 64.0 * np.finfo(float).eps * scale
    unresolved = []
    certified_bound = 0.0
    for triangle in cells:
        gaps = triangle @ normals.T - offsets
        maximum = np.max(gaps, axis=1)
        if float(np.min(maximum)) < -limit - margin:
            raise ValueError(
                "AFM-CONTACT-INITIAL-OVERLAP: a reference face vertex "
                "exceeds initial_penetration_limit."
            )
        dominant = np.flatnonzero(np.all(gaps >= maximum[:, None], axis=0))
        if not len(dominant):
            unresolved.append(triangle)
            continue
        lower = float(np.min(gaps[:, dominant[0]])) - margin
        if lower < -limit:
            unresolved.append(triangle)
            continue
        certified_bound = max(certified_bound, max(0.0, -lower))
    if not unresolved:
        return InitialGapBound(certified_bound, 3 * len(cells), 0)
    fallback = certify_initial_triangle_gap(
        np.asarray(unresolved), surface,
        penetration_limit=limit,
        maximum_depth=maximum_depth,
        maximum_samples=maximum_samples,
    )
    return InitialGapBound(
        max(certified_bound, fallback.maximum_penetration_bound),
        3 * len(cells) + fallback.sampled_point_count,
        fallback.maximum_refinement_depth,
    )


def certify_initial_triangle_gap(
    triangles,
    projector,
    *,
    penetration_limit: float,
    maximum_depth: int = 9,
    maximum_samples: int = 500_000,
) -> InitialGapBound:
    """Certify an initial limit, or fail without guessing an unsampled gap.

    ``projector`` must return a true 1-Lipschitz signed-distance field on the
    complete query region. This property is an explicit caller-side geometry
    admission requirement, not inferred from an arbitrary ``project`` method.
    """

    cells = np.asarray(triangles, dtype=float)
    limit = float(penetration_limit)
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or not np.all(np.isfinite(cells)):
        raise ValueError("Initial gap certificate requires finite planar triangles (n, 3, 3).")
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("Initial penetration limit must be finite and nonnegative.")
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int) or maximum_depth < 0:
        raise ValueError("Initial gap certificate maximum_depth must be nonnegative.")
    if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples < 1:
        raise ValueError("Initial gap certificate maximum_samples must be positive.")
    if len(cells) == 0:
        return InitialGapBound(0.0, 0, 0)
    scale = max(1.0, float(np.max(np.abs(cells))), limit)
    margin = 64.0 * np.finfo(float).eps * scale
    count = 0
    certified_bound = 0.0
    for depth in range(maximum_depth + 1):
        if count + len(cells) > maximum_samples:
            raise NotImplementedError(
                "AFM-CONTACT-INITIAL-UNCERTIFIED: signed-distance face bound "
                "exceeded its declared sample budget."
            )
        centers = np.mean(cells, axis=1)
        radii = np.max(np.linalg.norm(cells - centers[:, None, :], axis=2), axis=1)
        projected = projector.project(centers, factor=0.0)
        if not projected.all_valid:
            raise ValueError(
                "AFM-CONTACT-INITIAL-PROJECTION: signed-distance certificate "
                "encountered an invalid or ambiguous reference projection."
            )
        gaps = np.asarray(projected.signed_gaps, dtype=float)
        count += len(cells)
        if np.any(gaps < -limit - margin):
            raise ValueError(
                "AFM-CONTACT-INITIAL-OVERLAP: a reference face contains a "
                "sampled penetration exceeding initial_penetration_limit."
            )
        lower = gaps - radii - margin
        safe = lower >= -limit
        if np.any(safe):
            certified_bound = max(
                certified_bound,
                float(np.max(np.maximum(0.0, -lower[safe]))),
            )
        cells = cells[~safe]
        if not len(cells):
            return InitialGapBound(certified_bound, count, depth)
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
    raise NotImplementedError(
        "AFM-CONTACT-INITIAL-UNCERTIFIED: the conservative signed-distance "
        "face bound could not prove initial_penetration_limit within the "
        "declared refinement depth; no contact state was accepted."
    )


__all__ = [
    "InitialGapBound", "certify_initial_triangle_gap",
    "certify_initial_convex_triangle_gap", "initial_gap_bound_method",
    "certify_initial_reviewed_planar_branch_gap",
]
