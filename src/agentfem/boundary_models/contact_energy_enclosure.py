# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded reference-area penalty-energy diagnostic for a closed rigid tool.

This is deliberately *not* a contact Operator.  It neither supplies a nodal
residual nor a Newton tangent and must not make an unresolved face admissible
to a nonlinear Step.  Its purpose is to detect a missed positive-area contact
patch and to provide a reproducible target for feature-partition research.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid, _solid_winding_number
from .rigid import PrescribedRigidMotion


@dataclass(frozen=True)
class ContactPenaltyEnergyEnclosure:
    """Floating-point conservative-margin bounds, not formal interval arithmetic."""

    lower_bound: float
    upper_bound: float
    reference_area: float
    evaluated_points: int
    nonunique_nearest_points: int
    unresolved_signed_distance_points: int
    leaf_count: int
    maximum_reached_depth: int
    within_requested_tolerance: bool
    status: str
    tool_fingerprint: str


def _triangle_area(vertices: np.ndarray) -> float:
    return 0.5 * float(
        np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    )


def _subdivide(reference: np.ndarray, current: np.ndarray):
    a, b, c = np.arange(3)
    ab, bc, ca = (
        (reference[a] + reference[b]) / 2,
        (reference[b] + reference[c]) / 2,
        (reference[c] + reference[a]) / 2,
    )
    xab, xbc, xca = (
        (current[a] + current[b]) / 2,
        (current[b] + current[c]) / 2,
        (current[c] + current[a]) / 2,
    )
    return (
        (np.asarray((reference[a], ab, ca)), np.asarray((current[a], xab, xca))),
        (np.asarray((ab, reference[b], bc)), np.asarray((xab, current[b], xbc))),
        (np.asarray((ca, bc, reference[c])), np.asarray((xca, xbc, current[c]))),
        (np.asarray((ab, bc, ca)), np.asarray((xab, xbc, xca))),
    )


def _scalar_signed_distances(
    surface: ClosedTriangulatedRigidSolid,
    points: np.ndarray,
    motion: PrescribedRigidMotion | None,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep scalar distance at a nonunique nearest point; invent no normal."""

    reference_points = _to_tool_frame(points, motion, factor)
    candidates = surface._bvh._candidate_evidence(
        reference_points,
        None,
        require_same_normal=False,
    )
    gaps = np.full(len(points), np.nan)
    lower_box, upper_box = surface._bvh.bounds
    tolerance = surface.facets.tolerance
    for index, point in enumerate(reference_points):
        if not candidates.has_candidate[index]:
            continue
        radius = float(np.sqrt(candidates.squared_distances[index]))
        if radius <= tolerance:
            gaps[index] = 0.0
            continue
        if np.any((point < lower_box - tolerance) | (point > upper_box + tolerance)):
            gaps[index] = radius
            continue
        winding = _solid_winding_number(point, surface._triangles)
        if abs(winding) <= 1.0e-6:
            gaps[index] = radius
        elif abs(winding - 1.0) <= 1.0e-6:
            gaps[index] = -radius
    return gaps, candidates.ambiguous


def _to_tool_frame(
    points: np.ndarray,
    motion: PrescribedRigidMotion | None,
    factor: float,
) -> np.ndarray:
    if motion is None:
        return points
    pose = motion.state(factor)
    return np.asarray(motion.reference_point, dtype=float) + (
        points - np.asarray(pose["reference_point"], dtype=float)
    ) @ np.asarray(pose["rotation_matrix"], dtype=float)


def bound_closed_contact_penalty_energy(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    *,
    penalty: float,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.01,
    maximum_depth: int = 10,
    maximum_cells: int = 50_000,
) -> ContactPenaltyEnergyEnclosure:
    """Enclose ``0.5*k*int_ref max(-signed_distance, 0)**2 dA``.

    A reviewed embedded solid has a 1-Lipschitz signed distance. At a cell
    centroid with gap ``g`` and current-space radius ``r``, all cell gaps lie
    between ``g-r`` and ``g+r``. Monotonicity of the normal penalty density
    gives positive lower and upper cell integrals. If the centroid projection
    has nonunique nearest points, its *scalar* distance remains defined and
    the sign is independently resolved by closed-solid winding. A failed
    scalar sign check falls back to a finite tool-vertex upper bound, never
    to zero contact. The largest uncertain
    cell is split first. This does not establish contact force, tangent,
    complementarity, or convergence of a nonlinear solve.
    """

    reference = np.asarray(reference_vertices, dtype=float)
    current = np.asarray(current_vertices, dtype=float)
    if (
        reference.shape != (3, 3)
        or current.shape != (3, 3)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("Penalty energy enclosure requires finite P1 triangles.")
    area = _triangle_area(reference)
    current_area = _triangle_area(current)
    if (
        not np.isfinite(area)
        or not np.isfinite(current_area)
        or area <= 0.0
        or current_area <= 0.0
    ):
        raise ValueError("Penalty energy enclosure requires nondegenerate triangles.")
    if not isinstance(surface, ClosedTriangulatedRigidSolid):
        raise TypeError("Penalty energy enclosure requires a reviewed closed tool.")
    selected_penalty = float(penalty)
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    selected_factor = float(factor)
    if (
        not np.isfinite(selected_penalty)
        or selected_penalty <= 0.0
        or not np.isfinite(absolute)
        or absolute < 0.0
        or not np.isfinite(relative)
        or relative < 0.0
        or not np.isfinite(selected_factor)
        or not 0.0 <= selected_factor <= 1.0 + 1.0e-12
    ):
        raise ValueError("Penalty, tolerances and motion factor must be valid.")
    if (
        type(maximum_depth) is not int
        or not 0 <= maximum_depth <= 16
        or type(maximum_cells) is not int
        or maximum_cells < 1
    ):
        raise ValueError(
            "Penalty energy enclosure needs bounded depth and cell budget."
        )
    if motion is not None and (
        not isinstance(motion, PrescribedRigidMotion) or motion.dimension != 3
    ):
        raise TypeError("Penalty energy enclosure needs a 3D prescribed rigid motion.")
    tool_vertices = np.asarray(surface.facets.vertices, dtype=float)
    if motion is not None:
        pose = motion.state(selected_factor)
        tool_vertices = (
            np.asarray(pose["reference_point"], dtype=float)
            + (tool_vertices - np.asarray(motion.reference_point, dtype=float))
            @ np.asarray(pose["rotation_matrix"], dtype=float).T
        )
    world_scale = max(
        1.0,
        float(np.max(np.abs(current))),
        float(np.max(np.abs(tool_vertices))),
    )
    local_scale = max(
        float(np.max(np.linalg.norm(current[1:] - current[0], axis=1))),
        float(np.max(np.linalg.norm(tool_vertices - tool_vertices[0], axis=1))),
    )
    if 128.0 * np.finfo(float).eps * world_scale > 1.0e-4 * local_scale:
        raise ValueError(
            "Penalty energy geometry is unresolved at world-coordinate precision."
        )
    # Deliberate roundoff allowance: these bounds are numerical evidence,
    # not directed-rounding interval arithmetic.
    margin = max(
        128.0 * np.finfo(float).eps * world_scale,
        float(surface.facets.tolerance),
    )
    cells: dict[int, tuple[np.ndarray, np.ndarray, int, float, float]] = {}
    priorities: list[tuple[float, int]] = []
    next_id = 0
    evaluated = nonunique = unresolved = deepest = 0
    total_lower = total_upper = 0.0

    def add_cells(items):
        nonlocal next_id, evaluated, nonunique, unresolved, deepest
        nonlocal total_lower, total_upper
        centers = np.asarray([np.mean(x, axis=0) for _, x, _ in items])
        gaps, ambiguous = _scalar_signed_distances(
            surface,
            centers,
            motion,
            selected_factor,
        )
        evaluated += len(items)
        nonunique += int(np.count_nonzero(ambiguous))
        unresolved += int(np.count_nonzero(~np.isfinite(gaps)))
        for index, (ref, cur, depth) in enumerate(items):
            deepest = max(deepest, depth)
            cell_area = _triangle_area(ref)
            center = centers[index]
            radius = float(np.max(np.linalg.norm(cur - center, axis=1)))
            if np.isfinite(gaps[index]):
                gap = float(gaps[index])
                lower_penetration = max(0.0, -(gap + radius + margin))
                upper_penetration = max(0.0, -(gap - radius - margin))
                lower = 0.5 * selected_penalty * lower_penetration**2 * cell_area
                upper = 0.5 * selected_penalty * upper_penetration**2 * cell_area
            else:
                # Every tool vertex belongs to the boundary. Thus distance
                # to the boundary is no greater than distance to any one of
                # them; the triangle's largest vertex distance is an upper
                # bound for every affine point in the cell.
                vertex = tool_vertices[
                    np.argmin(np.linalg.norm(tool_vertices - center, axis=1))
                ]
                distance = float(np.max(np.linalg.norm(cur - vertex, axis=1))) + margin
                lower = 0.0
                upper = 0.5 * selected_penalty * distance**2 * cell_area
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("Penalty energy enclosure overflowed its finite bounds.")
            identity = next_id
            next_id += 1
            cells[identity] = (ref, cur, depth, lower, upper)
            total_lower += lower
            total_upper += upper
            heapq.heappush(priorities, (-(upper - lower), identity))

    add_cells(((reference, current, 0),))
    while True:
        if total_upper - total_lower <= absolute + relative * total_upper:
            status = "requested_energy_width_met"
            break
        if len(cells) + 3 > maximum_cells:
            status = "cell_budget_exhausted"
            break
        while priorities and (
            priorities[0][1] not in cells or cells[priorities[0][1]][2] >= maximum_depth
        ):
            heapq.heappop(priorities)
        if not priorities:
            status = "depth_budget_exhausted"
            break
        _, identity = heapq.heappop(priorities)
        ref, cur, depth, old_lower, old_upper = cells.pop(identity)
        total_lower -= old_lower
        total_upper -= old_upper
        add_cells(tuple((a, b, depth + 1) for a, b in _subdivide(ref, cur)))
    lower = math.fsum(item[3] for item in cells.values())
    upper = math.fsum(item[4] for item in cells.values())
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("Penalty energy enclosure lost finite bound ordering.")
    return ContactPenaltyEnergyEnclosure(
        lower_bound=max(0.0, float(np.nextafter(lower, -np.inf))),
        upper_bound=float(np.nextafter(upper, np.inf)),
        reference_area=area,
        evaluated_points=evaluated,
        nonunique_nearest_points=nonunique,
        unresolved_signed_distance_points=unresolved,
        leaf_count=len(cells),
        maximum_reached_depth=deepest,
        within_requested_tolerance=(
            status == "requested_energy_width_met"
            and upper - lower <= absolute + relative * upper
        ),
        status=status,
        tool_fingerprint=surface.geometry_fingerprint,
    )


__all__ = ["ContactPenaltyEnergyEnclosure", "bound_closed_contact_penalty_energy"]
