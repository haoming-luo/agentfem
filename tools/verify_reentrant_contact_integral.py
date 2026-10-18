# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent numerical oracle for a reentrant rigid-boundary face integral.

The rigid solid is the extrusion of an L polygon. A slave triangle spans a
physical reentrant edge, two adjacent planar features, and a small free notch.
Distances are computed directly from the six polygon segments, independently
of AgentFEM's closed-polyhedron projector. Uniform positive-area triangle
subdivision supplies a refinement reference, not a certified exact integral.
The three-point face rule is displayed to expose its sampling error.
"""

from __future__ import annotations

import json

import numpy as np


POLYGON = np.asarray(((0., 0.), (2., 0.), (2., 1.),
                      (1., 1.), (1., 2.), (0., 2.)))
SLAVE = np.asarray(((0.9, 0.9), (3.0, 0.9), (0.9, 3.0)))


def signed_gap(points: np.ndarray) -> np.ndarray:
    """Signed distance to the *boundary* of the closed extruded L polygon."""

    query = np.asarray(points, dtype=float)
    if query.ndim != 2 or query.shape[1] != 2 or not np.all(np.isfinite(query)):
        raise ValueError("Expected finite planar query points (n, 2).")
    distance_squared = np.full(len(query), np.inf)
    for first, second in zip(POLYGON, np.roll(POLYGON, -1, axis=0), strict=True):
        edge = second - first
        parameter = np.clip((query - first) @ edge / (edge @ edge), 0.0, 1.0)
        foot = first + parameter[:, None] * edge
        distance_squared = np.minimum(
            distance_squared, np.sum((query - foot) ** 2, axis=1),
        )
    x, y = query.T
    inside = (
        (x >= 0.0) & (y >= 0.0)
        & (((x <= 2.0) & (y <= 1.0)) | ((x <= 1.0) & (y <= 2.0)))
    )
    distance = np.sqrt(distance_squared)
    return np.where(inside, -distance, distance)


def triangle_centroids(subdivisions: int, vertices=SLAVE) -> np.ndarray:
    """Centroids of equal-area positive subtriangles of one affine face."""

    if type(subdivisions) is not int or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer.")
    selected = np.asarray(vertices, dtype=float)
    if selected.shape != (3, 2) or not np.all(np.isfinite(selected)):
        raise ValueError("Expected a finite triangle (3, 2).")
    points = []
    for i in range(subdivisions):
        for j in range(subdivisions - i):
            a = selected[0] + (i * (selected[1] - selected[0])
                               + j * (selected[2] - selected[0])) / subdivisions
            b = a + (selected[1] - selected[0]) / subdivisions
            c = a + (selected[2] - selected[0]) / subdivisions
            points.append((a + b + c) / 3.0)
            if i + j < subdivisions - 1:
                d = b + (selected[2] - selected[0]) / subdivisions
                points.append((b + c + d) / 3.0)
    return np.asarray(points)


def penalty_potential(*, subdivisions: int | None, shift_x: float = 0.0) -> float:
    """Integrate 0.5*negative_gap**2 with unit penalty, per unit extrusion."""

    vertices = SLAVE + np.asarray((float(shift_x), 0.0))
    area = 0.5 * abs(float(np.linalg.det(np.stack((
        vertices[1] - vertices[0], vertices[2] - vertices[0],
    )))))
    if subdivisions is None:
        barycentric = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                                  (1 / 6, 2 / 3, 1 / 6),
                                  (1 / 6, 1 / 6, 2 / 3)))
        query = barycentric @ vertices
    else:
        query = triangle_centroids(subdivisions, vertices)
    gaps = signed_gap(query)
    return float(area * np.mean(0.5 * np.maximum(-gaps, 0.0) ** 2))


def sampled_reference_p1_nodal_force(*, subdivisions: int) -> np.ndarray:
    """Independent centroid-refinement screen, not a certified force integral."""

    points = triangle_centroids(subdivisions)
    inside = signed_gap(points) < 0.0
    distance_squared = np.full(len(points), np.inf)
    nearest_foot = np.zeros_like(points)
    for first, second in zip(POLYGON, np.roll(POLYGON, -1, axis=0), strict=True):
        edge = second - first
        parameter = np.clip((points - first) @ edge / (edge @ edge), 0.0, 1.0)
        foot = first + parameter[:, None] * edge
        candidate = np.sum((points - foot) ** 2, axis=1)
        closer = candidate < distance_squared
        distance_squared[closer] = candidate[closer]
        nearest_foot[closer] = foot[closer]
    coordinates = (points - SLAVE[0]) @ np.linalg.inv(
        np.stack((SLAVE[1] - SLAVE[0], SLAVE[2] - SLAVE[0])),
    )
    shapes = np.column_stack((
        1.0 - np.sum(coordinates, axis=1), coordinates,
    ))
    traction = np.where(inside[:, None], points - nearest_foot, 0.0)
    area = 0.5 * abs(float(np.linalg.det(np.stack((
        SLAVE[1] - SLAVE[0], SLAVE[2] - SLAVE[0],
    )))))
    return area * np.mean(shapes[:, :, None] * traction[:, None, :], axis=0)


def bounded_potential_reference(
    *, vertices=SLAVE, maximum_depth: int = 8, maximum_cells: int = 200_000,
) -> dict[str, object]:
    """Bound the L-tool penalty potential with explicit uncertain subcells.

    Signed distance to the closed polygon boundary is 1-Lipschitz. At a
    subtriangle center with enclosing radius ``r``, its entire gap lies in
    ``[g-r, g+r]``. A fully interior cell whose nearest *segment* also beats
    every other segment by more than ``2r`` has one quadratic distance branch;
    a positive three-point triangle rule then integrates its potential
    exactly in real arithmetic. Unresolved cells retain lower/upper energy
    bounds rather than pretending a finer sampled quadrature is exact.

    The same cells bound the *reference-area P1 nodal traction integral*:
    a fixed segment has affine potential gradient ``x-foot``, while each
    unresolved component has magnitude at most the maximum penetration.
    Its nodal shape functions are carried through subdivision, rather than
    inferred from a point sample. This is a force oracle at the stated
    reference triangle, not a moving-surface Jacobian, Newton tangent or
    global Step.
    """

    selected = np.asarray(vertices, dtype=float)
    if selected.shape != (3, 2) or not np.all(np.isfinite(selected)):
        raise ValueError("Reference requires a finite planar triangle.")
    edge_scale = float(np.max(np.linalg.norm(selected[1:] - selected[0], axis=1)))
    double_area = abs(float(np.linalg.det(np.stack((
        selected[1] - selected[0], selected[2] - selected[0],
    )))))
    if (not np.isfinite(double_area) or double_area <=
            64.0 * np.finfo(float).eps * edge_scale**2):
        raise ValueError("Reference requires a nondegenerate planar triangle.")
    if (type(maximum_depth) is not int or maximum_depth < 0
            or type(maximum_cells) is not int or maximum_cells < 1):
        raise ValueError("Reference depth and cell budget must be nonnegative integers.")
    edges = np.roll(POLYGON, -1, axis=0) - POLYGON
    lengths_squared = np.einsum("ij,ij->i", edges, edges)
    if np.any(lengths_squared <= 0.0):
        raise ValueError("Reference polygon contains a degenerate segment.")
    stack = [(selected, np.eye(3), 0)]
    lower = upper = 0.0
    force_lower = np.zeros(2)
    force_upper = np.zeros(2)
    nodal_lower = np.zeros((3, 2))
    nodal_upper = np.zeros((3, 2))
    exact_active = exact_inactive = uncertain = processed = 0
    rule = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                       (1 / 6, 2 / 3, 1 / 6),
                       (1 / 6, 1 / 6, 2 / 3)))
    margin = 128.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(selected)))

    while stack:
        triangle, parent_shapes, depth = stack.pop()
        processed += 1
        center = np.mean(triangle, axis=0)
        radius = float(np.max(np.linalg.norm(triangle - center, axis=1)))
        area = 0.5 * abs(float(np.linalg.det(np.stack((
            triangle[1] - triangle[0], triangle[2] - triangle[0],
        )))))
        gap = float(signed_gap(center[None, :])[0])
        if gap - radius > margin:
            exact_inactive += 1
            continue
        if gap + radius < -margin:
            parameters = np.einsum(
                "ij,ij->i", center - POLYGON, edges,
            ) / lengths_squared
            feet = POLYGON + np.clip(parameters, 0.0, 1.0)[:, None] * edges
            distances = np.linalg.norm(center - feet, axis=1)
            order = np.argsort(distances, kind="stable")
            first, second = int(order[0]), int(order[1])
            if distances[second] - distances[first] > 2.0 * radius + margin:
                projections = (
                    (triangle - POLYGON[first]) @ edges[first]
                    / lengths_squared[first]
                )
                # The closest point along one segment is affine only while
                # all three vertices stay in the same face/end-point branch.
                if np.all(projections > margin) and np.all(projections < 1 - margin):
                    branch = "interior"
                elif np.all(projections < -margin):
                    branch = "start"
                elif np.all(projections > 1 + margin):
                    branch = "end"
                else:
                    branch = None
                if branch is not None:
                    points = rule @ triangle
                    if branch == "interior":
                        t = ((points - POLYGON[first]) @ edges[first]
                             / lengths_squared[first])
                        foot = POLYGON[first] + t[:, None] * edges[first]
                    else:
                        foot = POLYGON[
                            (first + 1) % len(POLYGON) if branch == "end" else first
                        ]
                    energy = area * float(np.mean(
                        0.5 * np.sum((points - foot) ** 2, axis=1),
                    ))
                    energy_slack = margin * max(1.0, distances[first]) * area
                    lower += max(0.0, energy - energy_slack)
                    upper += energy + energy_slack
                    force = area * np.mean(points - foot, axis=0)
                    force_lower += force - margin * area
                    force_upper += force + margin * area
                    rule_shapes = rule @ parent_shapes
                    nodal_force = area * np.mean(
                        rule_shapes[:, :, None] * (points - foot)[:, None, :],
                        axis=0,
                    )
                    nodal_slack = margin * area * np.mean(rule_shapes, axis=0)
                    nodal_lower += nodal_force - nodal_slack[:, None]
                    nodal_upper += nodal_force + nodal_slack[:, None]
                    exact_active += 1
                    continue
        if depth < maximum_depth and processed + len(stack) + 4 <= maximum_cells:
            a, b, c = triangle
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            na, nb, nc = parent_shapes
            nab, nbc, nca = (na + nb) / 2, (nb + nc) / 2, (nc + na) / 2
            stack.extend((np.asarray(child), np.asarray(shapes), depth + 1)
                         for child, shapes in (
                             ((a, ab, ca), (na, nab, nca)),
                             ((ab, b, bc), (nab, nb, nbc)),
                             ((ca, bc, c), (nca, nbc, nc)),
                             ((ab, bc, ca), (nab, nbc, nca)),
                         ))
            continue
        lower += area * 0.5 * max(-(gap + radius), 0.0) ** 2
        upper += area * 0.5 * max(-(gap - radius), 0.0) ** 2
        maximum_force = area * max(-(gap - radius), 0.0)
        force_lower -= maximum_force
        force_upper += maximum_force
        nodal_maximum = maximum_force * np.mean(parent_shapes, axis=0)
        nodal_lower -= nodal_maximum[:, None]
        nodal_upper += nodal_maximum[:, None]
        uncertain += 1
    return {
        "scope": "research_only_L_boundary_potential_per_unit_extrusion",
        "lower": lower, "upper": upper,
        "absolute_interval_width": upper - lower,
        "translation_force_lower": force_lower.tolist(),
        "translation_force_upper": force_upper.tolist(),
        "reference_p1_nodal_force_lower": nodal_lower.tolist(),
        "reference_p1_nodal_force_upper": nodal_upper.tolist(),
        "processed_subtriangles": processed,
        "exact_active_subtriangles": exact_active,
        "exact_inactive_subtriangles": exact_inactive,
        "uncertain_subtriangles": uncertain,
        "maximum_depth": maximum_depth,
        "maximum_cells": maximum_cells,
        "arithmetic": "floating_point_conservative_margin_not_formally_verified_interval",
        "nodal_force_interval_only": True,
        "newton_tangent_certified": False,
    }


def main() -> None:
    levels = (4, 8, 16, 32, 64, 128, 256, 512)
    values = {str(level): penalty_potential(subdivisions=level) for level in levels}
    sampled = penalty_potential(subdivisions=None)
    print(json.dumps({
        "scope": "research_only_reentrant_L_segment_distance_per_unit_extrusion",
        "unit_penalty": True,
        "three_point_face_rule": sampled,
        "positive_subtriangle_levels": values,
        "feature_aware_potential_interval": bounded_potential_reference(
            maximum_depth=10,
        ),
        "relative_three_point_difference_from_level_512": abs(
            sampled - values["512"]
        ) / values["512"],
        "reference_status": "numerically_refined_not_exact_or_certified",
    }, indent=2))


if __name__ == "__main__":
    main()
