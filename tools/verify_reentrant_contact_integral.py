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


def main() -> None:
    levels = (4, 8, 16, 32, 64, 128, 256, 512)
    values = {str(level): penalty_potential(subdivisions=level) for level in levels}
    sampled = penalty_potential(subdivisions=None)
    print(json.dumps({
        "scope": "research_only_reentrant_L_segment_distance_per_unit_extrusion",
        "unit_penalty": True,
        "three_point_face_rule": sampled,
        "positive_subtriangle_levels": values,
        "relative_three_point_difference_from_level_512": abs(
            sampled - values["512"]
        ) / values["512"],
        "reference_status": "numerically_refined_not_exact_or_certified",
    }, indent=2))


if __name__ == "__main__":
    main()
