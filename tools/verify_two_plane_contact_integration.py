# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Research-only exact two-affine-plane contact potential and full tangent.

This unit-reference-triangle oracle partitions both the zero-gap boundary and
the active feature switch.  It is intentionally not a DOLFINx Step: a reviewed
closed tool, arbitrary feature graph, State, and deterministic accepted output
need separate integration before promotion.
"""

from __future__ import annotations

import numpy as np


_TRIANGLE = tuple(np.eye(3))
_DEGREE_TWO = ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))
_LINE_GAUSS = (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))


def _clip(vertices, nodal_values):
    """Keep a convex barycentric polygon where an affine value is negative."""

    if not vertices:
        return []
    result = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        a, b = float(start @ nodal_values), float(end @ nodal_values)
        if (a < 0.0) != (b < 0.0):
            result.append((b * start - a * end) / (b - a))
        if b < 0.0:
            result.append(end)
    return result


def _polygon_rule(vertices):
    points, weights = [], []
    if len(vertices) < 3:
        return points, weights
    origin = vertices[0]
    for index in range(1, len(vertices) - 1):
        first, second = vertices[index:index + 2]
        jacobian = abs(np.linalg.det(np.column_stack((
            first[1:] - origin[1:], second[1:] - origin[1:],
        ))))
        if jacobian <= 0.0:
            continue
        for r, s in _DEGREE_TWO:
            points.append((1 - r - s) * origin + r * first + s * second)
            weights.append(jacobian / 6)
    return points, weights


def _feature_line(gaps):
    """Return the part of g0=g1 where their shared gap remains negative."""

    difference = gaps[0] - gaps[1]
    if ((np.min(difference) == 0.0 and np.max(difference) > 0.0)
            or (np.max(difference) == 0.0 and np.min(difference) < 0.0)):
        raise ValueError(
            "Two-plane feature switch touches a whole triangle boundary; "
            "the two-sided tangent is not defined there."
        )
    candidates = []
    for start, end in zip(_TRIANGLE, _TRIANGLE[1:] + _TRIANGLE[:1], strict=True):
        a, b = float(start @ difference), float(end @ difference)
        if a == 0.0:
            candidates.append(start)
        if a * b < 0.0:
            candidates.append((b * start - a * end) / (b - a))
    unique = []
    for point in candidates:
        if not any(np.linalg.norm(point - other) < 1e-13 for other in unique):
            unique.append(point)
    if len(unique) != 2:
        if not unique:
            return None
        raise ValueError("Two-plane oracle has a degenerate feature intersection.")
    start, end = unique
    a, b = float(start @ gaps[0]), float(end @ gaps[0])
    if a >= 0.0 and b >= 0.0:
        return None
    if (a < 0.0) != (b < 0.0):
        crossing = (b * start - a * end) / (b - a)
        if a >= 0.0:
            start = crossing
        else:
            end = crossing
    return start, end


def two_plane_triangle_contact(nodal_gaps, normals, *, penalty: float = 1.0):
    """Integrate k*min(max(g0,g1),0)^2/2 and exact nodal derivatives.

    Nodal gaps have shape (2,3), normals (2,3); weights refer to the unit
    triangle of area 1/2.  The tangent includes the distributional feature-
    switch line contribution, not merely the two bulk branch Hessians.
    """

    gaps = np.asarray(nodal_gaps, dtype=float)
    vectors = np.asarray(normals, dtype=float)
    k = float(penalty)
    if (gaps.shape != (2, 3) or vectors.shape != (2, 3)
            or not np.all(np.isfinite(gaps)) or not np.all(np.isfinite(vectors))
            or not np.isfinite(k) or k <= 0):
        raise ValueError("Two-plane oracle requires finite (2,3) gaps/normals and k>0.")
    if np.linalg.norm(vectors[0] - vectors[1]) < 1e-12:
        raise ValueError("Two-plane oracle needs distinct feature normals.")
    if np.max(np.abs(gaps[0] - gaps[1])) < 1e-13:
        raise ValueError("Two-plane oracle needs a unique feature partition.")
    potential = 0.0
    residual = np.zeros((3, 3))
    tangent = np.zeros((3, 3, 3, 3))
    for feature in range(2):
        other = 1 - feature
        region = _clip(list(_TRIANGLE), gaps[other] - gaps[feature])
        region = _clip(region, gaps[feature])
        points, weights = _polygon_rule(region)
        for shape, weight in zip(points, weights, strict=True):
            gap = float(shape @ gaps[feature])
            normal = vectors[feature]
            potential += 0.5 * k * weight * gap**2
            residual += k * weight * gap * shape[:, None] * normal[None, :]
            tangent += k * weight * np.einsum(
                "a,b,i,j->aibj", shape, shape, normal, normal,
            )
    feature_tangent = np.zeros_like(tangent)
    line = _feature_line(gaps)
    if line is not None:
        start, end = line
        grad = np.asarray((
            gaps[0, 1] - gaps[1, 1] - gaps[0, 0] + gaps[1, 0],
            gaps[0, 2] - gaps[1, 2] - gaps[0, 0] + gaps[1, 0],
        ))
        magnitude = float(np.linalg.norm(grad))
        if magnitude <= 1e-14:
            raise ValueError("Two-plane feature switch has zero surface gradient.")
        factor = float(np.linalg.norm((end - start)[1:])) / (2 * magnitude)
        jump = vectors[0] - vectors[1]
        for coordinate in _LINE_GAUSS:
            shape = (1 - coordinate) * start + coordinate * end
            gap = float(shape @ gaps[0])
            feature_tangent += k * factor * gap * np.einsum(
                "a,b,i,j->aibj", shape, shape, jump, jump,
            )
    tangent += feature_tangent
    return {
        "potential": potential,
        "nodal_residual": residual,
        "nodal_tangent": tangent.reshape(9, 9),
        "feature_tangent": feature_tangent.reshape(9, 9),
    }


__all__ = ["two_plane_triangle_contact"]
