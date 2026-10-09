# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded coplanar P1 common-refinement quadrature, not a mortar multiplier.

Intersections define integration cells only. Both original trace bases are
retained. Geometry is fixed; curved surfaces, Q1 traces and MPI are excluded.
"""

import numpy as np

from ._interface_pairing import _TRIANGLE_RULE, _make_pairing
from .boundary_models.rigid import TriangulatedRigidSurface
from .boundary_models.search import TriangleSurfaceBVH


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _intersection(subject, clip):
    """Convex triangle clipping in normalized planar coordinates."""
    polygon = list(subject)
    orientation = np.sign(_cross(clip[1] - clip[0], clip[2] - clip[0]))
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not polygon:
            break
        output = []
        previous = polygon[-1]
        old = orientation * _cross(b - a, previous - a)
        for current in polygon:
            new = orientation * _cross(b - a, current - a)
            if (new >= 0) != (old >= 0):
                output.append(previous + old / (old - new) * (current - previous))
            if new >= 0:
                output.append(current)
            previous, old = current, new
        polygon = output
    return np.asarray(polygon)


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    return (
        abs(
            sum(
                _cross(polygon[i] - polygon[0], polygon[i + 1] - polygon[0])
                for i in range(1, len(polygon) - 1)
            )
        )
        / 2
    )


def _barycentric(points, triangle):
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    tail = np.linalg.solve(matrix, (points - triangle[0]).T).T
    return np.column_stack((1 - tail.sum(axis=1), tail))


def planar_overlap_pairing(
    negative, positive, *, tolerance, maximum_points=200_000, quadrature_degree=2
):
    """Integrate coincident non-overlapping planar triangle partitions.

    Reject uncovered area and within-side overlapping elements. Coverage is
    checked on every facet, with relative area tolerance 1e-9. The supplied
    length tolerance checks planarity/coincidence, not arbitrary initial gaps.
    Three-point integration is exact for products of P1 traces on each overlap;
    nonlinear constitutive responses still need integration convergence.
    """
    if not all(isinstance(s, TriangulatedRigidSurface) for s in (negative, positive)):
        raise TypeError("Overlap pairing requires reviewed triangular surfaces.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("coincidence tolerance must be finite and positive.")
    if (
        isinstance(maximum_points, bool)
        or not isinstance(maximum_points, (int, np.integer))
        or maximum_points < 1
    ):
        raise ValueError("maximum_points must be a positive integer.")
    if (
        isinstance(quadrature_degree, bool)
        or not isinstance(quadrature_degree, (int, np.integer))
        or not 2 <= quadrature_degree <= 8
    ):
        raise ValueError("quadrature_degree must be an integer between 2 and 8.")
    if quadrature_degree == 2:
        rule, normalized_weights = _TRIANGLE_RULE, np.full(3, 1 / 3)
    else:
        import basix

        points, weights = basix.make_quadrature(
            basix.CellType.triangle, quadrature_degree
        )
        rule = np.column_stack((1 - points.sum(axis=1), points))
        normalized_weights = 2 * weights
    count = len(rule)
    origin = negative.vertices[negative.triangles[0, 0]]
    normal = negative.facet_normals[0]
    edge = negative.vertices[negative.triangles[0, 1]] - origin
    first = edge / np.linalg.norm(edge)
    basis = np.column_stack((first, np.cross(normal, first)))
    scale = max(
        np.linalg.norm(np.ptp(s.vertices, axis=0)) for s in (negative, positive)
    )
    planar = []
    for side, sign in ((negative, 1), (positive, -1)):
        if np.max(
            np.abs((side.vertices - origin) @ normal)
        ) > tolerance or not np.allclose(
            side.facet_normals, sign * normal, atol=1e-10, rtol=0
        ):
            raise ValueError("Overlap pairing requires opposing coplanar surfaces.")
        planar.append(((side.vertices - origin) @ basis / scale)[side.triangles])
    trees = [TriangleSurfaceBVH(s) for s in (negative, positive)]
    # No all-pairs matrix: reuse the geometry search owner's bounded tree.
    for surface, triangles, tree in zip((negative, positive), planar, trees):
        for i, xyz in enumerate(surface.vertices[surface.triangles]):
            for j in tree.overlapping_facets(
                xyz.min(axis=0) - tolerance, xyz.max(axis=0) + tolerance
            ):
                if j <= i:
                    continue
                if _area(_intersection(triangles[i], triangles[j])) > 1e-12 * min(
                    _area(triangles[i]), _area(triangles[j])
                ):
                    raise ValueError("Within-side triangle overlap is ambiguous.")
    covered = [np.zeros(len(s.triangles)) for s in (negative, positive)]
    chunks = {
        key: []
        for key in (
            "negative_nodes",
            "positive_nodes",
            "negative_weights",
            "positive_weights",
            "weights",
            "normals",
            "reference_mismatch",
        )
    }
    point_count = 0
    for i, xyz in enumerate(negative.vertices[negative.triangles]):
        for j in trees[1].overlapping_facets(
            xyz.min(axis=0) - tolerance, xyz.max(axis=0) + tolerance
        ):
            polygon = _intersection(planar[0][i], planar[1][j])
            for k in range(1, len(polygon) - 1):
                triangle = polygon[[0, k, k + 1]]
                area = _area(triangle)
                if area <= 1e-15 * min(_area(planar[0][i]), _area(planar[1][j])):
                    continue
                point_count += count
                if point_count > maximum_points:
                    raise ValueError("Interface overlap exceeds maximum_points.")
                points = rule @ triangle
                wn = _barycentric(points, planar[0][i])
                wp = _barycentric(points, planar[1][j])
                nn, pn = negative.triangles[i], positive.triangles[j]
                values = {
                    "negative_nodes": np.tile(nn, (count, 1)),
                    "positive_nodes": np.tile(pn, (count, 1)),
                    "negative_weights": wn,
                    "positive_weights": wp,
                    "weights": area * scale**2 * normalized_weights,
                    "normals": np.tile(normal, (count, 1)),
                    "reference_mismatch": wp @ positive.vertices[pn]
                    - wn @ negative.vertices[nn],
                }
                for key, value in values.items():
                    chunks[key].append(value)
                covered[0][i] += area
                covered[1][j] += area
    for triangles, actual in zip(planar, covered):
        expected = np.asarray([_area(t) for t in triangles])
        if np.any(np.abs(actual - expected) > 1e-9 * expected):
            raise ValueError(
                "Incomplete interface overlap: facet area coverage failed."
            )
    values = {key: np.concatenate(value) for key, value in chunks.items()}
    return _make_pairing(
        negative, positive, tolerance, values, "coplanar-triangle-common-refinement"
    )
