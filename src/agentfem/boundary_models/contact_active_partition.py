# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Positive reference-area quadrature on an affine P1 contact half-face.

This is a geometric integration rule, not a contact law or a new solver.  It
clips the slave reference triangle at an affine zero-gap contour.  A degree-two
rule on each retained subtriangle integrates the planar penalty potential,
residual and fixed-branch tangent exactly.  The moving clip boundary contributes
no first/second potential derivative because the gap vanishes there.
"""

from __future__ import annotations

import numpy as np

from .contact_trace import ContactTrace


_POINTS = np.asarray(((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)))


def affine_active_triangle_quadrature(nodal_gaps) -> tuple[np.ndarray, np.ndarray]:
    """Return barycentric shapes and positive weights on the active region.

    Weights refer to the unit reference triangle (area 1/2).  An entirely
    open face returns empty arrays; a caller may keep its original open-face
    points if its projection backend requires a nonempty global trace.
    """

    gaps = np.asarray(nodal_gaps, dtype=float)
    if gaps.shape != (3,) or not np.all(np.isfinite(gaps)):
        raise ValueError("Active contact partition requires three finite P1 gaps.")
    if np.all(gaps >= 0.0):
        return np.empty((0, 3)), np.empty((0,))
    vertices = [np.eye(3)[index] for index in range(3)]
    clipped: list[np.ndarray] = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        g_start = float(start @ gaps)
        g_end = float(end @ gaps)
        if (g_start < 0.0) != (g_end < 0.0):
            clipped.append((g_end * start - g_start * end) / (g_end - g_start))
        if g_end < 0.0:
            clipped.append(end)
    if len(clipped) < 3:
        return np.empty((0, 3)), np.empty((0,))
    points = []
    weights = []
    origin = clipped[0]
    for index in range(1, len(clipped) - 1):
        first, second = clipped[index:index + 2]
        jacobian = abs(np.linalg.det(np.column_stack((
            first[1:] - origin[1:], second[1:] - origin[1:],
        ))))
        if jacobian <= 0.0:
            continue
        for r, s in _POINTS:
            points.append(origin * (1.0 - r - s) + first * r + second * s)
            weights.append(jacobian / 6.0)
    if not points:
        return np.empty((0, 3)), np.empty((0,))
    result_points = np.asarray(points, dtype=float)
    result_weights = np.asarray(weights, dtype=float)
    if (not np.all(np.isfinite(result_points))
            or not np.all(np.isfinite(result_weights))
            or np.any(result_weights <= 0.0)
            or np.any(result_points < -1e-14)
            or not np.allclose(result_points.sum(axis=1), 1.0, atol=1e-14)):
        raise ValueError("Active contact partition produced invalid quadrature.")
    return result_points, result_weights


def partition_planar_p1_trace(
    trace: ContactTrace, nodal_positions, *, plane_point, plane_normal,
) -> ContactTrace:
    """Replace only crossing-face points by positive clipped quadrature.

    The input is the reviewed three-point P1 triangular trace.  Full and open
    faces retain their ordinary three points; this also keeps an all-open trace
    admissible to projection backends.  Crossing faces use the current planar
    signed gap but integrate on the *reference* face area.  New point IDs are
    deterministic within one trial; they are not historical State identities.
    """

    if not isinstance(trace, ContactTrace) or trace.measure_configuration != "reference":
        raise TypeError("Planar partition requires a reference ContactTrace.")
    ids = np.asarray(trace.point_ids, dtype=np.int64)
    nodes = np.asarray(trace.node_ids, dtype=np.int64)
    shapes = np.asarray(trace.shape_values, dtype=float)
    weights = np.asarray(trace.weights, dtype=float)
    if (ids.size % 3 != 0 or nodes.shape[1] != 3
            or not np.all(ids.reshape(-1, 3) % 3 == np.arange(3)[None, :])
            or not np.all(nodes.reshape(-1, 3, 3) == nodes.reshape(-1, 3, 3)[:, :1, :])
            or not np.allclose(shapes.reshape(-1, 3, 3).sum(axis=1), 1.0,
                               rtol=1e-12, atol=1e-14)
            or not np.allclose(weights.reshape(-1, 3).sum(axis=1),
                               3.0 * weights.reshape(-1, 3)[:, 0],
                               rtol=1e-12, atol=0.0)):
        raise NotImplementedError(
            "Planar partition requires the default three-point P1 triangle rule."
        )
    positions = np.asarray(nodal_positions, dtype=float)
    point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
            or point.shape != (3,) or normal.shape != (3,)
            or not np.all(np.isfinite(point)) or not np.all(np.isfinite(normal))
            or ids.size and int(nodes.max()) >= len(positions)):
        raise ValueError("Planar partition has invalid positions or plane geometry.")
    if ids.size and int(ids.max()) // 3 > np.iinfo(np.int64).max // 16:
        raise OverflowError("Planar partition point identity exceeds int64.")
    rows = nodes.reshape(-1, 3, 3)[:, 0, :]
    gaps = np.einsum("fij,j->fi", positions[rows] - point, normal)
    if not np.any((np.min(gaps, axis=1) < 0.0) & (np.max(gaps, axis=1) > 0.0)):
        return trace
    new_ids = []
    new_nodes = []
    new_shapes = []
    new_weights = []
    for facet_index, (facet_nodes, facet_gaps) in enumerate(zip(rows, gaps, strict=True)):
        start = 3 * facet_index
        facet_id = int(ids[start] // 3)
        crossing = float(np.min(facet_gaps)) < 0.0 < float(np.max(facet_gaps))
        if crossing:
            basis, reference_weights = affine_active_triangle_quadrature(facet_gaps)
            local_weights = 2.0 * float(np.sum(weights[start:start + 3])) * reference_weights
        else:
            basis = shapes[start:start + 3]
            local_weights = weights[start:start + 3]
        for local_index, (value, weight) in enumerate(zip(basis, local_weights, strict=True)):
            new_ids.append(facet_id * 16 + local_index)
            new_nodes.append(facet_nodes)
            new_shapes.append(value)
            new_weights.append(weight)
    return ContactTrace(
        point_ids=np.asarray(new_ids, dtype=np.int64),
        node_ids=np.asarray(new_nodes, dtype=np.int64).reshape(-1, 3),
        shape_values=np.asarray(new_shapes, dtype=float).reshape(-1, 3),
        weights=np.asarray(new_weights, dtype=float),
        measure_configuration="reference",
        source=f"{trace.source}_trial_planar_active_partition",
    )


__all__ = ["affine_active_triangle_quadrature", "partition_planar_p1_trace"]
