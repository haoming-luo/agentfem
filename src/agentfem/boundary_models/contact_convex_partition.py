# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded reference-face integration across a convex tool's sharp feature.

For an interior point of a certified convex tool, its signed gap is the
maximum of outward support-plane gaps. On an affine P1 slave triangle these
gaps are affine. The positive-area subcells are therefore polygonal, while a
compressed switch between two dominant planes contributes a *line* term to
the derivative of the integrated penalty force. This module handles at most
two positive-area dominant features per slave triangle. It deliberately does
not describe a nonconvex nearest-edge/vertex distance, friction State, or a
current-area follower law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_trace import ContactTrace
from .convex_polyhedron import ConvexPolyhedralRigidSurface
from .rigid import PrescribedRigidMotion


_TRIANGLE = tuple(np.eye(3))
_GAUSS_TRIANGLE = ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))
_GAUSS_LINE = (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))


def _clip_polygon(vertices, nodal_values):
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
        for r, s in _GAUSS_TRIANGLE:
            points.append((1 - r - s) * origin + r * first + s * second)
            weights.append(jacobian / 6)
    return points, weights


def _unique_support_planes(surface):
    normals = np.asarray(surface.facets.facet_normals, dtype=float)
    offsets = np.asarray(surface._offsets, dtype=float)
    selected = []
    # The projection tolerance is a *query* tolerance, not permission to
    # merge two distinct support planes. A deliberately loose CAD tolerance
    # must not erase a sharp feature from the integration law.
    tolerance = 128 * np.finfo(float).eps
    coordinate_scale = max(1.0, float(np.max(np.abs(surface.facets.vertices))))
    for index, (normal, offset) in enumerate(zip(normals, offsets, strict=True)):
        if not selected:
            selected.append(index)
            continue
        candidates = np.asarray(selected, dtype=np.int64)
        matches = (
            np.linalg.norm(normals[candidates] - normal, axis=1) <= tolerance
        ) & (np.abs(offsets[candidates] - offset) <= tolerance * coordinate_scale)
        if not np.any(matches):
            selected.append(index)
    unique_normals = normals[selected].copy()
    unique_offsets = offsets[selected].copy()
    unique_normals.setflags(write=False)
    unique_offsets.setflags(write=False)
    return unique_normals, unique_offsets


def _dominant_feature_line(gaps, first, second):
    """Clip a two-feature switching segment to the compressed dominant part."""

    difference = gaps[first] - gaps[second]
    lower, upper = float(np.min(difference)), float(np.max(difference))
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    if (-tolerance <= lower <= 0.0 and upper > tolerance) or (
        tolerance >= upper >= 0.0 and lower < -tolerance
    ):
        raise ValueError(
            "AFM-CONTACT-FEATURE-PARTITION: feature switch lies on a slave "
            "facet boundary, so its two-sided tangent is undefined."
        )
    if lower >= 0.0 or upper <= 0.0:
        return None
    intersections = []
    for start, end in zip(_TRIANGLE, _TRIANGLE[1:] + _TRIANGLE[:1], strict=True):
        a, b = float(start @ difference), float(end @ difference)
        if a == 0.0:
            intersections.append(start)
        if a * b < 0.0:
            intersections.append((b * start - a * end) / (b - a))
    unique = []
    for point in intersections:
        if not any(np.linalg.norm(point - other) < 1e-13 for other in unique):
            unique.append(point)
    if len(unique) != 2:
        raise ValueError("AFM-CONTACT-FEATURE-PARTITION: degenerate feature line.")
    start, end = unique
    # Intersect scalar inequalities h<=0 along one line parameter. Here
    # h=g_first for compression and h=g_other-g_first for dominance.
    t_low, t_high = 0.0, 1.0
    inequalities = [gaps[first]] + [
        gaps[index] - gaps[first]
        for index in range(len(gaps)) if index not in (first, second)
    ]
    for condition_index, values in enumerate(inequalities):
        a, b = float(start @ values), float(end @ values)
        if (condition_index > 0 and abs(a) <= tolerance
                and abs(b) <= tolerance):
            raise ValueError(
                "AFM-CONTACT-FEATURE-PARTITION: a third support plane ties "
                "along the compressed feature line."
            )
        if a <= 0.0 and b <= 0.0:
            continue
        if a >= 0.0 and b >= 0.0:
            return None
        crossing = a / (a - b)
        if a > 0.0:
            t_low = max(t_low, crossing)
        else:
            t_high = min(t_high, crossing)
    if t_high <= t_low:
        return None
    return ((1 - t_low) * start + t_low * end,
            (1 - t_high) * start + t_high * end)


@dataclass(frozen=True, eq=False)
class ConvexFeaturePartition:
    trace: ContactTrace
    selected_gaps: np.ndarray
    selected_normals: np.ndarray
    tangent_rows: np.ndarray
    tangent_columns: np.ndarray
    tangent_values: np.ndarray
    split_faces: int
    active_feature_lines: int


def partition_convex_p1_trace(
    trace: ContactTrace, nodal_positions, surface: ConvexPolyhedralRigidSurface,
    *, penalty: float, motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0, _support_planes=None,
) -> ConvexFeaturePartition:
    """Build a positive trial trace and the compressed feature-line Hessian."""

    if not isinstance(trace, ContactTrace) or trace.measure_configuration != "reference":
        raise TypeError("Convex feature partition requires a reference ContactTrace.")
    if not isinstance(surface, ConvexPolyhedralRigidSurface):
        raise TypeError("Convex feature partition requires certified convex geometry.")
    k = float(penalty)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("Convex feature partition requires a scalar positive penalty.")
    nodes = np.asarray(trace.node_ids, dtype=np.int64)
    shapes = np.asarray(trace.shape_values, dtype=float)
    weights = np.asarray(trace.weights, dtype=float)
    positions = np.asarray(nodal_positions, dtype=float)
    if (trace.point_count % 3 != 0 or nodes.shape[1] != 3
            or not np.all(nodes.reshape(-1, 3, 3) == nodes.reshape(-1, 3, 3)[:, :1, :])
            or not np.allclose(shapes.reshape(-1, 3, 3).sum(axis=1), 1.0)
            or not np.allclose(weights.reshape(-1, 3).sum(axis=1),
                               3 * weights.reshape(-1, 3)[:, 0])
            or positions.ndim != 2 or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
            or (nodes.size and (np.min(nodes) < 0 or np.max(nodes) >= len(positions)))):
        raise NotImplementedError(
            "AFM-CONTACT-FEATURE-PARTITION: requires the default three-point "
            "P1 reference triangle and finite three-dimensional positions."
        )
    normals, offsets = (
        _unique_support_planes(surface)
        if _support_planes is None else _support_planes
    )
    if motion is None:
        reference_positions = positions
        rotation = np.eye(3)
    else:
        pose = motion.state(factor)
        rotation = np.asarray(pose["rotation_matrix"], dtype=float)
        reference_positions = (
            np.asarray(motion.reference_point)
            + (positions - np.asarray(pose["reference_point"])) @ rotation
        )
    world_normals = normals @ rotation.T
    new_nodes, new_shapes, new_weights = [], [], []
    selected_gaps, selected_normals = [], []
    line_rows, line_columns, line_values = [], [], []
    split_faces = 0
    active_lines = 0
    for face_index, face_nodes in enumerate(nodes.reshape(-1, 3, 3)[:, 0, :]):
        first_point = 3 * face_index
        area_scale = 2.0 * float(np.sum(weights[first_point:first_point + 3]))
        gaps = reference_positions[face_nodes] @ normals.T - offsets
        gaps = gaps.T
        regions = []
        for feature in range(len(normals)):
            region = list(_TRIANGLE)
            for other in range(len(normals)):
                if other != feature:
                    region = _clip_polygon(region, gaps[other] - gaps[feature])
            region = _clip_polygon(region, gaps[feature])
            basis, reference_weights = _polygon_rule(region)
            if reference_weights and sum(reference_weights) > 0.0:
                regions.append((feature, basis, reference_weights))
        active = list(_TRIANGLE)
        for values in gaps:
            active = _clip_polygon(active, values)
        _, active_weights = _polygon_rule(active)
        total_area = sum(active_weights)
        if len(regions) > 2:
            raise NotImplementedError(
                "AFM-CONTACT-FEATURE-PARTITION: more than two dominant "
                "support planes cross one active slave triangle."
            )
        if not np.isclose(
            sum(sum(item[2]) for item in regions), total_area,
            rtol=1e-10, atol=1e-12,
        ):
            raise ValueError(
                "AFM-CONTACT-FEATURE-PARTITION: active reference-area "
                "coverage is incomplete."
            )
        if not regions:
            for shape, weight in zip(
                shapes[first_point:first_point + 3],
                weights[first_point:first_point + 3], strict=True,
            ):
                new_nodes.append(face_nodes)
                new_shapes.append(shape)
                new_weights.append(weight)
                selected_gaps.append(np.nan)
                selected_normals.append((np.nan, np.nan, np.nan))
            continue
        if len(regions) == 2:
            split_faces += 1
            first, second = regions[0][0], regions[1][0]
            segment = _dominant_feature_line(gaps, first, second)
            if segment is not None:
                start, end = segment
                gradient = np.asarray((
                    gaps[first, 1] - gaps[second, 1] - gaps[first, 0] + gaps[second, 0],
                    gaps[first, 2] - gaps[second, 2] - gaps[first, 0] + gaps[second, 0],
                ))
                magnitude = float(np.linalg.norm(gradient))
                if magnitude <= 1e-14:
                    raise ValueError("AFM-CONTACT-FEATURE-PARTITION: unresolved line gradient.")
                active_lines += 1
                length = float(np.linalg.norm((end - start)[1:]))
                difference = world_normals[first] - world_normals[second]
                block = np.zeros((3, 3, 3, 3))
                for coordinate in _GAUSS_LINE:
                    shape = (1 - coordinate) * start + coordinate * end
                    gap = float(shape @ gaps[first])
                    block += (
                        k * area_scale * length * gap / (2 * magnitude)
                        * np.einsum("a,b,i,j->aibj", shape, shape, difference, difference)
                    )
                dofs = (3 * face_nodes[:, None] + np.arange(3)[None, :]).reshape(-1)
                row, column = np.nonzero(block.reshape(9, 9))
                line_rows.extend(dofs[row])
                line_columns.extend(dofs[column])
                line_values.extend(block.reshape(9, 9)[row, column])
        for feature, basis, reference_weights in regions:
            for shape, weight in zip(basis, reference_weights, strict=True):
                new_nodes.append(face_nodes)
                new_shapes.append(shape)
                new_weights.append(area_scale * weight)
                selected_gaps.append(float(shape @ gaps[feature]))
                selected_normals.append(world_normals[feature])
    trial_trace = ContactTrace(
        point_ids=np.arange(len(new_nodes), dtype=np.int64),
        node_ids=np.asarray(new_nodes, dtype=np.int64).reshape(-1, 3),
        shape_values=np.asarray(new_shapes, dtype=float).reshape(-1, 3),
        weights=np.asarray(new_weights, dtype=float),
        measure_configuration="reference",
        source=f"{trace.source}_trial_convex_feature_partition",
    )
    return ConvexFeaturePartition(
        trace=trial_trace,
        selected_gaps=np.asarray(selected_gaps, dtype=float),
        selected_normals=np.asarray(selected_normals, dtype=float).reshape(-1, 3),
        tangent_rows=np.asarray(line_rows, dtype=np.int64),
        tangent_columns=np.asarray(line_columns, dtype=np.int64),
        tangent_values=np.asarray(line_values, dtype=float),
        split_faces=split_faces,
        active_feature_lines=active_lines,
    )


__all__ = ["ConvexFeaturePartition", "partition_convex_p1_trace"]
