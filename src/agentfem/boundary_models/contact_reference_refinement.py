# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Positive fixed-reference triangle subdivision for contact diagnostics.

This is a backend-neutral integration experiment, not a changing-trial
feature partition. It preserves the P1 slave interpolation and reference
area, but does not supply the moving feature-interface term in the tangent.
"""

from __future__ import annotations

import numpy as np

from .contact_trace import ContactTrace


_VERTICES = np.eye(3, dtype=float)
_POINTS = np.asarray((
    (2.0 / 3, 1.0 / 6, 1.0 / 6),
    (1.0 / 6, 2.0 / 3, 1.0 / 6),
    (1.0 / 6, 1.0 / 6, 2.0 / 3),
))


def _subtriangles(level: int) -> np.ndarray:
    triangles = _VERTICES[None, :, :]
    for _ in range(level):
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        triangles = np.concatenate((
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ))
    return triangles


def refine_p1_triangle_reference_trace(
    trace: ContactTrace, level: int, *, maximum_points: int = 50_000,
) -> ContactTrace:
    """Subdivide every affine reference face into ``4**level`` P1 triangles.

    Only the default positive three-point face rule is accepted, avoiding a
    false interpretation of an arbitrary three-point trace as one P1 face.
    The returned point IDs are local diagnostic IDs, not restart identities.
    """

    if not isinstance(trace, ContactTrace) or trace.measure_configuration != "reference":
        raise TypeError("Reference refinement requires a reference ContactTrace.")
    if (isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer))
            or not 0 <= int(level) <= 5):
        raise ValueError("Reference refinement level must be an integer from 0 to 5.")
    if (isinstance(maximum_points, bool) or not isinstance(maximum_points, int)
            or maximum_points < 1):
        raise ValueError("Reference refinement point budget must be positive.")
    if trace.point_count % 3 or trace.nodes_per_point != 3:
        raise NotImplementedError(
            "AFM-CONTACT-REFERENCE-REFINEMENT: requires default P1 triangular "
            "three-point reference integration."
        )
    nodes = trace.node_ids.reshape(-1, 3, 3)
    shapes = trace.shape_values.reshape(-1, 3, 3)
    weights = trace.weights.reshape(-1, 3)
    point_order = np.argsort(np.argmax(shapes, axis=2), axis=1)
    canonical_shapes = np.take_along_axis(shapes, point_order[:, :, None], axis=1)
    if (not np.all(nodes == nodes[:, :1, :])
            or not np.allclose(
                canonical_shapes, _POINTS[None, :, :],
                atol=1e-13, rtol=0.0,
            )
            or not np.allclose(weights, weights[:, :1], rtol=1e-13, atol=1e-14)):
        raise ValueError("Default P1 triangular trace grouping is inconsistent.")
    count = trace.point_count * 4 ** int(level)
    if count > maximum_points:
        raise ValueError(
            f"Contact reference refinement would create {count} points, "
            f"exceeding the {maximum_points}-point budget."
        )
    if level == 0:
        return trace
    barycentric = np.einsum("qb,sbc->sqc", _POINTS, _subtriangles(int(level)))
    per_face = barycentric.shape[0] * barycentric.shape[1]
    face_nodes = nodes[:, 0, :]
    return ContactTrace(
        point_ids=np.arange(count, dtype=np.int64),
        node_ids=np.repeat(face_nodes, per_face, axis=0),
        shape_values=np.tile(barycentric.reshape(-1, 3), (len(face_nodes), 1)),
        weights=np.repeat(np.sum(weights, axis=1) / per_face, per_face),
        measure_configuration="reference",
        source=f"{trace.source}_uniform_subdivision_level_{int(level)}",
    )


__all__ = ["refine_p1_triangle_reference_trace"]
