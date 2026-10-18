# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference-area Newton Operator for a reviewed closed triangle solid."""

from __future__ import annotations

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceEvaluation
from .implicit_contact import (
    ImplicitRigidContactContribution, integrate_implicit_rigid_contact,
)
from .rigid import PrescribedRigidMotion


def sampled_mixed_feature_slave_faces(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    surface: ClosedTriangulatedRigidSolid,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
) -> int:
    """Count slave faces whose active samples visit distinct geometric features.

    This is a *sampled warning*, not a proof that all feature-switch lines
    were found. Coplanar triangulation seams and subdivided straight edges
    count as one geometric feature. A changing branch within an unsampled
    subcell may still require refined integration and a line tangent.
    """

    if (
        not isinstance(evaluation, ContactTraceEvaluation)
        or not isinstance(record, ContactProjectionRecord)
        or not isinstance(response, FrictionlessPenaltyContactResponse)
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
        or not np.array_equal(evaluation.trace.point_ids, record.point_ids)
        or response.projection is not record.projection
        or not record.projection.all_valid
    ):
        raise ValueError("Sampled feature diagnosis requires one valid closed-solid trial.")
    projection = record.projection
    if projection.geometry_fingerprint != surface.geometry_fingerprint:
        raise ValueError("Sampled feature diagnosis uses a different rigid geometry.")

    scale = max(1.0, float(surface.facets.scale))
    tolerance = max(float(surface.facets.tolerance), 128.0 * np.finfo(float).eps * scale)
    feet = projection.closest_points
    if motion is not None:
        pose = motion.state(factor)
        rotation = np.asarray(pose["rotation_matrix"], dtype=float)
        reference = np.asarray(motion.reference_point, dtype=float)
        moved_reference = np.asarray(pose["reference_point"], dtype=float)
        feet = reference + (feet - moved_reference) @ rotation
    faces: dict[tuple[int, ...], list[int]] = {}
    for index, (nodes, active) in enumerate(zip(
        evaluation.trace.node_ids, response.active, strict=True,
    )):
        if active:
            faces.setdefault(tuple(int(node) for node in nodes), []).append(index)

    def same_feature(first: int, second: int) -> bool:
        slots = (
            surface._facet_slots[int(projection.entity_ids[first])],
            surface._facet_slots[int(projection.entity_ids[second])],
        )
        projectors = tuple(
            surface._geometric_feature_projector(slot, projection.local_coordinates[index])
            for slot, index in zip(slots, (first, second), strict=True)
        )
        if projectors[0][1] != projectors[1][1]:
            return False
        kind = projectors[0][1]
        if kind == "vertex":
            return bool(np.linalg.norm(feet[first] - feet[second]) <= tolerance)
        if kind == "edge":
            P, Q = projectors[0][0], projectors[1][0]
            return bool(
                np.linalg.norm(P - Q) <= tolerance / scale
                and np.linalg.norm(P @ feet[first] - Q @ feet[second]) <= tolerance
            )
        normals = surface.facets._facet_normals
        first_normal, second_normal = normals[slots[0]], normals[slots[1]]
        return bool(
            np.linalg.norm(first_normal - second_normal) <= tolerance / scale
            and abs(np.dot(first_normal, feet[first] - feet[second])) <= tolerance
        )

    return sum(
        any(not same_feature(indices[0], index) for index in indices[1:])
        for indices in faces.values() if len(indices) > 1
    )


def linearize_closed_rigid_solid_contact(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    surface: ClosedTriangulatedRigidSolid,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    surface_reference_point=None,
) -> ImplicitRigidContactContribution:
    """Integrate the selected exact face/edge/vertex distance branch.

    This is not a smoothed normal or an edge-averaged facet law. A medial-axis
    tie or ambiguous winding fails in the geometry owner before this function
    receives valid projection evidence. Topological closure alone does not
    certify embedding; the caller must separately review that prerequisite.
    """

    if not isinstance(surface, ClosedTriangulatedRigidSolid):
        raise TypeError("Closed-solid contact needs reviewed closed geometry.")
    return integrate_implicit_rigid_contact(
        evaluation, record, response,
        surface.gap_differential(
            record.projection, motion=motion, factor=factor,
        ).hessian,
        surface_reference_point=surface_reference_point,
    )


__all__ = ["linearize_closed_rigid_solid_contact"]
