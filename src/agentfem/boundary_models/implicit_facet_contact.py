# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded Newton linearization of a rigid, piecewise-planar contact trace.

This is a solver-neutral Operator contribution, not a contact Step.  Geometry
search and accepted/trial projection remain with their existing owners.  The
normal-penalty derivative is exact on an unchanged planar facet branch.  An
active non-smooth edge, vertex, or trimmed boundary is rejected rather than
assigned a fictitious normal derivative.
"""

from __future__ import annotations

import numpy as np

from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceEvaluation
from .convex_polyhedron import ConvexPolyhedralRigidSurface
from .implicit_contact import (
    ImplicitRigidContactContribution,
    integrate_implicit_rigid_contact,
)
from .rigid import TriangulatedRigidSurface

# Retain the historical internal type name for installed extensions that
# imported the bounded fixed-facet candidate before the common Operator seam.
ImplicitFacetContactContribution = ImplicitRigidContactContribution


def _smooth_active_facets(
    surface: TriangulatedRigidSurface | ConvexPolyhedralRigidSurface,
    record: ContactProjectionRecord,
    active: np.ndarray,
) -> None:
    """Reject active edge/vertex branches without a unique planar normal."""

    projection = record.projection
    if projection.surface_kind != "triangulated_rigid_surface":
        raise TypeError("Implicit facet contact requires triangulated projection evidence.")
    if projection.geometry_fingerprint != surface.geometry_fingerprint:
        raise ValueError("Contact projection belongs to a different surface geometry.")
    if projection.local_coordinates is None or projection.entity_ids is None:
        raise ValueError("Implicit facet contact requires facet IDs and barycentric coordinates.")
    # Barycentric tolerance inherits the geometry's length scale.  This only
    # classifies a branch; it does not change a projected point or contact gap.
    threshold = float(surface.tolerance) / float(surface.scale)
    active_points = np.flatnonzero(active)
    facet_order = np.argsort(surface.facet_ids)
    sorted_ids = surface.facet_ids[facet_order]
    selected_ids = projection.entity_ids[active_points]
    slots = np.searchsorted(sorted_ids, selected_ids)
    if np.any(slots >= len(sorted_ids)) or np.any(
        sorted_ids[np.minimum(slots, len(sorted_ids) - 1)] != selected_ids
    ):
        raise ValueError("Active projection references an unknown rigid facet.")
    selected_facets = facet_order[slots]
    edge_points = active_points[
        np.any(projection.local_coordinates[active_points] <= threshold, axis=1)
    ]
    if not len(edge_points):
        return
    facets_by_point = dict(zip(active_points.tolist(), selected_facets.tolist()))
    for point in edge_points:
        facet = facets_by_point[int(point)]
        barycentric = projection.local_coordinates[point]
        on_boundary = np.flatnonzero(barycentric <= threshold)
        if not len(on_boundary):
            continue
        selected_normal = surface.facet_normals[facet]
        if len(on_boundary) >= 2:
            # At a vertex, the two neighboring facets of the *selected*
            # triangle do not describe the full fan. A remote wedge may be a
            # free boundary or have a different normal. Both invalidate a
            # unique planar derivative even when the local two edges look
            # smooth, so inspect the whole incident star before admission.
            local_vertex = int(np.argmax(barycentric))
            vertex = int(surface.triangles[facet, local_vertex])
            incident = np.flatnonzero(
                np.any(surface.triangles == vertex, axis=1)
            )
            incident_local = np.argmax(
                surface.triangles[incident] == vertex, axis=1
            )
            edges_containing_vertex = (
                np.arange(3)[None, :] != incident_local[:, None]
            )
            if (
                np.any(np.linalg.norm(
                    surface.facet_normals[incident] - selected_normal,
                    axis=1,
                ) > threshold)
                or np.any(
                    (surface.edge_neighbors[incident] < 0)
                    & edges_containing_vertex
                )
            ):
                raise ValueError(
                    "AFM-IMPLICIT-CONTACT-BRANCH: active projection reaches a "
                    "free or non-smooth tool edge/vertex; cut back or refine "
                    f"the tool, contact_point_id={int(record.point_ids[point])}."
                )
        # An interior seam of coplanar facets has the same normal derivative
        # from both sides.  A free edge or a crease does not.
        for opposite in on_boundary:
            neighbor = int(surface.edge_neighbors[facet, opposite])
            if neighbor < 0 or (
                np.linalg.norm(surface.facet_normals[neighbor] - selected_normal)
                > threshold
            ):
                raise ValueError(
                    "AFM-IMPLICIT-CONTACT-BRANCH: active projection reaches a "
                    "free or non-smooth tool edge/vertex; cut back or refine "
                    f"the tool, contact_point_id={int(record.point_ids[point])}."
                )


def linearize_piecewise_planar_contact(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    surface: TriangulatedRigidSurface | ConvexPolyhedralRigidSurface,
    *,
    surface_reference_point=None,
) -> ImplicitFacetContactContribution:
    """Return the fixed-branch normal-penalty residual and exact Newton tangent.

    Reference-area weights and a fixed rigid pose are required.  Search and
    projection must be recomputed for every Newton trial; this function never
    reuses a previous gap, normal, or active set.  The resulting COO indices
    address local blocked-vector scalar dofs, including ghosts if present;
    a backend must still perform owner reduction and apply strong/MPC rows.
    """

    if not isinstance(evaluation, ContactTraceEvaluation):
        raise TypeError("Implicit facet contact requires ContactTraceEvaluation.")
    if not isinstance(record, ContactProjectionRecord):
        raise TypeError("Implicit facet contact requires ContactProjectionRecord.")
    if not isinstance(response, FrictionlessPenaltyContactResponse):
        raise TypeError("Implicit facet contact requires frictionless penalty response.")
    if not isinstance(surface, (TriangulatedRigidSurface, ConvexPolyhedralRigidSurface)):
        raise TypeError("Implicit facet contact requires reviewed planar geometry.")
    active = np.asarray(response.active, dtype=bool)
    if isinstance(surface, ConvexPolyhedralRigidSurface):
        projection = record.projection
        if (
            projection.surface_kind != "convex_polyhedral_rigid_surface"
            or projection.geometry_fingerprint != surface.geometry_fingerprint
        ):
            raise ValueError("Convex contact projection belongs to another geometry.")
        if projection.entity_ids is None or projection.local_coordinates is None:
            raise ValueError("Convex contact requires facet and local-coordinate evidence.")
        for point in np.flatnonzero(active):
            identity = int(projection.entity_ids[point])
            matches = np.flatnonzero(surface.facets.facet_ids == identity)
            if matches.size != 1:
                raise ValueError("Convex contact selected an unknown facet identity.")
            if not np.isclose(
                np.dot(
                    projection.query_points[point] - projection.closest_points[point],
                    projection.normals[point],
                ),
                projection.signed_gaps[point],
                atol=surface.facets.tolerance,
                rtol=0.0,
            ):
                raise ValueError("Convex contact gap and closest-point evidence disagree.")
    else:
        _smooth_active_facets(surface, record, active)
    return integrate_implicit_rigid_contact(
        evaluation, record, response,
        np.zeros((evaluation.trace.point_count, 3, 3)),
        surface_reference_point=surface_reference_point,
    )


__all__ = ["ImplicitFacetContactContribution", "linearize_piecewise_planar_contact"]
