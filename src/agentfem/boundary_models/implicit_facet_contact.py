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

from dataclasses import dataclass

import numpy as np

from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceAssembly, ContactTraceEvaluation
from .rigid import TriangulatedRigidSurface, _readonly_array


@dataclass(frozen=True, eq=False)
class ImplicitFacetContactContribution:
    """Integrated residual/evidence and scalar-DOF COO Newton contribution."""

    assembly: ContactTraceAssembly
    tangent_rows: object
    tangent_columns: object
    tangent_values: object
    active_points: int

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, ContactTraceAssembly):
            raise TypeError("Implicit contact requires integrated trace evidence.")
        rows = np.asarray(self.tangent_rows, dtype=np.int64)
        columns = np.asarray(self.tangent_columns, dtype=np.int64)
        values = np.asarray(self.tangent_values, dtype=float)
        if rows.ndim != 1 or rows.shape != columns.shape or rows.shape != values.shape:
            raise ValueError("Contact tangent COO arrays must have equal one-dimensional shapes.")
        ndofs = 3 * self.assembly.number_of_nodes
        if (
            np.any(rows < 0) or np.any(columns < 0)
            or np.any(rows >= ndofs) or np.any(columns >= ndofs)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Contact tangent contains an invalid row, column or value.")
        if isinstance(self.active_points, bool) or not 0 <= self.active_points <= len(self.assembly.point_ids):
            raise ValueError("Contact active-point count is invalid.")
        object.__setattr__(self, "tangent_rows", _readonly_array(rows, dtype=np.int64))
        object.__setattr__(self, "tangent_columns", _readonly_array(columns, dtype=np.int64))
        object.__setattr__(self, "tangent_values", _readonly_array(values))

    def tangent_action(self, direction) -> np.ndarray:
        """Apply the local tangent without assuming a solver or matrix format."""

        values = np.asarray(direction, dtype=float)
        shape = (self.assembly.number_of_nodes, 3)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError("Contact tangent direction must match local nodal displacement.")
        action = np.bincount(
            self.tangent_rows,
            weights=self.tangent_values * values.reshape(-1)[self.tangent_columns],
            minlength=3 * shape[0],
        )
        return action.reshape(shape)


def _smooth_active_facets(
    surface: TriangulatedRigidSurface,
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
    edge_neighbors: dict[tuple[int, int], list[int]] = {}
    for index, vertices in enumerate(surface.triangles):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(vertices[a]), int(vertices[b]))))
            edge_neighbors.setdefault(edge, []).append(index)
    for point in edge_points:
        facet = facets_by_point[int(point)]
        barycentric = projection.local_coordinates[point]
        on_boundary = np.flatnonzero(barycentric <= threshold)
        if not len(on_boundary):
            continue
        selected_vertices = surface.triangles[facet]
        selected_normal = surface.facet_normals[facet]
        # An interior seam of coplanar facets has the same normal derivative
        # from both sides.  A free edge or a crease does not.
        for opposite in on_boundary:
            edge = tuple(sorted(np.delete(selected_vertices, opposite).tolist()))
            neighbors = edge_neighbors.get(edge, [])
            if len(neighbors) != 2 or any(
                np.linalg.norm(surface.facet_normals[index] - selected_normal) > threshold
                for index in neighbors
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
    surface: TriangulatedRigidSurface,
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
    if not isinstance(surface, TriangulatedRigidSurface):
        raise TypeError("Implicit facet contact requires TriangulatedRigidSurface.")
    if evaluation.trace.measure_configuration != "reference":
        raise NotImplementedError(
            "Implicit facet contact does not yet linearize a current-area measure."
        )
    if not record.projection.all_valid:
        raise ValueError("Implicit facet contact refuses invalid projection evidence.")
    if not np.array_equal(record.point_ids, evaluation.trace.point_ids):
        raise ValueError("Contact projection identity differs from the trace.")
    if response.projection is not record.projection:
        raise ValueError("Contact response must use the current projection record.")

    active = np.asarray(response.active, dtype=bool)
    _smooth_active_facets(surface, record, active)
    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=surface_reference_point,
    )
    selected_nodes = evaluation.trace.node_ids[active]
    selected_shapes = evaluation.trace.shape_values[active]
    selected_normals = record.projection.normals[active]
    selected_weights = evaluation.trace.weights[active]
    selected_penalties = response.penalties[active]
    if not len(selected_nodes):
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=float)
        return ImplicitFacetContactContribution(
            assembly, empty_int, empty_int, empty_float, 0
        )
    dofs = (3 * selected_nodes[:, :, None] + np.arange(3)).reshape((len(selected_nodes), -1))
    gradients = (selected_shapes[:, :, None] * selected_normals[:, None, :]).reshape(dofs.shape)
    block = (
        (selected_weights * selected_penalties)[:, None, None]
        * gradients[:, :, None]
        * gradients[:, None, :]
    )
    width = dofs.shape[1]
    rows = np.broadcast_to(dofs[:, :, None], (len(dofs), width, width)).reshape(-1)
    columns = np.broadcast_to(dofs[:, None, :], (len(dofs), width, width)).reshape(-1)
    return ImplicitFacetContactContribution(
        assembly,
        rows,
        columns,
        block.reshape(-1),
        len(selected_nodes),
    )


__all__ = ["ImplicitFacetContactContribution", "linearize_piecewise_planar_contact"]
