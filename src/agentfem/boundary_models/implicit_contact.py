# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral Newton contribution for a rigid penalty-contact trace.

The trace owns integration, the surface owns gap geometry, and the contact
law owns the point response.  This module only differentiates their composed
reference-area potential at the current trial; it does not select an active
set, search a surface, or commit nonlinear state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceAssembly, ContactTraceEvaluation
from .rigid import _readonly_array


@dataclass(frozen=True, eq=False)
class ImplicitRigidContactContribution:
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
            raise ValueError(
                "Contact tangent COO arrays must have equal one-dimensional shapes."
            )
        dimension = self.assembly.nodal_structural_residual.shape[1]
        ndofs = dimension * self.assembly.number_of_nodes
        if (
            np.any(rows < 0)
            or np.any(columns < 0)
            or np.any(rows >= ndofs)
            or np.any(columns >= ndofs)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Contact tangent contains an invalid row, column or value."
            )
        if isinstance(self.active_points, bool) or not 0 <= self.active_points <= len(
            self.assembly.point_ids
        ):
            raise ValueError("Contact active-point count is invalid.")
        object.__setattr__(self, "tangent_rows", _readonly_array(rows, dtype=np.int64))
        object.__setattr__(
            self, "tangent_columns", _readonly_array(columns, dtype=np.int64)
        )
        object.__setattr__(self, "tangent_values", _readonly_array(values))

    def tangent_action(self, direction) -> np.ndarray:
        """Apply the local tangent without assuming a solver or matrix format."""

        values = np.asarray(direction, dtype=float)
        dimension = self.assembly.nodal_structural_residual.shape[1]
        shape = (self.assembly.number_of_nodes, dimension)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "Contact tangent direction must match local nodal displacement."
            )
        action = np.bincount(
            self.tangent_rows,
            weights=self.tangent_values * values.reshape(-1)[self.tangent_columns],
            minlength=dimension * shape[0],
        )
        return action.reshape(shape)


def integrate_implicit_rigid_contact(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    gap_hessian,
    *,
    surface_reference_point=None,
) -> ImplicitRigidContactContribution:
    """Integrate the exact active-branch Hessian of ``k*min(g, 0)**2/2``.

    The point Hessian is ``k * (n tensor n + g * Hessian(g))``.  The curvature
    term vanishes on a smooth planar branch, but not on a sphere or cylinder.
    At opening/closing and non-smooth geometry branches this is only a
    one-sided derivative; callers must handle those transitions explicitly.
    """

    if not isinstance(evaluation, ContactTraceEvaluation):
        raise TypeError("Implicit contact requires ContactTraceEvaluation.")
    if not isinstance(record, ContactProjectionRecord):
        raise TypeError("Implicit contact requires ContactProjectionRecord.")
    if not isinstance(response, FrictionlessPenaltyContactResponse):
        raise TypeError("Implicit contact requires frictionless penalty response.")
    if evaluation.trace.measure_configuration != "reference":
        raise NotImplementedError(
            "Implicit contact does not yet linearize a current-area measure."
        )
    if not record.projection.all_valid:
        raise ValueError("Implicit contact refuses invalid projection evidence.")
    if not np.array_equal(record.point_ids, evaluation.trace.point_ids):
        raise ValueError("Contact projection identity differs from the trace.")
    if response.projection is not record.projection:
        raise ValueError("Contact response must use the current projection record.")

    projection = record.projection
    dimension = projection.dimension
    hessian = np.asarray(gap_hessian, dtype=float)
    expected_shape = (evaluation.trace.point_count, dimension, dimension)
    if hessian.shape != expected_shape or not np.all(np.isfinite(hessian)):
        raise ValueError(
            "Gap Hessian must be one finite spatial matrix per trace point."
        )
    if not np.allclose(hessian, hessian.transpose(0, 2, 1), rtol=1e-12, atol=1e-12):
        raise ValueError("Gap Hessian must be symmetric on a smooth contact branch.")
    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=surface_reference_point,
    )
    active = np.asarray(response.active, dtype=bool)
    selected_nodes = evaluation.trace.node_ids[active]
    if not len(selected_nodes):
        return ImplicitRigidContactContribution(
            assembly,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            0,
        )

    normals = projection.normals[active]
    gaps = projection.signed_gaps[active]
    point_tangent = (
        normals[:, :, None] * normals[:, None, :]
        + gaps[:, None, None] * hessian[active]
    )
    shape = evaluation.trace.shape_values[active]
    weight = evaluation.trace.weights[active] * response.penalties[active]
    local_tangent = np.einsum(
        "q,qa,qb,qij->qaibj",
        weight,
        shape,
        shape,
        point_tangent,
    )
    dofs = (dimension * selected_nodes[:, :, None] + np.arange(dimension)).reshape(
        (len(selected_nodes), -1)
    )
    width = dofs.shape[1]
    rows = np.broadcast_to(dofs[:, :, None], (len(dofs), width, width)).reshape(-1)
    columns = np.broadcast_to(dofs[:, None, :], (len(dofs), width, width)).reshape(-1)
    return ImplicitRigidContactContribution(
        assembly,
        rows,
        columns,
        local_tangent.reshape(-1),
        len(selected_nodes),
    )


__all__ = ["ImplicitRigidContactContribution", "integrate_implicit_rigid_contact"]
