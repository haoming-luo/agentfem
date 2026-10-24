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
from .contact_trace import (
    ContactTraceAssembly, ContactTraceEvaluation, FrictionContactTraceAssembly,
)
from .implicit_friction import ImplicitPlanarFrictionTangent
from .rigid import _readonly_array


def _coalesce_trace_tangent(node_ids, local_tangent):
    """Sum quadrature matrices sharing one ordered nodal stencil.

    Only adjacent equal stencils are combined. Unordered, interleaved or
    pointwise traces remain valid without a global sort or cached geometry.
    This is finite-element integration before COO expansion, not reduced
    quadrature: every active point's current curvature and weight is retained.
    """
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(node_ids, axis=0), axis=1)) + 1]
    if len(starts) == len(node_ids):
        return node_ids, local_tangent
    return node_ids[starts], np.add.reduceat(local_tangent, starts, axis=0)


@dataclass(frozen=True, eq=False)
class ImplicitRigidContactContribution:
    """Integrated residual/evidence and scalar-DOF COO Newton contribution."""

    assembly: ContactTraceAssembly
    tangent_rows: object
    tangent_columns: object
    tangent_values: object
    active_points: int
    friction_assembly: FrictionContactTraceAssembly | None = None
    friction_tangent: ImplicitPlanarFrictionTangent | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, ContactTraceAssembly):
            raise TypeError("Implicit contact requires integrated trace evidence.")
        rows = np.asarray(self.tangent_rows, dtype=np.int64)
        columns = np.asarray(self.tangent_columns, dtype=np.int64)
        values = np.asarray(self.tangent_values, dtype=float)
        if (self.friction_assembly is None) != (self.friction_tangent is None):
            raise ValueError("Implicit friction requires both its trace and tangent.")
        if self.friction_assembly is not None:
            if (
                not isinstance(self.friction_assembly, FrictionContactTraceAssembly)
                or not isinstance(self.friction_tangent, ImplicitPlanarFrictionTangent)
                or not np.array_equal(
                    self.friction_assembly.point_ids, self.assembly.point_ids,
                )
                or self.friction_tangent.number_of_nodes != self.assembly.number_of_nodes
                or self.friction_tangent.dimension != self.assembly.nodal_structural_residual.shape[1]
            ):
                raise ValueError("Implicit friction does not share the normal trace layout.")
            rows = np.concatenate((rows, self.friction_tangent.rows))
            columns = np.concatenate((columns, self.friction_tangent.columns))
            values = np.concatenate((values, self.friction_tangent.values))
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

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        if self.friction_assembly is None:
            return self.assembly.nodal_structural_residual
        return (
            self.assembly.nodal_structural_residual
            + self.friction_assembly.nodal_structural_residual
        )

    @property
    def potential_energy(self) -> float:
        return float(
            self.assembly.potential_energy
            + (
                0.0 if self.friction_assembly is None
                else self.friction_assembly.recoverable_penalty_energy
            )
        )

    @property
    def contact_force_on_structure(self) -> np.ndarray:
        if self.friction_assembly is None:
            return self.assembly.contact_force_on_structure
        return (
            self.assembly.contact_force_on_structure
            + self.friction_assembly.contact_force_on_structure
        )

    @property
    def surface_generalized_moment(self) -> np.ndarray | None:
        normal = self.assembly.surface_generalized_moment
        if self.friction_assembly is None:
            return normal
        tangential = self.friction_assembly.surface_generalized_moment
        if normal is None or tangential is None:
            return None
        return normal + tangential


@dataclass(frozen=True, eq=False)
class CombinedImplicitRigidContactContribution:
    """Additive rigid-pair Operator, preserving each pair's scientific identity.

    This is a solver-neutral algebraic composition, not an admission of a
    multi-tool Step.  In particular, each provider still owns its own rigid
    coordinates, accepted work and checkpoint state.  Reusing contact point
    IDs across distinct pairs is valid because their names remain separate.
    """

    pairs: tuple[tuple[str, ImplicitRigidContactContribution], ...]

    def __post_init__(self) -> None:
        selected = tuple(self.pairs)
        if not selected:
            raise ValueError("Combined implicit contact requires at least one pair.")
        names = tuple(str(name).strip() for name, _ in selected)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                "Combined implicit contact pair names must be unique and nonempty."
            )
        if any(
            not isinstance(item, ImplicitRigidContactContribution)
            for _, item in selected
        ):
            raise TypeError(
                "Combined implicit contact accepts reviewed pair contributions only."
            )
        layout = selected[0][1].nodal_structural_residual.shape
        if any(
            item.nodal_structural_residual.shape != layout
            for _, item in selected
        ):
            raise ValueError(
                "Combined implicit contact pairs must share one nodal vector layout."
            )
        object.__setattr__(
            self, "pairs",
            tuple((name, item) for name, (_, item) in zip(names, selected)),
        )

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        return np.sum(
            [item.nodal_structural_residual for _, item in self.pairs], axis=0,
        )

    @property
    def potential_energy(self) -> float:
        return float(sum(item.potential_energy for _, item in self.pairs))

    @property
    def contact_force_on_structure(self) -> np.ndarray:
        return np.sum(
            [item.contact_force_on_structure for _, item in self.pairs], axis=0,
        )

    @property
    def active_points(self) -> int:
        return sum(item.active_points for _, item in self.pairs)

    def tangent_action(self, direction) -> np.ndarray:
        return np.sum([item.tangent_action(direction) for _, item in self.pairs], axis=0)


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
    # Keep pointwise projection, law and evidence above. Integrate repeated
    # facet stencils before allocating, copying and sorting scalar COO indices.
    active_points = len(selected_nodes)
    selected_nodes, local_tangent = _coalesce_trace_tangent(selected_nodes, local_tangent)
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
        active_points,
    )


__all__ = [
    "ImplicitRigidContactContribution",
    "CombinedImplicitRigidContactContribution",
    "integrate_implicit_rigid_contact",
]
