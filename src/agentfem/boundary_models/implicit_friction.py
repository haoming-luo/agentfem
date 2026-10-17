# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-old-state planar Coulomb tangent for a bounded implicit Operator.

This differentiates the existing penalty return map at one current Newton
trial.  It deliberately does not own accepted friction state, integration,
line search, work, or checkpointing.  Until those Procedure contracts exist,
the public implicit Step continues to reject frictional pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_friction import PenaltyCoulombFrictionLaw, PenaltyCoulombFrictionResponse
from .contact_response import FrictionlessPenaltyContactResponse, _penalty_values
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceEvaluation
from .rigid import RigidPlaneSurface, _readonly_array


@dataclass(frozen=True, eq=False)
class ImplicitPlanarFrictionTangent:
    """Tangential-return-map COO derivative on one fixed rigid-plane branch."""

    rows: object
    columns: object
    values: object
    number_of_nodes: int
    dimension: int
    sticking_points: int
    sliding_points: int

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        for name in ("number_of_nodes", "dimension", "sticking_points", "sliding_points"):
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
                raise TypeError(f"Planar friction {name} must be an integer.")
        count = int(self.number_of_nodes)
        dimension = int(self.dimension)
        if count < 0 or dimension != 3:
            raise ValueError("Planar friction tangent requires a 3D nodal layout.")
        if rows.ndim != 1 or rows.shape != columns.shape or rows.shape != values.shape:
            raise ValueError("Planar friction COO arrays must have matching vector shapes.")
        if (
            np.any(rows < 0) or np.any(columns < 0)
            or np.any(rows >= count * dimension)
            or np.any(columns >= count * dimension)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Planar friction COO data contain invalid entries.")
        if self.sticking_points < 0 or self.sliding_points < 0:
            raise ValueError("Planar friction branch counts must be nonnegative.")
        object.__setattr__(self, "rows", _readonly_array(rows, dtype=np.int64))
        object.__setattr__(self, "columns", _readonly_array(columns, dtype=np.int64))
        object.__setattr__(self, "values", _readonly_array(values))

    def tangent_action(self, direction) -> np.ndarray:
        values = np.asarray(direction, dtype=float)
        shape = (self.number_of_nodes, self.dimension)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError("Friction tangent direction must match nodal displacement.")
        return np.bincount(
            self.rows,
            weights=self.values * values.reshape(-1)[self.columns],
            minlength=self.number_of_nodes * self.dimension,
        ).reshape(shape)


def linearize_planar_penalty_friction(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    normal_response: FrictionlessPenaltyContactResponse,
    friction_response: PenaltyCoulombFrictionResponse,
    law: PenaltyCoulombFrictionLaw,
    surface: RigidPlaneSurface,
) -> ImplicitPlanarFrictionTangent:
    """Differentiate stick/slip traction at fixed accepted state and tool pose.

    In stick the point Jacobian is ``k_t P``. In slip it is
    ``-mu*k_n*t_hat tensor n + mu*p/|s_trial|*(P-t_hat tensor t_hat)``.
    ``P=I-n tensor n`` and ``p=-k_n*g`` on the active plane branch.  The
    nonsmooth cap and opening boundaries are rejected; a global Procedure
    must cut back or use a reviewed generalized derivative there.
    """

    if not isinstance(evaluation, ContactTraceEvaluation):
        raise TypeError("Implicit friction requires ContactTraceEvaluation.")
    if not isinstance(record, ContactProjectionRecord):
        raise TypeError("Implicit friction requires ContactProjectionRecord.")
    if not isinstance(normal_response, FrictionlessPenaltyContactResponse):
        raise TypeError("Implicit friction requires normal penalty response.")
    if not isinstance(friction_response, PenaltyCoulombFrictionResponse):
        raise TypeError("Implicit friction requires a tangential return-map response.")
    if not isinstance(law, PenaltyCoulombFrictionLaw) or not isinstance(surface, RigidPlaneSurface):
        raise TypeError("This implicit friction tangent is limited to a rigid plane.")
    if evaluation.trace.measure_configuration != "reference":
        raise NotImplementedError("Implicit friction does not linearize current-area weights.")
    if not record.projection.all_valid or normal_response.projection is not record.projection:
        raise ValueError("Implicit friction requires the current valid normal projection.")
    if record.projection.geometry_fingerprint != surface.geometry_fingerprint:
        raise ValueError("Implicit friction projection belongs to another rigid plane.")
    if not np.array_equal(record.projection.query_points, evaluation.query_points):
        raise ValueError("Implicit friction projection is stale for this Newton trial.")
    if not np.array_equal(record.point_ids, evaluation.trace.point_ids) or not np.array_equal(
        friction_response.record.point_ids, record.point_ids
    ):
        raise ValueError("Implicit friction point identities differ.")
    if not np.array_equal(friction_response.active, normal_response.active):
        raise ValueError("Friction and normal active sets differ.")
    if not np.allclose(
        friction_response.friction_limits,
        law.coefficient * normal_response.pressures,
        rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("Friction response does not use the current normal pressure cap.")

    count = evaluation.trace.point_count
    penalties = _penalty_values(law.tangential_penalty, count)
    normal_penalties = normal_response.penalties
    if not np.allclose(
        friction_response.record.normals, record.projection.normals,
        rtol=1e-12, atol=1e-14,
    ) or not np.allclose(
        friction_response.structural_residual_tractions,
        penalties[:, None] * friction_response.record.elastic_slips,
        rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("Friction return map differs from the current plane and law.")
    active = np.asarray(normal_response.active, dtype=bool)
    sticking = np.asarray(friction_response.sticking, dtype=bool)
    sliding = np.asarray(friction_response.sliding, dtype=bool)
    trial = (
        friction_response.record.elastic_slips
        + friction_response.plastic_slip_increments
    )
    norms = np.linalg.norm(trial, axis=1)
    trial_traction = penalties * norms
    limits = friction_response.friction_limits
    cap_tolerance = 128.0 * np.finfo(float).eps * np.maximum.reduce(
        (np.ones(count), trial_traction, limits)
    )
    if np.any(active & (np.abs(trial_traction - limits) <= cap_tolerance)):
        raise ValueError("AFM-IMPLICIT-FRICTION-BRANCH: stick/slip cap has no unique tangent.")
    geometry_scale = max(
        1.0,
        float(np.linalg.norm(surface.point)),
        float(np.max(np.abs(evaluation.query_points), initial=0.0)),
    )
    opening_tolerance = 128.0 * np.finfo(float).eps * geometry_scale
    if np.any(np.abs(record.projection.signed_gaps) <= opening_tolerance):
        raise ValueError("AFM-IMPLICIT-FRICTION-BRANCH: contact opening has no unique tangent.")

    normals = record.projection.normals
    projector = np.eye(3)[None, :, :] - normals[:, :, None] * normals[:, None, :]
    point_tangent = np.zeros((count, 3, 3), dtype=float)
    point_tangent[sticking] = penalties[sticking, None, None] * projector[sticking]
    nonzero_sliding = sliding & (norms > 0.0) & (law.coefficient > 0.0)
    if np.any(nonzero_sliding):
        direction = trial[nonzero_sliding] / norms[nonzero_sliding, None]
        point_tangent[nonzero_sliding] = (
            -law.coefficient * normal_penalties[nonzero_sliding, None, None]
            * direction[:, :, None] * normals[nonzero_sliding, None, :]
            + (limits[nonzero_sliding] / norms[nonzero_sliding])[:, None, None]
            * (projector[nonzero_sliding]
               - direction[:, :, None] * direction[:, None, :])
        )

    selected_nodes = evaluation.trace.node_ids[active]
    if not len(selected_nodes):
        return ImplicitPlanarFrictionTangent(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float), evaluation.number_of_nodes, 3, 0, 0,
        )
    shape = evaluation.trace.shape_values[active]
    weight = evaluation.trace.weights[active]
    local_tangent = np.einsum(
        "q,qa,qb,qij->qaibj", weight, shape, shape, point_tangent[active],
    )
    dofs = (3 * selected_nodes[:, :, None] + np.arange(3)).reshape(
        (len(selected_nodes), -1)
    )
    width = dofs.shape[1]
    rows = np.broadcast_to(dofs[:, :, None], (len(dofs), width, width)).reshape(-1)
    columns = np.broadcast_to(dofs[:, None, :], (len(dofs), width, width)).reshape(-1)
    return ImplicitPlanarFrictionTangent(
        rows, columns, local_tangent.reshape(-1),
        evaluation.number_of_nodes, 3,
        int(np.count_nonzero(sticking)), int(np.count_nonzero(sliding)),
    )


__all__ = ["ImplicitPlanarFrictionTangent", "linearize_planar_penalty_friction"]
