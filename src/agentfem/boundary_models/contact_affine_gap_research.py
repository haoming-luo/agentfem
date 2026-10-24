# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded affine-plane feasibility screen, not an AL/MPC admission rule.

Unlike a tangent-rank screen, this constructs the complete plane gap
``g(q) = A q + b`` from reference positions and the actual affine offset.
It proves no fact about curved gaps, large-strain material admissibility,
multiplier inf-sup stability, or points absent from the supplied trace.
"""

from dataclasses import dataclass

import numpy as np

from .contact_trace import ContactTrace
from .rigid import RigidPlaneSurface, _readonly_array


@dataclass(frozen=True, eq=False)
class AffinePlaneGapSystem:
    matrix: np.ndarray
    offset: np.ndarray
    point_ids: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        ids = np.asarray(self.point_ids)
        weights = np.asarray(self.weights, dtype=float)
        if (
            matrix.ndim != 2 or offset.shape != (len(matrix),)
            or ids.shape != offset.shape or ids.dtype.kind not in "iu"
            or weights.shape != offset.shape or np.any(weights <= 0)
            or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset))
            or not np.all(np.isfinite(weights))
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Affine plane gaps need finite arrays and unique point IDs.")
        for name, value in (("matrix", matrix), ("offset", offset),
                            ("point_ids", ids), ("weights", weights)):
            object.__setattr__(self, name, _readonly_array(value, dtype=value.dtype))

    def gaps(self, reduced_values):
        values = np.asarray(reduced_values, dtype=float)
        if values.shape != (self.matrix.shape[1],) or not np.all(np.isfinite(values)):
            raise ValueError("Reduced values do not match the affine plane gap system.")
        return self.matrix @ values + self.offset

    def stationary_penetration(self, *, gap_tolerance):
        """Identify *exactly* zero floating-point rows with a resolved overlap.

        This is an algebraic witness for the supplied floating-point affine
        system, not an interval certificate for uncertain geometry. A merely
        small nonzero row is never declared immovable.
        """
        tolerance = float(gap_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Gap tolerance must be finite and nonnegative.")
        stationary = np.all(self.matrix == 0., axis=1)
        blocked = stationary & (self.offset < -tolerance)
        return {
            "scope": "supplied_discrete_affine_plane_gaps",
            "stationary_point_ids": tuple(int(i) for i in self.point_ids[stationary]),
            "incompatible_point_ids": tuple(int(i) for i in self.point_ids[blocked]),
            "minimum_stationary_gap": (
                float(np.min(self.offset[stationary])) if np.any(stationary) else None
            ),
            "gap_tolerance": tolerance,
            "certifies_spatial_stability": False,
            "certifies_material_admissibility": False,
        }

    def find_feasible_values(self, *, gap_tolerance, time_limit=5.):
        """Return a checked numerical witness, or unresolved; never alter a Step.

        HiGHS' infeasible status alone is not promoted to a certificate.
        Reduced displacement variables must be free, not its default >= 0.
        """
        from scipy.optimize import linprog

        report = self.stationary_penetration(gap_tolerance=gap_tolerance)
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("The diagnostic time limit must be positive and finite.")
        if report["incompatible_point_ids"]:
            return {**report, "status": "stationary_incompatibility", "values": None}
        zero = np.zeros(self.matrix.shape[1])
        if np.all(self.gaps(zero) >= -gap_tolerance):
            return {**report, "status": "feasible_witness", "values": zero,
                    "minimum_gap": float(np.min(self.gaps(zero), initial=np.inf))}
        result = linprog(
            np.zeros(self.matrix.shape[1]), A_ub=-self.matrix, b_ub=self.offset,
            bounds=(None, None), method="highs",
            options={"time_limit": float(time_limit)},
        )
        values = None if result.x is None else np.asarray(result.x)
        valid = (result.success and values is not None
                 and np.all(np.isfinite(values))
                 and np.all(self.gaps(values) >= -gap_tolerance))
        return {
            **report, "status": "feasible_witness" if valid else "unresolved",
            "values": values.copy() if valid else None,
            "optimizer_status": int(result.status),
            "minimum_gap": float(np.min(self.gaps(values))) if valid else None,
        }


def affine_plane_gap_system(
    trace, reference_positions, displacement_reduction, displacement_offset,
    surface, *, maximum_points=256, maximum_dofs=4096,
):
    """Construct all supplied reference-area trace inequalities at one tool pose.

    T and u_bar must already include *all* essential and multi-point constraints.
    Tool translation/rotation is supplied as the actual plane at this station,
    never inferred from an arbitrary trial displacement or a loading scalar.
    """
    if not isinstance(trace, ContactTrace) or not isinstance(surface, RigidPlaneSurface):
        raise TypeError("Affine plane feasibility requires a fixed trace and an analytical plane.")
    if trace.measure_configuration != "reference":
        raise NotImplementedError("Affine plane feasibility requires a reference-area trace.")
    if any(type(v) is not int or v < 1 for v in (maximum_points, maximum_dofs)):
        raise ValueError("Diagnostic budgets must be positive integers.")
    positions = np.asarray(reference_positions, dtype=float)
    reduction = np.asarray(displacement_reduction, dtype=float)
    offset = np.asarray(displacement_offset, dtype=float)
    if (
        positions.ndim != 2 or positions.shape[1] != surface.dimension
        or offset.shape != (positions.size,)
        or reduction.ndim != 2 or reduction.shape[0] != positions.size
        or not np.all(np.isfinite(positions + offset.reshape(positions.shape)))
        or not np.all(np.isfinite(reduction))
    ):
        raise ValueError("Affine plane geometry, reduction and offset have inconsistent layouts.")
    if (
        trace.point_count > maximum_points or positions.size > maximum_dofs
        or reduction.shape[1] > maximum_dofs
    ):
        raise NotImplementedError("Affine plane diagnostic exceeds its dense point/DOF budget.")
    evaluation = trace.evaluate(positions + offset.reshape(positions.shape))
    projection = surface.project(evaluation.query_points)
    jacobian = np.zeros((trace.point_count, positions.size))
    dimension = positions.shape[1]
    for row in range(trace.point_count):
        for node, shape in zip(trace.node_ids[row], trace.shape_values[row], strict=True):
            start = int(node) * dimension
            jacobian[row, start:start+dimension] += shape * projection.normals[row]
    return AffinePlaneGapSystem(
        jacobian @ reduction, projection.signed_gaps, trace.point_ids, trace.weights,
    )
