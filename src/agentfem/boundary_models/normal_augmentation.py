# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-trace, solver-neutral augmented normal contact research contract.

The multiplier is an algorithmic State variable, not a material history or a
physical stored energy.  This module deliberately does not admit adaptive
contact quadrature or lower an ordinary Step: a changing point basis needs a
separately verified multiplier-transfer rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTrace, ContactTraceEvaluation
from .rigid import _readonly_array


def _point_ids(value) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in {"i", "u"}):
        raise ValueError("Normal multiplier point IDs must be one-dimensional integers.")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("Normal multiplier point IDs exceed signed 64-bit range.")
    ids = raw.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Normal multiplier point IDs must be unique and increasing.")
    return _readonly_array(ids, dtype=np.int64)


def _nonnegative_vector(value, count: int, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (count,) or not np.all(np.isfinite(selected)) or np.any(selected < 0):
        raise ValueError(f"{name} must have one finite nonnegative value per point.")
    return _readonly_array(selected)


def _nonnegative_scalar(value, *, name: str) -> float:
    selected = float(value)
    if not np.isfinite(selected) or selected < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return selected


def _trace_identity(
    trace: ContactTrace, reference_evaluation: ContactTraceEvaluation,
) -> str:
    """Bind multiplier meaning to the fixed reference quadrature basis."""

    if not isinstance(trace, ContactTrace) or trace.measure_configuration != "reference":
        raise TypeError("Normal multiplier State requires a fixed reference ContactTrace.")
    if (
        not isinstance(reference_evaluation, ContactTraceEvaluation)
        or reference_evaluation.trace is not trace
    ):
        raise ValueError("Normal multiplier reference geometry must use the fixed trace.")
    payload = {
        "point_ids": trace.point_ids.tolist(),
        "node_ids": trace.node_ids.tolist(),
        "shape_values": trace.shape_values.tolist(),
        "weights": trace.weights.tolist(),
        "measure_configuration": trace.measure_configuration,
        "source": trace.source,
        "reference_query_points": reference_evaluation.query_points.tolist(),
        "reference_number_of_nodes": reference_evaluation.number_of_nodes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, eq=False)
class AugmentedNormalPointResponse:
    """One fixed-multiplier Newton response at currently projected points."""

    evaluation: ContactTraceEvaluation
    record: ContactProjectionRecord
    multiplier: object
    penalty: float
    pressure: object
    active: object
    algorithmic_density: object

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evaluation, ContactTraceEvaluation)
            or not np.array_equal(self.evaluation.trace.point_ids, self.record.point_ids)
            or not np.array_equal(
                self.evaluation.query_points, self.record.projection.query_points,
            )
        ):
            raise ValueError("Augmented normal response trace and projection differ.")
        count = self.record.projection.point_count
        multiplier = _nonnegative_vector(self.multiplier, count, name="Multiplier")
        pressure = _nonnegative_vector(self.pressure, count, name="Pressure")
        active = np.asarray(self.active, dtype=bool)
        density = np.asarray(self.algorithmic_density, dtype=float)
        if active.shape != (count,) or density.shape != (count,) or not np.all(np.isfinite(density)):
            raise ValueError("Augmented normal response fields have the wrong shape or are non-finite.")
        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("Augmented normal penalty must be finite and positive.")
        if not self.record.projection.all_valid:
            raise ValueError("Augmented normal contact refuses invalid projections.")
        expected = np.maximum(0.0, multiplier - self.penalty * self.record.projection.signed_gaps)
        if not np.array_equal(active, expected > 0) or not np.allclose(pressure, expected, rtol=1e-12, atol=1e-12):
            raise ValueError("Augmented normal pressure does not match the frozen multiplier.")
        expected_density = (pressure**2 - multiplier**2) / (2.0 * self.penalty)
        if not np.allclose(density, expected_density, rtol=1e-12, atol=1e-12):
            raise ValueError("Augmented normal functional does not match the law.")
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "pressure", pressure)
        object.__setattr__(self, "active", _readonly_array(active, dtype=bool))
        object.__setattr__(self, "algorithmic_density", _readonly_array(density))


class FixedTraceNormalMultiplierState:
    """Accepted and outer-iteration multipliers on one immutable reference trace.

    The Procedure must hold the multiplier fixed during each Newton solve,
    call ``augment_after_converged`` only after that solve converges, and commit
    only when its *total* equilibrium test also passes. No contact state is
    accepted merely because the point law was evaluated successfully.
    ``penetration_tolerance`` has the declared length unit, whereas
    ``complementarity_tolerance`` bounds ``abs(multiplier * gap)`` and has
    pressure-times-length units. Neither tolerance is a force tolerance.
    """

    schema = "agentfem.fixed-trace-normal-augmentation.v1"

    def __init__(
        self,
        trace: ContactTrace,
        *,
        pair_identity: str,
        reference_evaluation: ContactTraceEvaluation,
        surface_fingerprint: str,
        penalty: float,
        penetration_tolerance: float,
        complementarity_tolerance: float,
        maximum_augmentations: int,
        length_unit: str,
        pressure_unit: str,
        initial_multiplier=None,
    ):
        self.trace = trace
        if not isinstance(pair_identity, str) or not pair_identity.strip():
            raise ValueError("Normal multiplier State requires a contact-pair identity.")
        self.pair_identity = pair_identity
        self.trace_identity = _trace_identity(trace, reference_evaluation)
        if not isinstance(surface_fingerprint, str) or not surface_fingerprint.strip():
            raise ValueError("Normal multiplier State requires a rigid-surface fingerprint.")
        self.surface_fingerprint = surface_fingerprint
        self.point_ids = _point_ids(trace.point_ids)
        self.penalty = float(penalty)
        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("Augmented normal penalty must be finite and positive.")
        self.penetration_tolerance = _nonnegative_scalar(
            penetration_tolerance, name="Penetration tolerance",
        )
        self.complementarity_tolerance = _nonnegative_scalar(
            complementarity_tolerance, name="Complementarity tolerance",
        )
        if (
            isinstance(maximum_augmentations, bool)
            or not isinstance(maximum_augmentations, (int, np.integer))
            or maximum_augmentations < 1
        ):
            raise ValueError("Maximum augmentations must be a positive integer.")
        self.maximum_augmentations = int(maximum_augmentations)
        for name, unit in (("Length", length_unit), ("Pressure", pressure_unit)):
            if not isinstance(unit, str) or not unit.strip():
                raise ValueError(f"{name} unit must be an explicit nonempty string.")
        self.length_unit = length_unit
        self.pressure_unit = pressure_unit
        initial = (
            np.zeros(len(self.point_ids))
            if initial_multiplier is None else initial_multiplier
        )
        self._accepted = _nonnegative_vector(initial, len(self.point_ids), name="Initial multiplier")
        self._iterate = self._accepted.copy()
        self._augmentation_count = 0
        self._converged_record: ContactProjectionRecord | None = None

    @property
    def accepted_multiplier(self) -> np.ndarray:
        return _readonly_array(self._accepted)

    @property
    def iterate_multiplier(self) -> np.ndarray:
        return _readonly_array(self._iterate)

    @property
    def augmentation_count(self) -> int:
        return self._augmentation_count

    def _check_record(
        self, evaluation: ContactTraceEvaluation, record: ContactProjectionRecord,
    ) -> None:
        if not isinstance(evaluation, ContactTraceEvaluation) or evaluation.trace is not self.trace:
            raise ValueError("Normal multiplier trace basis changed; transfer is not supported.")
        if not isinstance(record, ContactProjectionRecord):
            raise TypeError("Normal augmentation requires a projection record.")
        if not np.array_equal(record.point_ids, self.point_ids):
            raise ValueError("Normal multiplier and contact trace point identities differ.")
        if record.projection.geometry_fingerprint != self.surface_fingerprint:
            raise ValueError("Normal multiplier rigid-surface identity changed.")
        if not np.array_equal(evaluation.query_points, record.projection.query_points):
            raise ValueError("Normal multiplier projection does not match the current trace evaluation.")
        if not record.projection.all_valid:
            raise ValueError("Normal augmentation refuses invalid current projection.")

    def evaluate(
        self, evaluation: ContactTraceEvaluation, record: ContactProjectionRecord,
    ) -> AugmentedNormalPointResponse:
        """Return the fixed-multiplier response for one Newton evaluation."""

        self._check_record(evaluation, record)
        pressure = np.maximum(
            0.0, self._iterate - self.penalty * record.projection.signed_gaps,
        )
        density = (pressure**2 - self._iterate**2) / (2.0 * self.penalty)
        return AugmentedNormalPointResponse(
            evaluation, record, self._iterate, self.penalty,
            pressure, pressure > 0, density,
        )

    def mark_equilibrium_converged(
        self,
        evaluation: ContactTraceEvaluation,
        record: ContactProjectionRecord,
        *,
        total_residual_norm: float,
        total_residual_tolerance: float,
    ) -> None:
        """Bind a verified *total* Newton residual to the current multiplier."""

        self._check_record(evaluation, record)
        norm = float(total_residual_norm)
        tolerance = float(total_residual_tolerance)
        if not np.isfinite(norm) or not np.isfinite(tolerance) or tolerance < 0 or norm > tolerance:
            raise ValueError("Total equilibrium has not converged for this multiplier.")
        self._converged_record = record

    def augment_after_converged(
        self, evaluation: ContactTraceEvaluation, record: ContactProjectionRecord,
    ) -> np.ndarray:
        """Stage one outer update only after equilibrium with the old multiplier."""

        self._check_record(evaluation, record)
        if record is not self._converged_record:
            raise RuntimeError("Augmentation requires the current converged projection.")
        if self._augmentation_count >= self.maximum_augmentations:
            raise RuntimeError("Normal augmentation exceeded its declared outer-iteration limit.")
        self._iterate = np.maximum(
            0.0, self._iterate - self.penalty * record.projection.signed_gaps,
        )
        self._augmentation_count += 1
        self._converged_record = None
        return self.iterate_multiplier

    def acceptance_diagnostics(
        self, evaluation: ContactTraceEvaluation, record: ContactProjectionRecord,
    ) -> dict[str, float]:
        self._check_record(evaluation, record)
        gap = record.projection.signed_gaps
        return {
            "maximum_penetration": float(np.max(np.maximum(-gap, 0.0), initial=0.0)),
            "maximum_complementarity": float(np.max(np.abs(self._iterate * gap), initial=0.0)),
            "minimum_multiplier": float(np.min(self._iterate)) if self._iterate.size else 0.0,
        }

    def commit(
        self,
        evaluation: ContactTraceEvaluation,
        record: ContactProjectionRecord,
    ) -> dict[str, float]:
        """Accept only a converged global solve and an explicit contact test."""

        if record is not self._converged_record:
            raise ValueError("Total equilibrium must converge with the current multiplier before acceptance.")
        diagnostics = self.acceptance_diagnostics(evaluation, record)
        if (
            diagnostics["maximum_penetration"] > self.penetration_tolerance
            or diagnostics["maximum_complementarity"] > self.complementarity_tolerance
        ):
            raise ValueError("Normal augmentation has not met penetration/complementarity tolerances.")
        self._accepted = self._iterate.copy()
        self._augmentation_count = 0
        self._converged_record = None
        return diagnostics

    def rollback(self) -> None:
        self._iterate = self._accepted.copy()
        self._augmentation_count = 0
        self._converged_record = None

    def accepted_snapshot(self) -> dict[str, object]:
        """Serializable accepted State, not an on-disk checkpoint by itself."""

        return {
            "schema": self.schema,
            "pair_identity": self.pair_identity,
            "trace_identity": self.trace_identity,
            "surface_fingerprint": self.surface_fingerprint,
            "point_ids": self.point_ids.tolist(),
            "penalty": self.penalty,
            "penetration_tolerance": self.penetration_tolerance,
            "complementarity_tolerance": self.complementarity_tolerance,
            "maximum_augmentations": self.maximum_augmentations,
            "length_unit": self.length_unit,
            "pressure_unit": self.pressure_unit,
            "multiplier": self._accepted.tolist(),
        }

    def restore_accepted(self, snapshot: dict[str, object]) -> None:
        """Validate every field before atomically replacing accepted State."""

        if not isinstance(snapshot, dict) or snapshot.get("schema") != self.schema:
            raise ValueError("Normal multiplier checkpoint schema differs.")
        required = {
            "schema", "pair_identity", "trace_identity", "surface_fingerprint", "point_ids",
            "penalty", "penetration_tolerance", "complementarity_tolerance",
            "maximum_augmentations", "length_unit", "pressure_unit", "multiplier",
        }
        if set(snapshot) != required:
            raise ValueError("Normal multiplier checkpoint fields differ.")
        if snapshot["pair_identity"] != self.pair_identity:
            raise ValueError("Normal multiplier checkpoint contact pair differs.")
        if snapshot.get("trace_identity") != self.trace_identity:
            raise ValueError("Normal multiplier checkpoint trace basis differs.")
        if snapshot.get("surface_fingerprint") != self.surface_fingerprint:
            raise ValueError("Normal multiplier checkpoint rigid surface differs.")
        if not np.array_equal(_point_ids(snapshot.get("point_ids")), self.point_ids):
            raise ValueError("Normal multiplier checkpoint point IDs differ.")
        if float(snapshot.get("penalty")) != self.penalty:
            raise ValueError("Normal multiplier checkpoint penalty differs.")
        policy = (
            "penetration_tolerance", "complementarity_tolerance",
            "maximum_augmentations", "length_unit", "pressure_unit",
        )
        if any(snapshot.get(key) != getattr(self, key) for key in policy):
            raise ValueError("Normal multiplier checkpoint acceptance policy differs.")
        multiplier = _nonnegative_vector(
            snapshot.get("multiplier"), len(self.point_ids), name="Checkpoint multiplier",
        )
        self._accepted = multiplier.copy()
        self.rollback()


@dataclass(frozen=True, eq=False)
class AugmentedNormalContribution:
    """Reference-trace residual and fixed-multiplier Newton tangent.

    ``algorithmic_functional`` may be negative.  It is deliberately not named
    ``potential_energy`` and must not enter a physical energy ledger.
    """

    nodal_structural_residual: object
    contact_force_on_structure: object
    contact_force_on_surface: object
    surface_generalized_force: object
    surface_generalized_moment: object | None
    algorithmic_functional: float
    tangent_rows: object
    tangent_columns: object
    tangent_values: object
    active_points: int

    def __post_init__(self) -> None:
        nodal = np.asarray(self.nodal_structural_residual, dtype=float)
        if nodal.ndim != 2 or nodal.shape[1] not in {2, 3} or not np.all(np.isfinite(nodal)):
            raise ValueError("Augmented normal nodal residual must be finite 2D/3D vectors.")
        dimension = nodal.shape[1]
        structural = np.asarray(self.contact_force_on_structure, dtype=float)
        surface = np.asarray(self.contact_force_on_surface, dtype=float)
        generalized = np.asarray(self.surface_generalized_force, dtype=float)
        if (
            structural.shape != (dimension,) or surface.shape != (dimension,)
            or generalized.shape != (dimension,)
            or not np.all(np.isfinite(structural + surface + generalized))
        ):
            raise ValueError("Augmented normal force vectors have the wrong shape or value.")
        if (
            not np.allclose(structural, -np.sum(nodal, axis=0), rtol=1e-12, atol=1e-12)
            or not np.allclose(structural, -surface, rtol=1e-12, atol=1e-12)
            or not np.allclose(structural, generalized, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Augmented normal integrated action/reaction does not balance.")
        moment = self.surface_generalized_moment
        if moment is not None:
            selected = np.asarray(moment, dtype=float)
            if selected.shape != ((1,) if dimension == 2 else (3,)) or not np.all(np.isfinite(selected)):
                raise ValueError("Augmented normal moment has the wrong spatial shape.")
            object.__setattr__(self, "surface_generalized_moment", _readonly_array(selected))
        rows = np.asarray(self.tangent_rows, dtype=np.int64)
        columns = np.asarray(self.tangent_columns, dtype=np.int64)
        values = np.asarray(self.tangent_values, dtype=float)
        size = nodal.size
        if (rows.ndim != 1 or columns.shape != rows.shape or values.shape != rows.shape
                or np.any(rows < 0) or np.any(columns < 0) or np.any(rows >= size)
                or np.any(columns >= size) or not np.all(np.isfinite(values))):
            raise ValueError("Augmented normal tangent COO fields are invalid.")
        if not np.isfinite(self.algorithmic_functional) or self.active_points < 0:
            raise ValueError("Augmented normal functional or active count is invalid.")
        object.__setattr__(self, "nodal_structural_residual", _readonly_array(nodal))
        object.__setattr__(self, "contact_force_on_structure", _readonly_array(structural))
        object.__setattr__(self, "contact_force_on_surface", _readonly_array(surface))
        object.__setattr__(self, "surface_generalized_force", _readonly_array(generalized))
        object.__setattr__(self, "tangent_rows", _readonly_array(rows, dtype=np.int64))
        object.__setattr__(self, "tangent_columns", _readonly_array(columns, dtype=np.int64))
        object.__setattr__(self, "tangent_values", _readonly_array(values))

    def tangent_action(self, direction) -> np.ndarray:
        selected = np.asarray(direction, dtype=float)
        if selected.shape != self.nodal_structural_residual.shape or not np.all(np.isfinite(selected)):
            raise ValueError("Augmented normal direction must match the nodal layout.")
        action = np.bincount(
            self.tangent_rows,
            weights=self.tangent_values * selected.reshape(-1)[self.tangent_columns],
            minlength=selected.size,
        )
        return action.reshape(selected.shape)


def diagnose_fixed_trace_normal_constraint_rank(
    response: AugmentedNormalPointResponse, *, maximum_points: int = 256,
    maximum_dofs: int = 4096, relative_tolerance: float | None = None,
    displacement_reduction=None,
) -> dict[str, object]:
    """Audit active multiplier identifiability on a bounded fixed trace.

    The weighted gap Jacobian has one row per active contact point and one
    column per local displacement DOF. A deficient row rank means distinct
    point pressures can create the same nodal force. This small dense audit
    Optionally inspect ``J @ T`` where ``u = T q + u_bar`` is the *actual*
    Dirichlet/MPC displacement reduction. This remains a small dense negative
    screen: full row rank does not establish a mesh-uniform inf-sup constant.
    """

    if not isinstance(response, AugmentedNormalPointResponse):
        raise TypeError("Normal constraint rank requires a fixed-trace response.")
    if (
        isinstance(maximum_points, bool) or not isinstance(maximum_points, int)
        or maximum_points < 1 or isinstance(maximum_dofs, bool)
        or not isinstance(maximum_dofs, int) or maximum_dofs < 1
    ):
        raise ValueError("Normal constraint rank budgets must be positive integers.")
    trace = response.evaluation.trace
    active = np.flatnonzero(response.active)
    dimension = response.record.projection.dimension
    dofs = response.evaluation.number_of_nodes * dimension
    if len(active) > maximum_points or dofs > maximum_dofs:
        raise NotImplementedError(
            "AFM-CONTACT-MULTIPLIER-RANK: dense local rank audit exceeds "
            "its declared point/DOF budget; no stability claim was made."
        )
    jacobian = np.zeros((len(active), dofs), dtype=float)
    normals = response.record.projection.normals[active]
    for row, index in enumerate(active):
        scale = np.sqrt(trace.weights[index])
        for node, shape in zip(
            trace.node_ids[index], trace.shape_values[index], strict=True,
        ):
            begin = int(node) * dimension
            jacobian[row, begin:begin + dimension] += scale * shape * normals[row]
    if displacement_reduction is not None:
        reduction = np.asarray(displacement_reduction, dtype=float)
        if (
            reduction.ndim != 2 or reduction.shape[0] != dofs
            or reduction.shape[1] > maximum_dofs
            or not np.all(np.isfinite(reduction))
        ):
            raise ValueError(
                "Normal constraint rank reduction must be a finite dense "
                "full-DOF by reduced-DOF matrix within the declared budget."
        )
        jacobian = jacobian @ reduction
    if relative_tolerance is None:
        tolerance = max(jacobian.shape) * np.finfo(float).eps
    else:
        tolerance = float(relative_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Normal constraint rank tolerance must be finite and nonnegative.")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    rank = int(np.count_nonzero(singular > tolerance * largest)) if largest else 0
    return {
        "scope": (
            "bounded_active_fixed_trace_normal_constraint_jacobian"
            if displacement_reduction is None else
            "bounded_active_reduced_fixed_trace_normal_constraint_jacobian"
        ),
        "active_point_count": len(active),
        "displacement_dof_count": dofs,
        "reduced_displacement_dof_count": jacobian.shape[1],
        "displacement_reduction_applied": displacement_reduction is not None,
        "row_rank": rank,
        "row_nullity": len(active) - rank,
        "relative_tolerance": tolerance,
        "largest_singular_value": largest,
        "smallest_retained_singular_value": (
            float(singular[rank - 1]) if rank else None
        ),
        "certifies_spatial_stability": False,
    }


def integrate_fixed_trace_augmented_normal(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: AugmentedNormalPointResponse,
    gap_hessian,
    *,
    surface_reference_point=None,
) -> AugmentedNormalContribution:
    """Integrate one fixed-trace active branch without claiming physical energy."""

    if not isinstance(evaluation, ContactTraceEvaluation) or not isinstance(record, ContactProjectionRecord):
        raise TypeError("Augmented normal integration requires trace and projection evidence.")
    if (
        not isinstance(response, AugmentedNormalPointResponse)
        or response.record is not record
        or response.evaluation is not evaluation
    ):
        raise TypeError("Augmented normal integration requires one identical trace/projection trial.")
    trace = evaluation.trace
    projection = record.projection
    if trace.measure_configuration != "reference":
        raise NotImplementedError("Augmented normal integration requires reference-area weights.")
    if not np.array_equal(trace.point_ids, record.point_ids) or not np.array_equal(evaluation.query_points, projection.query_points):
        raise ValueError("Augmented normal evaluation and projection identities differ.")
    dimension = projection.dimension
    hessian = np.asarray(gap_hessian, dtype=float)
    if (hessian.shape != (trace.point_count, dimension, dimension)
            or not np.all(np.isfinite(hessian))
            or not np.allclose(hessian, hessian.transpose(0, 2, 1), rtol=1e-12, atol=1e-12)):
        raise ValueError("Augmented normal gap Hessian must be finite and symmetric.")

    point_residual = -trace.weights[:, None] * response.pressure[:, None] * projection.normals
    nodal = np.zeros((evaluation.number_of_nodes, dimension), dtype=float)
    for local in range(trace.nodes_per_point):
        np.add.at(
            nodal, trace.node_ids[:, local],
            trace.shape_values[:, local, None] * point_residual,
        )
    force_on_structure = -np.sum(point_residual, axis=0)
    moment = None
    if surface_reference_point is not None:
        reference = np.asarray(surface_reference_point, dtype=float)
        if reference.shape != (dimension,) or not np.all(np.isfinite(reference)):
            raise ValueError("Augmented normal moment reference point is invalid.")
        arm = projection.closest_points - reference
        point_surface_generalized = -point_residual
        if dimension == 2:
            moment = np.asarray([np.sum(
                arm[:, 0] * point_surface_generalized[:, 1]
                - arm[:, 1] * point_surface_generalized[:, 0]
            )])
        else:
            moment = np.sum(np.cross(arm, point_surface_generalized), axis=0)

    active = response.active
    nodes = trace.node_ids[active]
    if len(nodes):
        normal = projection.normals[active]
        pressure = response.pressure[active]
        point_tangent = (
            response.penalty * normal[:, :, None] * normal[:, None, :]
            - pressure[:, None, None] * hessian[active]
        )
        shape = trace.shape_values[active]
        local_tangent = np.einsum(
            "q,qa,qb,qij->qaibj", trace.weights[active], shape, shape, point_tangent,
        )
        dofs = (dimension * nodes[:, :, None] + np.arange(dimension)).reshape((len(nodes), -1))
        width = dofs.shape[1]
        rows = np.broadcast_to(dofs[:, :, None], (len(dofs), width, width)).reshape(-1)
        columns = np.broadcast_to(dofs[:, None, :], (len(dofs), width, width)).reshape(-1)
        values = local_tangent.reshape(-1)
    else:
        rows = np.empty(0, dtype=np.int64)
        columns = np.empty(0, dtype=np.int64)
        values = np.empty(0)
    return AugmentedNormalContribution(
        nodal_structural_residual=nodal,
        contact_force_on_structure=force_on_structure,
        contact_force_on_surface=-force_on_structure,
        surface_generalized_force=force_on_structure,
        surface_generalized_moment=moment,
        algorithmic_functional=float(np.dot(trace.weights, response.algorithmic_density)),
        tangent_rows=rows,
        tangent_columns=columns,
        tangent_values=values,
        active_points=int(np.count_nonzero(active)),
    )


__all__ = [
    "AugmentedNormalPointResponse",
    "AugmentedNormalContribution",
    "FixedTraceNormalMultiplierState",
    "diagnose_fixed_trace_normal_constraint_rank",
    "integrate_fixed_trace_augmented_normal",
]
