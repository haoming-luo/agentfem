# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral penalty Coulomb friction and accepted tangential state.

The objects in this module operate on already projected contact points.  They
do not search a surface, integrate a finite-element trace, choose a time step,
or provide an implicit tangent.  Those responsibilities remain with the
Backend, Operator, and Procedure layers respectively.

The state contract is deliberately explicit about three different energies:
recoverable tangential penalty energy, irreversible frictional dissipation,
and penalty energy released when contact opens.  Treating all three as
"friction work" would make an energy audit look complete while hiding a
numerical loss at separation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contact_response import FrictionlessPenaltyContactResponse, _penalty_values
from .contact_state import ContactProjectionRecord
from .rigid import PrescribedRigidMotion, _readonly_array


_RECORD_SCHEMA = "agentfem.tangential-contact-record.v1"
_STATE_SCHEMA = "agentfem.tangential-contact-state.v1"
_KINEMATIC_RECORD_SCHEMA = "agentfem.tangential-kinematic-record.v1"
_KINEMATIC_STATE_SCHEMA = "agentfem.tangential-kinematic-state.v1"


@dataclass(frozen=True, eq=False)
class TangentialKinematicRecord:
    """Accepted slave positions and rigid-motion coordinate by stable ID."""

    point_ids: object
    slave_positions: object
    motion_factor: float = 0.0

    def __post_init__(self) -> None:
        positions = np.asarray(self.slave_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
            raise ValueError("Tangential kinematics require 2D or 3D slave positions.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("Tangential slave positions must be finite.")
        ids, order = _canonical_point_order(self.point_ids, positions.shape[0])
        factor = float(self.motion_factor)
        if not np.isfinite(factor) or not 0.0 <= factor <= 1.0 + 1.0e-12:
            raise ValueError("Tangential kinematic motion factor must lie in [0, 1].")
        object.__setattr__(self, "point_ids", _readonly_array(ids, dtype=np.int64))
        object.__setattr__(self, "slave_positions", _readonly_array(positions[order]))
        object.__setattr__(self, "motion_factor", factor)

    @property
    def point_count(self) -> int:
        return int(self.point_ids.size)

    @property
    def dimension(self) -> int:
        return int(self.slave_positions.shape[1])

    @classmethod
    def from_projection(
        cls,
        record: ContactProjectionRecord,
        *,
        motion_factor: float = 0.0,
    ) -> TangentialKinematicRecord:
        if not isinstance(record, ContactProjectionRecord):
            raise TypeError("Tangential kinematics require ContactProjectionRecord.")
        return cls(
            point_ids=record.point_ids,
            slave_positions=record.projection.query_points,
            motion_factor=motion_factor,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": _KINEMATIC_RECORD_SCHEMA,
            "point_ids": self.point_ids.tolist(),
            "slave_positions": self.slave_positions.tolist(),
            "motion_factor": self.motion_factor,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> TangentialKinematicRecord:
        required = {"schema", "point_ids", "slave_positions", "motion_factor"}
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != required
            or snapshot.get("schema") != _KINEMATIC_RECORD_SCHEMA
        ):
            raise ValueError("Unsupported tangential kinematic record snapshot.")
        return cls(
            point_ids=snapshot["point_ids"],
            slave_positions=snapshot["slave_positions"],
            motion_factor=snapshot["motion_factor"],
        )


@dataclass
class TangentialKinematicState:
    """Atomic accepted/trial kinematics for explicit friction increments."""

    accepted: TangentialKinematicRecord | None = None
    trial: TangentialKinematicRecord | None = None

    def begin(
        self,
        projection: ContactProjectionRecord,
        *,
        motion_factor: float = 0.0,
    ) -> TangentialKinematicRecord:
        candidate = TangentialKinematicRecord.from_projection(
            projection,
            motion_factor=motion_factor,
        )
        if self.accepted is not None:
            if not np.array_equal(self.accepted.point_ids, candidate.point_ids):
                raise ValueError("Tangential kinematic point identity changed.")
            if self.accepted.dimension != candidate.dimension:
                raise ValueError("Tangential kinematic dimension changed.")
            if candidate.motion_factor + 1.0e-14 < self.accepted.motion_factor:
                raise ValueError("Tangential rigid-motion factor moved backward.")
        self.trial = candidate
        return candidate

    def commit(self) -> None:
        if self.trial is None:
            raise RuntimeError("No tangential kinematic trial is available to commit.")
        self.accepted = self.trial
        self.trial = None

    def rollback(self) -> None:
        self.trial = None

    def snapshot(self) -> dict[str, object]:
        if self.trial is not None:
            raise RuntimeError("Tangential kinematics require an accepted checkpoint.")
        return {
            "schema": _KINEMATIC_STATE_SCHEMA,
            "accepted": None if self.accepted is None else self.accepted.snapshot(),
        }

    def restore(self, snapshot: object) -> None:
        if self.trial is not None:
            raise RuntimeError("Rollback tangential kinematics before restore.")
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != {"schema", "accepted"}
            or snapshot.get("schema") != _KINEMATIC_STATE_SCHEMA
        ):
            raise ValueError("Unsupported tangential kinematic state snapshot.")
        raw = snapshot["accepted"]
        self.accepted = (
            None if raw is None else TangentialKinematicRecord.from_snapshot(raw)
        )


def relative_contact_displacement_increment(
    accepted: ContactProjectionRecord | TangentialKinematicRecord,
    current: ContactProjectionRecord,
    *,
    motion: PrescribedRigidMotion | None = None,
    accepted_motion_factor: float = 0.0,
    current_motion_factor: float = 0.0,
) -> np.ndarray:
    """Return slave-minus-master increments at stable contact-point IDs.

    The current closest point is pulled back to the rigid surface reference
    configuration and pushed to the accepted configuration. This follows the
    same master material point even when finite sliding changes the contacted
    facet. Subtracting two gap vectors would erase common tangential motion.
    """

    if not isinstance(
        accepted, (ContactProjectionRecord, TangentialKinematicRecord)
    ) or not isinstance(current, ContactProjectionRecord):
        raise TypeError(
            "Relative contact motion requires accepted and current projection records."
        )
    if not np.array_equal(accepted.point_ids, current.point_ids):
        raise ValueError("Relative contact motion point identity differs.")
    new = current.projection
    if isinstance(accepted, ContactProjectionRecord):
        old_positions = accepted.projection.query_points
        old_dimension = accepted.projection.dimension
        old_surface_name = accepted.projection.surface_name
        old_surface_kind = accepted.projection.surface_kind
        old_fingerprint = accepted.projection.geometry_fingerprint
        accepted_factor_default = accepted_motion_factor
        if not np.all(accepted.projection.valid):
            raise ValueError(
                "Relative contact motion requires valid accepted/current points."
            )
    else:
        old_positions = accepted.slave_positions
        old_dimension = accepted.dimension
        old_surface_name = new.surface_name
        old_surface_kind = new.surface_kind
        old_fingerprint = new.geometry_fingerprint
        accepted_factor_default = accepted.motion_factor
    if old_dimension != new.dimension:
        raise ValueError("Relative contact motion dimension differs.")
    if old_surface_name != new.surface_name or old_surface_kind != new.surface_kind:
        raise ValueError("Relative contact motion surface identity differs.")
    if old_fingerprint != new.geometry_fingerprint:
        raise ValueError("Relative contact motion geometry identity differs.")
    if not np.all(new.valid):
        raise ValueError("Relative contact motion requires valid accepted/current points.")

    dimension = new.dimension
    accepted_factor = float(accepted_factor_default)
    current_factor = float(current_motion_factor)
    if not np.isfinite(accepted_factor) or not np.isfinite(current_factor):
        raise ValueError("Rigid-motion factors must be finite.")
    if motion is None:
        if accepted_factor != 0.0 or current_factor != 0.0:
            raise ValueError("Fixed contact cannot use non-zero rigid-motion factors.")
        master_increment = np.zeros_like(new.closest_points)
    else:
        if not isinstance(motion, PrescribedRigidMotion):
            raise TypeError("Relative contact motion requires PrescribedRigidMotion.")
        if motion.dimension != dimension:
            raise ValueError("Rigid motion and contact projection dimensions differ.")
        old_state = motion.state(accepted_factor)
        new_state = motion.state(current_factor)
        reference = np.asarray(motion.reference_point, dtype=float)
        old_reference = np.asarray(old_state["reference_point"], dtype=float)
        new_reference = np.asarray(new_state["reference_point"], dtype=float)
        old_rotation = np.asarray(old_state["rotation_matrix"], dtype=float)
        new_rotation = np.asarray(new_state["rotation_matrix"], dtype=float)
        # Row-vector form of x = r(f) + R(f) (X-r0).
        material_reference_points = reference + (
            np.asarray(new.closest_points) - new_reference
        ) @ new_rotation
        previous_master_points = old_reference + (
            material_reference_points - reference
        ) @ old_rotation.T
        master_increment = np.asarray(new.closest_points) - previous_master_points

    slave_increment = np.asarray(new.query_points) - np.asarray(old_positions)
    return _readonly_array(slave_increment - master_increment)


def _canonical_point_order(point_ids, point_count: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(point_ids)
    if raw.ndim != 1 or raw.shape != (point_count,):
        raise ValueError("Tangential contact requires one point ID per point.")
    if raw.size and raw.dtype.kind not in {"i", "u"}:
        raise TypeError("Tangential contact point IDs must be explicit integers.")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("Tangential contact point IDs must fit signed 64-bit identity.")
    selected = raw.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.unique(selected).size != selected.size:
        raise ValueError("Tangential contact point IDs must be unique and non-negative.")
    order = np.argsort(selected, kind="stable")
    return selected[order], order


def _unit_normals(value, *, point_count: int, dimension: int) -> np.ndarray:
    normals = np.asarray(value, dtype=float)
    if normals.shape != (point_count, dimension) or not np.all(np.isfinite(normals)):
        raise ValueError("Contact normals must contain one finite vector per point.")
    lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(lengths, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Contact normals must be unit vectors.")
    return normals


def _transport_tangent_vectors(
    vectors: np.ndarray,
    old_normals: np.ndarray,
    new_normals: np.ndarray,
) -> np.ndarray:
    """Use the minimal normal rotation to preserve tangential vector length."""

    dimension = int(vectors.shape[1])
    if dimension not in {2, 3}:
        raise ValueError("Tangential history transport supports two or three dimensions.")
    transported = np.empty_like(vectors)
    tolerance = 128.0 * np.finfo(float).eps
    for index, (vector, old_normal, new_normal) in enumerate(
        zip(vectors, old_normals, new_normals, strict=True)
    ):
        cosine = float(np.clip(np.dot(old_normal, new_normal), -1.0, 1.0))
        if cosine < -1.0 + 1.0e-10:
            raise ValueError(
                "Contact normal reversed by approximately 180 degrees; tangential "
                "history transport is ambiguous and must not be guessed."
            )
        if dimension == 2:
            sine = float(
                old_normal[0] * new_normal[1]
                - old_normal[1] * new_normal[0]
            )
            rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=float)
            candidate = rotation @ vector
        else:
            axis = np.cross(old_normal, new_normal)
            sine = float(np.linalg.norm(axis))
            if sine <= tolerance:
                candidate = vector.copy()
            else:
                skew = np.array(
                    (
                        (0.0, -axis[2], axis[1]),
                        (axis[2], 0.0, -axis[0]),
                        (-axis[1], axis[0], 0.0),
                    ),
                    dtype=float,
                )
                rotation = (
                    np.eye(3)
                    + skew
                    + skew @ skew * ((1.0 - cosine) / (sine * sine))
                )
                candidate = rotation @ vector
        candidate -= np.dot(candidate, new_normal) * new_normal
        transported[index] = candidate
    return transported


@dataclass(frozen=True, eq=False)
class TangentialContactRecord:
    """Accepted tangential history keyed by stable contact-point identity."""

    point_ids: object
    normals: object
    elastic_slips: object
    cumulative_dissipation_densities: object
    cumulative_separation_release_densities: object

    def __post_init__(self) -> None:
        slips = np.asarray(self.elastic_slips, dtype=float)
        if slips.ndim != 2:
            raise ValueError("Tangential elastic slips must be a point-vector array.")
        count, dimension = slips.shape
        if dimension not in {2, 3} or not np.all(np.isfinite(slips)):
            raise ValueError("Tangential elastic slips must be finite 2D or 3D vectors.")
        point_ids, order = _canonical_point_order(self.point_ids, count)
        normals = _unit_normals(self.normals, point_count=count, dimension=dimension)[order]
        slips = slips[order]
        normal_components = np.einsum("ij,ij->i", slips, normals)
        if not np.allclose(normal_components, 0.0, rtol=0.0, atol=1.0e-11):
            raise ValueError("Stored tangential elastic slips must lie in the tangent plane.")
        cumulative = {}
        for name in (
            "cumulative_dissipation_densities",
            "cumulative_separation_release_densities",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (count,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} must contain one finite value per contact point."
                )
            value = value[order]
            if np.any(value < 0.0):
                raise ValueError(f"{name} cannot be negative.")
            cumulative[name] = value
        object.__setattr__(self, "point_ids", _readonly_array(point_ids, dtype=np.int64))
        object.__setattr__(self, "normals", _readonly_array(normals))
        object.__setattr__(self, "elastic_slips", _readonly_array(slips))
        for name, value in cumulative.items():
            object.__setattr__(self, name, _readonly_array(value))

    @property
    def point_count(self) -> int:
        return int(self.point_ids.size)

    @property
    def dimension(self) -> int:
        return int(self.elastic_slips.shape[1])

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": _RECORD_SCHEMA,
            "point_ids": self.point_ids.copy(),
            "normals": self.normals.copy(),
            "elastic_slips": self.elastic_slips.copy(),
            "cumulative_dissipation_densities": (
                self.cumulative_dissipation_densities.copy()
            ),
            "cumulative_separation_release_densities": (
                self.cumulative_separation_release_densities.copy()
            ),
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> TangentialContactRecord:
        if not isinstance(snapshot, Mapping):
            raise TypeError("Tangential contact record snapshot must be a mapping.")
        required = {
            "schema",
            "point_ids",
            "normals",
            "elastic_slips",
            "cumulative_dissipation_densities",
            "cumulative_separation_release_densities",
        }
        if set(snapshot) != required or snapshot.get("schema") != _RECORD_SCHEMA:
            raise ValueError("Unsupported tangential contact record snapshot.")
        return cls(
            point_ids=snapshot["point_ids"],
            normals=snapshot["normals"],
            elastic_slips=snapshot["elastic_slips"],
            cumulative_dissipation_densities=snapshot[
                "cumulative_dissipation_densities"
            ],
            cumulative_separation_release_densities=snapshot[
                "cumulative_separation_release_densities"
            ],
        )


@dataclass(frozen=True, eq=False)
class PenaltyCoulombFrictionResponse:
    """One canonical pointwise return-map result without FE integration."""

    record: TangentialContactRecord
    active: object
    sticking: object
    sliding: object
    friction_limits: object
    relative_tangential_increments: object
    plastic_slip_increments: object
    structural_residual_tractions: object
    surface_generalized_tractions: object
    recoverable_penalty_energy_densities: object
    dissipation_increment_densities: object
    separation_release_densities: object

    def __post_init__(self) -> None:
        if not isinstance(self.record, TangentialContactRecord):
            raise TypeError("Friction response requires TangentialContactRecord.")
        count = self.record.point_count
        dimension = self.record.dimension
        for name in ("active", "sticking", "sliding"):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != (count,):
                raise ValueError(f"{name} must contain one flag per point.")
            object.__setattr__(self, name, _readonly_array(value, dtype=bool))
        scalar_names = (
            "friction_limits",
            "recoverable_penalty_energy_densities",
            "dissipation_increment_densities",
            "separation_release_densities",
        )
        for name in scalar_names:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (count,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain one finite value per point.")
            if np.any(value < 0.0):
                raise ValueError(f"{name} cannot be negative.")
            object.__setattr__(self, name, _readonly_array(value))
        vector_names = (
            "relative_tangential_increments",
            "plastic_slip_increments",
            "structural_residual_tractions",
            "surface_generalized_tractions",
        )
        for name in vector_names:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (count, dimension) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain one finite vector per point.")
            object.__setattr__(self, name, _readonly_array(value))
        if np.any(self.sticking & self.sliding) or not np.array_equal(
            self.active,
            self.sticking | self.sliding,
        ):
            raise ValueError("Friction stick/slip flags are inconsistent.")
        if not np.allclose(
            self.structural_residual_tractions,
            -self.surface_generalized_tractions,
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        ):
            raise ValueError("Friction action and reaction tractions do not balance.")
        traction_norms = np.linalg.norm(self.structural_residual_tractions, axis=1)
        tolerance = 128.0 * np.finfo(float).eps * np.maximum(
            1.0,
            self.friction_limits,
        )
        if np.any(traction_norms > self.friction_limits + tolerance):
            raise ValueError("Friction traction exceeds the Coulomb cap.")
        if not np.allclose(
            self.plastic_slip_increments[self.sticking],
            0.0,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        ):
            raise ValueError("Sticking points cannot accumulate plastic slip.")
        if not np.allclose(
            traction_norms[self.sliding],
            self.friction_limits[self.sliding],
            rtol=1.0e-10,
            atol=64.0 * np.finfo(float).eps,
        ):
            raise ValueError("Sliding tractions must lie on the Coulomb cap.")
        if not np.allclose(
            self.structural_residual_tractions[~self.active],
            0.0,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        ):
            raise ValueError("Inactive contact points cannot carry friction traction.")

    @property
    def point_count(self) -> int:
        return self.record.point_count

    @property
    def contact_tractions_on_structure(self) -> np.ndarray:
        return -self.structural_residual_tractions

    def summary(self) -> dict[str, object]:
        return {
            "kind": "penalty_coulomb_friction_response",
            "point_count": self.point_count,
            "active_count": int(np.count_nonzero(self.active)),
            "sticking_count": int(np.count_nonzero(self.sticking)),
            "sliding_count": int(np.count_nonzero(self.sliding)),
            "response_level": "pointwise_unintegrated",
            "recoverable_energy": "tangential_penalty",
            "irreversible_energy": "coulomb_sliding_dissipation",
            "separation_release": "reported_separately",
            "linearization": "not_provided",
        }


@dataclass(frozen=True, eq=False)
class PenaltyCoulombFrictionLaw:
    """Elastic-stick/return-to-Coulomb-cap point law for explicit contact."""

    coefficient: float
    tangential_penalty: object
    name: str = "penalty_coulomb_friction"

    def __post_init__(self) -> None:
        coefficient = float(self.coefficient)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("Friction coefficient must be finite and non-negative.")
        raw = np.asarray(self.tangential_penalty, dtype=float)
        if raw.ndim > 1 or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
            raise ValueError(
                "Tangential contact penalty must be finite, positive, and at most 1D."
            )
        if not str(self.name).strip():
            raise ValueError("Friction law requires a name.")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(
            self,
            "tangential_penalty",
            float(raw) if raw.ndim == 0 else _readonly_array(raw.reshape(-1)),
        )
        object.__setattr__(self, "name", str(self.name))

    def evaluate(
        self,
        *,
        point_ids,
        normal_response: FrictionlessPenaltyContactResponse,
        relative_displacement_increment,
        accepted: TangentialContactRecord | None = None,
    ) -> PenaltyCoulombFrictionResponse:
        """Return one trial update while leaving accepted history untouched."""

        if not isinstance(normal_response, FrictionlessPenaltyContactResponse):
            raise TypeError("Friction requires a frictionless normal-contact response.")
        count = normal_response.point_count
        dimension = normal_response.dimension
        canonical_ids, order = _canonical_point_order(point_ids, count)
        increments = np.asarray(relative_displacement_increment, dtype=float)
        if increments.shape != (count, dimension) or not np.all(np.isfinite(increments)):
            raise ValueError(
                "Relative displacement increments must contain one finite vector per point."
            )
        increments = increments[order]
        normals = np.asarray(normal_response.projection.normals, dtype=float)[order]
        active = np.asarray(normal_response.active, dtype=bool)[order]
        pressures = np.asarray(normal_response.pressures, dtype=float)[order]
        penalties = _penalty_values(self.tangential_penalty, count)[order]

        tangential_increments = increments - np.einsum(
            "ij,ij->i",
            increments,
            normals,
        )[:, None] * normals
        if accepted is None:
            old_slips = np.zeros((count, dimension), dtype=float)
            cumulative = np.zeros(count, dtype=float)
            cumulative_release = np.zeros(count, dtype=float)
        else:
            if not isinstance(accepted, TangentialContactRecord):
                raise TypeError("Accepted friction state has the wrong type.")
            if not np.array_equal(accepted.point_ids, canonical_ids):
                raise ValueError("Accepted friction point identity differs.")
            if accepted.dimension != dimension:
                raise ValueError("Accepted friction state dimension differs.")
            old_slips = _transport_tangent_vectors(
                np.asarray(accepted.elastic_slips),
                np.asarray(accepted.normals),
                normals,
            )
            cumulative = np.asarray(
                accepted.cumulative_dissipation_densities,
                dtype=float,
            )
            cumulative_release = np.asarray(
                accepted.cumulative_separation_release_densities,
                dtype=float,
            )

        trial_slips = old_slips + tangential_increments
        trial_norms = np.linalg.norm(trial_slips, axis=1)
        limits = self.coefficient * pressures
        trial_traction_norms = penalties * trial_norms
        tolerance = 128.0 * np.finfo(float).eps * np.maximum(1.0, limits)
        sticking = active & (trial_traction_norms <= limits + tolerance)
        sliding = active & ~sticking

        new_slips = np.zeros_like(trial_slips)
        new_slips[sticking] = trial_slips[sticking]
        nonzero_sliding = sliding & (trial_norms > 0.0)
        new_slips[nonzero_sliding] = (
            limits[nonzero_sliding, None]
            / penalties[nonzero_sliding, None]
            * trial_slips[nonzero_sliding]
            / trial_norms[nonzero_sliding, None]
        )
        plastic_increment = np.zeros_like(trial_slips)
        plastic_increment[sliding] = trial_slips[sliding] - new_slips[sliding]

        residual_tractions = penalties[:, None] * new_slips
        surface_tractions = -residual_tractions
        recoverable = 0.5 * penalties * np.einsum(
            "ij,ij->i",
            new_slips,
            new_slips,
        )
        dissipation_increment = limits * np.linalg.norm(plastic_increment, axis=1)
        separation_release = np.zeros(count, dtype=float)
        separation_release[~active] = 0.5 * penalties[~active] * np.einsum(
            "ij,ij->i",
            old_slips[~active],
            old_slips[~active],
        )
        record = TangentialContactRecord(
            point_ids=canonical_ids,
            normals=normals,
            elastic_slips=new_slips,
            cumulative_dissipation_densities=cumulative + dissipation_increment,
            cumulative_separation_release_densities=(
                cumulative_release + separation_release
            ),
        )
        return PenaltyCoulombFrictionResponse(
            record=record,
            active=active,
            sticking=sticking,
            sliding=sliding,
            friction_limits=limits,
            relative_tangential_increments=tangential_increments,
            plastic_slip_increments=plastic_increment,
            structural_residual_tractions=residual_tractions,
            surface_generalized_tractions=surface_tractions,
            recoverable_penalty_energy_densities=recoverable,
            dissipation_increment_densities=dissipation_increment,
            separation_release_densities=separation_release,
        )

    def summary(self) -> dict[str, object]:
        penalty = np.asarray(self.tangential_penalty, dtype=float)
        return {
            "name": self.name,
            "kind": "penalty_coulomb_friction_law",
            "coefficient": self.coefficient,
            "tangential_penalty": (
                float(penalty) if penalty.ndim == 0 else penalty.tolist()
            ),
            "state": "tangential_elastic_slip",
            "normal_limit": "coefficient * normal_pressure",
            "integration": "backend_required",
            "linearization": "not_provided",
            "intended_procedure": "explicit",
        }


@dataclass
class TangentialContactState:
    """Atomic accepted/trial state for one frictional contact pair."""

    accepted: TangentialContactRecord | None = None
    trial: TangentialContactRecord | None = None

    def __post_init__(self) -> None:
        if self.accepted is not None and not isinstance(
            self.accepted,
            TangentialContactRecord,
        ):
            raise TypeError("Accepted tangential contact state has the wrong type.")
        if self.trial is not None:
            raise ValueError("Construct friction state at an accepted boundary.")

    def begin(
        self,
        response: PenaltyCoulombFrictionResponse,
    ) -> TangentialContactRecord:
        if not isinstance(response, PenaltyCoulombFrictionResponse):
            raise TypeError("Friction trial requires PenaltyCoulombFrictionResponse.")
        candidate = response.record
        if self.accepted is not None:
            if not np.array_equal(self.accepted.point_ids, candidate.point_ids):
                raise ValueError("Tangential contact point identity changed.")
            if self.accepted.dimension != candidate.dimension:
                raise ValueError("Tangential contact state dimension changed.")
        self.trial = candidate
        return candidate

    def commit(self) -> None:
        if self.trial is None:
            raise RuntimeError("No tangential contact trial is available to commit.")
        self.accepted = self.trial
        self.trial = None

    def rollback(self) -> None:
        self.trial = None

    def snapshot(self) -> dict[str, object]:
        if self.trial is not None:
            raise RuntimeError(
                "Tangential contact checkpoints require an accepted boundary."
            )
        return {
            "schema": _STATE_SCHEMA,
            "accepted": None if self.accepted is None else self.accepted.snapshot(),
        }

    def restore(self, snapshot: object) -> None:
        if self.trial is not None:
            raise RuntimeError("Rollback the active friction trial before restore.")
        if not isinstance(snapshot, Mapping):
            raise TypeError("Tangential contact state snapshot must be a mapping.")
        if set(snapshot) != {"schema", "accepted"} or snapshot.get(
            "schema"
        ) != _STATE_SCHEMA:
            raise ValueError("Unsupported tangential contact state snapshot.")
        raw = snapshot["accepted"]
        restored = None if raw is None else TangentialContactRecord.from_snapshot(raw)
        if self.accepted is not None and restored is None:
            raise ValueError("Restored tangential contact state is unexpectedly empty.")
        if self.accepted is not None and restored is not None:
            if not np.array_equal(self.accepted.point_ids, restored.point_ids):
                raise ValueError("Restored tangential contact point identity differs.")
            if self.accepted.dimension != restored.dimension:
                raise ValueError("Restored tangential contact dimension differs.")
        self.accepted = restored

    def summary(self) -> dict[str, object]:
        return {
            "kind": "tangential_contact_state",
            "point_count": 0 if self.accepted is None else self.accepted.point_count,
            "trial_available": self.trial is not None,
            "transaction": "trial_commit_rollback",
            "checkpoint_boundary": "accepted_only",
        }


def penalty_coulomb_friction_law(
    coefficient: float,
    tangential_penalty,
    *,
    name: str = "penalty_coulomb_friction",
) -> PenaltyCoulombFrictionLaw:
    return PenaltyCoulombFrictionLaw(
        coefficient=coefficient,
        tangential_penalty=tangential_penalty,
        name=name,
    )


def tangential_contact_state() -> TangentialContactState:
    return TangentialContactState()


__all__ = [
    "PenaltyCoulombFrictionLaw",
    "PenaltyCoulombFrictionResponse",
    "TangentialContactRecord",
    "TangentialContactState",
    "TangentialKinematicRecord",
    "TangentialKinematicState",
    "penalty_coulomb_friction_law",
    "relative_contact_displacement_increment",
    "tangential_contact_state",
]
