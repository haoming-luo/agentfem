# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit-dynamics residual adapter for the reviewed contact stack.

This module is deliberately a narrow Procedure consumer of the backend-neutral
surface, projection, response, trace, state, and accepted-work contracts. It
adds normal penalty contact and optional penalty Coulomb friction to an already
assembled DOLFINx residual. An implicit tangent remains a separate gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import operators

from .contact_lifecycle import ContactProjectionLifecycle
from .contact_projection_checkpoint import (
    global_projection_state_snapshot,
    local_projection_state_from_snapshot,
)
from .contact_friction import (
    PenaltyCoulombFrictionLaw,
    TangentialContactState,
    TangentialKinematicState,
    relative_contact_displacement_increment,
)
from .contact_friction_checkpoint import (
    global_friction_state_snapshot,
    local_friction_state_from_snapshot,
)
from .contact_pair import RigidContactPair
from .contact_response import (
    FrictionlessPenaltyContactLaw,
    frictionless_penalty_contact_law,
)
from .contact_stability import (
    CombinedExplicitStabilityEstimate,
    ContactStabilityEstimate,
    combine_explicit_stability_bounds,
)
from .contact_trace import ContactTraceAssembly
from .contact_work import (
    PrescribedContactWorkState,
    PrescribedContactWorkStation,
    PrescribedRigidMotionSchedule,
)
from .dolfinx_contact_trace import DolfinxContactTraceAdapter
from .dolfinx_contact_stability import (
    estimate_dolfinx_contact_stability,
)
from .rigid import TriangulatedRigidSurface, _readonly_array
from .rigid_body import RigidBody
from .search import (
    partition_triangle_surface,
    routed_distributed_triangle_surface_bvh,
    triangle_surface_bvh,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, eq=False)
class ExplicitContactEvidence:
    """MPI-global evidence from one contact residual evaluation."""

    potential_energy: float
    contact_force_on_structure: object
    contact_force_on_surface: object
    surface_generalized_moment: object | None
    active_point_count: int
    invalid_point_count: int
    maximum_penetration: float
    normal_potential_energy: float = 0.0
    tangential_potential_energy: float = 0.0
    friction_dissipation: float = 0.0
    separation_release: float = 0.0
    sticking_point_count: int = 0
    sliding_point_count: int = 0

    def __post_init__(self) -> None:
        potential = float(self.potential_energy)
        if not np.isfinite(potential) or potential < 0.0:
            raise ValueError("Contact potential energy must be finite and non-negative.")
        structure = np.asarray(self.contact_force_on_structure, dtype=float)
        surface = np.asarray(self.contact_force_on_surface, dtype=float)
        if structure.shape != (3,) or surface.shape != (3,):
            raise ValueError("Explicit contact resultants must be three-dimensional.")
        if not np.all(np.isfinite(structure)) or not np.all(np.isfinite(surface)):
            raise ValueError("Explicit contact resultants must be finite.")
        if not np.allclose(
            structure + surface,
            0.0,
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Explicit contact action and reaction do not balance.")
        moment = self.surface_generalized_moment
        if moment is not None:
            moment = np.asarray(moment, dtype=float)
            if moment.shape != (3,) or not np.all(np.isfinite(moment)):
                raise ValueError("Explicit contact moment must be one finite 3-vector.")
            object.__setattr__(self, "surface_generalized_moment", _readonly_array(moment))
        active = int(self.active_point_count)
        invalid = int(self.invalid_point_count)
        if active < 0 or invalid < 0:
            raise ValueError("Explicit contact point counts must be non-negative.")
        penetration = float(self.maximum_penetration)
        if not np.isfinite(penetration) or penetration < 0.0:
            raise ValueError(
                "Explicit contact maximum penetration must be finite and non-negative."
            )
        normal_potential = float(self.normal_potential_energy)
        tangential_potential = float(self.tangential_potential_energy)
        dissipation = float(self.friction_dissipation)
        release = float(self.separation_release)
        for label, value in (
            ("normal potential energy", normal_potential),
            ("tangential potential energy", tangential_potential),
            ("friction dissipation", dissipation),
            ("separation release", release),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Explicit contact {label} must be non-negative.")
        if not np.isclose(
            potential,
            normal_potential + tangential_potential,
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Explicit contact potential-energy channels do not sum.")
        sticking = int(self.sticking_point_count)
        sliding = int(self.sliding_point_count)
        if sticking < 0 or sliding < 0 or sticking + sliding > active:
            raise ValueError("Explicit contact stick/slip counts are inconsistent.")
        object.__setattr__(self, "potential_energy", potential)
        object.__setattr__(
            self,
            "contact_force_on_structure",
            _readonly_array(structure),
        )
        object.__setattr__(
            self,
            "contact_force_on_surface",
            _readonly_array(surface),
        )
        object.__setattr__(self, "active_point_count", active)
        object.__setattr__(self, "invalid_point_count", invalid)
        object.__setattr__(self, "maximum_penetration", penetration)
        object.__setattr__(self, "normal_potential_energy", normal_potential)
        object.__setattr__(self, "tangential_potential_energy", tangential_potential)
        object.__setattr__(self, "friction_dissipation", dissipation)
        object.__setattr__(self, "separation_release", release)
        object.__setattr__(self, "sticking_point_count", sticking)
        object.__setattr__(self, "sliding_point_count", sliding)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "explicit_contact_evidence",
            "potential_energy": self.potential_energy,
            "contact_force_on_structure": self.contact_force_on_structure.tolist(),
            "contact_force_on_surface": self.contact_force_on_surface.tolist(),
            "surface_generalized_moment": (
                None
                if self.surface_generalized_moment is None
                else self.surface_generalized_moment.tolist()
            ),
            "active_point_count": self.active_point_count,
            "invalid_point_count": self.invalid_point_count,
            "maximum_penetration": self.maximum_penetration,
            "normal_potential_energy": self.normal_potential_energy,
            "tangential_potential_energy": self.tangential_potential_energy,
            "friction_dissipation": self.friction_dissipation,
            "separation_release": self.separation_release,
            "sticking_point_count": self.sticking_point_count,
            "sliding_point_count": self.sliding_point_count,
        }

    @classmethod
    def from_summary(cls, summary: object) -> ExplicitContactEvidence:
        required = {
            "kind",
            "potential_energy",
            "contact_force_on_structure",
            "contact_force_on_surface",
            "surface_generalized_moment",
            "active_point_count",
            "invalid_point_count",
            "maximum_penetration",
            "normal_potential_energy",
            "tangential_potential_energy",
            "friction_dissipation",
            "separation_release",
            "sticking_point_count",
            "sliding_point_count",
        }
        if (
            not isinstance(summary, dict)
            or summary.get("kind") != "explicit_contact_evidence"
            or set(summary) != required
        ):
            raise ValueError("Unsupported explicit-contact evidence snapshot.")
        return cls(
            potential_energy=summary["potential_energy"],
            contact_force_on_structure=summary["contact_force_on_structure"],
            contact_force_on_surface=summary["contact_force_on_surface"],
            surface_generalized_moment=summary["surface_generalized_moment"],
            active_point_count=summary["active_point_count"],
            invalid_point_count=summary["invalid_point_count"],
            maximum_penetration=summary["maximum_penetration"],
            normal_potential_energy=summary["normal_potential_energy"],
            tangential_potential_energy=summary["tangential_potential_energy"],
            friction_dissipation=summary["friction_dissipation"],
            separation_release=summary["separation_release"],
            sticking_point_count=summary["sticking_point_count"],
            sliding_point_count=summary["sliding_point_count"],
        )


def _reviewed_projector_for_rigid_surface(surface, communicator):
    """Select search infrastructure without moving it into the Model asset."""

    if not isinstance(surface, TriangulatedRigidSurface):
        return surface
    if int(communicator.size) == 1:
        return triangle_surface_bvh(surface)
    partition = partition_triangle_surface(surface, communicator)
    return routed_distributed_triangle_surface_bvh(partition, communicator)


class DolfinxExplicitContactResidual:
    """Add reviewed trace contact to a DOLFINx explicit residual.

    The adapter owns no time integration.  ``ExplicitDynamicsStep`` evaluates
    it after displacement prediction and kinematic projection, then accepts or
    rejects its projection trial with the surrounding increment.  Its contact
    stability ceiling is either declared by the caller or produced by adding
    the trace/mass contact spectral contribution to an independent non-contact
    spectral bound. A penalty spring can reduce the central-difference
    critical time step below the body-wave estimate.
    """

    def __init__(
        self,
        base,
        *,
        adapter: DolfinxContactTraceAdapter,
        displacement,
        lifecycle: ContactProjectionLifecycle,
        law: FrictionlessPenaltyContactLaw,
        friction_law: PenaltyCoulombFrictionLaw | None = None,
        maximum_stable_time_increment: float,
        contact_stability_estimate: ContactStabilityEstimate | None = None,
        combined_stability_estimate: CombinedExplicitStabilityEstimate | None = None,
        declared_maximum_stable_time_increment: float | None = None,
        contact_pair: RigidContactPair | None = None,
        rigid_body: RigidBody | None = None,
        motion_schedule: PrescribedRigidMotionSchedule | None = None,
        surface_reference_point=None,
        projection_options=None,
        name: str = "dolfinx_explicit_contact_residual",
    ) -> None:
        if not isinstance(adapter, DolfinxContactTraceAdapter):
            raise TypeError("Explicit contact requires DolfinxContactTraceAdapter.")
        if displacement.function_space is not adapter.function_space:
            raise ValueError("Explicit contact displacement differs from its trace space.")
        if not isinstance(lifecycle, ContactProjectionLifecycle):
            raise TypeError("Explicit contact requires ContactProjectionLifecycle.")
        if not np.array_equal(lifecycle.point_ids, adapter.trace.point_ids):
            raise ValueError("Explicit contact lifecycle and trace identities differ.")
        if not isinstance(law, FrictionlessPenaltyContactLaw):
            raise TypeError("Explicit contact requires FrictionlessPenaltyContactLaw.")
        if friction_law is not None and not isinstance(
            friction_law, PenaltyCoulombFrictionLaw
        ):
            raise TypeError("Explicit friction requires PenaltyCoulombFrictionLaw.")
        if rigid_body is not None and not isinstance(rigid_body, RigidBody):
            raise TypeError("Explicit contact rigid_body must be RigidBody.")
        if contact_pair is not None and not isinstance(
            contact_pair, RigidContactPair
        ):
            raise TypeError("Explicit contact contact_pair must be RigidContactPair.")
        limit = float(maximum_stable_time_increment)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(
                "maximum_stable_time_increment must be finite and positive."
            )
        if contact_stability_estimate is not None:
            if not isinstance(contact_stability_estimate, ContactStabilityEstimate):
                raise TypeError(
                    "contact_stability_estimate must be ContactStabilityEstimate."
                )
            tolerance = 64.0 * np.finfo(float).eps * contact_stability_estimate.selected
            if limit > contact_stability_estimate.selected + tolerance:
                raise ValueError(
                    "Explicit contact limit exceeds its automatic stability estimate."
                )
        if combined_stability_estimate is not None:
            if not isinstance(
                combined_stability_estimate,
                CombinedExplicitStabilityEstimate,
            ):
                raise TypeError(
                    "combined_stability_estimate must be "
                    "CombinedExplicitStabilityEstimate."
                )
            tolerance = (
                64.0
                * np.finfo(float).eps
                * combined_stability_estimate.selected
            )
            if limit > combined_stability_estimate.selected + tolerance:
                raise ValueError(
                    "Explicit contact limit exceeds its combined non-contact/contact "
                    "stability estimate."
                )
        declared_limit = declared_maximum_stable_time_increment
        if declared_limit is not None:
            declared_limit = float(declared_limit)
            if not np.isfinite(declared_limit) or declared_limit <= 0.0:
                raise ValueError(
                    "declared_maximum_stable_time_increment must be finite and positive."
                )
            tolerance = 64.0 * np.finfo(float).eps * declared_limit
            if limit > declared_limit + tolerance:
                raise ValueError(
                    "Explicit contact limit exceeds its caller-declared ceiling."
                )
        options = {} if projection_options is None else dict(projection_options)
        if any(not isinstance(key, str) or not key for key in options):
            raise ValueError("Contact projection option names must be non-empty strings.")
        reference = (
            None
            if surface_reference_point is None
            else np.asarray(surface_reference_point, dtype=float).reshape(-1)
        )
        if reference is not None and (
            reference.shape != (3,) or not np.all(np.isfinite(reference))
        ):
            raise ValueError("Surface reference point must be one finite 3-vector.")
        if not str(name).strip():
            raise ValueError("Explicit contact residual requires a name.")
        if motion_schedule is not None:
            if not isinstance(motion_schedule, PrescribedRigidMotionSchedule):
                raise TypeError(
                    "Explicit contact motion requires PrescribedRigidMotionSchedule."
                )
            if motion_schedule.motion.dimension != 3:
                raise ValueError(
                    "The current DOLFINx explicit contact adapter requires 3D motion."
                )
            if "motion" in options or "factor" in options:
                raise ValueError(
                    "Motion-controlled projection options are owned by motion_schedule."
                )
        if rigid_body is not None:
            if rigid_body.dimension != 3:
                raise ValueError(
                    "The current DOLFINx explicit contact adapter requires a 3D body."
                )
            body_schedule = rigid_body.motion_schedule
            if _canonical_json(
                None if body_schedule is None else body_schedule.summary()
            ) != _canonical_json(
                None if motion_schedule is None else motion_schedule.summary()
            ):
                raise ValueError(
                    "Explicit contact motion differs from its rigid-body asset."
                )
            if rigid_body.reference_point is not None and (
                reference is None
                or not np.array_equal(reference, rigid_body.reference_point)
            ):
                raise ValueError(
                    "Explicit contact reference point differs from its rigid body."
                )
        if contact_pair is not None:
            if rigid_body is not contact_pair.rigid_body:
                raise ValueError(
                    "Explicit contact rigid body differs from its contact pair."
                )
            if law is not contact_pair.law:
                raise ValueError(
                    "Explicit contact law differs from its contact pair."
                )
            if adapter.boundary_name != contact_pair.slave_boundary.name:
                raise ValueError(
                    "Explicit contact trace boundary differs from its contact pair."
                )
        self.base = base
        self.adapter = adapter
        self.displacement = displacement
        self.lifecycle = lifecycle
        self.law = law
        self.friction_law = friction_law
        self.friction_state = (
            None if friction_law is None else TangentialContactState()
        )
        self.friction_kinematics = (
            None if friction_law is None else TangentialKinematicState()
        )
        self.maximum_stable_time_increment = limit
        self.contact_stability_estimate = contact_stability_estimate
        self.combined_stability_estimate = combined_stability_estimate
        self.declared_maximum_stable_time_increment = declared_limit
        self.contact_pair = contact_pair
        self.rigid_body = rigid_body
        self.motion_schedule = motion_schedule
        self.surface_reference_point = reference
        self.projection_options = options
        self.name = str(name)
        self.trial_evidence: ExplicitContactEvidence | None = None
        self.accepted_evidence: ExplicitContactEvidence | None = None
        self.accepted_evaluations = 0
        self.current_time = 0.0
        self.current_motion_factor = (
            0.0 if motion_schedule is None else motion_schedule.factor_at(0.0)
        )
        self.work_state = (
            None
            if motion_schedule is None
            else PrescribedContactWorkState(identity=self._work_identity())
        )
        self.checkpoint_state_required = bool(
            motion_schedule is not None
            or friction_law is not None
            or getattr(base, "checkpoint_state_required", False)
        )

    @property
    def communicator(self):
        return self.adapter.communicator

    @property
    def stability_controller(self) -> str:
        """Return which contact ceiling controls the explicit residual."""

        combined = self.combined_stability_estimate
        if combined is None:
            return "caller_declared"
        declared = self.declared_maximum_stable_time_increment
        if declared is not None:
            tolerance = 64.0 * np.finfo(float).eps * declared
            if declared < combined.selected - tolerance:
                return "caller_declared"
        return "combined_spectral_bound"

    def validate_time_increment(self, dt: float) -> None:
        """Reject a step exceeding the selected whole-system ceiling."""

        selected = float(dt)
        if not np.isfinite(selected) or selected <= 0.0:
            raise ValueError("Explicit contact requires a finite positive dt.")
        tolerance = 64.0 * np.finfo(float).eps * self.maximum_stable_time_increment
        if selected > self.maximum_stable_time_increment + tolerance:
            raise ValueError(
                "Explicit contact dt exceeds its declared stability limit "
                f"({selected:.6g} > {self.maximum_stable_time_increment:.6g})."
            )

    def update_time(self, time_value: float) -> None:
        """Set the physical time used by the next residual evaluation."""

        selected = float(time_value)
        if not np.isfinite(selected) or selected < 0.0:
            raise ValueError("Explicit contact time must be finite and non-negative.")
        if hasattr(self.base, "update_time"):
            self.base.update_time(selected)
        self.current_time = selected
        self.current_motion_factor = (
            0.0
            if self.motion_schedule is None
            else self.motion_schedule.factor_at(selected)
        )

    def _work_identity(self) -> str:
        payload = {
            "residual": self.name,
            "contact_pair_identity": (
                None
                if self.contact_pair is None
                else self.contact_pair.scientific_identity
            ),
            "rigid_body_identity": (
                None
                if self.rigid_body is None
                else self.rigid_body.scientific_identity
            ),
            "motion_schedule": (
                None
                if self.motion_schedule is None
                else self.motion_schedule.summary()
            ),
        }
        encoded = _canonical_json(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _projection_context(self) -> tuple[dict[str, object], object | None]:
        options = dict(self.projection_options)
        reference = self.surface_reference_point
        if self.motion_schedule is not None:
            motion = self.motion_schedule.motion
            options.update(motion=motion, factor=self.current_motion_factor)
            reference = motion.state(self.current_motion_factor)["reference_point"]
        return options, reference

    def _evaluate_local_contact(self):
        trace_evaluation = self.adapter.evaluate(self.displacement)
        options, reference = self._projection_context()
        projection = self.lifecycle.evaluate(
            trace_evaluation.query_points,
            **options,
        )
        response = self.law.evaluate(projection.record.projection)
        assembly = trace_evaluation.assemble(
            projection.record,
            response,
            surface_reference_point=reference,
        )
        friction_assembly = None
        friction_response = None
        if self.friction_law is not None:
            if self.friction_state is None or self.friction_kinematics is None:
                raise RuntimeError("Explicit friction State was not constructed.")
            accepted_kinematics = self.friction_kinematics.accepted
            if accepted_kinematics is None:
                relative_increment = np.zeros_like(
                    projection.record.projection.query_points
                )
            else:
                relative_increment = relative_contact_displacement_increment(
                    accepted_kinematics,
                    projection.record,
                    motion=(
                        None
                        if self.motion_schedule is None
                        else self.motion_schedule.motion
                    ),
                    current_motion_factor=self.current_motion_factor,
                )
            friction_response = self.friction_law.evaluate(
                point_ids=projection.record.point_ids,
                normal_response=response,
                relative_displacement_increment=relative_increment,
                accepted=self.friction_state.accepted,
            )
            friction_assembly = trace_evaluation.assemble_friction(
                projection.record,
                friction_response,
                surface_reference_point=reference,
            )
            self.friction_state.begin(friction_response)
            self.friction_kinematics.begin(
                projection.record,
                motion_factor=self.current_motion_factor,
            )
        return assembly, response, friction_assembly, friction_response

    def _collective_local_contact(self):
        local_error = None
        outcome = None
        try:
            outcome = self._evaluate_local_contact()
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        messages = tuple(self.communicator.allgather(local_error))
        failures = tuple(
            f"rank {rank}: {message}"
            for rank, message in enumerate(messages)
            if message is not None
        )
        if failures:
            self.lifecycle.rollback_increment()
            if self.friction_state is not None:
                self.friction_state.rollback()
                self.friction_kinematics.rollback()
            self.trial_evidence = None
            raise ValueError(
                "Explicit contact evaluation rejected collectively; "
                + "; ".join(failures)
            )
        if outcome is None:  # pragma: no cover - collective consensus guards this
            raise RuntimeError("Explicit contact consensus produced no local assembly.")
        return outcome

    def _global_evidence(
        self,
        assembly,
        response,
        friction_assembly=None,
        friction_response=None,
    ) -> ExplicitContactEvidence:
        total_structure = np.asarray(assembly.contact_force_on_structure, dtype=float)
        total_surface = np.asarray(assembly.contact_force_on_surface, dtype=float)
        moment = assembly.surface_generalized_moment
        total_moment = None if moment is None else np.asarray(moment, dtype=float)
        tangential_potential = 0.0
        friction_dissipation = 0.0
        separation_release = 0.0
        sticking_count = 0
        sliding_count = 0
        if friction_assembly is not None:
            total_structure = total_structure + np.asarray(
                friction_assembly.contact_force_on_structure,
                dtype=float,
            )
            total_surface = total_surface + np.asarray(
                friction_assembly.contact_force_on_surface,
                dtype=float,
            )
            friction_moment = friction_assembly.surface_generalized_moment
            if (total_moment is None) != (friction_moment is None):
                raise ValueError("Normal and friction moment contracts differ.")
            if total_moment is not None:
                total_moment = total_moment + np.asarray(friction_moment, dtype=float)
            tangential_potential = friction_assembly.recoverable_penalty_energy
            friction_dissipation = friction_assembly.cumulative_dissipation
            separation_release = friction_assembly.cumulative_separation_release
            sticking_count = int(np.count_nonzero(friction_response.sticking))
            sliding_count = int(np.count_nonzero(friction_response.sliding))
        local = np.concatenate(
            (
                np.asarray(
                    (
                        assembly.potential_energy,
                        tangential_potential,
                        friction_dissipation,
                        separation_release,
                    ),
                    dtype=float,
                ),
                total_structure,
                total_surface,
                np.zeros(3, dtype=float) if total_moment is None else total_moment,
                np.asarray(
                    (
                        np.count_nonzero(response.active),
                        np.count_nonzero(~response.projection.valid),
                        sticking_count,
                        sliding_count,
                    ),
                    dtype=float,
                ),
            )
        )
        global_values = np.empty_like(local)
        self.communicator.Allreduce(local, global_values, op=MPI.SUM)
        local_maximum_penetration = (
            0.0
            if response.penetration.size == 0
            else float(np.max(response.penetration))
        )
        maximum_penetration = self.communicator.allreduce(
            local_maximum_penetration,
            op=MPI.MAX,
        )
        return ExplicitContactEvidence(
            potential_energy=global_values[0] + global_values[1],
            normal_potential_energy=global_values[0],
            tangential_potential_energy=global_values[1],
            friction_dissipation=global_values[2],
            separation_release=global_values[3],
            contact_force_on_structure=global_values[4:7],
            contact_force_on_surface=global_values[7:10],
            surface_generalized_moment=(
                None if total_moment is None else global_values[10:13]
            ),
            active_point_count=int(round(global_values[13])),
            invalid_point_count=int(round(global_values[14])),
            sticking_point_count=int(round(global_values[15])),
            sliding_point_count=int(round(global_values[16])),
            maximum_penetration=maximum_penetration,
        )

    def _work_station(self, evidence: ExplicitContactEvidence) -> PrescribedContactWorkStation:
        if self.motion_schedule is None:
            raise RuntimeError("Fixed contact does not own prescribed-motion work.")
        moment = evidence.surface_generalized_moment
        if moment is None:
            raise RuntimeError("Moving rigid contact requires generalized moment evidence.")
        generalized_force = np.concatenate(
            (evidence.contact_force_on_structure, moment)
        )
        return PrescribedContactWorkStation(
            time=self.current_time,
            factor=self.current_motion_factor,
            generalized_force=generalized_force,
            generalized_coordinate=self.motion_schedule.generalized_coordinate_at(
                self.current_time
            ),
            contact_resultant=evidence.contact_force_on_structure,
            contact_potential_energy=evidence.potential_energy,
            active_point_count=evidence.active_point_count,
            invalid_point_count=evidence.invalid_point_count,
        )

    def initialize_accepted_state(self, *, time: float = 0.0) -> None:
        """Seed contact energy/work evidence at an accepted boundary."""

        if hasattr(self.base, "initialize_accepted_state"):
            self.base.initialize_accepted_state(time=time)
        if (
            (self.work_state is None and self.accepted_evidence is not None)
            or (self.work_state is not None and self.work_state.accepted)
        ):
            return
        self.update_time(time)
        try:
            assembly, response, friction_assembly, friction_response = (
                self._collective_local_contact()
            )
            evidence = self._global_evidence(
                assembly,
                response,
                friction_assembly,
                friction_response,
            )
            if self.work_state is not None:
                self.work_state.initialize(self._work_station(evidence))
            self.accepted_evidence = evidence
            if self.friction_state is not None:
                self.lifecycle.commit_increment()
                self.friction_state.commit()
                self.friction_kinematics.commit()
        except Exception:
            if self.lifecycle.state.trial is not None:
                self.lifecycle.rollback_increment()
            if self.friction_state is not None:
                self.friction_state.rollback()
                self.friction_kinematics.rollback()
            raise
        finally:
            # Frictionless projection is memoryless. Friction preserves only
            # the accepted boundary required by its incremental State.
            if self.friction_state is None:
                self.lifecycle.state.rollback()
            self.trial_evidence = None

    def assemble_vector(self):
        """Assemble base and contact residuals without double-counting ghosts."""

        return self._assemble_with_base(accepted=False)

    def assemble_accepted_vector(self):
        """Sample reactions without reintegrating a history material at dt=0."""
        if self.trial_evidence is not None or self.lifecycle.state.trial is not None:
            raise RuntimeError("Accepted contact force cannot replace an active trial.")
        if self.work_state is not None and (
            self.work_state.current is None
            or self.work_state.current.time != self.current_time
        ):
            raise RuntimeError("Accepted contact force requires the accepted motion time.")
        try:
            return self._assemble_with_base(accepted=True)
        finally:
            self.rollback()

    def _assemble_with_base(self, *, accepted):
        cached = getattr(self.base, "assemble_accepted_vector", None)
        vector = cached() if accepted and callable(cached) else operators.assemble_vector(self.base)
        try:
            assembly, response, friction_assembly, friction_response = (
                self._collective_local_contact()
            )
            contact = vector.duplicate()
            try:
                values = np.asarray(
                    assembly.nodal_structural_residual,
                    dtype=float,
                ).reshape(-1)
                if friction_assembly is not None:
                    values = values + np.asarray(
                        friction_assembly.nodal_structural_residual,
                        dtype=float,
                    ).reshape(-1)
                with contact.localForm() as local:
                    local.set(0.0)
                    if local.array.shape != values.shape:
                        raise ValueError(
                            "Contact trace residual and PETSc local layouts differ: "
                            f"trace={values.shape}, vector={local.array.shape}."
                        )
                    local.array[:] = values
                contact.ghostUpdate(
                    addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE,
                )
                if vector.array.shape != contact.array.shape:
                    raise ValueError("Contact and base owned vector layouts differ.")
                vector.array[:] += contact.array
            finally:
                contact.destroy()
            self.trial_evidence = self._global_evidence(
                assembly,
                response,
                friction_assembly,
                friction_response,
            )
            if self.work_state is not None:
                if not self.work_state.accepted:
                    raise RuntimeError(
                        "Moving contact work State was not initialized by its Procedure."
                    )
                station = self._work_station(self.trial_evidence)
                if station.time > self.work_state.current.time:
                    self.work_state.begin(station)
                elif not np.isclose(
                    station.time,
                    self.work_state.current.time,
                    rtol=0.0,
                    atol=64.0 * np.finfo(float).eps * max(1.0, station.time),
                ):
                    raise RuntimeError(
                        "Moving contact residual time precedes its accepted work State."
                    )
            return vector
        except Exception:
            vector.destroy()
            self.rollback()
            raise

    def commit(self) -> None:
        if self.trial_evidence is None or self.lifecycle.state.trial is None:
            raise RuntimeError("No explicit contact trial is available to commit.")
        if self.work_state is not None and self.work_state.trial is None:
            raise RuntimeError(
                "No moving-contact work trial is available to commit at this time."
            )
        if self.friction_state is not None and (
            self.friction_state.trial is None
            or self.friction_kinematics.trial is None
        ):
            raise RuntimeError("No explicit-friction trial is available to commit.")
        if hasattr(self.base, "commit"):
            self.base.commit()
        self.lifecycle.commit_increment()
        if self.friction_state is not None:
            self.friction_state.commit()
            self.friction_kinematics.commit()
        if self.work_state is not None:
            self.work_state.commit()
        self.accepted_evidence = self.trial_evidence
        self.trial_evidence = None
        self.accepted_evaluations += 1

    def rollback(self) -> None:
        if hasattr(self.base, "rollback"):
            self.base.rollback()
        if self.lifecycle.state.trial is not None:
            self.lifecycle.rollback_increment()
        if self.friction_state is not None:
            self.friction_state.rollback()
            self.friction_kinematics.rollback()
        if self.work_state is not None:
            self.work_state.rollback()
        self.trial_evidence = None

    def _global_friction_records(self) -> dict[str, object] | None:
        if self.friction_state is None or self.friction_kinematics is None:
            return None
        if (
            self.friction_state.accepted is None
            and self.friction_kinematics.accepted is None
        ):
            return None
        if (self.friction_state.accepted is None) != (
            self.friction_kinematics.accepted is None
        ):
            raise RuntimeError(
                "Friction constitutive and kinematic initialization differ."
            )
        return global_friction_state_snapshot(
            self.friction_state.accepted,
            self.friction_kinematics.accepted,
            self.communicator,
        )

    def _global_projection_record(self) -> dict[str, object] | None:
        accepted = self.lifecycle.state.accepted
        if accepted is None:
            return None
        return global_projection_state_snapshot(accepted, self.communicator)

    def _validated_global_projection_record(self, snapshot: object):
        if snapshot is None:
            return None
        restored = local_projection_state_from_snapshot(
            snapshot,
            point_ids=self.adapter.trace.point_ids,
        )
        projector_summary = self.lifecycle.projector.summary()
        expected_fingerprint = projector_summary.get(
            "global_geometry_fingerprint",
            projector_summary.get("geometry_fingerprint"),
        )
        if (
            expected_fingerprint is not None
            and restored.projection.geometry_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Contact-projection checkpoint geometry differs from its projector."
            )
        return restored

    def _validated_global_friction_records(self, snapshot: object):
        if self.friction_state is None or self.friction_kinematics is None:
            if snapshot is not None:
                raise ValueError("Frictionless contact cannot restore friction State.")
            return None
        if snapshot is None:
            return None
        return local_friction_state_from_snapshot(
            snapshot,
            point_ids=self.adapter.trace.point_ids,
            dimension=self.adapter.reference_nodal_positions.shape[1],
        )

    def snapshot(self) -> dict[str, object]:
        """Canonical contact records with provider-owned nested State."""

        return self._accepted_snapshot(
            "snapshot", canonical_base=not callable(getattr(self.base, "checkpoint_snapshot", None)),
        )

    def checkpoint_snapshot(self) -> dict[str, object]:
        """Preserve nested rank-local numeric State in the shared archive."""

        return self._accepted_snapshot("checkpoint_snapshot", canonical_base=False)

    def transaction_snapshot(self) -> dict[str, object]:
        """Keep numeric bulk rollback buffers out of JSON serialization."""

        return self._accepted_snapshot("transaction_snapshot", canonical_base=False)

    def checkpoint_capabilities(self):
        """A contact wrapper cannot broaden its nested State's portability."""
        from dataclasses import replace
        from ..checkpointing import CheckpointCapabilities, TRANSIENT_CHECKPOINT_SCHEMA

        components = ("accepted contact projection, friction and prescribed-motion work",)
        identities = ("contact pair, rigid geometry and prescribed motion",)
        declaration = getattr(self.base, "checkpoint_capabilities", None)
        if callable(declaration):
            nested = declaration()
            if not isinstance(nested, CheckpointCapabilities):
                raise TypeError("Nested residual must declare CheckpointCapabilities.")
            if nested.boundary != "accepted_step":
                raise ValueError("Contact and nested residual acceptance boundaries differ.")
            return replace(
                nested,
                state_components=nested.state_components + components,
                identity_scope=nested.identity_scope + identities,
            )
        # Preserve the existing global-record route for stateless forms and
        # legacy contact providers. Numeric providers must declare their own
        # partition contract before their capability can be advertised.
        has_numeric_state = callable(getattr(self.base, "checkpoint_snapshot", None))
        if has_numeric_state:
            raise ValueError("Numeric nested State requires an explicit checkpoint capability contract.")
        return CheckpointCapabilities(
            schemas=(TRANSIENT_CHECKPOINT_SCHEMA,),
            boundary="accepted_step", payload_scope="full_restart_state",
            state_components=components, atomic_publication=True,
            rank_count_portability="requires_portable_policy",
            identity_scope=identities,
            limitations=("Nested State retains its own restart restrictions.",),
        )

    def _accepted_snapshot(self, base_method: str, *, canonical_base: bool):

        if (
            self.trial_evidence is not None
            or self.lifecycle.state.trial is not None
            or (
                self.friction_state is not None
                and (
                    self.friction_state.trial is not None
                    or self.friction_kinematics.trial is not None
                )
            )
        ):
            raise RuntimeError("Explicit contact can only checkpoint an accepted boundary.")
        snapshot = {
            "schema": "agentfem.dolfinx-explicit-contact-residual.v6",
            "name": self.name,
            "contact_pair_identity": (
                None
                if self.contact_pair is None
                else self.contact_pair.scientific_identity
            ),
            "rigid_body_identity": (
                None
                if self.rigid_body is None
                else self.rigid_body.scientific_identity
            ),
            "motion_schedule": (
                None
                if self.motion_schedule is None
                else self.motion_schedule.summary()
            ),
            "work_state": (
                None if self.work_state is None else (
                    self.work_state.transaction_snapshot()
                    if base_method == "transaction_snapshot" else self.work_state.snapshot()
                )
            ),
            "accepted_evaluations": self.accepted_evaluations,
            "accepted_evidence": (
                None
                if self.accepted_evidence is None
                else self.accepted_evidence.summary()
            ),
            "friction_state": self._global_friction_records(),
            "projection_state": self._global_projection_record(),
            "base_state": (
                getattr(self.base, base_method, self.base.snapshot)()
                if hasattr(self.base, "snapshot") else None
            ),
        }
        # Contact records are global/canonical; a numeric material buffer is
        # owned by one partition. Its provider and the common array archive
        # validate that identity without gathering the large payload here.
        canonical = snapshot if canonical_base else {
            key: value for key, value in snapshot.items() if key != "base_state"
        }
        if base_method == "transaction_snapshot" and self.work_state is not None:
            canonical["work_state"] = self.work_state.summary()
        encoded = _canonical_json(canonical)
        copies = tuple(self.communicator.allgather(encoded))
        if any(item != copies[0] for item in copies[1:]):
            raise RuntimeError("Explicit contact checkpoint State differs across MPI ranks.")
        return snapshot

    def contact_energy_evidence(self) -> tuple[dict[str, object], ...]:
        """Return accepted per-pair terms for the shared dynamic energy ledger."""

        nested = getattr(self.base, "contact_energy_evidence", None)
        terms = list(nested()) if callable(nested) else []
        if self.accepted_evidence is None:
            raise RuntimeError(
                f"Explicit contact pair {self.name!r} has no accepted energy evidence."
            )
        terms.append(
            {
                "name": self.name,
                "contact_potential_energy": self.accepted_evidence.potential_energy,
                "contact_friction_dissipation": (
                    self.accepted_evidence.friction_dissipation
                ),
                "contact_separation_release": (
                    self.accepted_evidence.separation_release
                ),
                "contact_motion_work": (
                    0.0 if self.work_state is None else self.work_state.path_work
                ),
                "moving": self.work_state is not None,
            }
        )
        return tuple(terms)

    def contact_progress_evidence(self) -> tuple[dict[str, object], ...]:
        """Return compact accepted contact diagnostics for throttled progress."""

        nested = getattr(self.base, "contact_progress_evidence", None)
        terms = list(nested()) if callable(nested) else []
        if self.accepted_evidence is None:
            return tuple(terms)
        terms.append(
            {
                "name": self.name,
                "active_point_count": self.accepted_evidence.active_point_count,
                "invalid_point_count": self.accepted_evidence.invalid_point_count,
                "maximum_penetration": self.accepted_evidence.maximum_penetration,
                "contact_force_norm": float(
                    np.linalg.norm(
                        self.accepted_evidence.contact_force_on_structure
                    )
                ),
                "contact_motion_work": (
                    0.0 if self.work_state is None else self.work_state.path_work
                ),
                "contact_motion_power": (
                    0.0
                    if self.work_state is None
                    or self.work_state.latest_interval_power is None
                    else self.work_state.latest_interval_power
                ),
                "sticking_point_count": self.accepted_evidence.sticking_point_count,
                "sliding_point_count": self.accepted_evidence.sliding_point_count,
                "contact_friction_dissipation": (
                    self.accepted_evidence.friction_dissipation
                ),
            }
        )
        return tuple(terms)

    def _validate_restore(self, snapshot: object):
        """Local validation only, before any nested collective restoration."""

        required = {
            "schema",
            "name",
            "contact_pair_identity",
            "rigid_body_identity",
            "motion_schedule",
            "work_state",
            "accepted_evaluations",
            "accepted_evidence",
            "friction_state",
            "projection_state",
            "base_state",
        }
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema")
            != "agentfem.dolfinx-explicit-contact-residual.v6"
            or set(snapshot) != required
        ):
            raise ValueError("Unsupported explicit-contact residual snapshot.")
        expected_schedule = (
            None if self.motion_schedule is None else self.motion_schedule.summary()
        )
        expected_body_identity = (
            None
            if self.rigid_body is None
            else self.rigid_body.scientific_identity
        )
        expected_pair_identity = (
            None
            if self.contact_pair is None
            else self.contact_pair.scientific_identity
        )
        if (
            snapshot.get("name") != self.name
            or snapshot.get("contact_pair_identity") != expected_pair_identity
            or snapshot.get("rigid_body_identity") != expected_body_identity
            or _canonical_json(
                snapshot.get("motion_schedule")
            )
            != _canonical_json(expected_schedule)
        ):
            raise ValueError("Explicit-contact checkpoint identity differs.")
        raw_work = snapshot["work_state"]
        validated_work = None
        if self.work_state is None:
            if raw_work is not None:
                raise ValueError("Fixed contact cannot restore moving-contact work State.")
        else:
            if raw_work is None:
                raise ValueError("Moving contact checkpoint lacks its work State.")
            validated_work = PrescribedContactWorkState(
                identity=self.work_state.identity
            )
            validated_work.restore(raw_work)
        count = int(snapshot["accepted_evaluations"])
        if count < 0:
            raise ValueError("Accepted explicit-contact evaluation count is invalid.")
        raw_evidence = snapshot["accepted_evidence"]
        evidence = (
            None
            if raw_evidence is None
            else ExplicitContactEvidence.from_summary(raw_evidence)
        )
        raw_friction = snapshot["friction_state"]
        validated_friction = self._validated_global_friction_records(raw_friction)
        raw_projection = snapshot["projection_state"]
        validated_projection = self._validated_global_projection_record(
            raw_projection
        )
        raw_base = snapshot["base_state"]
        base_restore = getattr(self.base, "restore", None)
        if raw_base is not None and not callable(base_restore):
            raise ValueError(
                "Explicit-contact checkpoint has nested residual State, but the "
                "current base residual cannot restore it."
            )
        if raw_base is None and getattr(
            self.base,
            "checkpoint_state_required",
            False,
        ):
            raise ValueError(
                "Explicit-contact checkpoint lacks required nested residual State."
            )
        return (raw_base, validated_work, count, evidence,
                validated_friction, validated_projection)

    def restore(self, snapshot: object) -> None:
        """Reject rank-local outer corruption before nested State collectives."""
        from ..provenance import collective_call

        (raw_base, validated_work, count, evidence,
         validated_friction, validated_projection) = collective_call(
            lambda: self._validate_restore(snapshot), comm=self.communicator,
            label="Explicit-contact restore validation",
        )
        # Every outer field is validated before nested State is allowed to
        # mutate. The local work assignment below is then infallible.
        if raw_base is not None:
            self.base.restore(raw_base)
        if validated_friction is not None:
            friction, kinematics = validated_friction
            self.friction_state.accepted = friction
            self.friction_state.trial = None
            self.friction_kinematics.accepted = kinematics
            self.friction_kinematics.trial = None
        elif self.friction_state is not None:
            self.friction_state.accepted = None
            self.friction_state.trial = None
            self.friction_kinematics.accepted = None
            self.friction_kinematics.trial = None
        if self.work_state is not None:
            self.work_state.restore(validated_work.transaction_snapshot())
        self.accepted_evaluations = count
        self.accepted_evidence = evidence
        self.lifecycle.state.accepted = validated_projection
        self.lifecycle.state.rollback()
        self.trial_evidence = None

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "dolfinx_explicit_contact_residual",
            "procedure": "central_difference",
            "contact": (
                "frictionless_penalty"
                if self.friction_law is None
                else "normal_penalty_with_penalty_coulomb_friction"
            ),
            "surface_motion": (
                "fixed_only"
                if self.motion_schedule is None
                else "prescribed_proportional_rigid_motion"
            ),
            "maximum_stable_time_increment": self.maximum_stable_time_increment,
            "declared_maximum_stable_time_increment": (
                self.declared_maximum_stable_time_increment
            ),
            "stability": (
                (
                    "caller_supplied_normal_contact_limit_enforced"
                    if self.friction_law is None
                    else "caller_supplied_normal_and_tangential_contact_limit_enforced"
                )
                if self.contact_stability_estimate is None
                else "automatic_contact_spectral_bound_available"
            ),
            "stability_controller": self.stability_controller,
            "contact_stability_estimate": (
                None
                if self.contact_stability_estimate is None
                else self.contact_stability_estimate.summary()
            ),
            "combined_stability_estimate": (
                None
                if self.combined_stability_estimate is None
                else self.combined_stability_estimate.summary()
            ),
            "parallel_assembly": "ghost_reverse_add_then_owned_accumulation",
            "projection_update": "every_residual_evaluation",
            "restart": (
                "accepted_projection_evidence_restored_then_recomputed"
                if self.work_state is None
                else "projection_and_accepted_work_restored_then_recomputed"
            ),
            "accepted_evaluations": self.accepted_evaluations,
            "accepted_evidence": (
                None
                if self.accepted_evidence is None
                else self.accepted_evidence.summary()
            ),
            "motion_schedule": (
                None
                if self.motion_schedule is None
                else self.motion_schedule.summary()
            ),
            "rigid_body": (
                None if self.rigid_body is None else self.rigid_body.summary()
            ),
            "contact_pair": (
                None if self.contact_pair is None else self.contact_pair.summary()
            ),
            "prescribed_motion_work": (
                None if self.work_state is None else self.work_state.summary()
            ),
            "adapter": self.adapter.summary(),
            "lifecycle": self.lifecycle.summary(),
            "law": self.law.summary(),
            "friction_law": (
                None if self.friction_law is None else self.friction_law.summary()
            ),
            "friction_state": (
                None if self.friction_state is None else self.friction_state.summary()
            ),
            "base": self.base.summary() if hasattr(self.base, "summary") else repr(self.base),
        }


def dolfinx_explicit_contact_residual(
    base,
    *,
    adapter,
    displacement,
    projector=None,
    rigid_body=None,
    contact_pair=None,
    penalty=None,
    maximum_stable_time_increment=None,
    lumped_mass=None,
    contact_stability_estimate=None,
    noncontact_unsafed_stability_limit=None,
    contact_stability_safety_factor: float = 0.8,
    motion_schedule=None,
    invalid_policy: str = "reject",
    surface_reference_point=None,
    projection_options=None,
    name: str = "dolfinx_explicit_contact_residual",
) -> DolfinxExplicitContactResidual:
    """Build the reviewed first explicit contact residual consumer.

    Pass ``lumped_mass`` to derive a conservative contact spectral bound.
    Automatic time-step selection additionally requires the unsafed
    non-contact stability limit so the spectral bounds are added before
    selecting ``dt``. A caller-declared whole-system ceiling may be stricter.
    """

    if contact_pair is not None:
        if not isinstance(contact_pair, RigidContactPair):
            raise TypeError("contact_pair must be one RigidContactPair asset.")
        if rigid_body is not None:
            raise ValueError("rigid_body is owned by contact_pair when pair is used.")
        if penalty is not None:
            raise ValueError("penalty is owned by contact_pair when pair is used.")
        if invalid_policy != "reject":
            raise ValueError(
                "invalid_policy is owned by contact_pair when pair is used."
            )
        rigid_body = contact_pair.rigid_body
        law = contact_pair.law
        friction_law = contact_pair.friction
    else:
        if penalty is None:
            raise ValueError("Explicit contact requires penalty or contact_pair.")
        law = frictionless_penalty_contact_law(
            penalty,
            invalid_policy=invalid_policy,
            name=f"{name}_law",
        )
        friction_law = None
    if rigid_body is not None:
        if not isinstance(rigid_body, RigidBody):
            raise TypeError("rigid_body must be one RigidBody asset.")
        if motion_schedule is not None:
            raise ValueError(
                "motion_schedule is owned by rigid_body when that asset is used."
            )
        if surface_reference_point is not None:
            raise ValueError(
                "surface_reference_point is owned by rigid_body when that asset is used."
            )
        motion_schedule = rigid_body.motion_schedule
        surface_reference_point = rigid_body.reference_point
        if projector is None:
            projector = _reviewed_projector_for_rigid_surface(
                rigid_body.surface,
                adapter.communicator,
            )
        projector_summary = projector.summary()
        projected_fingerprint = projector_summary.get(
            "global_geometry_fingerprint",
            projector_summary.get("geometry_fingerprint"),
        )
        expected_fingerprint = rigid_body.surface.summary().get(
            "geometry_fingerprint"
        )
        if (
            projected_fingerprint is not None
            and expected_fingerprint is not None
            and projected_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Contact projector geometry differs from rigid_body.surface."
            )
    if projector is None:
        raise ValueError("Explicit contact requires projector or rigid_body.")
    stability_estimate = contact_stability_estimate
    combined_stability_estimate = None
    declared_limit = maximum_stable_time_increment
    if (
        lumped_mass is None
        and isinstance(contact_stability_estimate, ContactStabilityEstimate)
        and maximum_stable_time_increment is None
    ):
        maximum_stable_time_increment = contact_stability_estimate.selected
    if lumped_mass is not None:
        if contact_stability_estimate is not None:
            raise ValueError(
                "Pass either lumped_mass or contact_stability_estimate, not both."
            )
        projector_kind = projector.summary().get("kind")
        reviewed_stability_geometries = {
            "rigid_plane_surface",
            "triangulated_rigid_surface",
            "triangle_surface_bvh",
            "distributed_triangle_surface_bvh",
            "routed_distributed_triangle_surface_bvh",
        }
        if projector_kind not in reviewed_stability_geometries:
            raise NotImplementedError(
                "Automatic contact stability screening currently requires a "
                "fixed-normal or piecewise-planar reviewed projector; provide "
                "maximum_stable_time_increment for other geometry."
            )
        stability_estimate = estimate_dolfinx_contact_stability(
            adapter=adapter,
            lumped_mass=lumped_mass,
            normal_penalty=law.penalty,
            tangential_penalty=(
                None
                if friction_law is None
                else friction_law.tangential_penalty
            ),
            friction_coefficient=(
                0.0 if friction_law is None else friction_law.coefficient
            ),
            safety_factor=contact_stability_safety_factor,
        )
        if noncontact_unsafed_stability_limit is not None:
            combined_stability_estimate = combine_explicit_stability_bounds(
                stability_estimate,
                noncontact_unsafed_limit=noncontact_unsafed_stability_limit,
                safety_factor=contact_stability_safety_factor,
            )
        if maximum_stable_time_increment is None:
            if combined_stability_estimate is None:
                raise ValueError(
                    "Automatic explicit contact time-step selection requires "
                    "noncontact_unsafed_stability_limit so non-contact and "
                    "contact spectral bounds can be combined."
                )
            maximum_stable_time_increment = combined_stability_estimate.selected
        elif combined_stability_estimate is not None:
            maximum_stable_time_increment = min(
                float(maximum_stable_time_increment),
                combined_stability_estimate.selected,
            )
        else:
            maximum_stable_time_increment = min(
                float(maximum_stable_time_increment),
                stability_estimate.selected,
            )
    elif contact_stability_estimate is not None and not isinstance(
        contact_stability_estimate,
        ContactStabilityEstimate,
    ):
        raise TypeError(
            "contact_stability_estimate must be ContactStabilityEstimate."
        )
    elif noncontact_unsafed_stability_limit is not None:
        raise ValueError(
            "noncontact_unsafed_stability_limit requires lumped_mass so the "
            "contact spectral contribution can be assembled."
        )
    elif maximum_stable_time_increment is None:
        raise ValueError(
            "Explicit contact requires either maximum_stable_time_increment "
            "or lumped_mass for automatic stability screening."
        )
    elif not np.isclose(
        float(contact_stability_safety_factor),
        0.8,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "contact_stability_safety_factor requires lumped_mass; it does not "
            "modify a caller-declared ceiling."
        )
    lifecycle = ContactProjectionLifecycle(projector, adapter.trace.point_ids)
    return DolfinxExplicitContactResidual(
        base,
        adapter=adapter,
        displacement=displacement,
        lifecycle=lifecycle,
        law=law,
        friction_law=friction_law,
        maximum_stable_time_increment=maximum_stable_time_increment,
        contact_stability_estimate=stability_estimate,
        combined_stability_estimate=combined_stability_estimate,
        declared_maximum_stable_time_increment=declared_limit,
        contact_pair=contact_pair,
        rigid_body=rigid_body,
        motion_schedule=motion_schedule,
        surface_reference_point=surface_reference_point,
        projection_options=projection_options,
        name=name,
    )


__all__ = [
    "DolfinxExplicitContactResidual",
    "ExplicitContactEvidence",
    "dolfinx_explicit_contact_residual",
]
