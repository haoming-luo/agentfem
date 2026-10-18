# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private DOLFINx trial bridge for completely certified P1 contact faces.

This is not a separate nonlinear solver. The opt-in serial standard-J2
Procedure owns accepted increments, material rollback, path work, checkpoint
and Result publication. This private adapter supplies only the face Operator
and an identity-bound trial; it does not admit MPC or MPI. Named research
face pairs can be composed additively in one serial Step; mixing them with
point-sampled pairs is still refused.
Face-integrated evidence is never disguised as quadrature-point evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._petsc_contact_rows import add_contact_coo_tangent, add_contact_nodal_residual
from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .contact_face_operator_research import (
    CertifiedFaceContactOperator,
    evaluate_certified_closed_p1_contact_operator,
    evaluate_moving_certified_closed_p1_contact_operator,
    extract_complete_p1_face_layout,
)
from .contact_pair import RigidContactPair
from .dolfinx_contact_trace import DolfinxContactTraceAdapter
from .initial_gap_bound import InitialGapBound, certify_initial_triangle_gap


@dataclass(frozen=True, eq=False)
class CertifiedFaceContactTrial:
    """One immutable operator with the exact displacement/pose it certifies."""

    operator: CertifiedFaceContactOperator
    pair_identity: str
    current_nodal_positions: np.ndarray
    motion_factor: float

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        return self.operator.nodal_residual

    @property
    def potential_energy(self) -> float:
        return self.operator.potential_energy

    @property
    def friction_assembly(self) -> None:
        return None

    def tangent_action(self, direction) -> np.ndarray:
        return self.operator.tangent_action(direction)


@dataclass(eq=False)
class DolfinxCertifiedFaceContactTrialProvider:
    """Serial research adapter with fail-before-mutation trial checks."""

    pair: RigidContactPair
    adapter: DolfinxContactTraceAdapter
    displacement: object
    maximum_slave_faces: int = 128
    maximum_master_facets: int = 64
    order: int = 24
    current_factor: float = field(default=0.0, init=False)
    initial_gap_bound: InitialGapBound | None = field(default=None, init=False)
    contact_integration: str = field(default="certified_closed_two_feature", init=False)

    @property
    def name(self) -> str:
        return self.pair.name

    @property
    def surface(self) -> ClosedTriangulatedRigidSolid:
        return self.pair.rigid_body.surface

    @property
    def providers(self) -> tuple[DolfinxCertifiedFaceContactTrialProvider, ...]:
        return (self,)

    @property
    def has_prescribed_motion(self) -> bool:
        return self.pair.rigid_body.motion_schedule is not None

    def __post_init__(self) -> None:
        if not isinstance(self.pair, RigidContactPair) or not isinstance(
            self.adapter, DolfinxContactTraceAdapter
        ):
            raise TypeError(
                "Certified face trial requires a rigid pair and DOLFINx trace."
            )
        if (
            not isinstance(self.pair.rigid_body.surface, ClosedTriangulatedRigidSolid)
            or self.pair.friction is not None
            or self.pair.law.invalid_policy != "reject"
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
        ):
            raise NotImplementedError(
                "Research face trial admits only frictionless reviewed closed P1 "
                "triangles with scalar penalty."
            )
        if (
            self.pair.slave_boundary.domain is not self.displacement.function_space.mesh
            or self.adapter.function_space is not self.displacement.function_space
            or self.adapter.communicator.size != 1
        ):
            raise NotImplementedError(
                "Research face trial requires one serial mesh and displacement space."
            )
        layout = extract_complete_p1_face_layout(
            self.adapter.trace, self.adapter.reference_nodal_positions
        )
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is not None and (
            schedule.start_time != 0.0
            or schedule.end_time != 1.0
            or schedule.amplitude is not None
        ):
            raise NotImplementedError(
                "Research face trial requires normalized rigid motion without amplitude."
            )
        if self.pair.initial_penetration_limit is not None:
            reference_faces = self.adapter.reference_nodal_positions[
                layout.face_node_ids
            ]
            self.initial_gap_bound = certify_initial_triangle_gap(
                reference_faces,
                self.pair.rigid_body.surface,
                penetration_limit=self.pair.initial_penetration_limit,
            )

    def update_motion(self, factor: float) -> None:
        selected = float(factor)
        if not np.isfinite(selected) or not 0.0 <= selected <= 1.0 + 1e-12:
            raise ValueError("Research face trial factor must lie in [0, 1].")
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is not None:
            schedule.motion.state(selected)
        self.current_factor = selected

    def snapshot_motion(self) -> float:
        return self.current_factor

    def restore_motion(self, saved: float) -> None:
        self.update_motion(saved)

    def snapshot_runtime_state(self) -> float:
        return self.current_factor

    def restore_runtime_state(self, saved: float) -> None:
        self.update_motion(saved)

    def evaluate(self) -> CertifiedFaceContactTrial:
        current = self.adapter.current_nodal_positions(self.displacement)
        reference = self.adapter.reference_nodal_positions
        surface = self.pair.rigid_body.surface
        arguments = dict(
            penalty=float(np.asarray(self.pair.law.penalty)),
            order=self.order,
            maximum_slave_faces=self.maximum_slave_faces,
            maximum_master_facets=self.maximum_master_facets,
        )
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is None:
            operator = evaluate_certified_closed_p1_contact_operator(
                self.adapter.trace, reference, current, surface, **arguments
            )
        else:
            operator = evaluate_moving_certified_closed_p1_contact_operator(
                self.adapter.trace,
                reference,
                current,
                surface,
                motion=schedule.motion,
                factor=self.current_factor,
                **arguments,
            )
        saved = np.asarray(current, dtype=float).copy()
        saved.setflags(write=False)
        return CertifiedFaceContactTrial(
            operator=operator,
            pair_identity=self.pair.scientific_identity,
            current_nodal_positions=saved,
            motion_factor=self.current_factor,
        )

    def _check_current(self, trial: CertifiedFaceContactTrial) -> None:
        if (
            not isinstance(trial, CertifiedFaceContactTrial)
            or trial.pair_identity != self.pair.scientific_identity
            or trial.operator.tool_fingerprint
            != self.pair.rigid_body.surface.geometry_fingerprint
            or trial.motion_factor != self.current_factor
            or not np.array_equal(
                trial.current_nodal_positions,
                self.adapter.current_nodal_positions(self.displacement),
            )
        ):
            raise ValueError(
                "Certified face contribution belongs to another Newton trial or pair."
            )

    def add_residual(self, vector, trial: CertifiedFaceContactTrial) -> None:
        self._check_current(trial)
        add_contact_nodal_residual(vector, trial.nodal_structural_residual)

    def add_tangent(
        self, matrix, trial: CertifiedFaceContactTrial, *, constrained_dofs=()
    ) -> None:
        self._check_current(trial)
        operator = trial.operator
        add_contact_coo_tangent(
            matrix,
            operator.tangent_rows,
            operator.tangent_columns,
            operator.tangent_values,
            constrained_dofs=constrained_dofs,
        )

    def commit_accepted(self) -> None:
        """This frictionless law has no history beyond accepted U and tool pose."""

        self.validate_accepted_configuration()

    def validate_accepted_configuration(self) -> None:
        self.evaluate()

    def snapshot_accepted_state(self) -> None:
        return None

    def restore_accepted_state(self, saved: None) -> None:
        if saved is not None:
            raise ValueError("Frictionless certified face contact has no history State.")

    def current_face_diagnostics(self) -> dict[str, object]:
        """Expose extensive face evidence, never a guessed point pressure."""

        operator = self.evaluate().operator
        return {
            "schema": "agentfem.certified-face-contact-trial.v1",
            "location": "boundary_faces",
            "constraint_name": self.name,
            "reference_area_measure": True,
            "tool_geometry_fingerprint": operator.tool_fingerprint,
            "motion_factor": operator.motion_factor,
            "face_ids": operator.face_ids.tolist(),
            "activity": list(operator.face_activity_kinds),
            "feature_kinds": [
                None if item is None else list(item)
                for item in operator.face_feature_kinds
            ],
            "active_reference_areas": operator.face_active_reference_areas.tolist(),
            "potential_energies": operator.face_potential_energies.tolist(),
            "forces_on_structure": operator.face_forces_on_structure.tolist(),
            "moments_on_structure_about_origin": (
                operator.face_moments_on_structure_about_origin.tolist()
            ),
            "separation_lower_bounds": list(operator.face_separation_lower_bounds),
            "all_faces_certified_at_current_trial": True,
            "initial_contact": (
                None if self.initial_gap_bound is None else {
                    "method": "adaptive_closed_signed_distance_lipschitz",
                    "maximum_penetration_bound": (
                        self.initial_gap_bound.maximum_penetration_bound
                    ),
                    "sampled_point_count": (
                        self.initial_gap_bound.sampled_point_count
                    ),
                    "maximum_refinement_depth": (
                        self.initial_gap_bound.maximum_refinement_depth
                    ),
                }
            ),
        }

    def dual_evidence(self, problem):
        from dolfinx import fem

        from ..constraints import constraint_dual

        if getattr(problem, "solution", None) is not self.displacement:
            raise ValueError("Certified face contact dual needs its consuming problem.")
        if (
            getattr(problem, "last_solve_info", None) is None
            and getattr(problem, "_constraint_dual_state", None) != "accepted"
        ):
            raise RuntimeError("Certified face dual requires an accepted solve.")
        operator = self.evaluate().operator
        distribution = fem.Function(
            self.displacement.function_space, name=f"{self.name}_reaction"
        )
        distribution.x.array[:] = -operator.nodal_residual.reshape(-1)
        distribution.x.scatter_forward()
        force = operator.force_on_structure
        coordinate = None
        if self.has_prescribed_motion:
            motion = self.pair.rigid_body.motion_schedule.motion
            reference = motion.state(self.current_factor)["reference_point"]
            force = np.concatenate(
                (force, operator.moment_on_structure_about(reference))
            )
            coordinate = motion.generalized_coordinate(self.current_factor)
        return constraint_dual(
            self.pair,
            force=force,
            coordinate=coordinate,
            resultant=operator.force_on_structure,
            distribution=distribution,
            role="contact_constraint",
            source="dolfinx_certified_face_contact_research",
            diagnostics={
                "certified_face_count": len(operator.face_ids),
                "active_reference_area": operator.active_reference_area,
                "contact_potential": operator.potential_energy,
                "evidence_location": "boundary_faces",
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_certified_face_contact_research",
            "pair": self.pair.summary(),
            "trace": self.adapter.summary(),
            "contact_integration": self.contact_integration,
            "maximum_slave_faces": self.maximum_slave_faces,
            "maximum_master_facets": self.maximum_master_facets,
            "quadrature_order": self.order,
            "execution_scope": "serial_frictionless_standard_j2_research",
            "accepted_step_admitted": True,
            "product_release_admitted": False,
            "initial_gap_bound": (
                None if self.initial_gap_bound is None else {
                    "method": "adaptive_closed_signed_distance_lipschitz",
                    "maximum_penetration_bound": (
                        self.initial_gap_bound.maximum_penetration_bound
                    ),
                }
            ),
        }


@dataclass(frozen=True, eq=False)
class CombinedCertifiedFaceContactTrial:
    """Additive, identity-preserving trials from independently certified faces."""

    pairs: tuple[tuple[str, CertifiedFaceContactTrial], ...]

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        return np.sum(
            [trial.nodal_structural_residual for _, trial in self.pairs], axis=0
        )

    @property
    def potential_energy(self) -> float:
        return float(sum(trial.potential_energy for _, trial in self.pairs))

    @property
    def friction_assembly(self) -> None:
        return None

    def tangent_action(self, direction) -> np.ndarray:
        return np.sum(
            [trial.tangent_action(direction) for _, trial in self.pairs], axis=0
        )


@dataclass(eq=False)
class DolfinxCombinedCertifiedFaceContactTrialProvider:
    """Research-only serial sum; every pair must prove all its faces anew."""

    providers: tuple[DolfinxCertifiedFaceContactTrialProvider, ...]

    def __post_init__(self) -> None:
        self.providers = tuple(self.providers)
        if len(self.providers) < 2 or any(
            not isinstance(item, DolfinxCertifiedFaceContactTrialProvider)
            for item in self.providers
        ):
            raise ValueError("Combined certified faces require two or more face providers.")
        if len({item.name for item in self.providers}) != len(self.providers):
            raise ValueError("Combined certified face pair names must be unique.")
        displacement = self.providers[0].displacement
        if any(item.displacement is not displacement for item in self.providers):
            raise ValueError("Certified face pairs must share one displacement field.")

    @property
    def has_prescribed_motion(self) -> bool:
        return any(item.has_prescribed_motion for item in self.providers)

    def update_motion(self, factor: float) -> None:
        previous = self.snapshot_motion()
        try:
            for item in self.providers:
                item.update_motion(factor)
        except Exception:
            self.restore_motion(previous)
            raise

    def snapshot_motion(self) -> tuple[tuple[str, float], ...]:
        return tuple((item.name, item.snapshot_motion()) for item in self.providers)

    def restore_motion(self, saved) -> None:
        records = tuple(saved)
        if tuple(name for name, _ in records) != tuple(
            item.name for item in self.providers
        ):
            raise ValueError("Certified face rigid-pair pose identities differ.")
        previous = self.snapshot_motion()
        try:
            for item, (_, factor) in zip(self.providers, records, strict=True):
                item.restore_motion(factor)
        except Exception:
            for item, (_, factor) in zip(self.providers, previous, strict=True):
                item.restore_motion(factor)
            raise

    def snapshot_runtime_state(self):
        return self.snapshot_motion()

    def restore_runtime_state(self, saved) -> None:
        self.restore_motion(saved)

    def snapshot_accepted_state(self) -> None:
        return None

    def restore_accepted_state(self, saved: None) -> None:
        if saved is not None:
            raise ValueError("Frictionless certified face pairs have no history State.")

    def validate_accepted_configuration(self) -> None:
        self.evaluate()

    def commit_accepted(self) -> None:
        self.validate_accepted_configuration()

    def evaluate(self) -> CombinedCertifiedFaceContactTrial:
        # All pair evaluations finish before the sum can enter PETSc.
        return CombinedCertifiedFaceContactTrial(tuple(
            (item.name, item.evaluate()) for item in self.providers
        ))

    def _check_current(self, trial: CombinedCertifiedFaceContactTrial) -> None:
        if (
            not isinstance(trial, CombinedCertifiedFaceContactTrial)
            or tuple(name for name, _ in trial.pairs)
            != tuple(item.name for item in self.providers)
        ):
            raise ValueError("Combined certified face trial pair identities differ.")
        for item, (_, selected) in zip(self.providers, trial.pairs, strict=True):
            item._check_current(selected)

    def add_residual(self, vector, trial: CombinedCertifiedFaceContactTrial) -> None:
        self._check_current(trial)
        add_contact_nodal_residual(vector, trial.nodal_structural_residual)

    def add_tangent(
        self, matrix, trial: CombinedCertifiedFaceContactTrial,
        *, constrained_dofs=(),
    ) -> None:
        self._check_current(trial)
        for item, (_, selected) in zip(self.providers, trial.pairs, strict=True):
            item.add_tangent(matrix, selected, constrained_dofs=constrained_dofs)

    def dual_evidences(self, problem):
        return tuple(item.dual_evidence(problem) for item in self.providers)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_combined_certified_face_contact_research",
            "pairs": tuple(item.summary() for item in self.providers),
            "execution_scope": "serial_frictionless_standard_j2_research",
            "product_release_admitted": False,
        }
