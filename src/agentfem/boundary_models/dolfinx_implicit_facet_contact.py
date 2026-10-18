# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial DOLFINx lowering of bounded rigid-surface Newton contact.

The Model owns the pair, the geometry backend owns BVH search, and this
adapter recomputes contact for each current Newton trial.  It does not commit
state, choose load increments, alter Dirichlet rows or implement MPC reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .contact_pair import RigidContactPair
from .contact_friction import (
    TangentialContactRecord, TangentialContactState, TangentialKinematicRecord,
    TangentialKinematicState, relative_contact_displacement_increment,
)
from .contact_state import ContactProjectionRecord
from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .convex_polyhedron import ConvexPolyhedralRigidSurface
from .dolfinx_contact_trace import (
    DolfinxContactTraceAdapter,
    dolfinx_boundary_region_contact_trace,
)
from .implicit_facet_contact import (
    linearize_piecewise_planar_contact,
)
from .implicit_contact import ImplicitRigidContactContribution
from .implicit_contact import CombinedImplicitRigidContactContribution
from .implicit_closed_contact import linearize_closed_rigid_solid_contact
from .implicit_friction import linearize_piecewise_planar_penalty_friction
from .implicit_smooth_contact import linearize_smooth_rigid_contact
from .rigid import RigidPlaneSurface, SmoothRigidSurface, TriangulatedRigidSurface
from .search import TriangleSurfaceBVH, triangle_surface_bvh


@dataclass(eq=False)
class DolfinxImplicitRigidContact:
    """Current-trial rigid contact with bounded planar-branch tangential State."""

    pair: RigidContactPair
    adapter: DolfinxContactTraceAdapter
    displacement: object
    projector: TriangleSurfaceBVH | ConvexPolyhedralRigidSurface | ClosedTriangulatedRigidSolid | None
    motion_factor: object | None = None
    friction_state: TangentialContactState | None = field(default=None, init=False)
    friction_kinematics: TangentialKinematicState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pair, RigidContactPair):
            raise TypeError("Implicit rigid contact requires a RigidContactPair.")
        if not isinstance(self.adapter, DolfinxContactTraceAdapter):
            raise TypeError("Implicit rigid contact requires a DOLFINx trace adapter.")
        surface = self.pair.rigid_body.surface
        if isinstance(surface, (ConvexPolyhedralRigidSurface, ClosedTriangulatedRigidSolid)):
            if self.projector is not surface:
                raise ValueError("Closed-solid contact projector must share the tool geometry.")
        elif isinstance(surface, TriangulatedRigidSurface):
            if not isinstance(self.projector, TriangleSurfaceBVH):
                raise TypeError("Implicit facet contact requires a reviewed triangle BVH.")
            if self.projector.surface is not surface:
                raise ValueError("Contact search and pair must share one rigid geometry instance.")
        elif not isinstance(surface, SmoothRigidSurface):
            raise TypeError(
                "Implicit rigid contact requires smooth differentiable geometry, "
                "a reviewed triangle tool, or a reviewed closed polyhedral tool."
            )
        elif self.projector is not None:
            raise ValueError("Analytical curved contact does not use a triangle BVH.")
        if self.adapter.communicator.size != 1:
            raise NotImplementedError(
                "AFM-IMPLICIT-CONTACT-MPI: rigid contact has no distributed "
                "Newton matrix/owner reduction contract yet."
            )
        if self.pair.friction is not None:
            if not isinstance(surface, (
                RigidPlaneSurface, ConvexPolyhedralRigidSurface,
                TriangulatedRigidSurface, ClosedTriangulatedRigidSolid,
            )):
                raise NotImplementedError(
                    "AFM-IMPLICIT-CONTACT-FRICTION: only an analytical plane, "
                    "reviewed planar facet branch, or certified closed-convex "
                    "polyhedron or a smooth face of a reviewed closed solid "
                    "has a fixed-old-state Coulomb tangent."
                )
            initial = self.adapter.evaluate(self.displacement)
            projector = surface if self.projector is None else self.projector
            projection = projector.project(initial.query_points)
            if not projection.all_valid:
                raise ValueError("Initial planar-branch friction projection is invalid.")
            record = ContactProjectionRecord(initial.trace.point_ids, projection)
            self.friction_state = TangentialContactState()
            self.friction_kinematics = TangentialKinematicState(
                accepted=TangentialKinematicRecord.from_projection(record),
            )
        if self.pair.law.invalid_policy != "reject":
            raise NotImplementedError(
                "Implicit rigid contact requires invalid_policy='reject'; "
                "an unsupported or ambiguous projection cannot be inactive."
            )
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is not None:
            if (
                schedule.start_time != 0.0
                or schedule.end_time != 1.0
                or schedule.amplitude is not None
            ):
                raise NotImplementedError(
                    "AFM-IMPLICIT-CONTACT-MOTION: static J2 contact requires a "
                    "normalized 0-to-1 rigid-motion schedule without a separate "
                    "physical-time amplitude; the Step owns the load amplitude."
                )
            if self.motion_factor is None:
                raise ValueError("Moving rigid contact requires a mutable pose factor.")
        elif self.motion_factor is not None:
            raise ValueError("A fixed rigid contact cannot own a pose factor.")
        if self.pair.slave_boundary.domain is not self.displacement.function_space.mesh:
            raise ValueError("Contact slave and displacement must share one mesh.")
        if self.adapter.function_space is not self.displacement.function_space:
            raise ValueError("Contact trace and displacement must share one function space.")

    @property
    def surface(
        self,
    ) -> (
        TriangulatedRigidSurface | SmoothRigidSurface
        | ConvexPolyhedralRigidSurface | ClosedTriangulatedRigidSolid
    ):
        return self.pair.rigid_body.surface

    @property
    def name(self) -> str:
        return self.pair.name

    @property
    def has_prescribed_motion(self) -> bool:
        return self.pair.rigid_body.motion_schedule is not None

    @property
    def providers(self) -> tuple[DolfinxImplicitRigidContact, ...]:
        return (self,)

    def snapshot_motion(self) -> float | None:
        return (
            None if self.motion_factor is None
            else float(np.asarray(self.motion_factor.value).reshape(-1)[0])
        )

    def restore_motion(self, saved: float | None) -> None:
        if self.motion_factor is None:
            if saved is not None:
                raise ValueError("Fixed rigid contact has no pose state to restore.")
        elif saved is None:
            raise ValueError("Moving rigid contact requires a saved pose factor.")
        else:
            self.update_motion(saved)

    def dual_evidences(self, problem):
        return (self.dual_evidence(problem),)

    def snapshot_runtime_state(self):
        """Retain immutable records for a cheap in-memory rollback boundary."""

        if self.friction_state is None:
            return None
        if self.friction_kinematics is None or self.friction_kinematics.accepted is None:
            raise RuntimeError("Implicit friction has no accepted kinematic state.")
        if self.friction_state.trial is not None or self.friction_kinematics.trial is not None:
            raise RuntimeError("Cannot snapshot a provisional friction trial.")
        return (self.friction_state.accepted, self.friction_kinematics.accepted)

    def restore_runtime_state(self, saved) -> None:
        """Restore immutable accepted records without building JSON lists."""

        if self.friction_state is None:
            if saved is not None:
                raise ValueError("Frictionless rigid pair has no runtime State.")
            return
        if (
            not isinstance(saved, tuple) or len(saved) != 2
            or (saved[0] is not None and not isinstance(saved[0], TangentialContactRecord))
            or not isinstance(saved[1], TangentialKinematicRecord)
            or not np.array_equal(saved[1].point_ids, self.adapter.trace.point_ids)
            or (saved[0] is not None and not np.array_equal(
                saved[0].point_ids, saved[1].point_ids,
            ))
        ):
            raise ValueError("Implicit friction runtime State identity differs.")
        self.friction_state = TangentialContactState(accepted=saved[0])
        self.friction_kinematics = TangentialKinematicState(accepted=saved[1])

    def snapshot_accepted_state(self) -> dict[str, object] | None:
        """Return JSON-safe friction State at an accepted increment boundary."""

        if self.friction_state is None:
            return None
        if self.friction_kinematics is None or self.friction_kinematics.accepted is None:
            raise RuntimeError("Implicit friction has no accepted kinematic state.")
        if self.friction_state.trial is not None or self.friction_kinematics.trial is not None:
            raise RuntimeError("Implicit friction checkpoint cannot contain a trial.")
        accepted = self.friction_state.accepted
        kinematic = self.friction_kinematics.accepted
        return {
            "schema": "agentfem.implicit-planar-friction-state.v1",
            "pair": self.name,
            "point_ids": self.adapter.trace.point_ids.tolist(),
            "friction": (
                None if accepted is None else {
                    "schema": accepted.snapshot()["schema"],
                    "point_ids": accepted.point_ids.tolist(),
                    "normals": accepted.normals.tolist(),
                    "elastic_slips": accepted.elastic_slips.tolist(),
                    "cumulative_dissipation_densities": (
                        accepted.cumulative_dissipation_densities.tolist()
                    ),
                    "cumulative_separation_release_densities": (
                        accepted.cumulative_separation_release_densities.tolist()
                    ),
                }
            ),
            "kinematics": kinematic.snapshot() | {
                "point_ids": kinematic.point_ids.tolist(),
                "slave_positions": kinematic.slave_positions.tolist(),
            },
        }

    def restore_accepted_state(self, saved: dict[str, object] | None) -> None:
        """Validate all point-keyed State before atomically replacing it."""

        if self.friction_state is None:
            if saved is not None:
                raise ValueError("A frictionless rigid pair cannot restore friction State.")
            return
        if (
            not isinstance(saved, dict)
            or set(saved) != {"schema", "pair", "point_ids", "friction", "kinematics"}
            or saved["schema"] != "agentfem.implicit-planar-friction-state.v1"
            or saved["pair"] != self.name
            or not np.array_equal(saved["point_ids"], self.adapter.trace.point_ids)
        ):
            raise ValueError("Implicit friction checkpoint schema, pair or point IDs differ.")
        friction = (
            None if saved["friction"] is None else
            TangentialContactRecord.from_snapshot(saved["friction"])
        )
        kinematic = TangentialKinematicRecord.from_snapshot(saved["kinematics"])
        if not np.array_equal(kinematic.point_ids, self.adapter.trace.point_ids) or (
            friction is not None and not np.array_equal(
                friction.point_ids, kinematic.point_ids
            )
        ):
            raise ValueError("Implicit friction accepted point identity differs.")
        expected_factor = 0.0 if self.motion_factor is None else self.snapshot_motion()
        if abs(kinematic.motion_factor - expected_factor) > 1.0e-12:
            raise ValueError("Implicit friction checkpoint rigid pose differs.")
        self.friction_state = TangentialContactState(accepted=friction)
        self.friction_kinematics = TangentialKinematicState(accepted=kinematic)

    def validate_accepted_configuration(self) -> None:
        """Refuse a friction checkpoint detached from its accepted field."""

        if self.friction_state is None:
            return
        accepted = self.friction_kinematics.accepted
        current = self.adapter.evaluate(self.displacement)
        if not np.array_equal(accepted.point_ids, current.trace.point_ids) or (
            not np.allclose(
                accepted.slave_positions, current.query_points,
                rtol=1.0e-12, atol=1.0e-12,
            )
        ):
            raise ValueError(
                "Implicit friction accepted kinematics differ from the "
                "accepted displacement field."
            )
        factor = 0.0 if self.motion_factor is None else self.snapshot_motion()
        if abs(accepted.motion_factor - factor) > 1.0e-12:
            raise ValueError(
                "Implicit friction accepted rigid pose differs from the "
                "accepted kinematic State."
            )

    def commit_accepted(self) -> None:
        """Commit one friction trial from the current accepted displacement."""

        if self.friction_state is None:
            return
        if self.friction_kinematics is None or self.friction_kinematics.accepted is None:
            raise RuntimeError("Implicit friction has no accepted kinematic state.")
        evaluation, record, response, factor = self._current_point_response()
        schedule = self.pair.rigid_body.motion_schedule
        relative = relative_contact_displacement_increment(
            self.friction_kinematics.accepted, record,
            motion=None if schedule is None else schedule.motion,
            current_motion_factor=factor,
        )
        candidate = self.pair.friction.evaluate(
            point_ids=record.point_ids,
            normal_response=response,
            relative_displacement_increment=relative,
            accepted=self.friction_state.accepted,
        )
        # Validate integrated energy channels before any State mutation.
        evaluation.assemble_friction(record, candidate)
        self.friction_state.begin(candidate)
        try:
            self.friction_kinematics.begin(record, motion_factor=factor)
            self.friction_state.commit()
            self.friction_kinematics.commit()
        except Exception:
            self.friction_state.rollback()
            self.friction_kinematics.rollback()
            raise

    def update_motion(self, factor: float) -> None:
        selected = float(factor)
        if not np.isfinite(selected) or not 0.0 <= selected <= 1.0 + 1.0e-12:
            raise ValueError("Contact load factor must lie in [0, 1].")
        if self.motion_factor is not None:
            self.motion_factor.value = selected

    def set_load_factor(self, factor: float) -> None:
        """Follow a Procedure-owned accepted or trial load coordinate."""

        self.update_motion(factor)

    def capabilities(self):
        from ..constraints import ConstraintCapabilities

        return ConstraintCapabilities(
            kind="contact_constraint",
            enforcement="one_sided_rigid_surface_penalty",
            analyses=("nonlinear_static",),
            procedures=("incremental_newton",),
            strict=False,
            supports_parallel=False,
            reaction_evidence="provider_dual_required",
            work_evidence=(
                "provider_dual_path_required"
                if self.has_prescribed_motion else "internal_energy_operator"
            ),
        )

    def _current_point_response(self):
        """Reproject the current trial; never reuse a previous gap or pressure."""
        evaluation = self.adapter.evaluate(self.displacement)
        schedule = self.pair.rigid_body.motion_schedule
        factor = (
            0.0 if self.motion_factor is None
            else float(np.asarray(self.motion_factor.value).reshape(-1)[0])
        )
        projector = self.surface if self.projector is None else self.projector
        projection = projector.project(
            evaluation.query_points,
            motion=None if schedule is None else schedule.motion,
            factor=factor,
        )
        if not projection.all_valid:
            first = int(np.flatnonzero(~projection.valid)[0])
            raise ValueError(
                "AFM-IMPLICIT-CONTACT-PROJECTION: invalid projection at "
                f"contact_point_id={int(self.adapter.trace.point_ids[first])}, "
                f"status={projection.status_codes[first]}, "
                f"surface={self.surface.name}; the current trial is rejected "
                "without inventing a contact normal."
            )
        record = ContactProjectionRecord(self.adapter.trace.point_ids, projection)
        response = self.pair.law.evaluate(record.projection)
        return evaluation, record, response, factor

    def current_point_diagnostics(self) -> dict[str, float | int | str]:
        """Unsmoothed reference-trace diagnostics at the current trial pose."""

        evaluation, _record, response, _factor = self._current_point_response()
        active = np.asarray(response.active, dtype=bool)
        return {
            "location": "reference_boundary_quadrature",
            "pressure_measure": "penalty_force_per_reference_area_not_cauchy_pressure",
            "active_point_count": int(np.count_nonzero(active)),
            "active_reference_area": float(np.sum(evaluation.trace.weights[active])),
            "maximum_pressure": float(np.max(response.pressures, initial=0.0)),
            "maximum_penetration": float(np.max(response.penetration, initial=0.0)),
        }

    def accepted_point_samples(self) -> dict[str, object]:
        """Return unsmoothed point arrays without changing accepted State.

        The owning Procedure must first confirm that its displacement and
        load coordinate are accepted. This provider checks its own tangential
        State and recomputes the current projection; no smoothed nodal pressure
        or guessed Cauchy contact stress is produced.
        """

        self.validate_accepted_configuration()
        evaluation, record, normal, factor = self._current_point_response()
        point_count = evaluation.trace.point_count
        friction_traction = np.zeros((point_count, 3), dtype=float)
        elastic_slip = np.zeros((point_count, 3), dtype=float)
        cumulative_dissipation = np.zeros(point_count, dtype=float)
        if self.friction_state is not None:
            accepted = self.friction_state.accepted
            relative = relative_contact_displacement_increment(
                self.friction_kinematics.accepted, record,
                motion=(
                    None if self.pair.rigid_body.motion_schedule is None
                    else self.pair.rigid_body.motion_schedule.motion
                ),
                current_motion_factor=factor,
            )
            response = self.pair.friction.evaluate(
                point_ids=record.point_ids, normal_response=normal,
                relative_displacement_increment=relative, accepted=accepted,
            )
            friction_traction = np.asarray(response.surface_generalized_tractions)
            if accepted is not None:
                elastic_slip = np.asarray(accepted.elastic_slips)
                cumulative_dissipation = np.asarray(
                    accepted.cumulative_dissipation_densities
                )
        projection = record.projection
        return {
            "constraint_name": self.name,
            "surface_name": self.surface.name,
            "geometry_fingerprint": self.surface.geometry_fingerprint,
            "trace_source": evaluation.trace.source,
            "load_factor": factor,
            "point_ids": record.point_ids.copy(),
            "slave_positions": projection.query_points.copy(),
            "master_closest_points": projection.closest_points.copy(),
            "surface_normals": projection.normals.copy(),
            "signed_gaps": projection.signed_gaps.copy(),
            "reference_weights": evaluation.trace.weights.copy(),
            "normal_pressures": normal.pressures.copy(),
            "penetrations": normal.penetration.copy(),
            "active": normal.active.copy(),
            "entity_ids": (
                np.full(point_count, -1, dtype=np.int64)
                if projection.entity_ids is None else projection.entity_ids.copy()
            ),
            "surface_friction_tractions": friction_traction.copy(),
            "accepted_elastic_tangential_slips": elastic_slip.copy(),
            "cumulative_friction_dissipation_densities": cumulative_dissipation.copy(),
        }

    def evaluate(self) -> ImplicitRigidContactContribution:
        """Reproject the current displacement and assemble a fresh trial Operator."""

        evaluation, record, response, factor = self._current_point_response()
        schedule = self.pair.rigid_body.motion_schedule
        reference_point = self.pair.rigid_body.reference_point
        if schedule is not None:
            reference_point = schedule.motion.state(factor)["reference_point"]
        if self.projector is None:
            contribution = linearize_smooth_rigid_contact(
                evaluation, record, response, self.surface,
                motion=None if schedule is None else schedule.motion,
                factor=factor, surface_reference_point=reference_point,
            )
        elif isinstance(self.surface, ClosedTriangulatedRigidSolid):
            contribution = linearize_closed_rigid_solid_contact(
                evaluation, record, response, self.surface,
                motion=None if schedule is None else schedule.motion,
                factor=factor, surface_reference_point=reference_point,
            )
        else:
            contribution = linearize_piecewise_planar_contact(
                evaluation, record, response, self.surface,
                surface_reference_point=reference_point,
            )
        if self.friction_state is None:
            return contribution
        if self.friction_kinematics is None or self.friction_kinematics.accepted is None:
            raise RuntimeError("Implicit friction requires accepted slave kinematics.")
        relative = relative_contact_displacement_increment(
            self.friction_kinematics.accepted, record,
            motion=None if schedule is None else schedule.motion,
            current_motion_factor=factor,
        )
        friction_response = self.pair.friction.evaluate(
            point_ids=record.point_ids,
            normal_response=response,
            relative_displacement_increment=relative,
            accepted=self.friction_state.accepted,
        )
        friction_assembly = evaluation.assemble_friction(
            record, friction_response,
            surface_reference_point=reference_point,
        )
        friction_tangent = linearize_piecewise_planar_penalty_friction(
            evaluation, record, response, friction_response,
            self.pair.friction, self.surface,
            projector=(
                self.projector
                if isinstance(self.projector, TriangleSurfaceBVH) else None
            ),
            motion=None if schedule is None else schedule.motion,
            accepted_motion_factor=self.friction_kinematics.accepted.motion_factor,
            current_motion_factor=factor,
            cap_policy="stick_generalized",
            opening_policy="inactive_generalized",
        )
        return replace(
            contribution,
            friction_assembly=friction_assembly,
            friction_tangent=friction_tangent,
        )

    def add_residual(self, vector, contribution: ImplicitRigidContactContribution) -> None:
        """Add local scalar rows before the owning solver performs BC lifting."""

        values = contribution.nodal_structural_residual.reshape(-1)
        if vector.array.shape != values.shape:
            raise ValueError("Implicit contact residual does not match PETSc local rows.")
        vector.array[:] += values

    def add_tangent(
        self, matrix, contribution: ImplicitRigidContactContribution,
        *, constrained_dofs=(),
    ) -> None:
        """Insert free-row/column contact tangent before PETSc assembly.

        The material matrix is assembled with DOLFINx strong rows/columns
        already eliminated.  Contact lifting uses the *full* local tangent in
        the owning problem; adding constrained COO entries here would undo
        those strong rows and silently change the boundary condition.
        """

        from petsc4py import PETSc

        if contribution.tangent_values.size == 0:
            return
        # DOLFINx inserts the UFL matrix entries. PETSc requires a flush
        # before switching this same matrix to additive contact entries.
        matrix.assemble(PETSc.Mat.AssemblyType.FLUSH)
        selected = np.asarray(constrained_dofs, dtype=np.int64).reshape(-1)
        if selected.size:
            if np.any(selected < 0) or np.any(selected >= matrix.getSize()[0]):
                raise ValueError("Constrained contact DOFs must address the tangent matrix.")
            free = ~np.isin(contribution.tangent_rows, selected) & ~np.isin(
                contribution.tangent_columns, selected
            )
            rows = contribution.tangent_rows[free]
            columns = contribution.tangent_columns[free]
            values = contribution.tangent_values[free]
        else:
            rows = contribution.tangent_rows
            columns = contribution.tangent_columns
            values = contribution.tangent_values
        if not len(rows):
            return
        order = np.lexsort((columns, rows))
        rows = rows[order]
        columns = columns[order]
        values = values[order]
        stops = np.r_[np.flatnonzero(np.diff(rows)) + 1, len(rows)]
        start = 0
        for stop in stops:
            unique_columns, inverse = np.unique(columns[start:stop], return_inverse=True)
            row_values = np.bincount(inverse, weights=values[start:stop])
            matrix.setValues(
                [int(rows[start])],
                unique_columns.astype(np.int32),
                row_values,
                addv=PETSc.InsertMode.ADD_VALUES,
            )
            start = stop

    def dual_evidence(self, problem):
        """Observe one converged/accepted trial without changing contact state."""

        from dolfinx import fem

        from ..constraints import constraint_dual

        if getattr(problem, "solution", None) is not self.displacement:
            raise ValueError("Implicit contact dual requires its consuming problem.")
        if (
            getattr(problem, "last_solve_info", None) is None
            and getattr(problem, "_constraint_dual_state", None) != "accepted"
        ):
            raise RuntimeError("Implicit contact dual requires an accepted solve.")
        contribution = self.evaluate()
        distribution = fem.Function(
            self.displacement.function_space,
            name=f"{self.pair.name}_reaction",
        )
        distribution.x.array[:] = -contribution.nodal_structural_residual.reshape(-1)
        distribution.x.scatter_forward()
        force = contribution.contact_force_on_structure
        coordinate = None
        if self.has_prescribed_motion:
            moment = contribution.surface_generalized_moment
            if moment is None:
                raise RuntimeError("Moving rigid contact requires moment evidence.")
            factor = float(np.asarray(self.motion_factor.value).reshape(-1)[0])
            force = np.concatenate((force, moment))
            coordinate = self.pair.rigid_body.motion_schedule.motion.generalized_coordinate(
                factor
            )
        return constraint_dual(
            self.pair,
            force=force,
            coordinate=coordinate,
            resultant=contribution.contact_force_on_structure,
            distribution=distribution,
            role="contact_constraint",
            source="dolfinx_implicit_rigid_contact",
            diagnostics={
                "active_point_count": contribution.active_points,
                "contact_potential": contribution.potential_energy,
                "friction": (
                    None if contribution.friction_assembly is None
                    else contribution.friction_assembly.summary()
                ),
                "surface_generalized_moment": (
                    None if contribution.assembly.surface_generalized_moment is None
                    else contribution.assembly.surface_generalized_moment.tolist()
                ),
                "projection": (
                    "closed_concave_signed_distance_plus_exact_bvh"
                    if isinstance(self.surface, ClosedTriangulatedRigidSolid)
                    else "closed_convex_support_planes_plus_exact_exterior_bvh"
                    if isinstance(self.surface, ConvexPolyhedralRigidSurface)
                    else "current_geometry_bvh" if self.projector is not None
                    else "exact_analytical_current_pose"
                ),
                "linearization": self._linearization_name(),
            },
        )

    def summary(self) -> dict[str, object]:
        friction_geometry = (
            "closed_concave" if isinstance(self.surface, ClosedTriangulatedRigidSolid)
            else "closed_convex" if isinstance(self.surface, ConvexPolyhedralRigidSurface)
            else "triangulated_face_branch"
            if isinstance(self.surface, TriangulatedRigidSurface) else "plane"
        )
        return {
            "kind": "dolfinx_implicit_rigid_contact",
            "pair": self.pair.summary(),
            "trace": self.adapter.summary(),
            "search": (
                self.projector.summary() if self.projector is not None
                else {"kind": "exact_analytical_projection"}
            ),
            "linearization": self._linearization_name(),
            "execution_scope": (
                f"serial_moving_{friction_geometry}_coulomb_reference_area"
                if self.friction_state is not None and self.has_prescribed_motion
                else f"serial_fixed_{friction_geometry}_coulomb_reference_area"
                if self.friction_state is not None
                else "serial_frictionless_reference_area"
            ),
        }

    def _linearization_name(self) -> str:
        if self.friction_state is not None:
            geometry = (
                "closed_convex_piecewise_planar"
                if isinstance(self.surface, ConvexPolyhedralRigidSurface)
                else "triangulated_face_branch"
                if isinstance(self.surface, TriangulatedRigidSurface) else "plane"
            )
            return (
                f"moving_{geometry}_normal_plus_coulomb_semismooth_return_map"
                if self.has_prescribed_motion else
                f"fixed_{geometry}_normal_plus_coulomb_semismooth_return_map"
            )
        if isinstance(self.surface, ConvexPolyhedralRigidSurface):
            return "closed_convex_polyhedron_piecewise_affine_semismooth"
        if isinstance(self.surface, ClosedTriangulatedRigidSolid):
            return "closed_triangle_solid_exact_feature_distance_semismooth"
        if self.projector is not None:
            return "piecewise_planar_normal_branch_at_current_pose"
        return "smooth_signed_gap_with_curvature_at_current_pose"


@dataclass(eq=False)
class DolfinxImplicitRigidContacts:
    """Serial additive Operator with separate identity and motion per rigid pair.

    The owning J2 Procedure commits or rolls back the complete increment.
    Each pair keeps its own point-keyed tangential State and prescribed pose.
    Distributed assembly is intentionally excluded.
    """

    providers: tuple[DolfinxImplicitRigidContact, ...]

    def __post_init__(self) -> None:
        self.providers = tuple(self.providers)
        if len(self.providers) < 2 or any(
            not isinstance(item, DolfinxImplicitRigidContact)
            for item in self.providers
        ):
            raise ValueError("Multiple implicit tools require two or more reviewed rigid-pair providers.")
        names = tuple(item.name for item in self.providers)
        if len(set(names)) != len(names):
            raise ValueError("Implicit rigid contact pair names must be unique.")
        displacement = self.providers[0].displacement
        if any(item.displacement is not displacement for item in self.providers):
            raise ValueError("Implicit rigid pairs must consume one displacement field.")

    @property
    def has_prescribed_motion(self) -> bool:
        return any(item.has_prescribed_motion for item in self.providers)

    def update_motion(self, factor: float) -> None:
        for item in self.providers:
            item.update_motion(factor)

    def snapshot_motion(self) -> tuple[tuple[str, float | None], ...]:
        return tuple((item.name, item.snapshot_motion()) for item in self.providers)

    def restore_motion(self, saved) -> None:
        records = tuple(saved)
        if tuple(name for name, _ in records) != tuple(
            item.name for item in self.providers
        ):
            raise ValueError("Saved rigid-tool pose identities do not match the model.")
        for item, (_, factor) in zip(self.providers, records, strict=True):
            item.restore_motion(factor)

    def snapshot_runtime_state(self):
        """Keep each pair's immutable accepted records without JSON copying."""

        return tuple(
            (item.name, item.snapshot_runtime_state()) for item in self.providers
        )

    def restore_runtime_state(self, saved) -> None:
        records = tuple(saved)
        if tuple(name for name, _ in records) != tuple(
            item.name for item in self.providers
        ):
            raise ValueError("Implicit rigid-pair runtime identities differ.")
        previous = self.snapshot_runtime_state()
        try:
            for item, (_, state) in zip(self.providers, records, strict=True):
                item.restore_runtime_state(state)
        except Exception:
            for item, (_, original) in zip(self.providers, previous, strict=True):
                item.restore_runtime_state(original)
            raise

    def snapshot_accepted_state(self) -> dict[str, object] | None:
        pairs = {
            item.name: item.snapshot_accepted_state()
            for item in self.providers
        }
        if not any(value is not None for value in pairs.values()):
            return None
        return {
            "schema": "agentfem.implicit-rigid-pair-states.v1",
            "pairs": pairs,
        }

    def restore_accepted_state(self, saved: dict[str, object] | None) -> None:
        has_friction = any(item.friction_state is not None for item in self.providers)
        if not has_friction:
            if saved is not None:
                raise ValueError("Frictionless rigid pairs cannot restore friction State.")
            return
        if (
            not isinstance(saved, dict)
            or set(saved) != {"schema", "pairs"}
            or saved["schema"] != "agentfem.implicit-rigid-pair-states.v1"
            or not isinstance(saved["pairs"], dict)
            or set(saved["pairs"]) != {item.name for item in self.providers}
        ):
            raise ValueError("Implicit rigid-pair friction State identity differs.")
        previous = tuple(item.snapshot_accepted_state() for item in self.providers)
        try:
            for item in self.providers:
                item.restore_accepted_state(saved["pairs"][item.name])
        except Exception:
            for item, original in zip(self.providers, previous, strict=True):
                item.restore_accepted_state(original)
            raise

    def commit_accepted(self) -> None:
        previous = self.snapshot_runtime_state()
        try:
            for item in self.providers:
                item.commit_accepted()
        except Exception:
            self.restore_runtime_state(previous)
            raise

    def validate_accepted_configuration(self) -> None:
        for item in self.providers:
            item.validate_accepted_configuration()

    def evaluate(self) -> CombinedImplicitRigidContactContribution:
        return CombinedImplicitRigidContactContribution(tuple(
            (item.name, item.evaluate()) for item in self.providers
        ))

    def add_residual(self, vector, contribution: CombinedImplicitRigidContactContribution) -> None:
        values = contribution.nodal_structural_residual.reshape(-1)
        if vector.array.shape != values.shape:
            raise ValueError("Combined implicit contact residual does not match PETSc local rows.")
        vector.array[:] += values

    def add_tangent(self, matrix, contribution, *, constrained_dofs=()) -> None:
        for item, (name, selected) in zip(
            self.providers, contribution.pairs, strict=True
        ):
            if name != item.name:
                raise ValueError("Combined contact tangent pair identity changed.")
            item.add_tangent(matrix, selected, constrained_dofs=constrained_dofs)

    def dual_evidences(self, problem):
        return tuple(item.dual_evidence(problem) for item in self.providers)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_implicit_rigid_contacts",
            "pairs": tuple(item.summary() for item in self.providers),
            "execution_scope": (
                "serial_additive_with_fixed_plane_coulomb_reference_area"
                if any(item.friction_state is not None for item in self.providers)
                else "serial_frictionless_additive_reference_area"
            ),
        }


# Compatibility spelling for the original experimental facet adapter.
DolfinxImplicitFacetContact = DolfinxImplicitRigidContact


def dolfinx_implicit_rigid_contact(
    pair: RigidContactPair,
    displacement,
    *,
    contact_quadrature_degree: int | None = None,
) -> DolfinxImplicitRigidContact:
    """Lower one bounded rigid pair to a serial current-trial Newton Operator."""

    if not isinstance(pair, RigidContactPair):
        raise TypeError("Implicit rigid contact requires a RigidContactPair.")
    surface = pair.rigid_body.surface
    if not isinstance(surface, (
        TriangulatedRigidSurface, SmoothRigidSurface, ConvexPolyhedralRigidSurface,
        ClosedTriangulatedRigidSolid,
    )):
        raise TypeError("Implicit rigid contact requires reviewed smooth, triangle, or closed-solid geometry.")
    adapter = dolfinx_boundary_region_contact_trace(
        pair.slave_boundary, displacement.function_space,
        contact_quadrature_degree=contact_quadrature_degree,
    )
    from dolfinx import fem

    return DolfinxImplicitRigidContact(
        pair=pair, adapter=adapter, displacement=displacement,
        projector=(
            surface if isinstance(surface, (
                ConvexPolyhedralRigidSurface, ClosedTriangulatedRigidSolid,
            ))
            else triangle_surface_bvh(surface)
            if isinstance(surface, TriangulatedRigidSurface) else None
        ),
        motion_factor=(
            None if pair.rigid_body.motion_schedule is None
            else fem.Constant(displacement.function_space.mesh, 0.0)
        ),
    )


def dolfinx_implicit_rigid_contacts(
    pairs: tuple[RigidContactPair, ...],
    displacement,
    *,
    contact_quadrature_degree: int | None = None,
) -> DolfinxImplicitRigidContacts:
    """Lower independently named rigid pairs on one serial displacement field."""

    selected = tuple(pairs)
    if len(selected) < 2:
        raise ValueError("Multiple rigid contact requires at least two pairs.")
    if len({item.name for item in selected}) != len(selected):
        raise ValueError("Multiple rigid contact pair names must be unique.")
    return DolfinxImplicitRigidContacts(tuple(
        dolfinx_implicit_rigid_contact(
            item, displacement,
            contact_quadrature_degree=contact_quadrature_degree,
        ) for item in selected
    ))


def dolfinx_implicit_facet_contact(
    pair: RigidContactPair,
    displacement,
) -> DolfinxImplicitFacetContact:
    """Lower a bounded triangle tool to one serial Newton contribution."""

    if not isinstance(pair, RigidContactPair):
        raise TypeError("Implicit facet contact requires a RigidContactPair.")
    if not isinstance(pair.rigid_body.surface, TriangulatedRigidSurface):
        raise TypeError("Implicit facet contact requires TriangulatedRigidSurface.")
    return dolfinx_implicit_rigid_contact(pair, displacement)


__all__ = [
    "DolfinxImplicitFacetContact", "DolfinxImplicitRigidContact",
    "DolfinxImplicitRigidContacts", "dolfinx_implicit_facet_contact",
    "dolfinx_implicit_rigid_contact", "dolfinx_implicit_rigid_contacts",
]
