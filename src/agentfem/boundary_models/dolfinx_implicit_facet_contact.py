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
from .contact_integration_modes import resolve_contact_integration_modes
from .contact_active_partition import partition_planar_p1_trace
from .contact_convex_partition import (
    _unique_support_planes, partition_convex_p1_trace,
)
from .contact_friction import (
    TangentialContactRecord, TangentialContactState, TangentialKinematicRecord,
    TangentialKinematicState, relative_contact_displacement_increment,
    rigid_tangential_history_rotation,
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
from .implicit_closed_contact import (
    linearize_closed_rigid_solid_contact,
    sampled_mixed_feature_slave_faces,
)
from .implicit_friction import linearize_piecewise_planar_penalty_friction
from .implicit_smooth_contact import linearize_smooth_rigid_contact
from ._petsc_contact_rows import (
    add_contact_coo_tangent,
    add_contact_nodal_residual,
)
from .initial_gap_bound import (
    certify_initial_convex_triangle_gap, certify_initial_triangle_gap,
    certify_initial_radial_triangle_gap,
    certify_initial_reviewed_planar_branch_gap, initial_gap_bound_method,
)
from .contact_separation_bound import certify_current_triangle_separation
from .contact_planar_branch_bound import certify_reviewed_planar_branch
from .contact_reference_refinement import refine_p1_triangle_reference_trace
from .contact_radial_partition import (
    partition_radial_p1_trace, radial_partition_policy_summary,
)
from .contact_integration_modes import CONTACT_INTEGRATION_MODES
from .rigid import (
    RigidCylinderSurface, RigidPlaneSurface, RigidSphereSurface,
    SmoothRigidSurface, TriangulatedRigidSurface,
)
from .search import TriangleSurfaceBVH, triangle_surface_bvh


@dataclass(eq=False)
class DolfinxImplicitRigidContact:
    """Current-trial rigid contact with bounded planar-branch tangential State."""

    pair: RigidContactPair
    adapter: DolfinxContactTraceAdapter
    displacement: object
    projector: TriangleSurfaceBVH | ConvexPolyhedralRigidSurface | ClosedTriangulatedRigidSolid | None
    motion_factor: object | None = None
    contact_integration: str = "gauss"
    friction_state: TangentialContactState | None = field(default=None, init=False)
    friction_kinematics: TangentialKinematicState | None = field(default=None, init=False)
    _support_planes: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False,
    )
    _initial_contact_diagnostics: dict[str, object] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.pair, RigidContactPair):
            raise TypeError("Implicit rigid contact requires a RigidContactPair.")
        if not isinstance(self.adapter, DolfinxContactTraceAdapter):
            raise TypeError("Implicit rigid contact requires a DOLFINx trace adapter.")
        surface = self.pair.rigid_body.surface
        if self.contact_integration not in CONTACT_INTEGRATION_MODES:
            raise ValueError("Unsupported implicit contact integration method.")
        if self.contact_integration == "certified_closed_two_feature_research":
            raise NotImplementedError(
                "Certified face integration requires its face-level provider; "
                "the point-sampled contact adapter cannot lower this mode."
            )
        if self.contact_integration == "planar_active_partition" and (
            not isinstance(surface, RigidPlaneSurface)
            or self.pair.friction is not None
            or np.asarray(self.pair.law.penalty).ndim != 0
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
        ):
            raise NotImplementedError(
                "AFM-CONTACT-PARTITION: the trial active partition requires "
                "frictionless analytical-plane contact on a P1 tetrahedral "
                "triangle with the default positive reference-area rule."
            )
        if self.contact_integration == "convex_feature_partition" and (
            not isinstance(surface, ConvexPolyhedralRigidSurface)
            or self.pair.friction is not None
            or np.asarray(self.pair.law.penalty).ndim != 0
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
        ):
            raise NotImplementedError(
                "AFM-CONTACT-FEATURE-PARTITION: requires frictionless certified "
                "convex rigid geometry, scalar penalty, affine P1 tetrahedral "
                "slave triangles and the default positive reference-area rule."
            )
        if self.contact_integration == "radial_active_partition" and (
            not isinstance(surface, (RigidSphereSurface, RigidCylinderSurface))
            or surface.dimension != 3
            or self.pair.friction is not None
            or np.asarray(self.pair.law.penalty).ndim != 0
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
        ):
            raise NotImplementedError(
                "AFM-CONTACT-RADIAL-PARTITION: requires a frictionless analytical "
                "3D sphere or cylinder, scalar penalty, affine P1 slave triangles "
                "and the default positive reference-area rule."
            )
        planar_branch_mode = self.contact_integration in {
            "closed_planar_branch_partition", "reviewed_planar_branch_partition",
        }
        planar_branch_surfaces = (
            (ClosedTriangulatedRigidSolid,)
            if self.contact_integration == "closed_planar_branch_partition"
            else (ClosedTriangulatedRigidSolid, TriangulatedRigidSurface)
        )
        if planar_branch_mode and (
            not isinstance(surface, planar_branch_surfaces)
            or self.pair.friction is not None
            or np.asarray(self.pair.law.penalty).ndim != 0
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
        ):
            code = (
                "AFM-CONTACT-CLOSED-PLANAR-BRANCH"
                if self.contact_integration == "closed_planar_branch_partition"
                else "AFM-CONTACT-REVIEWED-PLANAR-BRANCH"
            )
            raise NotImplementedError(
                f"{code}: requires a frictionless "
                "reviewed triangle tool, scalar penalty, affine P1 triangular "
                "slave faces and the default positive reference-area rule."
            )
        if self.contact_integration == "convex_feature_partition":
            self._support_planes = _unique_support_planes(surface)
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
        self._audit_initial_contact()

    def _audit_initial_contact(self) -> None:
        """Sample reference geometry before Newton without moving the mesh.

        A prescribed initial overclosure is a modeling choice, not a hidden
        geometry adjustment. First-order reference geometry against an
        analytical plane has an exact vertex extrema bound even with a
        quadratic displacement field. This does not bound deformed P2 faces.
        For other geometry
        the sampled evidence is *not* a bound, so it cannot enforce a limit.
        """

        evaluation = self.adapter.trace.evaluate(
            self.adapter.reference_nodal_positions,
        )
        projector = self.surface if self.projector is None else self.projector
        projection = projector.project(evaluation.query_points, factor=0.0)
        valid = np.asarray(projection.valid, dtype=bool)
        gaps = np.asarray(projection.signed_gaps, dtype=float)
        penetrations = np.maximum(0.0, -gaps[valid])
        maximum = float(np.max(penetrations, initial=0.0))
        limit = self.pair.initial_penetration_limit
        method = initial_gap_bound_method(
            self.surface,
            interpolation_degree=self.adapter.interpolation_degree,
            geometry_degree=self.adapter.geometry_degree,
            facet_topology=self.adapter.facet_topology,
            contact_integration=self.contact_integration,
        )
        if method == "exact_plane_geometry_vertices" and self.adapter.reference_facet_vertices is None:
            # Expert-created adapters without authoritative face geometry may
            # still report samples, but cannot claim a reference-face bound.
            method = None
        certified = method is not None and limit is not None
        maximum_bound = None
        certificate_samples = 0
        certificate_depth = 0
        if method in {"exact_plane_nodal", "exact_plane_geometry_vertices"}:
            certified = True
            if method == "exact_plane_geometry_vertices":
                points = self.adapter.reference_facet_vertices
                if points is None:
                    raise NotImplementedError(
                        "AFM-CONTACT-INITIAL-BOUND: authoritative reference facet vertices are required.")
            else:
                boundary_nodes = np.unique(evaluation.trace.node_ids.reshape(-1))
                points = self.adapter.reference_nodal_positions[boundary_nodes]
            nodal = projector.project(points, factor=0.0)
            if not nodal.all_valid:
                raise ValueError(
                    "AFM-CONTACT-INITIAL-PROJECTION: analytical-plane "
                    "reference-node projection is invalid."
                )
            maximum_bound = float(np.max(
                np.maximum(0.0, -np.asarray(nodal.signed_gaps)), initial=0.0,
            ))
            certificate_samples = len(points)
        elif method in {
            "adaptive_closed_signed_distance_lipschitz",
            "convex_support_plane_or_adaptive_lipschitz",
        } and limit is not None:
            face_nodes = np.unique(evaluation.trace.node_ids, axis=0)
            certify = (
                certify_initial_convex_triangle_gap
                if method == "convex_support_plane_or_adaptive_lipschitz"
                else certify_initial_triangle_gap
            )
            certificate = certify(
                self.adapter.reference_nodal_positions[face_nodes],
                projector,
                penetration_limit=limit,
            )
            maximum_bound = certificate.maximum_penetration_bound
            certificate_samples = certificate.sampled_point_count
            certificate_depth = certificate.maximum_refinement_depth
        elif method == "certified_reviewed_planar_branch_nodal" and limit is not None:
            face_nodes = np.unique(evaluation.trace.node_ids, axis=0)
            certificate = certify_initial_reviewed_planar_branch_gap(
                self.adapter.reference_nodal_positions[face_nodes],
                self.surface,
                projector=self.projector,
                penetration_limit=limit,
            )
            maximum_bound = certificate.maximum_penetration_bound
            certificate_samples = certificate.sampled_point_count
            certificate_depth = certificate.maximum_refinement_depth
        elif method == "exact_radial_face_extrema" and limit is not None:
            face_nodes = np.unique(evaluation.trace.node_ids, axis=0)
            certificate = certify_initial_radial_triangle_gap(
                self.adapter.reference_nodal_positions[face_nodes],
                self.surface, penetration_limit=limit,
            )
            maximum_bound = certificate.maximum_penetration_bound
            certificate_samples = certificate.sampled_point_count
        self._initial_contact_diagnostics = {
            "sampling": "reference_boundary_quadrature_at_reference_rigid_pose",
            "sampled_point_count": int(valid.size),
            "invalid_projection_count": int(np.count_nonzero(~valid)),
            "penetrating_point_count": int(np.count_nonzero(penetrations > 0.0)),
            "maximum_penetration": maximum,
            "maximum_penetration_bound": maximum_bound,
            "face_wise_bound_certified": certified,
            "bound_method": method if certified else None,
            "bound_sampled_point_count": certificate_samples,
            "bound_refinement_depth": certificate_depth,
            "minimum_valid_signed_gap": (
                None if not np.any(valid) else float(np.min(gaps[valid]))
            ),
            "penetrating_reference_area": float(np.sum(
                evaluation.trace.weights[valid][penetrations > 0.0]
            )),
            "admission_limit": limit,
            "policy": "report" if limit is None else "reject_certified_above_limit",
        }
        if limit is None:
            return
        if not certified:
            raise NotImplementedError(
                "AFM-CONTACT-INITIAL-BOUND: an initial_penetration_limit "
                "requires first-order reference geometry with authoritative "
                "facet vertices for an analytical plane (CG1/CG2 displacement), "
                "or first-order slave interpolation and geometry with reviewed closed-solid "
                "signed distance, a bounded analytical radial partition, or "
                "an open sheet with a certified complete planar branch on "
                "triangular faces; sampled gaps are not a bound."
            )
        if not np.all(valid):
            first = int(np.flatnonzero(~valid)[0])
            raise ValueError(
                "AFM-CONTACT-INITIAL-PROJECTION: cannot certify the initial "
                "penetration limit because reference projection is invalid at "
                f"contact_point_id={int(evaluation.trace.point_ids[first])}, "
                f"surface={self.surface.name}."
            )
        if maximum_bound > limit:
            raise ValueError(
                "AFM-CONTACT-INITIAL-OVERLAP: reference-configuration "
                f"penetration bound {maximum_bound:.6g} exceeds the declared limit "
                f"{limit:.6g} on pair {self.name}; no mesh or tool is moved."
            )

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

    def _tangential_history_rotation(self, factor: float) -> np.ndarray | None:
        if self.friction_kinematics is None or self.friction_kinematics.accepted is None:
            raise RuntimeError("Implicit friction requires accepted kinematic State.")
        schedule = self.pair.rigid_body.motion_schedule
        return rigid_tangential_history_rotation(
            None if schedule is None else schedule.motion,
            accepted_motion_factor=self.friction_kinematics.accepted.motion_factor,
            current_motion_factor=factor,
        )

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
            history_rotation=self._tangential_history_rotation(factor),
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

    def _current_point_response(self, *, _include_partition=False):
        """Reproject the current trial; never reuse a previous gap or pressure."""
        schedule = self.pair.rigid_body.motion_schedule
        factor = (
            0.0 if self.motion_factor is None
            else float(np.asarray(self.motion_factor.value).reshape(-1)[0])
        )
        partition = None
        planar_bound = None
        if self.contact_integration == "planar_active_partition":
            current = self.adapter.current_nodal_positions(self.displacement)
            pose = self.surface.transformed(
                None if schedule is None else schedule.motion, factor,
            )
            trial_trace = partition_planar_p1_trace(
                self.adapter.trace, current,
                plane_point=pose["point"], plane_normal=pose["normal"],
            )
            evaluation = trial_trace.evaluate(current)
        elif self.contact_integration == "convex_feature_partition":
            current = self.adapter.current_nodal_positions(self.displacement)
            partition = partition_convex_p1_trace(
                self.adapter.trace, current, self.surface,
                penalty=float(np.asarray(self.pair.law.penalty)),
                motion=None if schedule is None else schedule.motion,
                factor=factor,
                _support_planes=self._support_planes,
            )
            evaluation = partition.trace.evaluate(current)
        elif self.contact_integration == "radial_active_partition":
            current = self.adapter.current_nodal_positions(self.displacement)
            trial_trace = partition_radial_p1_trace(
                self.adapter.trace, current, self.surface,
                penalty=float(np.asarray(self.pair.law.penalty)),
                motion=None if schedule is None else schedule.motion,
                factor=factor,
            )
            evaluation = trial_trace.evaluate(current)
        elif self.contact_integration in {
            "closed_planar_branch_partition", "reviewed_planar_branch_partition",
        }:
            current = self.adapter.current_nodal_positions(self.displacement)
            face_nodes = self.adapter.trace.node_ids.reshape(-1, 3, 3)[:, 0, :]
            planar_bound = certify_reviewed_planar_branch(
                current[face_nodes], self.surface,
                projector=(
                    self.projector
                    if isinstance(self.surface, TriangulatedRigidSurface) else None
                ),
                motion=None if schedule is None else schedule.motion,
                factor=factor,
            )
            if not planar_bound.certified:
                code = (
                    "AFM-CONTACT-CLOSED-PLANAR-BRANCH"
                    if self.contact_integration == "closed_planar_branch_partition"
                    else "AFM-CONTACT-REVIEWED-PLANAR-BRANCH"
                )
                raise NotImplementedError(
                    f"{code}: the current complete "
                    "slave trace cannot be proved to lie on a smooth master "
                    f"plane per face ({planar_bound.status}); no trial was accepted."
                )
            trial_trace = partition_planar_p1_trace(
                self.adapter.trace, current,
                plane_point=planar_bound.plane_points,
                plane_normal=planar_bound.plane_normals,
            )
            evaluation = trial_trace.evaluate(current)
        else:
            evaluation = self.adapter.evaluate(self.displacement)
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
                f"contact_point_id={int(evaluation.trace.point_ids[first])}, "
                f"status={projection.status_codes[first]}, "
                f"surface={self.surface.name}; the current trial is rejected "
                "without inventing a contact normal."
            )
        if partition is not None:
            # The convex projector has its own geometric query tolerance.
            # That tolerance may never silently replace the feature selected
            # by exact reference-area partitioning at a quadrature point.
            sampled = np.isfinite(partition.selected_gaps)
            scale = max(1.0, float(np.max(
                np.abs(evaluation.query_points), initial=0.0,
            )))
            gap_error = np.abs(
                projection.signed_gaps[sampled] - partition.selected_gaps[sampled]
            )
            normal_error = np.linalg.norm(
                projection.normals[sampled] - partition.selected_normals[sampled],
                axis=1,
            )
            mismatch = (
                (gap_error > 64 * np.finfo(float).eps * scale)
                | (normal_error > 1e-10)
            )
            if np.any(mismatch):
                sampled_indices = np.flatnonzero(sampled)
                first = int(sampled_indices[np.flatnonzero(mismatch)[0]])
                raise ValueError(
                    "AFM-CONTACT-FEATURE-PARTITION: geometry projection "
                    "disagrees with the selected support-plane branch at "
                    f"contact_point_id={int(evaluation.trace.point_ids[first])}, "
                    f"query_point={evaluation.query_points[first].tolist()}, "
                    f"projection_gap={projection.signed_gaps[first]:.6g}, "
                    f"partition_gap={partition.selected_gaps[first]:.6g}, "
                    f"projection_normal={projection.normals[first].tolist()}, "
                    f"partition_normal={partition.selected_normals[first].tolist()}; "
                    "the current trial is rejected."
                )
        if planar_bound is not None:
            original_ids = self.adapter.trace.point_ids.reshape(-1, 3)[:, 0] // 3
            face_slots = {int(identity): slot for slot, identity in enumerate(original_ids)}
            if len(face_slots) != len(original_ids):
                raise ValueError(
                    "AFM-CONTACT-REVIEWED-PLANAR-BRANCH: duplicate slave face identity."
                )
            identity_divisor = 3 if evaluation.trace is self.adapter.trace else 16
            try:
                point_slots = np.asarray([
                    face_slots[int(identity)]
                    for identity in evaluation.trace.point_ids // identity_divisor
                ], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(
                    "AFM-CONTACT-REVIEWED-PLANAR-BRANCH: trial point has no "
                    "certified parent slave face."
                ) from exc
            point_normals = planar_bound.plane_normals[point_slots]
            expected_gaps = np.einsum(
                "ij,ij->i",
                evaluation.query_points - planar_bound.plane_points[point_slots],
                point_normals,
            )
            scale = max(1.0, float(np.max(
                np.abs(evaluation.query_points), initial=0.0,
            )))
            gap_errors = np.abs(projection.signed_gaps - expected_gaps)
            normal_errors = np.linalg.norm(
                projection.normals - point_normals, axis=1,
            )
            mismatch = (
                (gap_errors > 128 * np.finfo(float).eps * scale)
                | (normal_errors > 512 * np.finfo(float).eps)
            )
            if np.any(mismatch):
                first = int(np.flatnonzero(mismatch)[0])
                raise ValueError(
                    "AFM-CONTACT-REVIEWED-PLANAR-BRANCH: current projection "
                    "disagrees with the certified exact plane at "
                    f"contact_point_id={int(evaluation.trace.point_ids[first])}, "
                    f"gap_error={gap_errors[first]:.6g}, "
                    f"normal_error={normal_errors[first]:.6g}; "
                    "the trial is rejected."
                )
        record = ContactProjectionRecord(evaluation.trace.point_ids, projection)
        response = self.pair.law.evaluate(record.projection)
        if _include_partition:
            return evaluation, record, response, factor, partition, planar_bound
        return evaluation, record, response, factor

    def current_point_diagnostics(self) -> dict[str, object]:
        """Unsmoothed reference-trace diagnostics at the current trial pose."""

        evaluation, record, response, factor, partition, planar_bound = self._current_point_response(
            _include_partition=True,
        )
        active = np.asarray(response.active, dtype=bool)
        evidence = {
            "location": "reference_boundary_quadrature",
            "pressure_measure": "penalty_force_per_reference_area_not_cauchy_pressure",
            "active_point_count": int(np.count_nonzero(active)),
            "active_reference_area": float(np.sum(evaluation.trace.weights[active])),
            "maximum_pressure": float(np.max(response.pressures, initial=0.0)),
            "maximum_penetration": float(np.max(response.penetration, initial=0.0)),
            "initial_contact": dict(self._initial_contact_diagnostics),
        }
        if partition is not None:
            evidence["split_slave_faces"] = partition.split_faces
            evidence["active_feature_lines"] = partition.active_feature_lines
        if isinstance(self.surface, ConvexPolyhedralRigidSurface):
            # An active (negative-gap) point inside a convex solid selects a
            # support plane. Coplanar triangulation seams share its normal
            # and offset; distinct planes on one slave face signal a sampled
            # feature transition. Absence of such samples is not a proof.
            projection = record.projection
            offsets = np.einsum(
                "ij,ij->i", projection.normals, projection.closest_points,
            )
            scale = max(1.0, float(np.max(
                np.abs(projection.closest_points), initial=0.0,
            )))
            faces: dict[tuple[int, ...], tuple[np.ndarray, float] | None] = {}
            mixed = set()
            for nodes, active_point, normal, offset in zip(
                evaluation.trace.node_ids, active, projection.normals, offsets,
                strict=True,
            ):
                if not active_point:
                    continue
                key = tuple(int(node) for node in nodes)
                if key in mixed:
                    continue
                first = faces.get(key)
                if first is None:
                    faces[key] = (normal, float(offset))
                elif (
                    np.linalg.norm(first[0] - normal) > 1e-10
                    or abs(first[1] - offset) > 1e-10 * scale
                ):
                    mixed.add(key)
            evidence["sampled_mixed_feature_slave_faces"] = len(mixed)
            evidence["feature_transition_integral_certified"] = (
                self.contact_integration == "convex_feature_partition"
            )
        if isinstance(self.surface, ClosedTriangulatedRigidSolid):
            # Do not infer this from nominal triangle IDs: a coplanar seam is
            # physically a face, while a subdivided straight edge remains an
            # edge. The same geometric feature classifier owns the tangent.
            counts = {"face": 0, "edge": 0, "vertex": 0}
            projection = record.projection
            for index in np.flatnonzero(active):
                _, feature = self.surface._geometric_feature_projector(
                    self.surface._facet_slots[int(projection.entity_ids[index])],
                    projection.local_coordinates[index],
                )
                counts[feature] += 1
            evidence["active_feature_counts"] = counts
            evidence["sampled_mixed_feature_slave_faces"] = (
                sampled_mixed_feature_slave_faces(
                    evaluation, record, response, self.surface,
                    motion=(
                        None if self.pair.rigid_body.motion_schedule is None
                        else self.pair.rigid_body.motion_schedule.motion
                    ),
                    factor=factor,
                )
            )
            # A sampled zero is not a face-wise zero. Only a conservative
            # signed-distance bound over every current affine slave triangle
            # may certify that this accepted configuration has no contact.
            # This does not admit any active nonconvex feature integral.
            if planar_bound is not None:
                evidence["whole_face_planar_branch"] = {
                    "certified": True,
                    "status": planar_bound.status,
                    "slave_face_count": planar_bound.slave_face_count,
                    "examined_master_facet_count": planar_bound.examined_master_facet_count,
                    "minimum_competitor_clearance": (
                        planar_bound.minimum_competitor_clearance
                    ),
                    "scope": "current_affine_slave_faces_one_master_plane_per_face",
                }
                evidence["feature_transition_integral_certified"] = True
            elif (
                not np.any(active)
                and self.adapter.interpolation_degree == 1
                and self.adapter.geometry_degree == 1
                and self.adapter.facet_topology == "triangle"
            ):
                face_nodes = np.unique(evaluation.trace.node_ids, axis=0)
                current = self.adapter.current_nodal_positions(self.displacement)
                separation = certify_current_triangle_separation(
                    current[face_nodes], self.surface,
                    motion=None if self.pair.rigid_body.motion_schedule is None
                    else self.pair.rigid_body.motion_schedule.motion,
                    factor=factor,
                )
                evidence["whole_face_separation"] = {
                    "certified": separation.certified,
                    "minimum_gap_lower_bound": separation.minimum_gap_lower_bound,
                    "sampled_point_count": separation.sampled_point_count,
                    "maximum_refinement_depth": separation.maximum_refinement_depth,
                    "status": separation.status,
                    "scope": "current_affine_slave_faces_only",
                }
                evidence["feature_transition_integral_certified"] = separation.certified
            else:
                evidence["feature_transition_integral_certified"] = False
        elif isinstance(self.surface, TriangulatedRigidSurface) and planar_bound is not None:
            evidence["active_feature_counts"] = {
                "face": int(np.count_nonzero(active)), "edge": 0, "vertex": 0,
            }
            evidence["sampled_mixed_feature_slave_faces"] = 0
            evidence["whole_face_planar_branch"] = {
                "certified": True,
                "status": planar_bound.status,
                "slave_face_count": planar_bound.slave_face_count,
                "examined_master_facet_count": planar_bound.examined_master_facet_count,
                "minimum_competitor_clearance": planar_bound.minimum_competitor_clearance,
                "scope": "current_affine_slave_faces_one_oriented_master_plane_per_face",
            }
            evidence["feature_transition_integral_certified"] = True
        return evidence

    def quadrature_sensitivity(
        self, degrees=(None, 5, 7, 9, 11),
    ) -> dict[str, object]:
        """Compare positive face rules on one *fixed* current configuration.

        This diagnostic changes neither the displacement nor contact State.
        It isolates integration-rule sensitivity from the feedback of
        independently re-solving the nonlinear problem. Agreement between
        sampled rules is not a continuous feature-boundary certificate.
        """

        try:
            selected = tuple(degrees)
        except TypeError as exc:
            raise ValueError(
                "Contact quadrature audit requires a sequence of supported rules."
            ) from exc
        if (not selected or any(
            value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) not in (5, 7, 9, 11)
            ) for value in selected
        )):
            raise ValueError("Contact quadrature audit requires unique supported rules.")
        selected = tuple(None if value is None else int(value) for value in selected)
        if len(set(selected)) != len(selected):
            raise ValueError("Contact quadrature audit requires unique supported rules.")
        if (self.contact_integration != "gauss" or self.pair.friction is not None
                or np.asarray(self.pair.law.penalty).ndim != 0):
            raise NotImplementedError(
                "AFM-CONTACT-QUADRATURE-AUDIT: requires a frictionless scalar-"
                "penalty pair with fixed Gaussian reference integration."
            )
        schedule = self.pair.rigid_body.motion_schedule
        factor = (
            0.0 if self.motion_factor is None
            else float(np.asarray(self.motion_factor.value).reshape(-1)[0])
        )
        reference_point = self.pair.rigid_body.reference_point
        if schedule is not None:
            reference_point = schedule.motion.state(factor)["reference_point"]
        evaluations = []
        for degree in selected:
            alternative = dolfinx_implicit_rigid_contact(
                self.pair, self.displacement,
                contact_quadrature_degree=degree,
            )
            alternative.update_motion(factor)
            evaluation, record, response, _ = alternative._current_point_response()
            assembly = evaluation.assemble(
                record, response, surface_reference_point=reference_point,
            )
            active = np.asarray(response.active, dtype=bool)
            evaluations.append({
                "degree": degree,
                "quadrature_rule": alternative.adapter.quadrature_rule,
                "sample_count": evaluation.trace.point_count,
                "active_point_count": int(np.count_nonzero(active)),
                "active_reference_area": float(np.sum(evaluation.trace.weights[active])),
                "maximum_sampled_penetration": float(np.max(
                    response.penetration, initial=0.0,
                )),
                "contact_force_on_structure": assembly.contact_force_on_structure.tolist(),
                "contact_potential_energy": assembly.potential_energy,
            })
        return {
            "method": "fixed_configuration_positive_rule_sensitivity",
            "load_factor": factor,
            "measure": "reference_area",
            "certifies_feature_transition_integral": False,
            "evaluations": evaluations,
        }

    def closed_penalty_energy_enclosure(
        self, *, absolute_tolerance=0.0, relative_tolerance=0.01,
        maximum_depth=8, maximum_cells_per_face=4096, maximum_faces=32,
    ) -> dict[str, object]:
        """Bound the current closed-tool penalty energy without changing State.

        Unlike quadrature comparisons, a positive lower bound can establish
        that the *fixed* configuration has a missed active-area contribution.
        This does not return forces, a Newton tangent or a corrected Step.
        Its finite bounds carry conservative floating-point margins, not a
        formal interval-arithmetic guarantee.
        """

        from .contact_energy_enclosure import bound_closed_contact_penalty_energy
        from .contact_face_operator_research import extract_complete_p1_face_layout

        if (
            not isinstance(self.surface, ClosedTriangulatedRigidSolid)
            or self.pair.friction is not None
            or np.asarray(self.pair.law.penalty).ndim != 0
            or self.adapter.communicator.size != 1
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
            or type(maximum_faces) is not int
            or maximum_faces < 1
        ):
            raise NotImplementedError(
                "AFM-CONTACT-ENERGY-ENCLOSURE: requires serial, frictionless, "
                "scalar-penalty contact on a reviewed closed rigid solid with "
                "the complete default reference P1 triangular trace."
            )
        reference = self.adapter.reference_nodal_positions
        layout = extract_complete_p1_face_layout(self.adapter.trace, reference)
        if len(layout.face_ids) > maximum_faces:
            raise NotImplementedError(
                "AFM-CONTACT-ENERGY-ENCLOSURE: complete slave face count "
                "exceeds the declared diagnostic budget."
            )
        current = self.adapter.current_nodal_positions(self.displacement)
        schedule = self.pair.rigid_body.motion_schedule
        factor = 0.0 if self.motion_factor is None else self.snapshot_motion()
        faces = []
        for face_id, nodes in zip(layout.face_ids, layout.face_node_ids, strict=True):
            bound = bound_closed_contact_penalty_energy(
                reference[nodes], current[nodes], self.surface,
                penalty=float(np.asarray(self.pair.law.penalty)),
                motion=None if schedule is None else schedule.motion,
                factor=factor,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                maximum_depth=maximum_depth,
                maximum_cells=maximum_cells_per_face,
            )
            faces.append({
                "face_id": int(face_id),
                "lower_bound": bound.lower_bound,
                "upper_bound": bound.upper_bound,
                "status": bound.status,
                "within_requested_tolerance": bound.within_requested_tolerance,
                "evaluated_points": bound.evaluated_points,
                "nonunique_nearest_points": bound.nonunique_nearest_points,
                "unresolved_signed_distance_points": (
                    bound.unresolved_signed_distance_points
                ),
                "leaf_count": bound.leaf_count,
            })
        return {
            "method": "fixed_configuration_closed_penalty_energy_enclosure",
            "load_factor": factor,
            "measure": "reference_area",
            "arithmetic": "floating_point_conservative_margin_not_formal_interval",
            "certifies_feature_transition_integral": False,
            "returns_contact_residual_or_tangent": False,
            "lower_bound": float(sum(item["lower_bound"] for item in faces)),
            "upper_bound": float(sum(item["upper_bound"] for item in faces)),
            "faces": faces,
        }

    def reference_subdivision_sensitivity(
        self, levels=(0, 1, 2, 3), *, maximum_points=50_000,
    ) -> dict[str, object]:
        """Screen a fixed P1 triangle rule under positive reference subdivision.

        This changes no global solve and adds no feature-switch line tangent.
        It is an accuracy diagnostic, not an admissible default contact law.
        """

        selected = tuple(levels)
        if (not selected or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= 5
            for value in selected
        )):
            raise ValueError("Contact reference subdivision requires levels 0 to 5.")
        selected = tuple(int(value) for value in selected)
        if len(set(selected)) != len(selected):
            raise ValueError("Contact reference subdivision levels must be unique.")
        if (self.contact_integration != "gauss" or self.pair.friction is not None
                or np.asarray(self.pair.law.penalty).ndim != 0
                or self.adapter.facet_topology != "triangle"
                or self.adapter.interpolation_degree != 1
                or self.adapter.geometry_degree != 1):
            raise NotImplementedError(
                "AFM-CONTACT-REFERENCE-REFINEMENT: requires frictionless scalar-"
                "penalty Gauss contact on an affine P1 triangular trace."
            )
        schedule = self.pair.rigid_body.motion_schedule
        factor = (
            0.0 if self.motion_factor is None
            else float(np.asarray(self.motion_factor.value).reshape(-1)[0])
        )
        reference_point = self.pair.rigid_body.reference_point
        if schedule is not None:
            reference_point = schedule.motion.state(factor)["reference_point"]
        positions = self.adapter.current_nodal_positions(self.displacement)
        projector = self.surface if self.projector is None else self.projector
        base_trace = (
            self.adapter.trace
            if self.adapter.quadrature_rule == "triangle_degree_two_three_point"
            else dolfinx_boundary_region_contact_trace(
                self.pair.slave_boundary, self.displacement.function_space,
            ).trace
        )
        evaluations = []
        for level in selected:
            trace = refine_p1_triangle_reference_trace(
                base_trace, level, maximum_points=maximum_points,
            )
            evaluation = trace.evaluate(positions)
            projection = projector.project(
                evaluation.query_points,
                motion=None if schedule is None else schedule.motion,
                factor=factor,
            )
            if not projection.all_valid:
                raise ValueError(
                    "AFM-CONTACT-REFERENCE-REFINEMENT: a diagnostic point "
                    "has an invalid or ambiguous rigid projection."
                )
            record = ContactProjectionRecord(trace.point_ids, projection)
            response = self.pair.law.evaluate(projection)
            assembly = evaluation.assemble(
                record, response, surface_reference_point=reference_point,
            )
            active = np.asarray(response.active, dtype=bool)
            evaluations.append({
                "subdivision_level": level,
                "sample_count": trace.point_count,
                "active_point_count": int(np.count_nonzero(active)),
                "active_reference_area": float(np.sum(trace.weights[active])),
                "maximum_sampled_penetration": float(np.max(
                    response.penetration, initial=0.0,
                )),
                "contact_force_on_structure": assembly.contact_force_on_structure.tolist(),
                "contact_potential_energy": assembly.potential_energy,
            })
        return {
            "method": "fixed_configuration_uniform_reference_subdivision",
            "load_factor": factor,
            "measure": "reference_area",
            "certifies_feature_transition_integral": False,
            "evaluations": evaluations,
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
                history_rotation=self._tangential_history_rotation(factor),
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
            "reference_positions": evaluation.trace.evaluate(
                self.adapter.reference_nodal_positions
            ).query_points.copy(),
            "slave_positions": projection.query_points.copy(),
            "master_closest_points": projection.closest_points.copy(),
            "surface_normals": projection.normals.copy(),
            "signed_gaps": projection.signed_gaps.copy(),
            "reference_weights": evaluation.trace.weights.copy(),
            "measure_configuration": evaluation.trace.measure_configuration,
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

    def _current_friction_response(self, record, response, factor):
        """Evaluate the same fixed-old-State law for mechanics or observation."""
        if self.friction_kinematics is None or self.friction_kinematics.accepted is None:
            raise RuntimeError("Implicit friction requires accepted slave kinematics.")
        schedule = self.pair.rigid_body.motion_schedule
        relative = relative_contact_displacement_increment(
            self.friction_kinematics.accepted, record,
            motion=None if schedule is None else schedule.motion,
            current_motion_factor=factor,
        )
        return self.pair.friction.evaluate(
            point_ids=record.point_ids,
            normal_response=response,
            relative_displacement_increment=relative,
            accepted=self.friction_state.accepted,
            history_rotation=self._tangential_history_rotation(factor),
        )

    def current_energy_components(self) -> dict[str, float]:
        """Integrate current energy channels without assembling a tangent.

        Current geometry and the same material law are still evaluated. This
        method neither advances State nor caches old contact forces/projections.
        """
        evaluation, record, response, factor = self._current_point_response()
        weights = evaluation.trace.weights
        potential = float(np.dot(weights, response.potential_densities))
        dissipated = released = 0.
        if self.friction_state is not None:
            tangential = self._current_friction_response(record, response, factor)
            potential += float(np.dot(weights, tangential.recoverable_penalty_energy_densities))
            dissipated = float(np.dot(weights, tangential.record.cumulative_dissipation_densities))
            released = float(np.dot(weights, tangential.record.cumulative_separation_release_densities))
        return {"contact_potential": potential, "friction_dissipation": dissipated,
                "friction_separation_release": released}

    def evaluate(self) -> ImplicitRigidContactContribution:
        """Reproject the current displacement and assemble a fresh trial Operator."""

        evaluation, record, response, factor, partition, _ = self._current_point_response(
            _include_partition=True,
        )
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
        if partition is not None and partition.tangent_values.size:
            contribution = replace(
                contribution,
                tangent_rows=np.r_[contribution.tangent_rows, partition.tangent_rows],
                tangent_columns=np.r_[
                    contribution.tangent_columns, partition.tangent_columns,
                ],
                tangent_values=np.r_[
                    contribution.tangent_values, partition.tangent_values,
                ],
            )
        if self.friction_state is None:
            return contribution
        friction_response = self._current_friction_response(record, response, factor)
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

        add_contact_nodal_residual(vector, contribution.nodal_structural_residual)

    def add_current_residual(self, vector) -> None:
        """Current smooth frictionless residual without an unused Newton tangent.

        Faceted, frictional and derived-law providers retain full evaluation,
        including their feature/linearization admission checks. In particular,
        an augmented provider must not inherit a penalty-law shortcut. This
        hook does not cache a projection or manufacture an empty tangent.
        """
        if (type(self) is not DolfinxImplicitRigidContact
                or not isinstance(self.surface, SmoothRigidSurface)
                or self.friction_state is not None):
            self.add_residual(vector, self.evaluate())
            return
        evaluation, record, response, factor = self._current_point_response()
        schedule = self.pair.rigid_body.motion_schedule
        reference_point = (self.pair.rigid_body.reference_point if schedule is None
                           else schedule.motion.state(factor)["reference_point"])
        assembly = evaluation.assemble(record, response, surface_reference_point=reference_point)
        add_contact_nodal_residual(vector, assembly.nodal_structural_residual)

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

        add_contact_coo_tangent(
            matrix,
            contribution.tangent_rows,
            contribution.tangent_columns,
            contribution.tangent_values,
            constrained_dofs=constrained_dofs,
        )

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
            if isinstance(self.surface, TriangulatedRigidSurface)
            else "analytical_radial"
            if isinstance(self.surface, (RigidSphereSurface, RigidCylinderSurface))
            else "plane"
        )
        summary = {
            "kind": "dolfinx_implicit_rigid_contact",
            "pair": self.pair.summary(),
            "trace": self.adapter.summary(),
            "contact_integration": self.contact_integration,
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
        if self.contact_integration == "radial_active_partition":
            summary["integration_policy"] = radial_partition_policy_summary()
        return summary

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
        add_contact_nodal_residual(vector, contribution.nodal_structural_residual)

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
    contact_integration: str = "gauss",
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
    if contact_integration == "certified_closed_two_feature_research":
        if contact_quadrature_degree is not None:
            raise ValueError(
                "Certified research faces require the complete default P1 trace."
            )
        from ._certified_face_trial_provider import (
            DolfinxCertifiedFaceContactTrialProvider,
        )

        return DolfinxCertifiedFaceContactTrialProvider(
            pair=pair, adapter=adapter, displacement=displacement,
        )
    from dolfinx import fem

    return DolfinxImplicitRigidContact(
        pair=pair, adapter=adapter, displacement=displacement,
        contact_integration=contact_integration,
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
    contact_integration="gauss",
) -> object:
    """Lower independently named rigid pairs on one serial displacement field."""

    selected = tuple(pairs)
    if len(selected) < 2:
        raise ValueError("Multiple rigid contact requires at least two pairs.")
    if len({item.name for item in selected}) != len(selected):
        raise ValueError("Multiple rigid contact pair names must be unique.")
    modes = resolve_contact_integration_modes(selected, contact_integration)
    if "certified_closed_two_feature_research" in modes:
        if any(mode != "certified_closed_two_feature_research" for mode in modes):
            from ._mixed_face_point_trial_provider import (
                DolfinxMixedFacePointContactTrialProvider,
            )

            return DolfinxMixedFacePointContactTrialProvider(tuple(
                dolfinx_implicit_rigid_contact(
                    item, displacement,
                    contact_quadrature_degree=contact_quadrature_degree,
                    contact_integration=mode,
                ) for item, mode in zip(selected, modes, strict=True)
            ))
        from ._certified_face_trial_provider import (
            DolfinxCombinedCertifiedFaceContactTrialProvider,
        )

        return DolfinxCombinedCertifiedFaceContactTrialProvider(tuple(
            dolfinx_implicit_rigid_contact(
                item, displacement,
                contact_quadrature_degree=contact_quadrature_degree,
                contact_integration=mode,
            ) for item, mode in zip(selected, modes, strict=True)
        ))
    return DolfinxImplicitRigidContacts(tuple(
        dolfinx_implicit_rigid_contact(
            item, displacement,
            contact_quadrature_degree=contact_quadrature_degree,
            contact_integration=mode,
        ) for item, mode in zip(selected, modes, strict=True)
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
