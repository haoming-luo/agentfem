# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial DOLFINx lowering of bounded rigid-surface Newton contact.

The Model owns the pair, the geometry backend owns BVH search, and this
adapter recomputes contact for each current Newton trial.  It does not commit
state, choose load increments, alter Dirichlet rows or implement MPC reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_pair import RigidContactPair
from .contact_state import ContactProjectionRecord
from .dolfinx_contact_trace import (
    DolfinxContactTraceAdapter,
    dolfinx_boundary_region_contact_trace,
)
from .implicit_facet_contact import (
    linearize_piecewise_planar_contact,
)
from .implicit_contact import ImplicitRigidContactContribution
from .implicit_smooth_contact import linearize_smooth_rigid_contact
from .rigid import SmoothRigidSurface, TriangulatedRigidSurface
from .search import TriangleSurfaceBVH, triangle_surface_bvh


@dataclass(eq=False)
class DolfinxImplicitRigidContact:
    """Current-trial contact with one reviewed frictionless rigid tool."""

    pair: RigidContactPair
    adapter: DolfinxContactTraceAdapter
    displacement: object
    projector: TriangleSurfaceBVH | None
    motion_factor: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair, RigidContactPair):
            raise TypeError("Implicit rigid contact requires a RigidContactPair.")
        if not isinstance(self.adapter, DolfinxContactTraceAdapter):
            raise TypeError("Implicit rigid contact requires a DOLFINx trace adapter.")
        surface = self.pair.rigid_body.surface
        if isinstance(surface, TriangulatedRigidSurface):
            if not isinstance(self.projector, TriangleSurfaceBVH):
                raise TypeError("Implicit facet contact requires a reviewed triangle BVH.")
            if self.projector.surface is not surface:
                raise ValueError("Contact search and pair must share one rigid geometry instance.")
        elif not isinstance(surface, SmoothRigidSurface):
            raise TypeError("Implicit rigid contact requires smooth differentiable geometry or a reviewed triangle tool.")
        elif self.projector is not None:
            raise ValueError("Analytical curved contact does not use a triangle BVH.")
        if self.adapter.communicator.size != 1:
            raise NotImplementedError(
                "AFM-IMPLICIT-CONTACT-MPI: rigid contact has no distributed "
                "Newton matrix/owner reduction contract yet."
            )
        if self.pair.friction is not None:
            raise NotImplementedError("Implicit rigid contact does not yet linearize friction.")
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
    def surface(self) -> TriangulatedRigidSurface | SmoothRigidSurface:
        return self.pair.rigid_body.surface

    @property
    def name(self) -> str:
        return self.pair.name

    @property
    def has_prescribed_motion(self) -> bool:
        return self.pair.rigid_body.motion_schedule is not None

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

    def evaluate(self) -> ImplicitRigidContactContribution:
        """Reproject the current displacement and assemble a fresh trial Operator."""

        evaluation, record, response, factor = self._current_point_response()
        schedule = self.pair.rigid_body.motion_schedule
        reference_point = self.pair.rigid_body.reference_point
        if schedule is not None:
            reference_point = schedule.motion.state(factor)["reference_point"]
        if self.projector is None:
            return linearize_smooth_rigid_contact(
                evaluation, record, response, self.surface,
                motion=None if schedule is None else schedule.motion,
                factor=factor, surface_reference_point=reference_point,
            )
        return linearize_piecewise_planar_contact(
            evaluation, record, response, self.surface,
            surface_reference_point=reference_point,
        )

    def add_residual(self, vector, contribution: ImplicitRigidContactContribution) -> None:
        """Add local scalar rows before the owning solver performs BC lifting."""

        values = contribution.assembly.nodal_structural_residual.reshape(-1)
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
        distribution.x.array[:] = -contribution.assembly.nodal_structural_residual.reshape(-1)
        distribution.x.scatter_forward()
        force = contribution.assembly.contact_force_on_structure
        coordinate = None
        if self.has_prescribed_motion:
            moment = contribution.assembly.surface_generalized_moment
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
            resultant=contribution.assembly.contact_force_on_structure,
            distribution=distribution,
            role="contact_constraint",
            source="dolfinx_implicit_rigid_contact",
            diagnostics={
                "active_point_count": contribution.active_points,
                "contact_potential": contribution.assembly.potential_energy,
                "surface_generalized_moment": (
                    None if contribution.assembly.surface_generalized_moment is None
                    else contribution.assembly.surface_generalized_moment.tolist()
                ),
                "projection": (
                    "current_geometry_bvh" if self.projector is not None
                    else "exact_analytical_current_pose"
                ),
                "linearization": self._linearization_name(),
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_implicit_rigid_contact",
            "pair": self.pair.summary(),
            "trace": self.adapter.summary(),
            "search": (
                self.projector.summary() if self.projector is not None
                else {"kind": "exact_analytical_projection"}
            ),
            "linearization": self._linearization_name(),
            "execution_scope": "serial_frictionless_reference_area",
        }

    def _linearization_name(self) -> str:
        return (
            "piecewise_planar_normal_branch_at_current_pose"
            if self.projector is not None
            else "smooth_signed_gap_with_curvature_at_current_pose"
        )


# Compatibility spelling for the original experimental facet adapter.
DolfinxImplicitFacetContact = DolfinxImplicitRigidContact


def dolfinx_implicit_rigid_contact(
    pair: RigidContactPair,
    displacement,
) -> DolfinxImplicitRigidContact:
    """Lower one bounded rigid pair to a serial current-trial Newton Operator."""

    if not isinstance(pair, RigidContactPair):
        raise TypeError("Implicit rigid contact requires a RigidContactPair.")
    surface = pair.rigid_body.surface
    if not isinstance(surface, (TriangulatedRigidSurface, SmoothRigidSurface)):
        raise TypeError("Implicit rigid contact requires smooth differentiable geometry or a reviewed triangle tool.")
    adapter = dolfinx_boundary_region_contact_trace(
        pair.slave_boundary, displacement.function_space,
    )
    from dolfinx import fem

    return DolfinxImplicitRigidContact(
        pair=pair, adapter=adapter, displacement=displacement,
        projector=(triangle_surface_bvh(surface) if isinstance(surface, TriangulatedRigidSurface) else None),
        motion_factor=(
            None if pair.rigid_body.motion_schedule is None
            else fem.Constant(displacement.function_space.mesh, 0.0)
        ),
    )


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
    "dolfinx_implicit_facet_contact", "dolfinx_implicit_rigid_contact",
]
