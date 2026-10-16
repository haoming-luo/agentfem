# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial DOLFINx lowering of the bounded rigid-facet Newton Operator.

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
    ImplicitFacetContactContribution,
    linearize_piecewise_planar_contact,
)
from .rigid import TriangulatedRigidSurface
from .search import TriangleSurfaceBVH, triangle_surface_bvh


@dataclass(eq=False)
class DolfinxImplicitFacetContact:
    """Current-trial facet contact, limited to a fixed frictionless rigid tool."""

    pair: RigidContactPair
    adapter: DolfinxContactTraceAdapter
    displacement: object
    projector: TriangleSurfaceBVH

    def __post_init__(self) -> None:
        if not isinstance(self.pair, RigidContactPair):
            raise TypeError("Implicit facet contact requires a RigidContactPair.")
        if not isinstance(self.adapter, DolfinxContactTraceAdapter):
            raise TypeError("Implicit facet contact requires a DOLFINx trace adapter.")
        if not isinstance(self.projector, TriangleSurfaceBVH):
            raise TypeError("Implicit facet contact requires a reviewed triangle BVH.")
        if self.adapter.communicator.size != 1:
            raise NotImplementedError(
                "AFM-IMPLICIT-CONTACT-MPI: facet contact has no distributed "
                "Newton matrix/owner reduction contract yet."
            )
        if self.pair.friction is not None:
            raise NotImplementedError("Implicit facet contact does not yet linearize friction.")
        if self.pair.law.invalid_policy != "reject":
            raise NotImplementedError(
                "Implicit facet contact requires invalid_policy='reject'; "
                "an unsupported or ambiguous projection cannot be inactive."
            )
        if self.pair.rigid_body.motion_schedule is not None:
            raise NotImplementedError(
                "Implicit facet contact does not yet provide prescribed-tool "
                "work and pose rollback."
            )
        if not isinstance(self.pair.rigid_body.surface, TriangulatedRigidSurface):
            raise TypeError("Implicit facet contact requires a triangulated rigid tool.")
        if self.projector.surface is not self.pair.rigid_body.surface:
            raise ValueError("Contact search and pair must share one rigid geometry instance.")
        if self.pair.slave_boundary.domain is not self.displacement.function_space.mesh:
            raise ValueError("Contact slave and displacement must share one mesh.")
        if self.adapter.function_space is not self.displacement.function_space:
            raise ValueError("Contact trace and displacement must share one function space.")

    @property
    def surface(self) -> TriangulatedRigidSurface:
        return self.pair.rigid_body.surface

    @property
    def name(self) -> str:
        return self.pair.name

    @property
    def has_prescribed_motion(self) -> bool:
        return False

    @property
    def motion_factor(self):
        return None

    def update_motion(self, factor: float) -> None:
        if not np.isfinite(float(factor)):
            raise ValueError("Contact load factor must be finite.")

    def capabilities(self):
        from ..constraints import ConstraintCapabilities

        return ConstraintCapabilities(
            kind="contact_constraint",
            enforcement="one_sided_rigid_facet_penalty",
            analyses=("nonlinear_static",),
            procedures=("incremental_newton",),
            strict=False,
            supports_parallel=False,
            reaction_evidence="provider_dual_required",
            work_evidence="internal_energy_operator",
        )

    def evaluate(self) -> ImplicitFacetContactContribution:
        """Reproject the current displacement and assemble a fresh trial Operator."""

        evaluation = self.adapter.evaluate(self.displacement)
        projection = self.projector.project(evaluation.query_points)
        record = ContactProjectionRecord(self.adapter.trace.point_ids, projection)
        response = self.pair.law.evaluate(record.projection)
        return linearize_piecewise_planar_contact(
            evaluation,
            record,
            response,
            self.surface,
            surface_reference_point=self.pair.rigid_body.reference_point,
        )

    def add_residual(self, vector, contribution: ImplicitFacetContactContribution) -> None:
        """Add local scalar rows before the owning solver performs BC lifting."""

        values = contribution.assembly.nodal_structural_residual.reshape(-1)
        if vector.array.shape != values.shape:
            raise ValueError("Implicit contact residual does not match PETSc local rows.")
        vector.array[:] += values

    def add_tangent(self, matrix, contribution: ImplicitFacetContactContribution) -> None:
        """Insert the branch tangent before PETSc final assembly (serial only)."""

        from petsc4py import PETSc

        if contribution.tangent_values.size == 0:
            return
        # DOLFINx inserts the UFL matrix entries. PETSc requires a flush
        # before switching this same matrix to additive contact entries.
        matrix.assemble(PETSc.Mat.AssemblyType.FLUSH)
        order = np.lexsort((contribution.tangent_columns, contribution.tangent_rows))
        rows = contribution.tangent_rows[order]
        columns = contribution.tangent_columns[order]
        values = contribution.tangent_values[order]
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
        return constraint_dual(
            self.pair,
            force=contribution.assembly.contact_force_on_structure,
            resultant=contribution.assembly.contact_force_on_structure,
            distribution=distribution,
            role="contact_constraint",
            source="dolfinx_implicit_facet_contact",
            diagnostics={
                "active_point_count": contribution.active_points,
                "contact_potential": contribution.assembly.potential_energy,
                "surface_generalized_moment": (
                    None if contribution.assembly.surface_generalized_moment is None
                    else contribution.assembly.surface_generalized_moment.tolist()
                ),
                "projection": "current_geometry_bvh",
                "linearization": "fixed_piecewise_planar_normal_branch",
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_implicit_facet_contact",
            "pair": self.pair.summary(),
            "trace": self.adapter.summary(),
            "search": self.projector.summary(),
            "linearization": "fixed_piecewise_planar_normal_branch",
            "execution_scope": "serial_fixed_frictionless_reference_area",
        }


def dolfinx_implicit_facet_contact(
    pair: RigidContactPair,
    displacement,
) -> DolfinxImplicitFacetContact:
    """Lower a reviewed fixed triangle tool to one serial Newton contribution."""

    if not isinstance(pair, RigidContactPair):
        raise TypeError("Implicit facet contact requires a RigidContactPair.")
    if not isinstance(pair.rigid_body.surface, TriangulatedRigidSurface):
        raise TypeError("Implicit facet contact requires TriangulatedRigidSurface.")
    adapter = dolfinx_boundary_region_contact_trace(
        pair.slave_boundary,
        displacement.function_space,
    )
    return DolfinxImplicitFacetContact(
        pair=pair,
        adapter=adapter,
        displacement=displacement,
        projector=triangle_surface_bvh(pair.rigid_body.surface),
    )


__all__ = ["DolfinxImplicitFacetContact", "dolfinx_implicit_facet_contact"]
