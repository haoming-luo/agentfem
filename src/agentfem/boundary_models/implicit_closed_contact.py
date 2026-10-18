# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference-area Newton Operator for a reviewed closed triangle solid."""

from __future__ import annotations

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceEvaluation
from .implicit_contact import (
    ImplicitRigidContactContribution, integrate_implicit_rigid_contact,
)
from .rigid import PrescribedRigidMotion


def linearize_closed_rigid_solid_contact(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    surface: ClosedTriangulatedRigidSolid,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    surface_reference_point=None,
) -> ImplicitRigidContactContribution:
    """Integrate the selected exact face/edge/vertex distance branch.

    This is not a smoothed normal or an edge-averaged facet law. A medial-axis
    tie or ambiguous winding fails in the geometry owner before this function
    receives valid projection evidence. Topological closure alone does not
    certify embedding; the caller must separately review that prerequisite.
    """

    if not isinstance(surface, ClosedTriangulatedRigidSolid):
        raise TypeError("Closed-solid contact needs reviewed closed geometry.")
    return integrate_implicit_rigid_contact(
        evaluation, record, response,
        surface.gap_differential(
            record.projection, motion=motion, factor=factor,
        ).hessian,
        surface_reference_point=surface_reference_point,
    )


__all__ = ["linearize_closed_rigid_solid_contact"]
