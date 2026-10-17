# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact active-branch Newton contact for analytical rigid curved surfaces."""

from __future__ import annotations

import numpy as np

from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .contact_trace import ContactTraceEvaluation
from .implicit_contact import (
    ImplicitRigidContactContribution,
    integrate_implicit_rigid_contact,
)
from .rigid import (
    PrescribedRigidMotion,
    SmoothRigidSurface,
    SurfaceGapDifferential,
    SurfaceProjection,
)


def smooth_gap_hessian(
    surface: SmoothRigidSurface,
    projection: SurfaceProjection,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
) -> np.ndarray:
    """Return current-pose curvature supplied by a smooth geometry provider."""

    if not isinstance(surface, SmoothRigidSurface):
        raise TypeError("Smooth implicit contact requires SmoothRigidSurface.")
    differential = surface.gap_differential(
        projection, motion=motion, factor=factor,
    )
    if not isinstance(differential, SurfaceGapDifferential):
        raise TypeError("Smooth geometry must return SurfaceGapDifferential.")
    if differential.projection is not projection:
        raise ValueError("Gap differential must bind the current projection instance.")
    return differential.hessian


def linearize_smooth_rigid_contact(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    surface: SmoothRigidSurface,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
    surface_reference_point=None,
) -> ImplicitRigidContactContribution:
    """Integrate a smooth rigid tool's penalty force and consistent curvature."""

    return integrate_implicit_rigid_contact(
        evaluation,
        record,
        response,
        smooth_gap_hessian(surface, record.projection, motion=motion, factor=factor),
        surface_reference_point=surface_reference_point,
    )


__all__ = ["linearize_smooth_rigid_contact", "smooth_gap_hessian"]
