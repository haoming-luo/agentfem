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
    RigidCylinderSurface,
    RigidSphereSurface,
    SurfaceProjection,
    _transformed_point_and_direction,
)


def smooth_gap_hessian(
    surface: RigidSphereSurface | RigidCylinderSurface,
    projection: SurfaceProjection,
    *,
    motion: PrescribedRigidMotion | None = None,
    factor: float = 0.0,
) -> np.ndarray:
    """Differentiate the *signed* gap, including current rigid orientation.

    The normal of a sphere/cylinder points toward the admissible side.  The
    center/axis singularity and a finite cylinder rim are rejected by the
    existing projector rather than assigned a fictitious curvature.
    """

    if not isinstance(surface, (RigidSphereSurface, RigidCylinderSurface)):
        raise TypeError(
            "Smooth implicit contact supports analytical spheres/cylinders."
        )
    if not isinstance(projection, SurfaceProjection):
        raise TypeError("Smooth gap Hessian requires projection evidence.")
    if (
        projection.geometry_fingerprint != surface.geometry_fingerprint
        or projection.dimension != surface.dimension
        or not projection.all_valid
    ):
        raise ValueError(
            "Smooth gap Hessian requires valid matching surface projection."
        )
    direction = 1.0 if surface.admissible_side == "exterior" else -1.0
    radial_distance = surface.radius + direction * projection.signed_gaps
    if np.any(radial_distance <= 64 * np.finfo(float).eps * max(1.0, surface.radius)):
        raise ValueError("Smooth gap Hessian is undefined at the surface axis/center.")
    dimension = projection.dimension
    projector = np.eye(dimension)
    if isinstance(surface, RigidCylinderSurface):
        _, axis = _transformed_point_and_direction(
            np.asarray(surface.axis_point),
            np.asarray(surface.axis_direction),
            motion=motion,
            factor=factor,
        )
        projector -= np.outer(axis, axis)
    radial = direction * projection.normals
    hessian = (
        direction
        / radial_distance[:, None, None]
        * (projector[None, :, :] - radial[:, :, None] * radial[:, None, :])
    )
    return hessian


def linearize_smooth_rigid_contact(
    evaluation: ContactTraceEvaluation,
    record: ContactProjectionRecord,
    response: FrictionlessPenaltyContactResponse,
    surface: RigidSphereSurface | RigidCylinderSurface,
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
