# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared capability gate for the bounded affine implicit-contact Operator."""

from __future__ import annotations

from .contact_pair import RigidContactPair
from .rigid import RigidCylinderSurface, RigidSphereSurface, TriangulatedRigidSurface


def affine_rigid_contact_supported(model) -> bool:
    """One serial rigid pair may share the affine J2 residual and work path."""

    boundaries = tuple(getattr(model, "boundary_models", ()))
    if not boundaries:
        return True
    if len(boundaries) != 1 or not isinstance(boundaries[0], RigidContactPair):
        return False
    pair = boundaries[0]
    return (
        pair.slave_boundary.domain.comm.size == 1
        and isinstance(
            pair.rigid_body.surface,
            (TriangulatedRigidSurface, RigidSphereSurface, RigidCylinderSurface),
        )
        and (
            pair.rigid_body.motion_schedule is None
            or (
                pair.rigid_body.motion_schedule.start_time == 0.0
                and pair.rigid_body.motion_schedule.end_time == 1.0
                and pair.rigid_body.motion_schedule.amplitude is None
            )
        )
        and pair.friction is None
        and pair.law.invalid_policy == "reject"
    )


def affine_fixed_contact_supported(model) -> bool:
    """Preserve the historical fixed-tool-only predicate for extensions."""

    return affine_rigid_contact_supported(model) and all(
        pair.rigid_body.motion_schedule is None
        for pair in getattr(model, "boundary_models", ())
    )


__all__ = ["affine_fixed_contact_supported", "affine_rigid_contact_supported"]
