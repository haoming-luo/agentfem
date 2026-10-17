# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared capability gate for the bounded affine implicit-contact Operator."""

from __future__ import annotations

from .contact_pair import RigidContactPair
from .rigid import RigidCylinderSurface, RigidSphereSurface, TriangulatedRigidSurface


def affine_fixed_contact_supported(model) -> bool:
    """Only one fixed, frictionless, serial rigid pair has a reviewed tangent."""

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
        and pair.rigid_body.motion_schedule is None
        and pair.friction is None
        and pair.law.invalid_policy == "reject"
    )


__all__ = ["affine_fixed_contact_supported"]
