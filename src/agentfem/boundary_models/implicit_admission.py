# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared capability gate for the bounded affine implicit-contact Operator."""

from __future__ import annotations

from .contact_pair import RigidContactPair
from .rigid import SmoothRigidSurface, TriangulatedRigidSurface


def affine_rigid_contact_assets_supported(boundaries) -> bool:
    """Validate the bounded affine contact assets independently of a Model."""

    boundaries = tuple(boundaries)
    if not boundaries:
        return True
    if not all(isinstance(pair, RigidContactPair) for pair in boundaries):
        return False
    if len({pair.name for pair in boundaries}) != len(boundaries):
        return False
    return all(
        pair.slave_boundary.domain.comm.size == 1
        and isinstance(pair.rigid_body.surface, (TriangulatedRigidSurface, SmoothRigidSurface))
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
        for pair in boundaries
    )


def affine_rigid_contact_supported(model) -> bool:
    """Named serial frictionless rigid pairs may share the affine J2 path."""

    return affine_rigid_contact_assets_supported(
        getattr(model, "boundary_models", ()),
    )


def affine_fixed_contact_supported(model) -> bool:
    """Preserve the historical fixed-tool-only predicate for extensions."""

    return affine_rigid_contact_supported(model) and all(
        pair.rigid_body.motion_schedule is None
        for pair in getattr(model, "boundary_models", ())
    )


__all__ = [
    "affine_fixed_contact_supported",
    "affine_rigid_contact_assets_supported",
    "affine_rigid_contact_supported",
]
