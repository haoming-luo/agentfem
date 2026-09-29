# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable weak boundary models."""

from . import absorbing
from . import mechanical
from . import rigid
from . import thermal
from .mechanical import (
    ElasticFoundation,
    RigidObstaclePenaltyContact,
    elastic_foundation,
    rigid_obstacle_contact,
)
from .rigid import (
    PrescribedRigidMotion,
    RigidPlaneSurface,
    RigidSurface,
    SurfaceProjection,
    prescribed_rigid_motion,
    rigid_plane,
)
from .thermal import ConvectionBoundary, convection

__all__ = [
    "ConvectionBoundary",
    "ElasticFoundation",
    "PrescribedRigidMotion",
    "RigidPlaneSurface",
    "RigidSurface",
    "RigidObstaclePenaltyContact",
    "SurfaceProjection",
    "absorbing",
    "convection",
    "elastic_foundation",
    "mechanical",
    "prescribed_rigid_motion",
    "rigid",
    "rigid_plane",
    "rigid_obstacle_contact",
    "thermal",
]
