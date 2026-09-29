# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable weak boundary models."""

from . import absorbing
from . import mechanical
from . import rigid
from . import search
from . import thermal
from .mechanical import (
    ElasticFoundation,
    RigidObstaclePenaltyContact,
    elastic_foundation,
    rigid_obstacle_contact,
)
from .rigid import (
    PrescribedRigidMotion,
    RigidCylinderSurface,
    RigidPlaneSurface,
    RigidSphereSurface,
    RigidSurface,
    SurfaceProjection,
    TriangulatedRigidSurface,
    prescribed_rigid_motion,
    rigid_cylinder,
    rigid_plane,
    rigid_sphere,
    triangulated_rigid_surface,
)
from .search import (
    TriangleSearchDiagnostics,
    TriangleSearchOutcome,
    TriangleSurfaceBVH,
    triangle_surface_bvh,
)
from .thermal import ConvectionBoundary, convection

__all__ = [
    "ConvectionBoundary",
    "ElasticFoundation",
    "PrescribedRigidMotion",
    "RigidCylinderSurface",
    "RigidPlaneSurface",
    "RigidSphereSurface",
    "RigidSurface",
    "RigidObstaclePenaltyContact",
    "SurfaceProjection",
    "TriangulatedRigidSurface",
    "TriangleSearchDiagnostics",
    "TriangleSearchOutcome",
    "TriangleSurfaceBVH",
    "absorbing",
    "convection",
    "elastic_foundation",
    "mechanical",
    "prescribed_rigid_motion",
    "rigid",
    "rigid_cylinder",
    "rigid_plane",
    "rigid_sphere",
    "rigid_obstacle_contact",
    "search",
    "triangle_surface_bvh",
    "triangulated_rigid_surface",
    "thermal",
]
