# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable weak boundary models."""

from . import absorbing
from . import dolfinx_adapter
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
from .dolfinx_adapter import dolfinx_exterior_triangle_partition
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
    DistributedTriangleSearchDiagnostics,
    DistributedTriangleSearchOutcome,
    DistributedTriangleSurfaceBVH,
    RoutedDistributedTriangleSurfaceBVH,
    RoutedTriangleSearchDiagnostics,
    RoutedTriangleSearchOutcome,
    TriangleSearchDiagnostics,
    TriangleSearchOutcome,
    TriangleSurfacePartition,
    TriangleSurfaceBVH,
    distributed_triangle_surface_bvh,
    partition_triangle_surface,
    routed_distributed_triangle_surface_bvh,
    triangle_surface_bvh,
)
from .thermal import ConvectionBoundary, convection

__all__ = [
    "ConvectionBoundary",
    "DistributedTriangleSearchDiagnostics",
    "DistributedTriangleSearchOutcome",
    "DistributedTriangleSurfaceBVH",
    "ElasticFoundation",
    "PrescribedRigidMotion",
    "RigidCylinderSurface",
    "RigidPlaneSurface",
    "RigidSphereSurface",
    "RigidSurface",
    "RoutedDistributedTriangleSurfaceBVH",
    "RoutedTriangleSearchDiagnostics",
    "RoutedTriangleSearchOutcome",
    "RigidObstaclePenaltyContact",
    "SurfaceProjection",
    "TriangulatedRigidSurface",
    "TriangleSearchDiagnostics",
    "TriangleSearchOutcome",
    "TriangleSurfacePartition",
    "TriangleSurfaceBVH",
    "absorbing",
    "convection",
    "dolfinx_adapter",
    "dolfinx_exterior_triangle_partition",
    "elastic_foundation",
    "distributed_triangle_surface_bvh",
    "mechanical",
    "prescribed_rigid_motion",
    "partition_triangle_surface",
    "rigid",
    "rigid_cylinder",
    "rigid_plane",
    "rigid_sphere",
    "routed_distributed_triangle_surface_bvh",
    "rigid_obstacle_contact",
    "search",
    "triangle_surface_bvh",
    "triangulated_rigid_surface",
    "thermal",
]
