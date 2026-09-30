# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable weak boundary models."""

from . import absorbing
from . import contact_state
from . import contact_lifecycle
from . import contact_friction
from . import contact_pair
from . import contact_response
from . import contact_trace
from . import contact_work
from . import dolfinx_adapter
from . import dolfinx_contact_trace
from . import dolfinx_explicit_contact
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
from .contact_state import ContactProjectionRecord, ContactProjectionState
from .contact_lifecycle import ContactProjectionEvaluation, ContactProjectionLifecycle
from .contact_friction import (
    PenaltyCoulombFrictionLaw,
    PenaltyCoulombFrictionResponse,
    TangentialContactRecord,
    TangentialContactState,
    TangentialKinematicRecord,
    TangentialKinematicState,
    penalty_coulomb_friction_law,
    relative_contact_displacement_increment,
    tangential_contact_state,
)
from .contact_pair import RigidContactPair, rigid_contact_pair
from .contact_response import (
    FrictionlessPenaltyContactLaw,
    FrictionlessPenaltyContactResponse,
    frictionless_penalty_contact_law,
)
from .contact_trace import (
    ContactTrace,
    ContactTraceAssembly,
    ContactTraceEvaluation,
    FrictionContactTraceAssembly,
)
from .contact_work import (
    PrescribedContactWorkState,
    PrescribedContactWorkStation,
    PrescribedRigidMotionSchedule,
    prescribed_contact_work_state,
    prescribed_rigid_motion_schedule,
)
from .dolfinx_adapter import (
    dolfinx_boundary_region_triangle_partition,
    dolfinx_exterior_triangle_partition,
    dolfinx_tagged_exterior_triangle_partition,
)
from .dolfinx_contact_trace import (
    DolfinxContactTraceAdapter,
    dolfinx_boundary_region_contact_trace,
)
from .dolfinx_explicit_contact import (
    DolfinxExplicitContactResidual,
    ExplicitContactEvidence,
    dolfinx_explicit_contact_residual,
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
from .rigid_body import RigidBody, rigid_body
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
    "ContactProjectionRecord",
    "ContactProjectionState",
    "ContactProjectionEvaluation",
    "ContactProjectionLifecycle",
    "PenaltyCoulombFrictionLaw",
    "PenaltyCoulombFrictionResponse",
    "TangentialContactRecord",
    "TangentialContactState",
    "TangentialKinematicRecord",
    "TangentialKinematicState",
    "RigidContactPair",
    "ContactTrace",
    "ContactTraceAssembly",
    "ContactTraceEvaluation",
    "FrictionContactTraceAssembly",
    "PrescribedContactWorkState",
    "PrescribedContactWorkStation",
    "PrescribedRigidMotionSchedule",
    "FrictionlessPenaltyContactLaw",
    "FrictionlessPenaltyContactResponse",
    "DistributedTriangleSearchDiagnostics",
    "DistributedTriangleSearchOutcome",
    "DistributedTriangleSurfaceBVH",
    "DolfinxContactTraceAdapter",
    "DolfinxExplicitContactResidual",
    "ElasticFoundation",
    "ExplicitContactEvidence",
    "PrescribedRigidMotion",
    "RigidCylinderSurface",
    "RigidBody",
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
    "contact_state",
    "contact_lifecycle",
    "contact_friction",
    "contact_pair",
    "contact_response",
    "contact_trace",
    "contact_work",
    "convection",
    "dolfinx_adapter",
    "dolfinx_contact_trace",
    "dolfinx_explicit_contact",
    "dolfinx_explicit_contact_residual",
    "dolfinx_boundary_region_contact_trace",
    "dolfinx_boundary_region_triangle_partition",
    "dolfinx_exterior_triangle_partition",
    "dolfinx_tagged_exterior_triangle_partition",
    "elastic_foundation",
    "frictionless_penalty_contact_law",
    "penalty_coulomb_friction_law",
    "relative_contact_displacement_increment",
    "distributed_triangle_surface_bvh",
    "mechanical",
    "prescribed_rigid_motion",
    "prescribed_contact_work_state",
    "prescribed_rigid_motion_schedule",
    "partition_triangle_surface",
    "rigid",
    "rigid_cylinder",
    "rigid_contact_pair",
    "rigid_body",
    "rigid_plane",
    "rigid_sphere",
    "routed_distributed_triangle_surface_bvh",
    "rigid_obstacle_contact",
    "search",
    "triangle_surface_bvh",
    "triangulated_rigid_surface",
    "thermal",
    "tangential_contact_state",
]
