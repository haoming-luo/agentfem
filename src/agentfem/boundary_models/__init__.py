# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable weak boundary models."""
from .q1_execution import Q1ContactExecution
from .._exact_rational import exact_geometry_arithmetic

from . import absorbing
from . import contact_state
from . import contact_lifecycle
from . import contact_projection_checkpoint
from . import contact_friction
from . import contact_pair
from . import contact_response
from . import contact_stability
from . import contact_trace
from . import contact_work
from . import dolfinx_adapter
from . import dolfinx_contact_trace
from . import dolfinx_contact_stability
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
from .contact_projection_checkpoint import (
    global_projection_state_snapshot,
    local_projection_state_from_snapshot,
)
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
from .closed_polyhedron import (
    ClosedTriangulatedRigidSolid, closed_triangulated_rigid_solid,
    closed_triangulated_rigid_solid_from_mesh,
)
from .convex_polyhedron import (
    ConvexPolyhedralRigidSurface,
    convex_polyhedral_rigid_surface,
    convex_polyhedral_rigid_surface_from_mesh,
)
from .deformable_pair import DeformableContactPair, deformable_contact_pair
from .trace_policy import RigidContactTracePolicy
from .contact_response import (
    FrictionlessPenaltyContactLaw,
    FrictionlessPenaltyContactResponse,
    frictionless_penalty_contact_law,
)
from .contact_stability import (
    CombinedExplicitStabilityEstimate,
    ContactStabilityEstimate,
    combine_explicit_stability_bounds,
    contact_penalty_local_row_sums,
    contact_stability_estimate_from_bound,
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
from .dolfinx_contact_stability import (
    estimate_dolfinx_contact_stability,
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
    SmoothRigidSurface,
    SurfaceGapDifferential,
    SurfaceGapDifferentialCheck,
    SurfaceProjection,
    TriangulatedRigidSurface,
    prescribed_rigid_motion,
    rigid_cylinder,
    rigid_plane,
    rigid_sphere,
    triangulated_rigid_surface,
    triangulated_rigid_surface_from_mesh,
    check_surface_gap_differential,
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
    "Q1ContactExecution",
    "exact_geometry_arithmetic",
    "RigidContactTracePolicy",
    "DeformableContactPair", "deformable_contact_pair",
    "ConvectionBoundary",
    "ContactProjectionRecord",
    "ConvexPolyhedralRigidSurface",
    "ClosedTriangulatedRigidSolid",
    "closed_triangulated_rigid_solid",
    "closed_triangulated_rigid_solid_from_mesh",
    "ContactProjectionState",
    "ContactProjectionEvaluation",
    "ContactProjectionLifecycle",
    "CombinedExplicitStabilityEstimate",
    "ContactStabilityEstimate",
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
    "SmoothRigidSurface",
    "SurfaceGapDifferential",
    "SurfaceGapDifferentialCheck",
    "check_surface_gap_differential",
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
    "contact_projection_checkpoint",
    "contact_friction",
    "contact_pair",
    "contact_response",
    "contact_stability",
    "contact_trace",
    "contact_work",
    "global_projection_state_snapshot",
    "local_projection_state_from_snapshot",
    "convection",
    "convex_polyhedral_rigid_surface",
    "convex_polyhedral_rigid_surface_from_mesh",
    "dolfinx_adapter",
    "dolfinx_contact_trace",
    "dolfinx_contact_stability",
    "dolfinx_explicit_contact",
    "dolfinx_explicit_contact_residual",
    "dolfinx_boundary_region_contact_trace",
    "estimate_dolfinx_contact_stability",
    "dolfinx_boundary_region_triangle_partition",
    "dolfinx_exterior_triangle_partition",
    "dolfinx_tagged_exterior_triangle_partition",
    "elastic_foundation",
    "frictionless_penalty_contact_law",
    "contact_penalty_local_row_sums",
    "contact_stability_estimate_from_bound",
    "combine_explicit_stability_bounds",
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
    "triangulated_rigid_surface_from_mesh",
    "q1_contact_runtime",
    "thermal",
    "tangential_contact_state",
]


def q1_contact_runtime(*, backend="auto", geometry_workers=0, projection_workers=1,
                       separation_reuse="incremental", geometry_timeout=300.0,
                       sign_policy="two_rays"):
    """Instance-owned Q1 arithmetic; use as a context around the Step lifetime.

    Pass runtime.execution to the ordinary deformable-contact pair. Native
    algorithms are optional. Reference fallback preserves geometry, contact
    laws, ambiguity rules and state ownership; it is not a second solver.
    geometry_timeout bounds a reference-worker batch in wall-clock seconds;
    None disables that operational deadline. It does not alter physical time.
    sign_policy="interior_normal" optionally classifies strict smooth interiors
    of reviewed connected outward closed Q1 solids from current geometry.
    Edges, vertices and unresolved signs retain the two-ray reference policy.
    """
    from .native_execution import Q1ContactRuntime
    return Q1ContactRuntime(backend=backend, geometry_workers=geometry_workers,
                            projection_workers=projection_workers,
                            separation_reuse=separation_reuse,
                            geometry_timeout=geometry_timeout,
                            sign_policy=sign_policy)
