# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""A bounded, closed-convex sharp rigid tool with a signed-distance branch.

The interior signed distance to the complement of a convex intersection of
half-spaces is the maximum of its outward, unit-normal plane gaps.  This is a
piecewise-affine potential, not a smoothed triangle normal.  Tied active faces
select one deterministic limiting branch of its generalized derivative.
Open, nonconvex, or inconsistently oriented triangle tools are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .rigid import (
    PrescribedRigidMotion,
    RigidSurface,
    SurfaceProjection,
    TriangulatedRigidSurface,
    _closest_point_on_triangle,
    _finite_nonnegative,
    _triangle_barycentric_coordinates,
)


@dataclass(frozen=True, eq=False)
class ConvexPolyhedralRigidSurface(RigidSurface):
    """Closed convex triangle tool, with an exact active interior gap.

    The exterior query is projected to the nearest actual triangle, but only
    interior points contribute penalty traction.  The tool is serial and the
    exhaustive projector is a correctness reference, not a scalable CAD path.
    """

    facets: TriangulatedRigidSurface
    name: str = "convex_polyhedral_rigid_surface"
    _geometry_fingerprint: str = field(init=False, repr=False, compare=False)
    _offsets: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        facets = self.facets
        if not isinstance(facets, TriangulatedRigidSurface):
            raise TypeError("Convex tool requires reviewed oriented triangles.")
        if not str(self.name).strip():
            raise ValueError("Convex tool requires a name.")
        if not facets.summary()["closed"]:
            raise ValueError("Convex tool requires a closed triangle manifold.")
        first = facets.vertices[facets.triangles[:, 0]]
        normals = facets.facet_normals
        offsets = np.einsum("ij,ij->i", first, normals)
        tolerance = max(float(facets.tolerance), 64.0 * np.finfo(float).eps * facets.scale)
        # Every vertex must lie behind every outward face.  This simultaneously
        # rejects inward orientation and nonconvex shape without changing any
        # supplied geometry.  It is a one-time O(facets * vertices) audit.
        for index, (normal, offset) in enumerate(zip(normals, offsets, strict=True)):
            if np.max(facets.vertices @ normal - offset) > tolerance:
                raise ValueError(
                    "Convex tool has a nonconvex or inward-oriented facet "
                    f"at local index {index}."
                )
        identity = hashlib.sha256(
            b"convex_polyhedral_signed_distance_v1\0"
            + bytes.fromhex(facets.geometry_fingerprint)
        ).hexdigest()
        offsets = np.frombuffer(offsets.tobytes(), dtype=np.float64)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(self, "_geometry_fingerprint", identity)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def geometry_fingerprint(self) -> str:
        return self._geometry_fingerprint

    def _project_reference(self, query: np.ndarray, maximum_distance: float | None):
        count = len(query)
        closest = np.full((count, 3), np.nan, dtype=float)
        normals = np.full((count, 3), np.nan, dtype=float)
        gaps = np.full(count, np.nan, dtype=float)
        valid = np.zeros(count, dtype=bool)
        status = np.full(count, "no_candidate", dtype="<U20")
        entity_ids = np.full(count, -1, dtype=np.int64)
        coordinates = np.full((count, 3), np.nan, dtype=float)
        facets = self.facets
        triangle_points = facets.vertices[facets.triangles]
        tolerance = float(facets.tolerance)
        relative_tolerance = tolerance / float(facets.scale)
        limit = (
            float("inf") if maximum_distance is None
            else _finite_nonnegative(maximum_distance, name="Maximum projection distance")
        )
        for point_index, point in enumerate(query):
            plane_gaps = facets.facet_normals @ point - self._offsets
            maximum = float(np.max(plane_gaps))
            if maximum < 0.0:
                # Distance to the union of half-space complements is the
                # smallest distance to one supporting plane.  Choose a face
                # containing the foot; coplanar triangulation is immaterial.
                candidates = np.flatnonzero(plane_gaps >= maximum - tolerance)
                selected = None
                for index in sorted(candidates, key=lambda item: int(facets.facet_ids[item])):
                    foot = point - float(plane_gaps[index]) * facets.facet_normals[index]
                    barycentric = _triangle_barycentric_coordinates(
                        foot, triangle_points[index],
                    )
                    if np.min(barycentric) >= -relative_tolerance:
                        selected = int(index)
                        break
                if selected is None:
                    status[point_index] = "invalid_convex_foot"
                    continue
                gap = float(plane_gaps[selected])
                normal = facets.facet_normals[selected]
                projection = point - gap * normal
            else:
                # Outside the convex solid, the penalty is inactive.  Still
                # publish a real nearest point rather than pretending that a
                # supporting-plane foot outside its facet is a closest point.
                feet = np.asarray([
                    _closest_point_on_triangle(point, *triangle)
                    for triangle in triangle_points
                ])
                distances = np.linalg.norm(feet - point, axis=1)
                minimum = float(np.min(distances))
                ties = np.flatnonzero(distances <= minimum + tolerance)
                selected = int(ties[np.argmin(facets.facet_ids[ties])])
                projection = feet[selected]
                gap = minimum
                normal = (
                    (point - projection) / gap if gap > tolerance
                    else facets.facet_normals[selected]
                )
                barycentric = _triangle_barycentric_coordinates(
                    projection, triangle_points[selected],
                )
            if abs(gap) > limit:
                continue
            closest[point_index] = projection
            normals[point_index] = normal
            gaps[point_index] = gap
            valid[point_index] = True
            status[point_index] = "ok"
            entity_ids[point_index] = int(facets.facet_ids[selected])
            coordinates[point_index] = barycentric
        return closest, normals, gaps, valid, status, entity_ids, coordinates

    def project(
        self, points, *, motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0, maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        query = np.asarray(points, dtype=float)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
            raise ValueError("Convex tool queries must have finite shape (count, 3).")
        if motion is None:
            query_reference = query
            rotation = np.eye(3)
            reference = np.zeros(3)
            translated_reference = reference
        else:
            if motion.dimension != 3:
                raise ValueError("Convex tool motion must be three-dimensional.")
            pose = motion.state(factor)
            rotation = np.asarray(pose["rotation_matrix"], dtype=float)
            reference = np.asarray(motion.reference_point, dtype=float)
            translated_reference = np.asarray(pose["reference_point"], dtype=float)
            query_reference = reference + (query - translated_reference) @ rotation
        closest, normals, gaps, valid, status, entity_ids, coordinates = (
            self._project_reference(query_reference, maximum_distance)
        )
        if motion is not None:
            closest[valid] = translated_reference + (closest[valid] - reference) @ rotation.T
            normals[valid] = normals[valid] @ rotation.T
        return SurfaceProjection(
            surface_name=self.name,
            surface_kind="convex_polyhedral_rigid_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=valid,
            status_codes=status,
            method="closed_convex_signed_distance_semismooth",
            entity_ids=entity_ids,
            geometry_fingerprint=self.geometry_fingerprint,
            local_coordinates=coordinates,
            local_coordinate_system="triangle_barycentric_connectivity_order",
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "convex_polyhedral_rigid_surface",
            "dimension": 3,
            "geometry_fingerprint": self.geometry_fingerprint,
            "facet_geometry_fingerprint": self.facets.geometry_fingerprint,
            "facet_count": int(len(self.facets.triangles)),
            "projection": "exhaustive_closed_convex_signed_distance",
            "active_feature_branch": "minimum_facet_id_limiting_normal",
            "signed_gap_convention": "positive_admissible_negative_penetration",
            "execution_scope": "serial_reference_geometry",
        }


def convex_polyhedral_rigid_surface(
    *, vertices, triangles, facet_ids=None, name="convex_polyhedral_rigid_surface",
) -> ConvexPolyhedralRigidSurface:
    """Create an explicitly certified closed-convex sharp rigid tool."""

    facets = TriangulatedRigidSurface(
        vertices=vertices, triangles=triangles,
        facet_ids=facet_ids, name=f"{name}_facets",
    )
    return ConvexPolyhedralRigidSurface(facets=facets, name=name)


__all__ = ["ConvexPolyhedralRigidSurface", "convex_polyhedral_rigid_surface"]
