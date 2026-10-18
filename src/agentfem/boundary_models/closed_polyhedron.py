# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Signed-distance reference for a reviewed closed, possibly concave tool.

This deliberately bounded serial geometry is distinct from an oriented open
triangle sheet. A caller must attest that the closed triangle manifold is an
embedded, outward-oriented solid: topological closure alone does not detect
self-intersection. Query-time winding evidence must still be unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from math import pi
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .rigid import (
    PrescribedRigidMotion, RigidSurface, SurfaceGapDifferential,
    SurfaceProjection, TriangulatedRigidSurface, _finite_nonnegative,
    triangulated_rigid_surface_from_mesh,
)
from .search import TriangleSurfaceBVH, triangle_surface_bvh


def _solid_winding_number(point: np.ndarray, triangles: np.ndarray) -> float:
    """Evaluate the oriented solid-angle sum, with no approximate thresholding."""

    a = triangles[:, 0, :] - point
    b = triangles[:, 1, :] - point
    c = triangles[:, 2, :] - point
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return float(np.sum(2.0 * np.arctan2(numerator, denominator)) / (4.0 * pi))


def _feature_projector(
    triangle: np.ndarray, barycentric: np.ndarray, tolerance: float,
) -> tuple[np.ndarray, str]:
    """Return the tangent projector of one nearest face/edge/vertex branch."""

    on_edge = np.flatnonzero(barycentric <= tolerance)
    if len(on_edge) == 0:
        return np.zeros((3, 3)), "face"
    if len(on_edge) >= 2:
        return np.eye(3), "vertex"
    remaining = np.flatnonzero(np.arange(3) != on_edge[0])
    edge = triangle[remaining[1]] - triangle[remaining[0]]
    edge /= np.linalg.norm(edge)
    return np.eye(3) - np.outer(edge, edge), "edge"


@dataclass(frozen=True, eq=False)
class ClosedTriangulatedRigidSolid(RigidSurface):
    """Exact nearest-feature signed gap for a reviewed embedded triangle solid.

    Nearest-feature search is repeated at every current query. Queries not
    certified exterior by the complete AABB use direct winding. No old contact
    gap or normal is cached. The winding path is an O(NF) *reference* sign
    classifier and is not a production large-CAD backend.
    Medial-axis ties with distinct nearest points fail closed.
    """

    facets: TriangulatedRigidSurface
    reviewed_embedded: bool
    name: str = "closed_triangulated_rigid_solid"
    _geometry_fingerprint: str = field(init=False, repr=False, compare=False)
    _bvh: TriangleSurfaceBVH = field(init=False, repr=False, compare=False)
    _triangles: np.ndarray = field(init=False, repr=False, compare=False)
    _facet_slots: Mapping[int, int] = field(init=False, repr=False, compare=False)
    _vertex_facets: tuple[np.ndarray, ...] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        facets = self.facets
        if not isinstance(facets, TriangulatedRigidSurface):
            raise TypeError("Closed rigid solid requires reviewed triangles.")
        if self.reviewed_embedded is not True:
            raise ValueError(
                "Closed rigid solid requires reviewed_embedded=True: topology "
                "checks do not certify the absence of self-intersection."
            )
        if not facets.summary()["closed"]:
            raise ValueError("Closed rigid solid requires a closed triangle manifold.")
        if not str(self.name).strip():
            raise ValueError("Closed rigid solid requires a name.")
        triangles = np.asarray(facets.vertices[facets.triangles], dtype=float)
        triangles.setflags(write=False)
        origin = np.mean(facets.vertices, axis=0)
        centered = triangles - origin
        volume_terms = np.einsum(
            "ij,ij->i", centered[:, 0],
            np.cross(centered[:, 1], centered[:, 2]),
        ) / 6.0
        signed_volume = float(np.sum(volume_terms))
        if not np.isfinite(signed_volume) or signed_volume <= (
            max(float(np.sum(np.abs(volume_terms))), np.finfo(float).tiny)
            * 64.0 * np.finfo(float).eps
        ):
            raise ValueError(
                "Closed rigid solid requires outward orientation and positive "
                "enclosed volume."
            )
        identity = hashlib.sha256(
            b"closed_triangulated_signed_distance_v1\0"
            + bytes.fromhex(facets.geometry_fingerprint)
        ).hexdigest()
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "_geometry_fingerprint", identity)
        object.__setattr__(self, "_bvh", triangle_surface_bvh(facets))
        object.__setattr__(self, "_triangles", triangles)
        object.__setattr__(self, "_facet_slots", MappingProxyType({
            int(identity): index for index, identity in enumerate(facets.facet_ids)
        }))
        incident: list[list[int]] = [[] for _ in range(len(facets.vertices))]
        for slot, triangle in enumerate(facets.triangles):
            for vertex in triangle:
                incident[int(vertex)].append(slot)
        vertex_facets = tuple(
            np.asarray(slots, dtype=np.int64) for slots in incident
        )
        for slots in vertex_facets:
            slots.setflags(write=False)
        object.__setattr__(self, "_vertex_facets", vertex_facets)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def geometry_fingerprint(self) -> str:
        return self._geometry_fingerprint

    def _geometric_feature_projector(
        self, slot: int, barycentric: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Ignore artificial coplanar seams and subdivided straight edges."""

        tolerance = self.facets.tolerance / self.facets.scale
        angular_tolerance = max(128.0 * np.finfo(float).eps, tolerance)
        opposite = np.flatnonzero(barycentric <= tolerance)
        if len(opposite) == 0:
            return np.zeros((3, 3)), "face"
        if len(opposite) == 1:
            neighbor = int(self.facets.edge_neighbors[slot, opposite[0]])
            if neighbor >= 0 and np.linalg.norm(
                self.facets._facet_normals[slot]
                - self.facets._facet_normals[neighbor]
            ) <= angular_tolerance:
                return np.zeros((3, 3)), "face"
            return _feature_projector(
                self._triangles[slot].copy(), barycentric, tolerance,
            )
        vertex = int(self.facets.triangles[slot, int(np.argmax(barycentric))])
        normals = self.facets._facet_normals[self._vertex_facets[vertex]]
        _, singular, directions = np.linalg.svd(normals, full_matrices=True)
        rank = int(np.count_nonzero(
            singular > singular[0] * angular_tolerance,
        ))
        if rank == 1:
            return np.zeros((3, 3)), "face"
        if rank == 2:
            edge = directions[-1]
            return np.eye(3) - np.outer(edge, edge), "edge"
        return np.eye(3), "vertex"

    def project(
        self, points, *, motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0, maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        query = np.asarray(points, dtype=float)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
            raise ValueError("Closed-solid queries require finite shape (count, 3).")
        if maximum_distance is not None:
            _finite_nonnegative(maximum_distance, name="Maximum projection distance")
        if motion is None:
            query_reference = query
            rotation = np.eye(3)
            reference = np.zeros(3)
            moved_reference = reference
        else:
            if motion.dimension != 3:
                raise ValueError("Closed-solid motion must be three-dimensional.")
            pose = motion.state(factor)
            rotation = np.asarray(pose["rotation_matrix"], dtype=float)
            reference = np.asarray(motion.reference_point, dtype=float)
            moved_reference = np.asarray(pose["reference_point"], dtype=float)
            query_reference = reference + (query - moved_reference) @ rotation

        candidates = self._bvh._candidate_evidence(
            query_reference, maximum_distance, require_same_normal=False,
        )
        count = len(query)
        closest = np.full((count, 3), np.nan)
        normals = np.full((count, 3), np.nan)
        gaps = np.full(count, np.nan)
        valid = np.zeros(count, dtype=bool)
        status = np.full(count, "no_candidate", dtype="<U32")
        identity = np.full(count, -1, dtype=np.int64)
        local = np.full((count, 3), np.nan)
        tolerance = float(self.facets.tolerance)
        lower, upper = self._bvh.bounds
        status[candidates.ambiguous] = "nonunique_nearest_feature"
        for index in np.flatnonzero(candidates.has_candidate & ~candidates.ambiguous):
            point = query_reference[index]
            foot = candidates.closest_points[index]
            radius = float(np.sqrt(candidates.squared_distances[index]))
            barycentric = candidates.local_coordinates[index]
            if radius <= tolerance:
                _, feature = self._geometric_feature_projector(
                    self._facet_slots[int(candidates.entity_ids[index])],
                    barycentric,
                )
                if feature != "face":
                    status[index] = "nonsmooth_on_surface"
                    continue
                gap = 0.0
                normal = candidates.normals[index]
            else:
                # A point strictly beyond the complete tool AABB cannot be
                # inside any embedded closed surface. The tolerance margin
                # only shrinks this certified exterior region; it never
                # changes the sign of a point requiring winding evidence.
                beyond_bounds = np.any(
                    (point < lower - tolerance) | (point > upper + tolerance)
                )
                winding = (
                    0.0 if beyond_bounds
                    else _solid_winding_number(point, self._triangles)
                )
                if abs(winding) <= 1.0e-6:
                    sign = 1.0
                elif abs(winding - 1.0) <= 1.0e-6:
                    sign = -1.0
                else:
                    status[index] = "ambiguous_solid_winding"
                    continue
                gap = sign * radius
                normal = sign * (point - foot) / radius
            closest[index] = foot
            normals[index] = normal
            gaps[index] = gap
            valid[index] = True
            status[index] = "ok"
            identity[index] = candidates.entity_ids[index]
            local[index] = barycentric
        if motion is not None:
            closest[valid] = moved_reference + (closest[valid] - reference) @ rotation.T
            normals[valid] = normals[valid] @ rotation.T
        return SurfaceProjection(
            surface_name=self.name,
            surface_kind="closed_triangulated_rigid_solid",
            query_points=query, closest_points=closest, normals=normals,
            signed_gaps=gaps, valid=valid, status_codes=status,
            method="exact_bvh_nearest_feature_plus_certified_aabb_or_winding",
            entity_ids=identity, geometry_fingerprint=self.geometry_fingerprint,
            local_coordinates=local,
            local_coordinate_system="triangle_barycentric_connectivity_order",
        )

    def gap_differential(
        self, projection: SurfaceProjection, *,
        motion: PrescribedRigidMotion | None = None, factor: float = 0.0,
    ) -> SurfaceGapDifferential:
        if (
            not isinstance(projection, SurfaceProjection)
            or projection.geometry_fingerprint != self.geometry_fingerprint
            or projection.surface_kind != "closed_triangulated_rigid_solid"
            or not projection.all_valid
        ):
            raise ValueError("Closed-solid differential needs its valid projection.")
        rotation = (
            np.eye(3) if motion is None
            else np.asarray(motion.state(factor)["rotation_matrix"], dtype=float)
        )
        hessian = np.zeros((projection.point_count, 3, 3))
        for index, (gap, barycentric, identity) in enumerate(zip(
            projection.signed_gaps, projection.local_coordinates,
            projection.entity_ids, strict=True,
        )):
            radius = abs(float(gap))
            P, feature = self._geometric_feature_projector(
                self._facet_slots[int(identity)], barycentric,
            )
            if feature == "face":
                continue
            if radius <= self.facets.tolerance:
                raise ValueError("Closed-solid feature Hessian is singular at contact.")
            normal = projection.normals[index]
            world_P = rotation @ P @ rotation.T
            hessian[index] = np.sign(gap) * (
                world_P - np.outer(normal, normal)
            ) / radius
        return SurfaceGapDifferential(projection, hessian)

    def summary(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "name": self.name,
            "kind": "closed_triangulated_rigid_solid",
            "dimension": 3,
            "facet_count": int(len(self.facets.triangles)),
            "geometry_fingerprint": self.geometry_fingerprint,
            "facet_geometry_fingerprint": self.facets.geometry_fingerprint,
            "projection": "nearest_feature_plus_certified_aabb_or_winding",
            "self_intersection_certificate": False,
            "embedded_surface_review": "caller_attested",
            "execution_scope": "serial_reference_geometry",
        }
        if self.facets.source_format is not None:
            summary["source"] = self.facets.summary()["source"]
        return summary


def closed_triangulated_rigid_solid(
    *, vertices, triangles, reviewed_embedded: bool, facet_ids=None,
    name: str = "closed_triangulated_rigid_solid",
) -> ClosedTriangulatedRigidSolid:
    """Build a bounded concave-capable solid, requiring embedding review."""

    facets = TriangulatedRigidSurface(
        vertices=vertices, triangles=triangles, facet_ids=facet_ids,
        name=f"{name}_facets",
    )
    return ClosedTriangulatedRigidSolid(
        facets=facets, reviewed_embedded=reviewed_embedded, name=name,
    )


def closed_triangulated_rigid_solid_from_mesh(
    path, *, coordinate_scale: float, reviewed_embedded: bool,
    input_format: str | None = None, flip_normals: bool = False,
    name: str | None = None, tolerance: float | None = None,
    ambiguity_tolerance: float | None = None,
) -> ClosedTriangulatedRigidSolid:
    """Import a reviewed triangle solid without inferring embedding validity."""

    facets = triangulated_rigid_surface_from_mesh(
        path, coordinate_scale=coordinate_scale, input_format=input_format,
        flip_normals=flip_normals, name=name, tolerance=tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
    )
    return ClosedTriangulatedRigidSolid(
        facets=facets, reviewed_embedded=reviewed_embedded,
        name=facets.name,
    )


__all__ = [
    "ClosedTriangulatedRigidSolid",
    "closed_triangulated_rigid_solid",
    "closed_triangulated_rigid_solid_from_mesh",
]
