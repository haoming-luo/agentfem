# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic broad-phase search for reviewed rigid surfaces.

The search layer accelerates candidate discovery without changing projection
semantics.  Exact closest points, gap orientation, ambiguity, and stable facet
identity remain owned by the reviewed :class:`TriangulatedRigidSurface`.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .rigid import (
    PrescribedRigidMotion,
    SurfaceProjection,
    TriangulatedRigidSurface,
    _closest_point_on_triangle,
    _finite_nonnegative,
    _projection_query,
    _readonly_array,
    _triangle_barycentric_coordinates,
)


def _bbox_squared_distance(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    offset = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return float(np.dot(offset, offset))


@dataclass(frozen=True)
class TriangleSearchDiagnostics:
    """Per-query work evidence for one immutable triangle BVH."""

    geometry_fingerprint: str
    tree_node_count: int
    facet_count: int
    visited_node_counts: tuple[int, ...]
    evaluated_facet_counts: tuple[int, ...]
    method: str = "deterministic_aabb_bvh"

    def __post_init__(self) -> None:
        fingerprint = str(self.geometry_fingerprint).lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("Search diagnostics require a SHA-256 geometry identity.")
        tree_node_count = int(self.tree_node_count)
        facet_count = int(self.facet_count)
        visited = tuple(int(value) for value in self.visited_node_counts)
        evaluated = tuple(int(value) for value in self.evaluated_facet_counts)
        if tree_node_count < 1 or facet_count < 1:
            raise ValueError("Search diagnostics require a non-empty tree and surface.")
        if len(visited) != len(evaluated):
            raise ValueError("Search diagnostics require aligned per-query counts.")
        if any(value < 0 or value > tree_node_count for value in visited):
            raise ValueError("Visited-node counts must lie within the tree size.")
        if any(value < 0 or value > facet_count for value in evaluated):
            raise ValueError("Evaluated-facet counts must lie within the facet count.")
        if not str(self.method).strip():
            raise ValueError("Search diagnostics require a method.")
        object.__setattr__(self, "geometry_fingerprint", fingerprint)
        object.__setattr__(self, "tree_node_count", tree_node_count)
        object.__setattr__(self, "facet_count", facet_count)
        object.__setattr__(self, "visited_node_counts", visited)
        object.__setattr__(self, "evaluated_facet_counts", evaluated)
        object.__setattr__(self, "method", str(self.method))

    @property
    def query_count(self) -> int:
        return len(self.visited_node_counts)

    def summary(self) -> dict[str, object]:
        evaluated = self.evaluated_facet_counts
        visited = self.visited_node_counts
        return {
            "method": self.method,
            "geometry_fingerprint": self.geometry_fingerprint,
            "query_count": self.query_count,
            "tree_node_count": self.tree_node_count,
            "facet_count": self.facet_count,
            "maximum_visited_nodes": max(visited, default=0),
            "mean_visited_nodes": (
                float(np.mean(visited)) if visited else 0.0
            ),
            "maximum_evaluated_facets": max(evaluated, default=0),
            "mean_evaluated_facets": (
                float(np.mean(evaluated)) if evaluated else 0.0
            ),
            "distributed_ownership": False,
        }


@dataclass(frozen=True)
class TriangleSearchOutcome:
    """Projection evidence paired with broad-phase work diagnostics."""

    projection: SurfaceProjection
    diagnostics: TriangleSearchDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.projection, SurfaceProjection):
            raise TypeError("Search outcome requires SurfaceProjection evidence.")
        if not isinstance(self.diagnostics, TriangleSearchDiagnostics):
            raise TypeError("Search outcome requires TriangleSearchDiagnostics.")
        if self.projection.point_count != self.diagnostics.query_count:
            raise ValueError("Projection and search diagnostics point counts differ.")
        if (
            self.projection.geometry_fingerprint
            != self.diagnostics.geometry_fingerprint
        ):
            raise ValueError("Projection and search diagnostics geometry differ.")


class TriangleSurfaceBVH:
    """Immutable deterministic AABB tree over a reviewed triangle surface.

    The current object is replicated and process-local.  It is safe to build on
    each MPI rank, but it does not claim distributed facet ownership or ghost
    exchange.  Those remain a separate promotion gate.
    """

    def __init__(self, surface: TriangulatedRigidSurface):
        if not isinstance(surface, TriangulatedRigidSurface):
            raise TypeError("TriangleSurfaceBVH requires a TriangulatedRigidSurface.")
        self._surface = surface
        triangle_points = np.asarray(surface.vertices[surface.triangles], dtype=float)
        facet_lower = np.min(triangle_points, axis=1)
        facet_upper = np.max(triangle_points, axis=1)
        centroids = np.mean(triangle_points, axis=1)
        facet_ids = np.asarray(surface.facet_ids, dtype=np.int64)

        lower_nodes: list[np.ndarray] = []
        upper_nodes: list[np.ndarray] = []
        left_nodes: list[int] = []
        right_nodes: list[int] = []
        facet_nodes: list[int] = []

        def build(indices: np.ndarray) -> int:
            node = len(lower_nodes)
            lower_nodes.append(np.min(facet_lower[indices], axis=0))
            upper_nodes.append(np.max(facet_upper[indices], axis=0))
            left_nodes.append(-1)
            right_nodes.append(-1)
            facet_nodes.append(-1)
            if indices.size == 1:
                facet_nodes[node] = int(indices[0])
                return node

            spans = np.ptp(centroids[indices], axis=0)
            axis = int(np.argmax(spans))
            order = np.lexsort((facet_ids[indices], centroids[indices, axis]))
            ordered = indices[order]
            middle = ordered.size // 2
            left_nodes[node] = build(ordered[:middle])
            right_nodes[node] = build(ordered[middle:])
            return node

        root = build(np.arange(triangle_points.shape[0], dtype=np.int64))
        self._triangle_points = _readonly_array(triangle_points)
        self._facet_normals = _readonly_array(surface.facet_normals)
        self._lower = _readonly_array(lower_nodes)
        self._upper = _readonly_array(upper_nodes)
        self._left = _readonly_array(left_nodes, dtype=np.int64)
        self._right = _readonly_array(right_nodes, dtype=np.int64)
        self._facet = _readonly_array(facet_nodes, dtype=np.int64)
        self._root = root

    @property
    def surface(self) -> TriangulatedRigidSurface:
        return self._surface

    @property
    def node_count(self) -> int:
        return int(self._facet.size)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "triangle_surface_bvh",
            "method": "deterministic_aabb_bvh",
            "geometry_fingerprint": self.surface.geometry_fingerprint,
            "tree_node_count": self.node_count,
            "leaf_count": int(self.surface.triangles.shape[0]),
            "leaf_capacity": 1,
            "exact_narrow_phase": "reviewed_triangle_closest_point",
            "ambiguity_policy": "preserve_all_equidistant_candidates",
            "execution_scope": "replicated_process_local",
            "distributed_ownership": False,
        }

    def _project_reference(
        self,
        query: np.ndarray,
        maximum_distance: float | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
    ]:
        count = int(query.shape[0])
        closest = np.full((count, 3), np.nan, dtype=float)
        normals = np.full((count, 3), np.nan, dtype=float)
        gaps = np.full(count, np.nan, dtype=float)
        valid = np.zeros(count, dtype=bool)
        statuses = np.full(count, "no_candidate", dtype="<U20")
        entity_ids = np.full(count, -1, dtype=np.int64)
        local_coordinates = np.full((count, 3), np.nan, dtype=float)
        visited_counts: list[int] = []
        evaluated_counts: list[int] = []
        maximum_squared = (
            float("inf")
            if maximum_distance is None
            else _finite_nonnegative(
                maximum_distance,
                name="Maximum projection distance",
            )
            ** 2
        )
        ambiguity_squared = float(self.surface.ambiguity_tolerance) ** 2
        facet_ids = np.asarray(self.surface.facet_ids, dtype=np.int64)

        for point_index, point in enumerate(query):
            queue = [
                (
                    _bbox_squared_distance(
                        point,
                        self._lower[self._root],
                        self._upper[self._root],
                    ),
                    self._root,
                )
            ]
            candidates: list[tuple[int, np.ndarray, float]] = []
            best = maximum_squared
            visited = 0
            evaluated = 0
            while queue:
                lower_bound, node = heapq.heappop(queue)
                threshold = np.nextafter(best + ambiguity_squared, np.inf)
                if lower_bound > threshold:
                    break
                visited += 1
                facet = int(self._facet[node])
                if facet >= 0:
                    candidate = _closest_point_on_triangle(
                        point,
                        *self._triangle_points[facet],
                    )
                    squared_distance = float(np.sum((candidate - point) ** 2))
                    evaluated += 1
                    if squared_distance <= np.nextafter(maximum_squared, np.inf):
                        candidates.append((facet, candidate, squared_distance))
                        best = min(best, squared_distance)
                    continue
                for child in (int(self._left[node]), int(self._right[node])):
                    child_distance = _bbox_squared_distance(
                        point,
                        self._lower[child],
                        self._upper[child],
                    )
                    if child_distance <= threshold:
                        heapq.heappush(queue, (child_distance, child))

            visited_counts.append(visited)
            evaluated_counts.append(evaluated)
            if not candidates:
                continue
            minimum = min(item[2] for item in candidates)
            tied = [
                item
                for item in candidates
                if item[2] <= np.nextafter(minimum + ambiguity_squared, np.inf)
            ]
            selected_facet, selected_point, _ = min(
                tied,
                key=lambda item: int(facet_ids[item[0]]),
            )
            if len(tied) > 1:
                same_point = all(
                    np.linalg.norm(candidate - selected_point)
                    <= self.surface.ambiguity_tolerance
                    for _, candidate, _ in tied
                )
                selected_normal = self._facet_normals[selected_facet]
                same_normal = all(
                    np.linalg.norm(self._facet_normals[facet] - selected_normal)
                    <= self.surface.ambiguity_tolerance / self.surface.scale
                    for facet, _, _ in tied
                )
                if not (same_point and same_normal):
                    statuses[point_index] = "ambiguous_projection"
                    continue
            closest[point_index] = selected_point
            normals[point_index] = self._facet_normals[selected_facet]
            gaps[point_index] = float(
                np.dot(point - selected_point, self._facet_normals[selected_facet])
            )
            valid[point_index] = True
            statuses[point_index] = "ok"
            entity_ids[point_index] = int(facet_ids[selected_facet])
            local_coordinates[point_index] = _triangle_barycentric_coordinates(
                selected_point,
                self._triangle_points[selected_facet],
            )

        return (
            closest,
            normals,
            gaps,
            valid,
            statuses,
            entity_ids,
            local_coordinates,
            tuple(visited_counts),
            tuple(evaluated_counts),
        )

    def project_with_diagnostics(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> TriangleSearchOutcome:
        """Project points and return both physical and search evidence."""

        query = _projection_query(
            points,
            dimension=3,
            geometry="Triangle-surface BVH",
        )
        if motion is None:
            query_reference = query
            rotation = np.eye(3, dtype=float)
            reference = np.zeros(3, dtype=float)
            translated_reference = reference
        else:
            if motion.dimension != 3:
                raise ValueError("Triangulated surface motion must be three-dimensional.")
            state = motion.state(factor)
            rotation = np.asarray(state["rotation_matrix"], dtype=float)
            reference = np.asarray(motion.reference_point, dtype=float)
            translated_reference = np.asarray(state["reference_point"], dtype=float)
            query_reference = reference + (query - translated_reference) @ rotation

        (
            closest,
            normals,
            gaps,
            valid,
            statuses,
            entity_ids,
            local_coordinates,
            visited,
            evaluated,
        ) = self._project_reference(query_reference, maximum_distance)
        if motion is not None:
            closest[valid] = (
                translated_reference + (closest[valid] - reference) @ rotation.T
            )
            normals[valid] = normals[valid] @ rotation.T
        projection = SurfaceProjection(
            surface_name=self.surface.name,
            surface_kind="triangulated_rigid_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=valid,
            status_codes=statuses,
            method="aabb_bvh_exact_triangle_projection",
            entity_ids=entity_ids,
            geometry_fingerprint=self.surface.geometry_fingerprint,
            local_coordinates=local_coordinates,
            local_coordinate_system="triangle_barycentric_connectivity_order",
        )
        diagnostics = TriangleSearchDiagnostics(
            geometry_fingerprint=self.surface.geometry_fingerprint,
            tree_node_count=self.node_count,
            facet_count=int(self.surface.triangles.shape[0]),
            visited_node_counts=visited,
            evaluated_facet_counts=evaluated,
        )
        return TriangleSearchOutcome(projection=projection, diagnostics=diagnostics)

    def project(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        """Project points through the accelerated tree."""

        return self.project_with_diagnostics(
            points,
            motion=motion,
            factor=factor,
            maximum_distance=maximum_distance,
        ).projection


def triangle_surface_bvh(surface: TriangulatedRigidSurface) -> TriangleSurfaceBVH:
    """Build a deterministic process-local BVH for one reviewed surface."""

    return TriangleSurfaceBVH(surface)


__all__ = [
    "TriangleSearchDiagnostics",
    "TriangleSearchOutcome",
    "TriangleSurfaceBVH",
    "triangle_surface_bvh",
]
