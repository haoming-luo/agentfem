# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic broad-phase search for reviewed rigid surfaces.

The search layer accelerates candidate discovery without changing projection
semantics.  Exact closest points, gap orientation, ambiguity, and stable facet
identity remain owned by the reviewed :class:`TriangulatedRigidSurface`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


def _facet_identity_fingerprint(values) -> str:
    selected = np.sort(np.asarray(values, dtype="<i8").reshape(-1))
    digest = hashlib.sha256()
    digest.update(np.asarray(selected.shape, dtype="<i8").tobytes())
    digest.update(selected.tobytes())
    return digest.hexdigest()


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


@dataclass(frozen=True)
class TriangleSurfacePartition:
    """One deterministic facet shard of a globally identified rigid surface.

    This first distributed-search contract starts from a reviewed surface that
    is replicated while the partition is constructed.  Only the compact local
    shard is retained by the search object.  Importers may later construct the
    same contract directly from distributed mesh ownership.
    """

    local_surface: TriangulatedRigidSurface
    global_geometry_fingerprint: str
    global_facet_identity_fingerprint: str
    global_surface_name: str
    global_facet_count: int
    global_scale: float
    global_ambiguity_tolerance: float
    rank: int
    rank_count: int
    ownership_method: str = "spatial_centroid_contiguous"

    def __post_init__(self) -> None:
        if not isinstance(self.local_surface, TriangulatedRigidSurface):
            raise TypeError("Triangle partition requires a local triangle surface.")
        fingerprint = str(self.global_geometry_fingerprint).lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("Triangle partition requires a global SHA-256 identity.")
        facet_fingerprint = str(self.global_facet_identity_fingerprint).lower()
        if len(facet_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in facet_fingerprint
        ):
            raise ValueError(
                "Triangle partition requires a global facet-identity digest."
            )
        rank = int(self.rank)
        rank_count = int(self.rank_count)
        global_facet_count = int(self.global_facet_count)
        if rank_count < 1 or rank < 0 or rank >= rank_count:
            raise ValueError("Triangle partition rank must lie within its communicator.")
        if global_facet_count < rank_count:
            raise ValueError(
                "Reference triangle partitioning requires at least one facet per rank."
            )
        if not str(self.global_surface_name).strip():
            raise ValueError("Triangle partition requires a global surface name.")
        scale = float(self.global_scale)
        ambiguity = float(self.global_ambiguity_tolerance)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Triangle partition requires a positive global scale.")
        if not np.isfinite(ambiguity) or ambiguity <= 0.0:
            raise ValueError(
                "Triangle partition requires a positive ambiguity tolerance."
            )
        if not str(self.ownership_method).strip():
            raise ValueError("Triangle partition requires an ownership method.")
        object.__setattr__(self, "global_geometry_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "global_facet_identity_fingerprint",
            facet_fingerprint,
        )
        object.__setattr__(self, "global_surface_name", str(self.global_surface_name))
        object.__setattr__(self, "global_facet_count", global_facet_count)
        object.__setattr__(self, "global_scale", scale)
        object.__setattr__(self, "global_ambiguity_tolerance", ambiguity)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "rank_count", rank_count)
        object.__setattr__(self, "ownership_method", str(self.ownership_method))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "triangle_surface_partition",
            "global_geometry_fingerprint": self.global_geometry_fingerprint,
            "global_facet_identity_fingerprint": (
                self.global_facet_identity_fingerprint
            ),
            "global_surface_name": self.global_surface_name,
            "global_facet_count": self.global_facet_count,
            "local_facet_count": int(self.local_surface.triangles.shape[0]),
            "local_facet_ids": tuple(
                int(value) for value in self.local_surface.facet_ids
            ),
            "rank": self.rank,
            "rank_count": self.rank_count,
            "ownership_method": self.ownership_method,
            "distributed_ownership": True,
        }


@dataclass(frozen=True)
class DistributedTriangleSearchDiagnostics:
    """Communication and local-work evidence for the reference MPI search."""

    geometry_fingerprint: str
    rank: int
    rank_count: int
    local_query_count: int
    global_query_count: int
    local_facet_count: int
    global_facet_count: int
    gathered_query_bytes: int
    gathered_candidate_bytes: int
    visited_node_counts: tuple[int, ...]
    evaluated_facet_counts: tuple[int, ...]
    method: str = "allgather_reference_distributed_bvh"

    def __post_init__(self) -> None:
        fingerprint = str(self.geometry_fingerprint).lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError(
                "Distributed search diagnostics require a SHA-256 identity."
            )
        integer_fields = (
            "rank",
            "rank_count",
            "local_query_count",
            "global_query_count",
            "local_facet_count",
            "global_facet_count",
            "gathered_query_bytes",
            "gathered_candidate_bytes",
        )
        values = {name: int(getattr(self, name)) for name in integer_fields}
        if values["rank_count"] < 1 or not (
            0 <= values["rank"] < values["rank_count"]
        ):
            raise ValueError("Distributed diagnostics require a valid rank.")
        if any(values[name] < 0 for name in integer_fields[2:]):
            raise ValueError("Distributed diagnostic counts must be non-negative.")
        if values["local_query_count"] > values["global_query_count"]:
            raise ValueError("Local query count cannot exceed the global count.")
        if values["local_facet_count"] > values["global_facet_count"]:
            raise ValueError("Local facet count cannot exceed the global count.")
        visited = tuple(int(value) for value in self.visited_node_counts)
        evaluated = tuple(int(value) for value in self.evaluated_facet_counts)
        if len(visited) != values["local_query_count"] or len(evaluated) != len(
            visited
        ):
            raise ValueError(
                "Distributed work counts must contain one value per local query."
            )
        if any(value < 0 for value in (*visited, *evaluated)):
            raise ValueError("Distributed work counts must be non-negative.")
        if not str(self.method).strip():
            raise ValueError("Distributed diagnostics require a method.")
        object.__setattr__(self, "geometry_fingerprint", fingerprint)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "visited_node_counts", visited)
        object.__setattr__(self, "evaluated_facet_counts", evaluated)
        object.__setattr__(self, "method", str(self.method))

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method,
            "geometry_fingerprint": self.geometry_fingerprint,
            "rank": self.rank,
            "rank_count": self.rank_count,
            "local_query_count": self.local_query_count,
            "global_query_count": self.global_query_count,
            "local_facet_count": self.local_facet_count,
            "global_facet_count": self.global_facet_count,
            "gathered_query_bytes": self.gathered_query_bytes,
            "gathered_candidate_bytes": self.gathered_candidate_bytes,
            "maximum_visited_nodes": max(self.visited_node_counts, default=0),
            "maximum_evaluated_facets": max(
                self.evaluated_facet_counts, default=0
            ),
            "distributed_ownership": True,
            "collective_pattern": "allgather_queries_and_nearest_candidates",
            "scalable_neighbor_routing": False,
        }


@dataclass(frozen=True)
class DistributedTriangleSearchOutcome:
    """Distributed projection paired with explicit communication evidence."""

    projection: SurfaceProjection
    diagnostics: DistributedTriangleSearchDiagnostics

    def __post_init__(self) -> None:
        if self.projection.point_count != self.diagnostics.local_query_count:
            raise ValueError("Distributed projection and diagnostics counts differ.")
        if (
            self.projection.geometry_fingerprint
            != self.diagnostics.geometry_fingerprint
        ):
            raise ValueError("Distributed projection and diagnostics geometry differ.")


@dataclass(frozen=True)
class RoutedTriangleSearchDiagnostics:
    """Sparse rank-routing evidence for one local query batch."""

    geometry_fingerprint: str
    rank: int
    rank_count: int
    local_query_count: int
    local_facet_count: int
    global_facet_count: int
    phase_one_query_messages: int
    phase_two_query_messages: int
    candidate_response_messages: int
    queried_rank_counts: tuple[int, ...]
    visited_node_counts: tuple[int, ...]
    evaluated_facet_counts: tuple[int, ...]
    estimated_query_payload_bytes: int
    estimated_candidate_payload_bytes: int
    method: str = "rank_aabb_two_stage_sparse_routing"

    def __post_init__(self) -> None:
        fingerprint = str(self.geometry_fingerprint).lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("Routed diagnostics require a SHA-256 identity.")
        rank = int(self.rank)
        rank_count = int(self.rank_count)
        local_query_count = int(self.local_query_count)
        if rank_count < 1 or not 0 <= rank < rank_count:
            raise ValueError("Routed diagnostics require a valid rank.")
        counts = {
            "local_query_count": local_query_count,
            "local_facet_count": int(self.local_facet_count),
            "global_facet_count": int(self.global_facet_count),
            "phase_one_query_messages": int(self.phase_one_query_messages),
            "phase_two_query_messages": int(self.phase_two_query_messages),
            "candidate_response_messages": int(self.candidate_response_messages),
            "estimated_query_payload_bytes": int(
                self.estimated_query_payload_bytes
            ),
            "estimated_candidate_payload_bytes": int(
                self.estimated_candidate_payload_bytes
            ),
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError("Routed diagnostic counts must be non-negative.")
        queried = tuple(int(value) for value in self.queried_rank_counts)
        visited = tuple(int(value) for value in self.visited_node_counts)
        evaluated = tuple(int(value) for value in self.evaluated_facet_counts)
        if not (
            len(queried)
            == len(visited)
            == len(evaluated)
            == local_query_count
        ):
            raise ValueError("Routed diagnostics require one record per local query.")
        if any(value < 1 or value > rank_count for value in queried):
            raise ValueError("Each routed query must visit one or more valid ranks.")
        if any(value < 0 for value in (*visited, *evaluated)):
            raise ValueError("Routed work counts must be non-negative.")
        if not str(self.method).strip():
            raise ValueError("Routed diagnostics require a method.")
        object.__setattr__(self, "geometry_fingerprint", fingerprint)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "rank_count", rank_count)
        for name, value in counts.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "queried_rank_counts", queried)
        object.__setattr__(self, "visited_node_counts", visited)
        object.__setattr__(self, "evaluated_facet_counts", evaluated)
        object.__setattr__(self, "method", str(self.method))

    @property
    def avoided_query_messages(self) -> int:
        return self.local_query_count * self.rank_count - (
            self.phase_one_query_messages + self.phase_two_query_messages
        )

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method,
            "geometry_fingerprint": self.geometry_fingerprint,
            "rank": self.rank,
            "rank_count": self.rank_count,
            "local_query_count": self.local_query_count,
            "local_facet_count": self.local_facet_count,
            "global_facet_count": self.global_facet_count,
            "phase_one_query_messages": self.phase_one_query_messages,
            "phase_two_query_messages": self.phase_two_query_messages,
            "candidate_response_messages": self.candidate_response_messages,
            "avoided_query_messages": self.avoided_query_messages,
            "maximum_queried_ranks": max(self.queried_rank_counts, default=0),
            "mean_queried_ranks": (
                float(np.mean(self.queried_rank_counts))
                if self.queried_rank_counts
                else 0.0
            ),
            "maximum_visited_nodes": max(self.visited_node_counts, default=0),
            "maximum_evaluated_facets": max(
                self.evaluated_facet_counts, default=0
            ),
            "estimated_query_payload_bytes": self.estimated_query_payload_bytes,
            "estimated_candidate_payload_bytes": (
                self.estimated_candidate_payload_bytes
            ),
            "distributed_ownership": True,
            "collective_pattern": "two_stage_sparse_object_alltoall",
            "packed_numeric_transport": False,
            "partition_independent_oracle": "allgather_reference_distributed_bvh",
        }


@dataclass(frozen=True)
class RoutedTriangleSearchOutcome:
    """Sparse-routed projection and its communication/work evidence."""

    projection: SurfaceProjection
    diagnostics: RoutedTriangleSearchDiagnostics

    def __post_init__(self) -> None:
        if self.projection.point_count != self.diagnostics.local_query_count:
            raise ValueError("Routed projection and diagnostics counts differ.")
        if (
            self.projection.geometry_fingerprint
            != self.diagnostics.geometry_fingerprint
        ):
            raise ValueError("Routed projection and diagnostics geometry differ.")


@dataclass(frozen=True)
class _TriangleCandidateBatch:
    closest_points: np.ndarray
    normals: np.ndarray
    squared_distances: np.ndarray
    has_candidate: np.ndarray
    ambiguous: np.ndarray
    entity_ids: np.ndarray
    local_coordinates: np.ndarray
    visited_node_counts: tuple[int, ...]
    evaluated_facet_counts: tuple[int, ...]

    @property
    def byte_count(self) -> int:
        return sum(
            int(value.nbytes)
            for value in (
                self.closest_points,
                self.normals,
                self.squared_distances,
                self.has_candidate,
                self.ambiguous,
                self.entity_ids,
                self.local_coordinates,
            )
        )


def _projection_arrays_from_candidates(
    query: np.ndarray,
    candidates: _TriangleCandidateBatch,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    count = int(query.shape[0])
    closest = np.full((count, 3), np.nan, dtype=float)
    normals = np.full((count, 3), np.nan, dtype=float)
    gaps = np.full(count, np.nan, dtype=float)
    valid = candidates.has_candidate & ~candidates.ambiguous
    statuses = np.full(count, "no_candidate", dtype="<U20")
    statuses[candidates.ambiguous] = "ambiguous_projection"
    statuses[valid] = "ok"
    entity_ids = np.full(count, -1, dtype=np.int64)
    local_coordinates = np.full((count, 3), np.nan, dtype=float)
    closest[valid] = candidates.closest_points[valid]
    normals[valid] = candidates.normals[valid]
    gaps[valid] = np.einsum(
        "ij,ij->i",
        query[valid] - candidates.closest_points[valid],
        candidates.normals[valid],
    )
    entity_ids[valid] = candidates.entity_ids[valid]
    local_coordinates[valid] = candidates.local_coordinates[valid]
    return (
        closest,
        normals,
        gaps,
        valid,
        statuses,
        entity_ids,
        local_coordinates,
    )


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

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return immutable lower/upper bounds of the complete local tree."""

        return self._lower[self._root], self._upper[self._root]

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

    def _candidate_evidence(
        self,
        query: np.ndarray,
        maximum_distance: float | None,
        *,
        ambiguity_tolerance: float | None = None,
        normal_scale: float | None = None,
    ) -> _TriangleCandidateBatch:
        count = int(query.shape[0])
        closest = np.full((count, 3), np.nan, dtype=float)
        normals = np.full((count, 3), np.nan, dtype=float)
        squared_distances = np.full(count, np.inf, dtype=float)
        has_candidate = np.zeros(count, dtype=bool)
        ambiguous = np.zeros(count, dtype=bool)
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
        ambiguity_tolerance = (
            float(self.surface.ambiguity_tolerance)
            if ambiguity_tolerance is None
            else _finite_nonnegative(
                ambiguity_tolerance,
                name="Projection ambiguity tolerance",
            )
        )
        scale = self.surface.scale if normal_scale is None else float(normal_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Projection normal scale must be finite and positive.")
        ambiguity_squared = ambiguity_tolerance**2
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
            selected_normal = self._facet_normals[selected_facet]
            has_candidate[point_index] = True
            closest[point_index] = selected_point
            normals[point_index] = selected_normal
            squared_distances[point_index] = minimum
            entity_ids[point_index] = int(facet_ids[selected_facet])
            local_coordinates[point_index] = _triangle_barycentric_coordinates(
                selected_point,
                self._triangle_points[selected_facet],
            )
            if len(tied) > 1:
                same_point = all(
                    np.linalg.norm(candidate - selected_point)
                    <= ambiguity_tolerance
                    for _, candidate, _ in tied
                )
                same_normal = all(
                    np.linalg.norm(self._facet_normals[facet] - selected_normal)
                    <= ambiguity_tolerance / scale
                    for facet, _, _ in tied
                )
                if not (same_point and same_normal):
                    ambiguous[point_index] = True

        return _TriangleCandidateBatch(
            closest_points=closest,
            normals=normals,
            squared_distances=squared_distances,
            has_candidate=has_candidate,
            ambiguous=ambiguous,
            entity_ids=entity_ids,
            local_coordinates=local_coordinates,
            visited_node_counts=tuple(visited_counts),
            evaluated_facet_counts=tuple(evaluated_counts),
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

        candidates = self._candidate_evidence(query_reference, maximum_distance)
        (
            closest,
            normals,
            gaps,
            valid,
            statuses,
            entity_ids,
            local_coordinates,
        ) = _projection_arrays_from_candidates(query_reference, candidates)
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
            visited_node_counts=candidates.visited_node_counts,
            evaluated_facet_counts=candidates.evaluated_facet_counts,
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


def partition_triangle_surface(
    surface: TriangulatedRigidSurface,
    comm,
    *,
    ownership_method: str = "spatial_centroid_contiguous",
) -> TriangleSurfacePartition:
    """Partition a replicated reviewed surface by stable facet identity.

    The routine is intentionally a correctness-reference adapter.  It proves
    that search results are independent of facet ownership while avoiding a
    dependency on one mesh importer.  Large imported tools should eventually
    construct :class:`TriangleSurfacePartition` without first replicating the
    complete geometry.
    """

    if not isinstance(surface, TriangulatedRigidSurface):
        raise TypeError("Triangle partitioning requires a TriangulatedRigidSurface.")
    rank = int(comm.rank)
    rank_count = int(comm.size)
    fingerprints = tuple(comm.allgather(surface.geometry_fingerprint))
    if len(set(fingerprints)) != 1:
        raise ValueError(
            "All ranks must partition the same reviewed triangle geometry."
        )
    facet_count = int(surface.triangles.shape[0])
    if facet_count < rank_count:
        raise ValueError(
            "Reference triangle partitioning requires at least one facet per rank."
        )
    facet_ids = np.asarray(surface.facet_ids, dtype=np.int64)
    ownership_method = str(ownership_method)
    if ownership_method == "spatial_centroid_contiguous":
        centroids = np.mean(surface.vertices[surface.triangles], axis=1)
        ordered = np.lexsort(
            (facet_ids, centroids[:, 2], centroids[:, 1], centroids[:, 0])
        )
        owned = np.array_split(ordered, rank_count)[rank]
    elif ownership_method == "stable_facet_id_round_robin":
        ordered = np.argsort(facet_ids, kind="stable")
        owned = ordered[np.arange(facet_count) % rank_count == rank]
    else:
        raise ValueError(
            "Triangle ownership method must be 'spatial_centroid_contiguous' "
            "or 'stable_facet_id_round_robin'."
        )
    owned = owned[np.argsort(facet_ids[owned], kind="stable")]
    global_triangles = np.asarray(surface.triangles[owned], dtype=np.int64)
    used_vertices = np.unique(global_triangles.reshape(-1))
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(used_vertices.tolist())
    }
    local_triangles = np.asarray(
        [
            [global_to_local[int(global_index)] for global_index in triangle]
            for triangle in global_triangles
        ],
        dtype=np.int64,
    )
    local_surface = TriangulatedRigidSurface(
        vertices=surface.vertices[used_vertices],
        triangles=local_triangles,
        facet_ids=facet_ids[owned],
        tolerance=surface.tolerance,
        ambiguity_tolerance=surface.ambiguity_tolerance,
        name=f"{surface.name}__rank_{rank}_partition",
    )
    return TriangleSurfacePartition(
        local_surface=local_surface,
        global_geometry_fingerprint=surface.geometry_fingerprint,
        global_facet_identity_fingerprint=_facet_identity_fingerprint(facet_ids),
        global_surface_name=surface.name,
        global_facet_count=facet_count,
        global_scale=surface.scale,
        global_ambiguity_tolerance=surface.ambiguity_tolerance,
        rank=rank,
        rank_count=rank_count,
        ownership_method=ownership_method,
    )


def _merge_distributed_candidates(
    batches: tuple[_TriangleCandidateBatch, ...],
    *,
    query: np.ndarray,
    ambiguity_tolerance: float,
    normal_scale: float,
) -> _TriangleCandidateBatch:
    count = int(query.shape[0])
    closest = np.full((count, 3), np.nan, dtype=float)
    normals = np.full((count, 3), np.nan, dtype=float)
    squared_distances = np.full(count, np.inf, dtype=float)
    has_candidate = np.zeros(count, dtype=bool)
    ambiguous = np.zeros(count, dtype=bool)
    entity_ids = np.full(count, -1, dtype=np.int64)
    local_coordinates = np.full((count, 3), np.nan, dtype=float)
    ambiguity_squared = float(ambiguity_tolerance) ** 2
    normal_tolerance = float(ambiguity_tolerance) / float(normal_scale)

    for point_index in range(count):
        participating = [
            batch
            for batch in batches
            if bool(batch.has_candidate[point_index])
        ]
        if not participating:
            continue
        minimum = min(
            float(batch.squared_distances[point_index])
            for batch in participating
        )
        tied = [
            batch
            for batch in participating
            if float(batch.squared_distances[point_index])
            <= np.nextafter(minimum + ambiguity_squared, np.inf)
        ]
        selected = min(tied, key=lambda batch: int(batch.entity_ids[point_index]))
        has_candidate[point_index] = True
        closest[point_index] = selected.closest_points[point_index]
        normals[point_index] = selected.normals[point_index]
        squared_distances[point_index] = minimum
        entity_ids[point_index] = selected.entity_ids[point_index]
        local_coordinates[point_index] = selected.local_coordinates[point_index]
        if any(bool(batch.ambiguous[point_index]) for batch in tied):
            ambiguous[point_index] = True
            continue
        selected_point = selected.closest_points[point_index]
        selected_normal = selected.normals[point_index]
        same_point = all(
            np.linalg.norm(batch.closest_points[point_index] - selected_point)
            <= ambiguity_tolerance
            for batch in tied
        )
        same_normal = all(
            np.linalg.norm(batch.normals[point_index] - selected_normal)
            <= normal_tolerance
            for batch in tied
        )
        ambiguous[point_index] = not (same_point and same_normal)

    return _TriangleCandidateBatch(
        closest_points=closest,
        normals=normals,
        squared_distances=squared_distances,
        has_candidate=has_candidate,
        ambiguous=ambiguous,
        entity_ids=entity_ids,
        local_coordinates=local_coordinates,
        visited_node_counts=tuple(
            sum(batch.visited_node_counts[index] for batch in batches)
            for index in range(count)
        ),
        evaluated_facet_counts=tuple(
            sum(batch.evaluated_facet_counts[index] for batch in batches)
            for index in range(count)
        ),
    )


def _exchange_routed_queries(
    *,
    comm,
    local_search: TriangleSurfaceBVH,
    outgoing: list[list[tuple[int, int, float, float, float]]],
    maximum_distance: float | None,
    ambiguity_tolerance: float,
    normal_scale: float,
) -> list[tuple[object, ...]]:
    received_groups = comm.alltoall(outgoing)
    received = [record for group in received_groups for record in group]
    points = (
        np.asarray([record[2:5] for record in received], dtype=float)
        if received
        else np.empty((0, 3), dtype=float)
    )
    candidates = local_search._candidate_evidence(
        points,
        maximum_distance,
        ambiguity_tolerance=ambiguity_tolerance,
        normal_scale=normal_scale,
    )
    responses: list[list[tuple[object, ...]]] = [
        [] for _ in range(int(comm.size))
    ]
    for record_index, request in enumerate(received):
        origin_rank, origin_index = int(request[0]), int(request[1])
        responses[origin_rank].append(
            (
                origin_index,
                int(comm.rank),
                bool(candidates.has_candidate[record_index]),
                bool(candidates.ambiguous[record_index]),
                float(candidates.squared_distances[record_index]),
                int(candidates.entity_ids[record_index]),
                tuple(float(value) for value in candidates.closest_points[record_index]),
                tuple(float(value) for value in candidates.normals[record_index]),
                tuple(
                    float(value) for value in candidates.local_coordinates[record_index]
                ),
                int(candidates.visited_node_counts[record_index]),
                int(candidates.evaluated_facet_counts[record_index]),
            )
        )
    returned_groups = comm.alltoall(responses)
    return [record for group in returned_groups for record in group]


def _merge_routed_candidate_records(
    records: list[tuple[object, ...]],
    *,
    query_count: int,
    ambiguity_tolerance: float,
    normal_scale: float,
) -> _TriangleCandidateBatch:
    grouped: list[list[tuple[object, ...]]] = [[] for _ in range(query_count)]
    for record in records:
        index = int(record[0])
        if index < 0 or index >= query_count:
            raise ValueError("Routed candidate references an invalid query index.")
        grouped[index].append(record)

    closest = np.full((query_count, 3), np.nan, dtype=float)
    normals = np.full((query_count, 3), np.nan, dtype=float)
    squared_distances = np.full(query_count, np.inf, dtype=float)
    has_candidate = np.zeros(query_count, dtype=bool)
    ambiguous = np.zeros(query_count, dtype=bool)
    entity_ids = np.full(query_count, -1, dtype=np.int64)
    local_coordinates = np.full((query_count, 3), np.nan, dtype=float)
    visited: list[int] = []
    evaluated: list[int] = []
    ambiguity_squared = float(ambiguity_tolerance) ** 2
    normal_tolerance = float(ambiguity_tolerance) / float(normal_scale)

    for point_index, point_records in enumerate(grouped):
        if not point_records:
            raise ValueError("Every routed query requires at least one response.")
        if len({int(record[1]) for record in point_records}) != len(point_records):
            raise ValueError("A routed query received duplicate rank responses.")
        visited.append(sum(int(record[9]) for record in point_records))
        evaluated.append(sum(int(record[10]) for record in point_records))
        participating = [record for record in point_records if bool(record[2])]
        if not participating:
            continue
        minimum = min(float(record[4]) for record in participating)
        tied = [
            record
            for record in participating
            if float(record[4])
            <= np.nextafter(minimum + ambiguity_squared, np.inf)
        ]
        selected = min(tied, key=lambda record: int(record[5]))
        selected_point = np.asarray(selected[6], dtype=float)
        selected_normal = np.asarray(selected[7], dtype=float)
        has_candidate[point_index] = True
        closest[point_index] = selected_point
        normals[point_index] = selected_normal
        squared_distances[point_index] = minimum
        entity_ids[point_index] = int(selected[5])
        local_coordinates[point_index] = np.asarray(selected[8], dtype=float)
        if any(bool(record[3]) for record in tied):
            ambiguous[point_index] = True
            continue
        same_point = all(
            np.linalg.norm(np.asarray(record[6], dtype=float) - selected_point)
            <= ambiguity_tolerance
            for record in tied
        )
        same_normal = all(
            np.linalg.norm(np.asarray(record[7], dtype=float) - selected_normal)
            <= normal_tolerance
            for record in tied
        )
        ambiguous[point_index] = not (same_point and same_normal)

    return _TriangleCandidateBatch(
        closest_points=closest,
        normals=normals,
        squared_distances=squared_distances,
        has_candidate=has_candidate,
        ambiguous=ambiguous,
        entity_ids=entity_ids,
        local_coordinates=local_coordinates,
        visited_node_counts=tuple(visited),
        evaluated_facet_counts=tuple(evaluated),
    )


class DistributedTriangleSurfaceBVH:
    """Correctness-first collective search over partitioned triangle facets.

    Queries and one nearest-candidate record per rank are currently exchanged
    with ``allgather``.  This removes global facet replication from the search
    object and establishes partition-independent physics, but deliberately
    does not claim scalable neighbor routing.
    """

    def __init__(self, partition: TriangleSurfacePartition, comm):
        if not isinstance(partition, TriangleSurfacePartition):
            raise TypeError(
                "Distributed triangle search requires TriangleSurfacePartition."
            )
        if int(comm.rank) != partition.rank or int(comm.size) != partition.rank_count:
            raise ValueError("Triangle partition does not match the communicator.")
        records = tuple(
            comm.allgather(
                (
                    partition.global_geometry_fingerprint,
                    partition.global_facet_count,
                    partition.global_facet_identity_fingerprint,
                    tuple(int(value) for value in partition.local_surface.facet_ids),
                )
            )
        )
        if len({record[0] for record in records}) != 1 or len(
            {record[1] for record in records}
        ) != 1:
            raise ValueError("Distributed triangle partitions disagree globally.")
        if len({record[2] for record in records}) != 1:
            raise ValueError("Distributed triangle facet registries disagree.")
        owned_ids = tuple(value for record in records for value in record[3])
        if (
            len(owned_ids) != partition.global_facet_count
            or len(set(owned_ids)) != len(owned_ids)
            or _facet_identity_fingerprint(owned_ids)
            != partition.global_facet_identity_fingerprint
        ):
            raise ValueError(
                "Distributed triangle facet ownership must be complete and unique."
            )
        self._partition = partition
        self._comm = comm
        self._local_search = TriangleSurfaceBVH(partition.local_surface)

    @property
    def partition(self) -> TriangleSurfacePartition:
        return self._partition

    def summary(self) -> dict[str, object]:
        return {
            "kind": "distributed_triangle_surface_bvh",
            "method": "allgather_reference_distributed_bvh",
            "global_geometry_fingerprint": (
                self.partition.global_geometry_fingerprint
            ),
            "global_facet_count": self.partition.global_facet_count,
            "local_facet_count": int(
                self.partition.local_surface.triangles.shape[0]
            ),
            "rank": self.partition.rank,
            "rank_count": self.partition.rank_count,
            "distributed_ownership": True,
            "retains_replicated_global_geometry": False,
            "collective_pattern": "allgather_queries_and_nearest_candidates",
            "scalable_neighbor_routing": False,
        }

    def project_with_diagnostics(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> DistributedTriangleSearchOutcome:
        error = None
        try:
            query = _projection_query(
                points,
                dimension=3,
                geometry="Distributed triangle-surface BVH",
            )
            selected_factor = float(factor)
            if not np.isfinite(selected_factor):
                raise ValueError("Rigid-motion factor must be finite.")
            selected_maximum = (
                None
                if maximum_distance is None
                else _finite_nonnegative(
                    maximum_distance,
                    name="Maximum projection distance",
                )
            )
            if motion is not None and not isinstance(motion, PrescribedRigidMotion):
                raise TypeError("Rigid motion must be PrescribedRigidMotion.")
            motion_identity = (
                None
                if motion is None
                else (
                    tuple(motion.translation),
                    tuple(motion.rotation),
                    tuple(motion.reference_point),
                    motion.name,
                )
            )
        except (TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            query = np.empty((0, 3), dtype=float)
            selected_factor = 0.0
            selected_maximum = None
            motion_identity = None
        errors = tuple(self._comm.allgather(error))
        if any(value is not None for value in errors):
            raise ValueError(
                "Distributed triangle query validation failed collectively: "
                + "; ".join(
                    f"rank {rank}: {value}"
                    for rank, value in enumerate(errors)
                    if value is not None
                )
            )
        configs = tuple(
            self._comm.allgather(
                (selected_factor, selected_maximum, motion_identity)
            )
        )
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError(
                "Distributed triangle search requires identical motion and "
                "distance configuration on every rank."
            )

        if motion is None:
            query_reference = query
            rotation = np.eye(3, dtype=float)
            reference = np.zeros(3, dtype=float)
            translated_reference = reference
        else:
            if motion.dimension != 3:
                raise ValueError("Triangulated surface motion must be three-dimensional.")
            state = motion.state(selected_factor)
            rotation = np.asarray(state["rotation_matrix"], dtype=float)
            reference = np.asarray(motion.reference_point, dtype=float)
            translated_reference = np.asarray(state["reference_point"], dtype=float)
            query_reference = reference + (query - translated_reference) @ rotation

        gathered_queries = tuple(self._comm.allgather(query_reference))
        query_counts = tuple(int(value.shape[0]) for value in gathered_queries)
        global_query = (
            np.vstack(gathered_queries)
            if sum(query_counts) > 0
            else np.empty((0, 3), dtype=float)
        )
        local_candidates = self._local_search._candidate_evidence(
            global_query,
            selected_maximum,
            ambiguity_tolerance=self.partition.global_ambiguity_tolerance,
            normal_scale=self.partition.global_scale,
        )
        gathered_candidates = tuple(self._comm.allgather(local_candidates))
        merged = _merge_distributed_candidates(
            gathered_candidates,
            query=global_query,
            ambiguity_tolerance=self.partition.global_ambiguity_tolerance,
            normal_scale=self.partition.global_scale,
        )
        start = sum(query_counts[: self.partition.rank])
        stop = start + query_counts[self.partition.rank]
        local_candidates_merged = _TriangleCandidateBatch(
            closest_points=merged.closest_points[start:stop],
            normals=merged.normals[start:stop],
            squared_distances=merged.squared_distances[start:stop],
            has_candidate=merged.has_candidate[start:stop],
            ambiguous=merged.ambiguous[start:stop],
            entity_ids=merged.entity_ids[start:stop],
            local_coordinates=merged.local_coordinates[start:stop],
            visited_node_counts=merged.visited_node_counts[start:stop],
            evaluated_facet_counts=merged.evaluated_facet_counts[start:stop],
        )
        (
            closest,
            normals,
            gaps,
            valid,
            statuses,
            entity_ids,
            local_coordinates,
        ) = _projection_arrays_from_candidates(
            query_reference,
            local_candidates_merged,
        )
        if motion is not None:
            closest[valid] = (
                translated_reference + (closest[valid] - reference) @ rotation.T
            )
            normals[valid] = normals[valid] @ rotation.T
        projection = SurfaceProjection(
            surface_name=self.partition.global_surface_name,
            surface_kind="triangulated_rigid_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=valid,
            status_codes=statuses,
            method="distributed_aabb_bvh_exact_triangle_projection",
            entity_ids=entity_ids,
            geometry_fingerprint=self.partition.global_geometry_fingerprint,
            local_coordinates=local_coordinates,
            local_coordinate_system="triangle_barycentric_connectivity_order",
        )
        diagnostics = DistributedTriangleSearchDiagnostics(
            geometry_fingerprint=self.partition.global_geometry_fingerprint,
            rank=self.partition.rank,
            rank_count=self.partition.rank_count,
            local_query_count=int(query.shape[0]),
            global_query_count=int(global_query.shape[0]),
            local_facet_count=int(
                self.partition.local_surface.triangles.shape[0]
            ),
            global_facet_count=self.partition.global_facet_count,
            gathered_query_bytes=sum(int(value.nbytes) for value in gathered_queries),
            gathered_candidate_bytes=sum(
                value.byte_count for value in gathered_candidates
            ),
            visited_node_counts=local_candidates_merged.visited_node_counts,
            evaluated_facet_counts=local_candidates_merged.evaluated_facet_counts,
        )
        return DistributedTriangleSearchOutcome(projection, diagnostics)

    def project(self, points, **kwargs) -> SurfaceProjection:
        """Project rank-local points through the collective reference search."""

        return self.project_with_diagnostics(points, **kwargs).projection


class RoutedDistributedTriangleSurfaceBVH:
    """Two-stage sparse rank-AABB routing over partitioned triangle facets.

    The first phase asks the rank with the smallest bounding-box lower bound
    for an exact candidate.  That distance becomes an upper bound; the second
    phase asks only ranks whose boxes can still tie or improve it.  Exact
    candidates are reduced with the same ambiguity semantics as the all-gather
    oracle.  Transport currently uses Python-object ``alltoall`` and is
    reported as such; packed numeric ``Alltoallv`` remains a later optimization.
    """

    def __init__(self, partition: TriangleSurfacePartition, comm):
        self._reference = DistributedTriangleSurfaceBVH(partition, comm)
        self._partition = partition
        self._comm = comm
        self._local_search = self._reference._local_search
        local_lower, local_upper = self._local_search.bounds
        gathered_bounds = tuple(
            comm.allgather(
                (
                    tuple(float(value) for value in local_lower),
                    tuple(float(value) for value in local_upper),
                )
            )
        )
        self._rank_lowers = _readonly_array(
            [value[0] for value in gathered_bounds]
        )
        self._rank_uppers = _readonly_array(
            [value[1] for value in gathered_bounds]
        )

    @property
    def partition(self) -> TriangleSurfacePartition:
        return self._partition

    @property
    def correctness_oracle(self) -> DistributedTriangleSurfaceBVH:
        """Return the all-gather reference using the identical local shard."""

        return self._reference

    def summary(self) -> dict[str, object]:
        return {
            "kind": "routed_distributed_triangle_surface_bvh",
            "method": "rank_aabb_two_stage_sparse_routing",
            "global_geometry_fingerprint": (
                self.partition.global_geometry_fingerprint
            ),
            "global_facet_count": self.partition.global_facet_count,
            "local_facet_count": int(
                self.partition.local_surface.triangles.shape[0]
            ),
            "rank": self.partition.rank,
            "rank_count": self.partition.rank_count,
            "distributed_ownership": True,
            "retains_replicated_global_geometry": False,
            "collective_pattern": "two_stage_sparse_object_alltoall",
            "packed_numeric_transport": False,
            "exact_narrow_phase": "reviewed_triangle_closest_point",
            "partition_independent_oracle": (
                "allgather_reference_distributed_bvh"
            ),
        }

    def project_with_diagnostics(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> RoutedTriangleSearchOutcome:
        error = None
        try:
            query = _projection_query(
                points,
                dimension=3,
                geometry="Routed distributed triangle-surface BVH",
            )
            selected_factor = float(factor)
            if not np.isfinite(selected_factor):
                raise ValueError("Rigid-motion factor must be finite.")
            selected_maximum = (
                None
                if maximum_distance is None
                else _finite_nonnegative(
                    maximum_distance,
                    name="Maximum projection distance",
                )
            )
            if motion is not None and not isinstance(motion, PrescribedRigidMotion):
                raise TypeError("Rigid motion must be PrescribedRigidMotion.")
            motion_identity = (
                None
                if motion is None
                else (
                    tuple(motion.translation),
                    tuple(motion.rotation),
                    tuple(motion.reference_point),
                    motion.name,
                )
            )
        except (TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            query = np.empty((0, 3), dtype=float)
            selected_factor = 0.0
            selected_maximum = None
            motion_identity = None
        errors = tuple(self._comm.allgather(error))
        if any(value is not None for value in errors):
            raise ValueError(
                "Routed triangle query validation failed collectively: "
                + "; ".join(
                    f"rank {rank}: {value}"
                    for rank, value in enumerate(errors)
                    if value is not None
                )
            )
        configs = tuple(
            self._comm.allgather(
                (selected_factor, selected_maximum, motion_identity)
            )
        )
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError(
                "Routed triangle search requires identical motion and distance "
                "configuration on every rank."
            )

        if motion is None:
            query_reference = query
            rotation = np.eye(3, dtype=float)
            reference = np.zeros(3, dtype=float)
            translated_reference = reference
        else:
            if motion.dimension != 3:
                raise ValueError("Triangulated surface motion must be three-dimensional.")
            state = motion.state(selected_factor)
            rotation = np.asarray(state["rotation_matrix"], dtype=float)
            reference = np.asarray(motion.reference_point, dtype=float)
            translated_reference = np.asarray(state["reference_point"], dtype=float)
            query_reference = reference + (query - translated_reference) @ rotation

        query_count = int(query_reference.shape[0])
        rank_count = self.partition.rank_count
        first_ranks = np.empty(query_count, dtype=np.int64)
        lower_bounds = np.empty((query_count, rank_count), dtype=float)
        phase_one_outgoing: list[list[tuple[int, int, float, float, float]]] = [
            [] for _ in range(rank_count)
        ]
        for point_index, point in enumerate(query_reference):
            lower_bounds[point_index] = [
                _bbox_squared_distance(point, lower, upper)
                for lower, upper in zip(self._rank_lowers, self._rank_uppers)
            ]
            first_rank = int(np.argmin(lower_bounds[point_index]))
            first_ranks[point_index] = first_rank
            phase_one_outgoing[first_rank].append(
                (
                    self.partition.rank,
                    point_index,
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                )
            )
        phase_one_records = _exchange_routed_queries(
            comm=self._comm,
            local_search=self._local_search,
            outgoing=phase_one_outgoing,
            maximum_distance=selected_maximum,
            ambiguity_tolerance=self.partition.global_ambiguity_tolerance,
            normal_scale=self.partition.global_scale,
        )
        if len(phase_one_records) != query_count:
            raise ValueError("Routed phase one must return one candidate per query.")
        phase_one_by_index = {int(record[0]): record for record in phase_one_records}
        if len(phase_one_by_index) != query_count:
            raise ValueError("Routed phase one returned duplicate query evidence.")
        maximum_squared = (
            float("inf")
            if selected_maximum is None
            else float(selected_maximum) ** 2
        )
        ambiguity_squared = self.partition.global_ambiguity_tolerance**2
        phase_two_outgoing: list[list[tuple[int, int, float, float, float]]] = [
            [] for _ in range(rank_count)
        ]
        for point_index, point in enumerate(query_reference):
            initial = phase_one_by_index[point_index]
            upper_bound = (
                float(initial[4]) if bool(initial[2]) else maximum_squared
            )
            threshold = np.nextafter(upper_bound + ambiguity_squared, np.inf)
            for destination in range(rank_count):
                if destination == int(first_ranks[point_index]):
                    continue
                if lower_bounds[point_index, destination] <= threshold:
                    phase_two_outgoing[destination].append(
                        (
                            self.partition.rank,
                            point_index,
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                        )
                    )
        phase_two_records = _exchange_routed_queries(
            comm=self._comm,
            local_search=self._local_search,
            outgoing=phase_two_outgoing,
            maximum_distance=selected_maximum,
            ambiguity_tolerance=self.partition.global_ambiguity_tolerance,
            normal_scale=self.partition.global_scale,
        )
        records = phase_one_records + phase_two_records
        merged = _merge_routed_candidate_records(
            records,
            query_count=query_count,
            ambiguity_tolerance=self.partition.global_ambiguity_tolerance,
            normal_scale=self.partition.global_scale,
        )
        (
            closest,
            normals,
            gaps,
            valid,
            statuses,
            entity_ids,
            local_coordinates,
        ) = _projection_arrays_from_candidates(query_reference, merged)
        if motion is not None:
            closest[valid] = (
                translated_reference + (closest[valid] - reference) @ rotation.T
            )
            normals[valid] = normals[valid] @ rotation.T
        projection = SurfaceProjection(
            surface_name=self.partition.global_surface_name,
            surface_kind="triangulated_rigid_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=valid,
            status_codes=statuses,
            method="routed_distributed_aabb_bvh_exact_triangle_projection",
            entity_ids=entity_ids,
            geometry_fingerprint=self.partition.global_geometry_fingerprint,
            local_coordinates=local_coordinates,
            local_coordinate_system="triangle_barycentric_connectivity_order",
        )
        queried_rank_counts = tuple(
            1
            + sum(
                int(record[0]) == point_index for record in phase_two_records
            )
            for point_index in range(query_count)
        )
        query_messages = query_count + len(phase_two_records)
        diagnostics = RoutedTriangleSearchDiagnostics(
            geometry_fingerprint=self.partition.global_geometry_fingerprint,
            rank=self.partition.rank,
            rank_count=rank_count,
            local_query_count=query_count,
            local_facet_count=int(
                self.partition.local_surface.triangles.shape[0]
            ),
            global_facet_count=self.partition.global_facet_count,
            phase_one_query_messages=query_count,
            phase_two_query_messages=len(phase_two_records),
            candidate_response_messages=len(records),
            queried_rank_counts=queried_rank_counts,
            visited_node_counts=merged.visited_node_counts,
            evaluated_facet_counts=merged.evaluated_facet_counts,
            estimated_query_payload_bytes=query_messages * 40,
            estimated_candidate_payload_bytes=len(records) * 122,
        )
        return RoutedTriangleSearchOutcome(projection, diagnostics)

    def project(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        """Project rank-local points through sparse two-stage routing."""

        return self.project_with_diagnostics(
            points,
            motion=motion,
            factor=factor,
            maximum_distance=maximum_distance,
        ).projection


def distributed_triangle_surface_bvh(
    partition: TriangleSurfacePartition,
    comm,
) -> DistributedTriangleSurfaceBVH:
    """Build the correctness-reference collective search for one partition."""

    return DistributedTriangleSurfaceBVH(partition, comm)


def routed_distributed_triangle_surface_bvh(
    partition: TriangleSurfacePartition,
    comm,
) -> RoutedDistributedTriangleSurfaceBVH:
    """Build sparse two-stage rank-AABB routing for one partition."""

    return RoutedDistributedTriangleSurfaceBVH(partition, comm)


def triangle_surface_bvh(surface: TriangulatedRigidSurface) -> TriangleSurfaceBVH:
    """Build a deterministic process-local BVH for one reviewed surface."""

    return TriangleSurfaceBVH(surface)


__all__ = [
    "DistributedTriangleSearchDiagnostics",
    "DistributedTriangleSearchOutcome",
    "DistributedTriangleSurfaceBVH",
    "RoutedDistributedTriangleSurfaceBVH",
    "RoutedTriangleSearchDiagnostics",
    "RoutedTriangleSearchOutcome",
    "TriangleSearchDiagnostics",
    "TriangleSearchOutcome",
    "TriangleSurfacePartition",
    "TriangleSurfaceBVH",
    "distributed_triangle_surface_bvh",
    "partition_triangle_surface",
    "routed_distributed_triangle_surface_bvh",
    "triangle_surface_bvh",
]
