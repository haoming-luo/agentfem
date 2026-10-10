# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded process-local Q1 candidate search; no contact-law admission."""

import numpy as np

from ._q1_surface_geometry import q1_closest_point, q1_triangulation_distance_bound
from .boundary_models.rigid import TriangulatedRigidSurface
from .boundary_models.search import TriangleSurfaceBVH


class Q1SurfaceSearch:
    """Reuse triangular BVH broad phase, refine on the original Q1 geometry.

    Triangle bounds are expanded by the Q1 surrogate distance bound. Query
    radius is a caller-specified physical search radius, not a bond tolerance.
    Returns one facet's original four-node weights. A missing/ambiguous query
    rejects the batch; shared seams with identical points/normals choose the
    lower input facet index. This scalar narrow-phase reference is not MPI or
    a performance-qualified production search.
    """

    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float, copy=True)
        bounds = q1_triangulation_distance_bound(vertices)
        if not len(bounds):
            raise ValueError("Q1 search requires a nonempty surface.")
        self.padding = float(bounds.max())
        self.vertices.flags.writeable = False
        ids = np.arange(len(bounds) * 4).reshape(-1, 4)
        triangles = np.stack((ids[:, [0, 1, 2]], ids[:, [0, 2, 3]]), axis=1).reshape(
            -1, 3
        )
        nodes, inverse = np.unique(
            self.vertices.reshape(-1, 3), axis=0, return_inverse=True
        )
        self.tree = TriangleSurfaceBVH(
            TriangulatedRigidSurface(nodes, inverse[triangles])
        )

    def project(self, points, *, maximum_distance, maximum_candidates=128):
        raw = np.asarray(points)
        if np.iscomplexobj(raw):
            raise ValueError("Q1 queries must be real.")
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Q1 queries must be finite (queries,3) arrays.")
        if not np.isfinite(maximum_distance) or maximum_distance <= 0:
            raise ValueError("maximum_distance must be positive and finite.")
        if (
            isinstance(maximum_candidates, bool)
            or not isinstance(maximum_candidates, int)
            or maximum_candidates < 1
        ):
            raise ValueError("maximum_candidates must be a positive integer.")
        found, counts = [], []
        for index, point in enumerate(points):
            radius = maximum_distance + self.padding
            facets = np.unique(
                self.tree.overlapping_facets(point - radius, point + radius) // 2
            )
            if len(facets) > maximum_candidates:
                raise ValueError(f"Q1 query {index}: candidate budget exceeded.")
            counts.append(len(facets))
            choices = [
                (int(f), q1_closest_point(self.vertices[f], point)) for f in facets
            ]
            choices = [(f, r) for f, r in choices if r[-1] <= maximum_distance]
            if not choices:
                raise ValueError(f"Q1 query {index}: no surface within search radius.")
            choices.sort(key=lambda item: (item[1][-1], item[0]))
            best = choices[0]
            tolerance = 1e-10 * maximum_distance
            for candidate in choices[1:]:
                if abs(candidate[1][-1] - best[1][-1]) <= tolerance:
                    if (
                        np.linalg.norm(candidate[1][2] - best[1][2]) > tolerance
                        or np.linalg.norm(candidate[1][3] - best[1][3]) > 1e-8
                    ):
                        raise ValueError(
                            f"Q1 query {index}: ambiguous surface projection."
                        )
                    if candidate[0] < best[0]:
                        best = candidate
            found.append(best)
        return {
            "facet_indices": np.array([f for f, _ in found], dtype=np.int64),
            "weights": np.array([r[1] for _, r in found]).reshape(-1, 4),
            "points": np.array([r[2] for _, r in found]).reshape(-1, 3),
            "normals": np.array([r[3] for _, r in found]).reshape(-1, 3),
            "distances": np.array([r[4] for _, r in found]),
            "candidate_counts": np.array(counts, dtype=np.int64),
        }
