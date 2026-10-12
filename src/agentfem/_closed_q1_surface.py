# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private closed-solid Q1 signed distance with independently checked rays.

The geometry owner must supply an outward oriented non-self-intersecting solid
boundary. Closure/orientation and queried winding are checked here; arbitrary
self-intersection detection is not claimed. Actual Q1 closest coordinates and
ray intersections are retained, including warped facets.
"""

import numpy as np

from ._q1_ray import AmbiguousRay, q1_ray_crossings
from ._q1_surface_search import Q1SurfaceSearch


class ClosedQ1Surface:
    def __init__(self, vertices, quadrilaterals, *, execution=None):
        self.execution = execution
        if np.iscomplexobj(vertices):
            raise ValueError("Closed Q1 geometry must be real.")
        x, cells = np.asarray(vertices, dtype=float), np.asarray(quadrilaterals)
        if (
            x.ndim != 2
            or x.shape[1:] != (3,)
            or not np.isfinite(x).all()
            or cells.ndim != 2
            or cells.shape[1:] != (4,)
            or cells.dtype.kind not in "iu"
            or not len(cells)
            or np.any(cells < 0)
            or np.any(cells >= len(x))
        ):
            raise ValueError(
                "Closed Q1 surface requires finite vertices and connectivity."
            )
        edges = {}
        for face, row in enumerate(cells):
            for a, b in zip(row, np.roll(row, -1), strict=True):
                key = tuple(sorted((int(a), int(b))))
                edges.setdefault(key, []).append((face, int(a) < int(b)))
        if any(len(v) != 2 or v[0][1] == v[1][1] for v in edges.values()):
            raise ValueError(
                "Closed Q1 solid requires two oppositely oriented faces per edge."
            )
        # Edge manifoldness alone admits two shells pinched at one vertex.
        incident = {}
        for (a, b), uses in edges.items():
            for node in (a, b):
                incident.setdefault(node, []).append((uses[0][0], uses[1][0]))
        for pairs in incident.values():
            adjacency = {}
            for a, b in pairs:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
            seen, stack = set(), [next(iter(adjacency))]
            while stack:
                face = stack.pop()
                if face not in seen:
                    seen.add(face)
                    stack.extend(adjacency[face] - seen)
            if len(seen) != len(adjacency):
                raise ValueError("Closed Q1 solid has a nonmanifold vertex.")
        self.search = Q1SurfaceSearch(x[cells], node_ids=cells, execution=execution)
        self.lower, self.upper = x.min(axis=0), x.max(axis=0)
        self.scale = float(np.linalg.norm(self.upper - self.lower))
        self._vertex_shape = x.shape

    def refitted(self, vertices):
        """Update geometry without rechecking unchanged closed-shell topology."""
        raw = np.asarray(vertices)
        if (np.iscomplexobj(raw) or raw.shape != self._vertex_shape
                or not np.isfinite(raw).all()):
            raise ValueError("Closed Q1 refit requires unchanged finite real vertices.")
        x = np.asarray(raw, dtype=float)
        result = object.__new__(ClosedQ1Surface)
        result.execution = self.execution
        result.search = Q1SurfaceSearch(
            x[self.search.node_ids], node_ids=self.search.node_ids, _previous=self.search,
            execution=self.execution)
        result.lower, result.upper = x.min(axis=0), x.max(axis=0)
        result.scale = float(np.linalg.norm(result.upper - result.lower))
        result._vertex_shape = x.shape
        return result

    def _inside(self, point):
        # Fixed directions make restart deterministic. Two nonsingular rays
        # must agree; edge hits do not count as evidence in either direction.
        directions = np.array(
            [
                [1.0, 0.371, 0.193],
                [0.217, 1.0, 0.419],
                [0.613, 0.293, 1.0],
                [-1.0, 0.523, 0.317],
                [0.731, -1.0, 0.239],
                [0.419, 0.617, -1.0],
                [-0.317, -0.719, 1.0],
                [1.0, -0.233, -0.871],
            ]
        )
        answers = []
        length = 2 * (
            self.scale + np.linalg.norm(point - (self.lower + self.upper) / 2)
        )
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            end = point + length * direction
            pad = self.search.padding + 128 * np.finfo(float).eps * self.scale
            facets = np.unique(
                self.search.tree.segment_overlapping_facets(point, end, padding=pad)
                // 2
            )
            try:
                hits = q1_ray_crossings(self.search.vertices[facets], point, direction)
            except AmbiguousRay:
                from ._q1_ray_exact import exact_q1_ray_crossings

                try:
                    hits = exact_q1_ray_crossings(
                        self.search.vertices[facets], point, direction
                    )
                except AmbiguousRay:
                    continue
            winding = int(hits[:, 4].sum())
            if winding not in (0, 1):
                raise ValueError(
                    "Closed Q1 winding is inconsistent with an outward solid."
                )
            answers.append(winding)
            if len(answers) == 2:
                if answers[0] != answers[1]:
                    raise ValueError("Independent Q1 ray classifications disagree.")
                return bool(winding)
        raise ValueError("No two unambiguous Q1 rays classify this query.")

    def zero_gap_resolution(self, points):
        """Existing binary64 distance resolution, not a penetration allowance."""
        return 128 * np.finfo(float).eps * max(
            self.scale, float(np.max(np.abs(points), initial=0))
        )

    def _inside_batch(self, points):
        """Reference batch seam; every point retains the two-ray contract."""
        if self.execution is not None:
            return self.execution.inside(self, points, self._inside_batch_reference)
        return self._inside_batch_reference(points)

    def _inside_batch_reference(self, points):
        return np.fromiter((self._inside(p) for p in points), dtype=bool,
                           count=len(points))

    def project(self, points, *, maximum_distance):
        found = self.search.project(
            points, maximum_distance=maximum_distance, require_unique_normal=False
        )
        query = np.asarray(points, dtype=float)
        delta = query - found["points"]
        distance = np.linalg.norm(delta, axis=1)
        tolerance = self.zero_gap_resolution(query)
        gaps = np.zeros(len(query))
        normal = found["normals"].copy()
        resolved = np.flatnonzero(distance > tolerance)
        if len(resolved):
            inside = self._inside_batch(query[resolved])
            gaps[resolved] = np.where(inside, -distance[resolved], distance[resolved])
            # Signed-distance gradients, not averaged facet normals: pair
            # forces remain central even at edges/corners. Zero-gap points
            # are not classified by artificial rays launched on the surface.
            normal[resolved] = delta[resolved] / gaps[resolved, None]
        # Numerical closest distance controls refinement effort only. Returned
        # admission evidence comes exclusively from convex Q1 enclosures.
        # Only exterior points consume a positive separation certificate.
        # Interior/on-surface points retain the signed gap below; recursively
        # refining their enclosure (often toward zero) cannot change that result.
        exterior = gaps > 0
        lower = np.zeros(len(query))
        if np.any(exterior):
            lower[exterior] = self.search.distance_lower_bounds(
                query[exterior], refinement_distance=distance[exterior] / 4
            )
        return {
            **found,
            "normals": normal,
            "signed_gaps": gaps,
            "separation_lower_bounds": np.where(gaps > 0, lower, gaps),
            "sign_method": "two_true_q1_oriented_rays",
        }
