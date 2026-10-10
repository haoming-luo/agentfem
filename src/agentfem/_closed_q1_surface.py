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
    def __init__(self, vertices, quadrilaterals):
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
        self.search = Q1SurfaceSearch(x[cells], node_ids=cells)
        self.lower, self.upper = x.min(axis=0), x.max(axis=0)
        self.scale = float(np.linalg.norm(self.upper - self.lower))

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
                self.search.tree.overlapping_facets(
                    np.minimum(point, end) - pad, np.maximum(point, end) + pad
                )
                // 2
            )
            try:
                hits = q1_ray_crossings(self.search.vertices[facets], point, direction)
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

    def project(self, points, *, maximum_distance):
        found = self.search.project(
            points, maximum_distance=maximum_distance, require_unique_normal=False
        )
        query = np.asarray(points, dtype=float)
        delta = query - found["points"]
        distance = np.linalg.norm(delta, axis=1)
        tolerance = (
            128
            * np.finfo(float).eps
            * max(self.scale, float(np.max(np.abs(query), initial=0)))
        )
        gaps = np.zeros(len(query))
        normal = found["normals"].copy()
        for i in np.flatnonzero(distance > tolerance):
            gaps[i] = -distance[i] if self._inside(query[i]) else distance[i]
            # This is the signed-distance gradient, not an averaged facet
            # normal. It makes pair forces central even at edges/corners.
            normal[i] = delta[i] / gaps[i]
        return {
            **found,
            "normals": normal,
            "signed_gaps": gaps,
            "sign_method": "two_true_q1_oriented_rays",
        }
