# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded process-local Q1 candidate search; no contact-law admission."""

import heapq

import numpy as np

from ._q1_surface_geometry import q1_closest_point, q1_triangulation_distance_bound
from .boundary_models.rigid import TriangulatedRigidSurface
from .boundary_models.search import TriangleSurfaceBVH, _bbox_squared_distance


def _patch_enclosure_bound(vertices, point, pad, target, depth=0):
    box = np.sqrt(
        _bbox_squared_distance(point, vertices.min(axis=0), vertices.max(axis=0))
    )
    normal = np.cross(vertices[2] - vertices[0], vertices[3] - vertices[1])
    length = np.linalg.norm(normal)
    if length:
        normal = normal / length
    axes = np.vstack(
        (normal, np.cross(np.roll(vertices, -1, axis=0) - vertices, normal))
    )
    lengths = np.linalg.norm(axes, axis=1)
    axes /= np.where(lengths > 0, lengths, 1)[:, None]
    offsets = (vertices - vertices[0]) @ axes.T
    offset = (point - vertices[0]) @ axes.T
    slab = max(
        float(np.max(offsets.min(axis=0) - offset)),
        float(np.max(offset - offsets.max(axis=0))),
        0.0,
    )
    lower = max(0.0, max(box, slab) - pad)
    if lower >= target or depth >= 6:
        return lower
    # Exact Q1 restriction to four parameter subrectangles; convex enclosures
    # tighten without substituting a triangular field for the bilinear patch.
    a, b, c, d = vertices
    ab, bc, cd, da, center = (
        (a + b) / 2,
        (b + c) / 2,
        (c + d) / 2,
        (d + a) / 2,
        vertices.mean(axis=0),
    )
    return min(
        _patch_enclosure_bound(np.array(v), point, pad, target, depth + 1)
        for v in (
            (a, ab, center, da),
            (ab, b, bc, center),
            (center, bc, c, cd),
            (da, center, cd, d),
        )
    )


class Q1SurfaceSearch:
    """Reuse triangular BVH broad phase, refine on the original Q1 geometry.

    Triangle bounds are expanded by the Q1 surrogate distance bound. Query
    radius is a caller-specified physical search radius, not a bond tolerance.
    Returns one facet's original four-node weights. A missing/ambiguous query
    rejects the batch; shared seams with identical points/normals choose the
    lower input facet index. This scalar narrow-phase reference is not MPI or
    a performance-qualified production search.
    """

    def __init__(self, vertices, *, node_ids=None):
        self.vertices = np.array(vertices, dtype=float, copy=True)
        self.node_ids = None
        if node_ids is not None:
            ids = np.asarray(node_ids)
            if (
                ids.shape != self.vertices.shape[:2]
                or ids.dtype.kind not in "iu"
                or np.any(ids < 0)
            ):
                raise ValueError(
                    "Q1 node identities must be nonnegative facet connectivity."
                )
            self.node_ids = ids.copy()
            self.node_ids.flags.writeable = False
            order = np.argsort(ids.ravel(), kind="stable")
            ordered_ids = ids.ravel()[order]
            ordered_points = self.vertices.reshape(-1, 3)[order]
            if np.any(
                (ordered_ids[1:] == ordered_ids[:-1])[:, None]
                & (ordered_points[1:] != ordered_points[:-1])
            ):
                raise ValueError("One Q1 node identity has inconsistent coordinates.")
        bounds = q1_triangulation_distance_bound(vertices)
        if not len(bounds):
            raise ValueError("Q1 search requires a nonempty surface.")
        self.padding = float(bounds.max())
        self.facet_scales = np.linalg.norm(np.ptp(self.vertices, axis=1), axis=1)
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
        self._bound_lower = self.vertices.min(axis=1)
        self._bound_upper = self.vertices.max(axis=1)
        self._bound_origins = self.vertices[:, 0]
        normals = np.cross(
            self.vertices[:, 2] - self.vertices[:, 0],
            self.vertices[:, 3] - self.vertices[:, 1],
        )
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.where(lengths > 0, lengths, 1)[:, None]
        axes = np.concatenate(
            (
                normals[:, None],
                np.cross(
                    np.roll(self.vertices, -1, axis=1) - self.vertices, normals[:, None]
                ),
            ),
            axis=1,
        )
        lengths = np.linalg.norm(axes, axis=2)
        self._bound_normals = axes / np.where(lengths > 0, lengths, 1)[:, :, None]
        offsets = np.einsum(
            "nvi,nai->nva",
            self.vertices - self._bound_origins[:, None],
            self._bound_normals,
        )
        self._slab_lower, self._slab_upper = offsets.min(axis=1), offsets.max(axis=1)

    def distance_lower_bounds(self, points, *, refinement_distance=None):
        """Conservative distance to convex enclosures of the original Q1 faces.

        Each face lies inside its vertex box and a normal slab because Q1
        weights are nonnegative. The BVH's triangulation padding is used only
        for branch pruning, never subtracted globally from the final leaf bound.
        This is not a closest-point projection or an inside/outside classifier.
        """
        raw = np.asarray(points)
        if (
            np.iscomplexobj(raw)
            or raw.ndim != 2
            or raw.shape[1:] != (3,)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Q1 distance bounds require finite real (points,3) data.")
        tree = self.tree
        targets = (
            np.zeros(len(raw))
            if refinement_distance is None
            else np.asarray(refinement_distance)
        )
        if (
            targets.shape != (len(raw),)
            or np.iscomplexobj(targets)
            or not np.isfinite(targets).all()
            or np.any(targets < 0)
        ):
            raise ValueError(
                "Refinement distances must be finite nonnegative point values."
            )
        scale = float(np.max(np.abs(self.vertices)))
        result = np.zeros(len(raw))
        for i, point in enumerate(raw):
            pad = 256 * np.finfo(float).eps * max(scale, float(np.max(np.abs(point))))
            pending = [(0.0, tree._root)]
            best = float("inf")
            visited = set()
            while pending:
                bound, node = heapq.heappop(pending)
                if bound >= best:
                    break
                facet = int(tree._facet[node])
                if facet >= 0:
                    face = facet // 2
                    if face in visited:
                        continue
                    visited.add(face)
                    box = np.sqrt(
                        _bbox_squared_distance(
                            point, self._bound_lower[face], self._bound_upper[face]
                        )
                    )
                    offset = self._bound_normals[face] @ (
                        point - self._bound_origins[face]
                    )
                    slab = max(
                        float(np.max(self._slab_lower[face] - offset)),
                        float(np.max(offset - self._slab_upper[face])),
                        0.0,
                    )
                    leaf = max(0.0, max(box, slab) - pad)
                    if leaf < targets[i]:
                        leaf = _patch_enclosure_bound(
                            self.vertices[face], point, pad, targets[i]
                        )
                    best = min(best, leaf)
                else:
                    for child in (tree._left[node], tree._right[node]):
                        lower = max(
                            0.0,
                            np.sqrt(
                                _bbox_squared_distance(
                                    point, tree._lower[child], tree._upper[child]
                                )
                            )
                            - self.padding
                            - pad,
                        )
                        if lower < best:
                            heapq.heappush(pending, (lower, int(child)))
            result[i] = best
        if not np.isfinite(result).all():
            raise ValueError("Q1 distance bound overflowed.")
        result.flags.writeable = False
        return result

    def project(
        self,
        points,
        *,
        maximum_distance,
        maximum_candidates=128,
        require_unique_normal=True,
    ):
        if not isinstance(require_unique_normal, bool):
            raise ValueError("Normal uniqueness must be a boolean policy.")
        if not require_unique_normal and self.node_ids is None:
            raise ValueError(
                "Normal-independent selection requires explicit node identities."
            )
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
            # A physical search radius is not an ambiguity tolerance. Close to
            # a shared edge, an interior minimum and its neighboring edge
            # minimum can differ by far less than 1e-10*radius while remaining
            # numerically resolvable. Use local geometric roundoff instead.
            scale = max(maximum_distance, float(self.facet_scales[facets].max()))
            tolerance = 64 * np.finfo(float).eps * scale
            point_tolerance = (
                64 * np.finfo(float).eps * max(scale, np.linalg.norm(point))
            )
            for candidate in choices[1:]:
                if abs(candidate[1][-1] - best[1][-1]) <= tolerance:
                    if np.linalg.norm(
                        candidate[1][2] - best[1][2]
                    ) > point_tolerance or (
                        require_unique_normal
                        and np.linalg.norm(candidate[1][3] - best[1][3]) > 1e-8
                    ):
                        raise ValueError(
                            f"Q1 query {index}: ambiguous surface projection."
                        )
                    if not require_unique_normal:
                        coefficients = {}
                        for (facet, record), sign in ((best, 1.0), (candidate, -1.0)):
                            for node, weight in zip(
                                self.node_ids[facet], record[1], strict=True
                            ):
                                coefficients[int(node)] = (
                                    coefficients.get(int(node), 0.0) + sign * weight
                                )
                        if max(map(abs, coefficients.values()), default=0.0) > 1e-10:
                            raise ValueError(
                                f"Q1 query {index}: ambiguous trace interpolation."
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
            "unique_normal_required": require_unique_normal,
        }
