# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded process-local Q1 candidate search; no contact-law admission."""

import heapq

import numpy as np

from ._q1_surface_geometry import q1_closest_point, q1_triangulation_distance_bound
from .boundary_models.rigid import TriangulatedRigidSurface
from .boundary_models.search import TriangleSurfaceBVH, _bbox_squared_distance


class AmbiguousSurfaceProjection(ValueError):
    """A nearest-point choice is unresolved; not evidence of penetration."""

    def __init__(self, query_index, *, facets=None, distances=None,
                 point_separation=None, distance_tolerance=None):
        self.query_index = int(query_index)
        self.facets = None if facets is None else tuple(int(f) for f in facets)
        self.distances = None if distances is None else tuple(float(d) for d in distances)
        self.point_separation = point_separation
        self.distance_tolerance = distance_tolerance
        super().__init__(f"Q1 query {query_index}: ambiguous surface projection.")


class SurfaceOutsideSearchRadius(ValueError):
    """No projection in the requested radius; does not classify solid side."""

    def __init__(self, query_index):
        self.query_index = int(query_index)
        super().__init__(f"Q1 query {query_index}: no surface within search radius.")


def _interior_dominates_shared_edge(vertices, node_ids, point, interior, edge,
                                  *, allow_exact=True):
    """Resolve a candidate versus a neighboring patch sharing a physical edge.

    Near an edge, distance differences are quadratic in edge distance and can
    disappear in a distance-based tie tolerance. Topology plus a strongly
    convex squared-distance objective supplies an independent ordering screen.
    A filtered exact convex-boundary proof also covers near-corner edge
    candidates. Overlapping/disconnected/nonconvex alternatives remain unresolved.
    """
    if node_ids is None:
        return False
    fi, ri = interior
    fe, re = edge
    shared = set(node_ids[fi]) & set(node_ids[fe])
    if len(shared) not in (1, 2):
        return False
    for node in shared:
        vi = vertices[fi, np.flatnonzero(node_ids[fi] == node)]
        ve = vertices[fe, np.flatnonzero(node_ids[fe] == node)]
        if vi.shape != (1, 3) or not np.array_equal(vi, ve):
            return False
    if len(shared) == 1:
        # Corner-adjacent patches need no invented common edge. The exact
        # comparison proves global ordering independently on the other patch.
        from ._q1_exact_order import witness_beats_boundary_minimum

        return allow_exact and witness_beats_boundary_minimum(
            vertices[fi], point, ri[0], vertices[fe])
    for row in (node_ids[fi], node_ids[fe]):
        positions = [i for i, node in enumerate(row) if node in shared]
        if (positions[1] - positions[0]) % 4 not in (1, 3):
            return False
    from ._q1_exact_order import witness_beats_boundary_minimum

    def exact_order():
        return allow_exact and witness_beats_boundary_minimum(
            vertices[fi], point, ri[0], vertices[fe])

    eps = np.finfo(float).eps
    outside = [w for node, w in zip(node_ids[fe], re[1], strict=True)
               if node not in shared]
    if max(map(abs, outside), default=1.) > 64 * eps:
        return exact_order()
    s, t = ri[0]
    margin = min(s, t, 1 - s, 1 - t)
    if margin <= 256 * eps:
        return exact_order()
    v = vertices[fi]
    a = v[0] - point
    b, c = v[1] - v[0], v[3] - v[0]
    d = (v[2] - v[3]) - b

    def minimum_norm_squared(direction):
        dd = float(d @ d)
        coordinate = 0. if dd == 0 else np.clip(-float(direction @ d) / dd, 0., 1.)
        tangent = direction + coordinate * d
        return float(tangent @ tangent)

    off = max(abs(float(b @ c + a @ d + 2*x*(b @ d)
                        + 2*y*(c @ d) + 2*x*y*(d @ d)))
              for x in (0., 1.) for y in (0., 1.))
    curvature = min(minimum_norm_squared(b), minimum_norm_squared(c)) - off
    scale = max(np.linalg.norm(b), np.linalg.norm(c), np.linalg.norm(d))
    if curvature <= 256 * eps * scale**2:
        return False
    delta = a + b*s + c*t + d*s*t
    gradient = np.array([delta @ (b+d*t), delta @ (c+d*s)])
    # Include translated-coordinate cancellation, not only local arithmetic.
    error = 256 * eps * max(scale, np.linalg.norm(point), np.max(abs(v))) * scale
    if np.linalg.norm(gradient) + error < curvature * margin / 16:
        return True
    # A cancellation-limited tie need not be physically ambiguous. Only this
    # rare filtered case pays for exact binary64-rational arithmetic; prove
    # global ordering against the neighboring convex patch, or keep rejecting.
    return exact_order()


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
    best = float("inf")
    for vertices_child in (
        (a, ab, center, da), (ab, b, bc, center),
        (center, bc, c, cd), (da, center, cd, d),
    ):
        best = min(best, _patch_enclosure_bound(
            np.array(vertices_child), point, pad, target, depth + 1
        ))
        # Every child bound is nonnegative. Once zero is found, additional
        # subdivision cannot lower the minimum. This is an exact early exit,
        # not a relaxed distance tolerance or an assumed closest projection.
        if best == 0.0:
            break
    return best


def _weld_coordinates(points):
    """Exact coordinate equality, with vertex storage in first-occurrence order."""
    nodes, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    return nodes[order], renumber[inverse]


class Q1SurfaceSearch:
    """Reuse triangular BVH broad phase, refine on the original Q1 geometry.

    Triangle bounds are expanded by the Q1 surrogate distance bound. Query
    radius is a caller-specified physical search radius, not a bond tolerance.
    Returns one facet's original four-node weights. A missing/ambiguous query
    rejects the batch; shared seams with identical points/normals choose the
    lower input facet index. This scalar narrow-phase reference is not MPI or
    a performance-qualified production search.
    """

    def __init__(self, vertices, *, node_ids=None, _previous=None, execution=None):
        self.execution = execution
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
        # Coordinate welding is unchanged, but storage follows first facet
        # occurrence rather than lexicographic current coordinates. Ordinary
        # motion must not gratuitously renumber unchanged triangle topology.
        nodes, inverse = _weld_coordinates(self.vertices.reshape(-1, 3))
        surface = TriangulatedRigidSurface(nodes, inverse[triangles])
        if _previous is not None:
            if (not isinstance(_previous, Q1SurfaceSearch)
                    or self.vertices.shape != _previous.vertices.shape
                    or not np.array_equal(self.node_ids, _previous.node_ids)):
                raise ValueError("Q1 refit requires unchanged ordered trace topology.")
            self.tree = _previous.tree._refitted(surface)
        else:
            self.tree = TriangleSurfaceBVH(surface)
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
        if self.execution is not None:
            return self.execution.bounds(self, points, self._distance_lower_bounds_reference,
                                          refinement_distance=refinement_distance)
        return self._distance_lower_bounds_reference(points, refinement_distance=refinement_distance)

    def _distance_lower_bounds_reference(self, points, *, refinement_distance=None):
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
        self, points, *, maximum_distance, maximum_candidates=128,
        require_unique_normal=True,
    ):
        options = dict(maximum_distance=maximum_distance,
                       maximum_candidates=maximum_candidates,
                       require_unique_normal=require_unique_normal)
        # Input admission must be identical before an optional backend runs.
        self._validate_project(points, **options)
        if self.execution is not None and self.execution.project_batch is not None:
            return self.execution.project(self, np.asarray(points, dtype=float),
                                          self._project_reference, **options)
        return self._project_reference(points, **options)

    def _validate_project(
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

    def _project_reference(self, points, *, maximum_distance, maximum_candidates=128,
                           require_unique_normal=True, _closest=None, _overlap=None):
        self._validate_project(points, maximum_distance=maximum_distance,
                               maximum_candidates=maximum_candidates,
                               require_unique_normal=require_unique_normal)
        points = np.asarray(points, dtype=float)
        overlap = self.tree.overlapping_facets if _overlap is None else _overlap
        def closest(vertices, point):
            if _closest is not None:
                return _closest(vertices, point)
            if self.execution is None:
                return q1_closest_point(vertices, point)
            return self.execution.closest(vertices, point, q1_closest_point)
        # Closed-solid callers can reject a provably exterior ambiguous point
        # and retry the remaining batch. Reuse only successful prefix queries
        # on this immutable geometry, with identical search/normal policies.
        # The cache is emptied on success and never transferred on refit.
        policy = (float(maximum_distance), maximum_candidates, require_unique_normal)
        previous = getattr(self, "_retry_projection_prefix", {})
        self._retry_projection_prefix = {}
        found, counts = [], []
        for index, point in enumerate(points):
            key = (policy, point.tobytes())
            if key in previous:
                best, count = previous[key]
                found.append(best)
                counts.append(count)
                self._retry_projection_prefix[key] = (best, count)
                continue
            radius = maximum_distance + self.padding
            facets = np.unique(
                overlap(point - radius, point + radius) // 2
            )
            if len(facets) > maximum_candidates:
                raise ValueError(f"Q1 query {index}: candidate budget exceeded.")
            counts.append(len(facets))
            choices = [
                (int(f), closest(self.vertices[f], point)) for f in facets
            ]
            choices = [(f, r) for f, r in choices if r[-1] <= maximum_distance]
            if not choices:
                raise SurfaceOutsideSearchRadius(index)
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
                    # Two boundary projections already equivalent under the
                    # geometric AND trace contracts need no rational ordering
                    # proof. Strict interiors still use the convex screen.
                    coincident_boundary = False
                    if (self.node_ids is not None
                            and all(np.any((r[0] == 0) | (r[0] == 1))
                                    for _, r in (best, candidate))
                            and np.linalg.norm(candidate[1][2]-best[1][2]) <= point_tolerance
                            and (not require_unique_normal or
                                 np.linalg.norm(candidate[1][3]-best[1][3]) <= 1e-8)):
                        coefficients = {}
                        for (facet, record), sign in ((best, 1.), (candidate, -1.)):
                            for node, weight in zip(self.node_ids[facet], record[1], strict=True):
                                coefficients[int(node)] = coefficients.get(int(node), 0.) + sign*weight
                        coincident_boundary = max(map(abs, coefficients.values()), default=0.) <= 1e-10
                    if _interior_dominates_shared_edge(
                        self.vertices, self.node_ids, point, best, candidate,
                        allow_exact=not coincident_boundary,
                    ):
                        continue
                    if _interior_dominates_shared_edge(
                        self.vertices, self.node_ids, point, candidate, best,
                        allow_exact=not coincident_boundary,
                    ):
                        best = candidate
                        continue
                    if not coincident_boundary and np.linalg.norm(
                        candidate[1][2] - best[1][2]
                    ) > point_tolerance:
                        from ._q1_exact_order import convex_minimum_precedes

                        if convex_minimum_precedes(
                            self.vertices[best[0]], point, best[1][0],
                            self.vertices[candidate[0]], candidate[1][0],
                        ):
                            continue
                        if convex_minimum_precedes(
                            self.vertices[candidate[0]], point, candidate[1][0],
                            self.vertices[best[0]], best[1][0],
                        ):
                            best = candidate
                            continue
                    if np.linalg.norm(
                        candidate[1][2] - best[1][2]
                    ) > point_tolerance or (
                        require_unique_normal
                        and np.linalg.norm(candidate[1][3] - best[1][3]) > 1e-8
                    ):
                        raise AmbiguousSurfaceProjection(
                            index, facets=(best[0], candidate[0]),
                            distances=(best[1][-1], candidate[1][-1]),
                            point_separation=float(np.linalg.norm(
                                candidate[1][2] - best[1][2])),
                            distance_tolerance=float(tolerance),
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
            self._retry_projection_prefix[key] = (best, counts[-1])
        self._retry_projection_prefix = {}
        return {
            "facet_indices": np.array([f for f, _ in found], dtype=np.int64),
            "weights": np.array([r[1] for _, r in found]).reshape(-1, 4),
            "points": np.array([r[2] for _, r in found]).reshape(-1, 3),
            "normals": np.array([r[3] for _, r in found]).reshape(-1, 3),
            "distances": np.array([r[4] for _, r in found]),
            "candidate_counts": np.array(counts, dtype=np.int64),
            "unique_normal_required": require_unique_normal,
        }
