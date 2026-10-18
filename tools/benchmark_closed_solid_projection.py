# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible local profiling of the bounded closed-solid geometry path.

This is not a whole-Step benchmark. It reports search/sign/projection time
separately so a proposed sign accelerator cannot claim to speed up Newton
without measuring its actual share on the same fixed geometry and queries.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from agentfem.boundary_models import closed_triangulated_rigid_solid
from agentfem.boundary_models.closed_polyhedron import _solid_winding_number


_PARITY_RAY = np.asarray((1.0, np.sqrt(2.0), np.sqrt(3.0)))
_PARITY_RAY /= np.linalg.norm(_PARITY_RAY)


def _experimental_ray_parity_sign(point, triangles, bvh, tolerance):
    """Research-only BVH parity classifier; never used by the solver.

    Strict face-interior crossings are counted. Numerical grazing cases
    defer to direct winding. This does not detect all invalid embeddings,
    so a microbenchmark win is insufficient for product admission.
    """

    lower, upper = bvh.bounds
    span = float(np.linalg.norm(upper - lower))
    length = float(np.linalg.norm(point - (lower + upper) / 2.0) + 2.0 * span)
    endpoint = point + length * _PARITY_RAY
    if not np.isfinite(endpoint).all() or np.array_equal(endpoint, point):
        return None
    slots = bvh.segment_overlapping_facets(
        point, endpoint, padding=tolerance,
    )
    if len(slots) == 0:
        return 1.0
    selected = triangles[slots]
    edge1 = selected[:, 1] - selected[:, 0]
    edge2 = selected[:, 2] - selected[:, 0]
    normal = np.cross(edge1, edge2)
    determinant = np.einsum("ij,ij->i", edge1, np.cross(_PARITY_RAY, edge2))
    area_scale = np.linalg.norm(normal, axis=1)
    parallel = np.abs(determinant) <= area_scale * 1.0e-12
    if np.any(parallel):
        start_plane = np.einsum(
            "ij,ij->i", normal[parallel], point - selected[parallel, 0],
        )
        end_plane = start_plane + length * (normal[parallel] @ _PARITY_RAY)
        margin = tolerance * area_scale[parallel]
        if np.any(
            (start_plane <= margin) & (end_plane >= -margin)
            | (end_plane <= margin) & (start_plane >= -margin)
        ):
            return None
    regular = ~parallel
    if not np.any(regular):
        return 1.0
    selected = selected[regular]
    edge1 = edge1[regular]
    edge2 = edge2[regular]
    reciprocal = 1.0 / determinant[regular]
    offset = point - selected[:, 0]
    pvec = np.cross(_PARITY_RAY, edge2)
    u = np.einsum("ij,ij->i", offset, pvec) * reciprocal
    qvec = np.cross(offset, edge1)
    v = (qvec @ _PARITY_RAY) * reciprocal
    t = np.einsum("ij,ij->i", edge2, qvec) * reciprocal
    margin = 1.0e-10
    possible = (
        (t >= -tolerance) & (t <= length + tolerance)
        & (u >= -margin) & (v >= -margin)
        & (u + v <= 1.0 + margin)
    )
    if np.any(possible & (
        (t <= tolerance) | (t >= length - tolerance)
        | (u <= margin) | (v <= margin) | (1.0 - u - v <= margin)
    )):
        return None
    return -1.0 if int(np.count_nonzero(possible)) % 2 else 1.0


def _refined_l_prism(levels: int):
    polygon = np.asarray(
        ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)),
        dtype=float,
    )
    vertices = np.vstack((
        np.column_stack((polygon, np.zeros(6))),
        np.column_stack((polygon, np.ones(6))),
    )).tolist()
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in top]
    triangles += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend(((first, second, second + 6),
                          (first, second + 6, first + 6)))
    for _ in range(levels):
        edges = {}
        refined = []

        def midpoint(a, b, edges=edges):
            key = tuple(sorted((a, b)))
            if key not in edges:
                edges[key] = len(vertices)
                vertices.append((
                    (np.asarray(vertices[a]) + np.asarray(vertices[b])) / 2
                ).tolist())
            return edges[key]

        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (ab, b, bc),
                            (ca, bc, c), (ab, bc, ca)))
        triangles = refined
    return closed_triangulated_rigid_solid(
        vertices=vertices, triangles=triangles,
        reviewed_embedded=True, name="benchmark_l_prism",
    )


def _median_seconds(fn, repeats):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations)), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--outside-aabb", action="store_true",
        help="Place the outside half strictly beyond the full tool AABB",
    )
    args = parser.parse_args()
    if args.levels < 0 or args.queries < 2 or args.repeats < 1:
        parser.error("levels >= 0, queries >= 2 and repeats >= 1 are required")
    tool = _refined_l_prism(args.levels)
    rng = np.random.default_rng(405)
    count = args.queries // 2
    inside = np.column_stack((
        rng.uniform(0.1, 0.4, count),
        rng.uniform(0.15, 0.45, count),
        rng.uniform(0.2, 0.8, count),
    ))
    outside = np.column_stack((
        rng.uniform(1.2, 1.8, args.queries - count),
        rng.uniform(1.1, 1.8, args.queries - count),
        rng.uniform(0.2, 0.8, args.queries - count),
    ))
    if args.outside_aabb:
        outside[:, 0] += 1.0
    queries = np.vstack((inside, outside))
    tool.project(queries[:10])
    full, projection = _median_seconds(
        lambda: tool.project(queries), args.repeats,
    )
    search, _ = _median_seconds(
        lambda: tool._bvh._candidate_evidence(
            queries, None, require_same_normal=False,
        ), args.repeats,
    )
    winding, direct_values = _median_seconds(
        lambda: [_solid_winding_number(point, tool._triangles)
                 for point in queries], args.repeats,
    )
    ray, ray_values = _median_seconds(
        lambda: [_experimental_ray_parity_sign(
            point, tool._triangles, tool._bvh, tool.facets.tolerance,
        ) for point in queries], args.repeats,
    )
    direct_signs = np.where(np.asarray(direct_values) > 0.5, -1.0, 1.0)
    conclusive = np.asarray([value is not None for value in ray_values])
    if not np.array_equal(
        np.asarray(ray_values, dtype=object)[conclusive].astype(float),
        direct_signs[conclusive],
    ):
        raise RuntimeError("Ray parity disagrees with direct winding on fixture")
    print(json.dumps({
        "scope": "local_projection_only_not_whole_step",
        "facet_count": len(tool.facets.triangles),
        "query_count": len(queries),
        "outside_aabb": bool(args.outside_aabb),
        "repeats": args.repeats,
        "geometry_fingerprint": tool.geometry_fingerprint,
        "valid_count": int(np.sum(projection.valid)),
        "full_median_seconds": full,
        "search_median_seconds": search,
        "direct_winding_median_seconds": winding,
        "ray_parity_median_seconds": ray,
        "ray_parity_conclusive_count": int(np.sum(conclusive)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
