# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded reuse of coordinate-independent, exact triangle topology checks."""
from functools import lru_cache

import numpy as np


def _check(vertex_count, payload):
    triangles = np.frombuffer(payload, dtype=np.int64).reshape(-1, 3)
    if np.any(triangles < 0) or np.any(triangles >= vertex_count):
        raise ValueError("Surface triangle index lies outside the vertex table.")
    if np.unique(triangles).size != vertex_count:
        raise ValueError("Every surface vertex must belong to at least one triangle.")
    if np.any((triangles[:, 0] == triangles[:, 1])
              | (triangles[:, 1] == triangles[:, 2])
              | (triangles[:, 2] == triangles[:, 0])):
        raise ValueError("Surface triangles must reference three distinct vertices.")
    if np.unique(np.sort(triangles, axis=1), axis=0).shape[0] != len(triangles):
        raise ValueError("Triangulated surface contains duplicate facets.")
    directed = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges, first, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0,
        return_index=True, return_inverse=True, return_counts=True,
    )
    nonmanifold = np.flatnonzero(counts > 2)
    if nonmanifold.size:
        edge = tuple(int(v) for v in edges[nonmanifold[np.argmin(first[nonmanifold])]])
        raise ValueError(
            "Triangulated surface contains non-manifold edges; first edge "
            f"is {edge}."
        )
    orientation = np.zeros(len(edges), dtype=np.int64)
    np.add.at(orientation, inverse, np.where(directed[:, 0] < directed[:, 1], 1, -1))
    conflicts = np.flatnonzero((counts == 2) & (orientation != 0))
    if conflicts.size:
        edge = tuple(int(v) for v in edges[conflicts[np.argmin(first[conflicts])]])
        raise ValueError(
            "Adjacent surface facets have inconsistent orientation; first "
            f"edge is {edge}."
        )
    return len(edges), int(np.count_nonzero(counts == 1))


_cached_check = lru_cache(maxsize=4)(_check)


def reviewed_triangle_topology(vertex_count, triangles):
    """Return immutable counts; never cache coordinates, normals or validity.

    Byte equality, not an object ID or an unchecked hash, gates reuse. At most
    four <=4 MiB connectivity payloads are retained. Large inputs are checked
    without retention. The caller owns dtype/shape conversion and geometry.
    """
    payload = np.asarray(triangles, dtype=np.int64).tobytes(order="C")
    checker = _cached_check if len(payload) <= 4*1024*1024 else _check
    return checker(int(vertex_count), payload)
