# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded reuse of coordinate-independent, exact triangle topology checks."""
from functools import lru_cache

import numpy as np


def _check(vertex_count, payload, require_connected_vertex_links):
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
    if require_connected_vertex_links:
        # A closed solid must be a 2-manifold at vertices too: two shells can
        # meet at one vertex while every edge still has degree two. An open
        # reference patch may intentionally be point-adjacent, so this check
        # belongs to the signed-solid admission rather than every triangle
        # search user.
        ordered = np.argsort(inverse, kind="stable")
        paired = inverse[ordered[1:]] == inverse[ordered[:-1]]
        first_directed = ordered[:-1][paired]
        second_directed = ordered[1:][paired]
        neighbors = np.full((len(triangles), 3), -1, dtype=np.int64)
        neighbors.reshape(-1)[first_directed] = second_directed // 3
        neighbors.reshape(-1)[second_directed] = first_directed // 3
        incident: list[list[int]] = [[] for _ in range(vertex_count)]
        for slot, triangle in enumerate(triangles):
            for vertex in triangle:
                incident[int(vertex)].append(slot)
        for vertex, slots in enumerate(incident):
            remaining = set(slots)
            pending = [slots[0]]
            while pending:
                slot = pending.pop()
                if slot not in remaining:
                    continue
                remaining.remove(slot)
                local = int(np.flatnonzero(triangles[slot] == vertex)[0])
                for directed_edge in (local, (local + 2) % 3):
                    neighbor = int(neighbors[slot, directed_edge])
                    if neighbor >= 0:
                        pending.append(neighbor)
            if remaining:
                raise ValueError(
                    "Triangulated surface contains a non-manifold vertex link; "
                    f"first vertex is {vertex}."
                )
    return len(edges), int(np.count_nonzero(counts == 1))


_cached_check = lru_cache(maxsize=4)(_check)


def _edge_neighbors(payload):
    """Coordinate-independent adjacency for an already reviewed topology."""
    triangles = np.frombuffer(payload, dtype=np.int64).reshape(-1, 3)
    edges = triangles[:, ((1, 2), (2, 0), (0, 1))].reshape(-1, 2)
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    paired = inverse[order[1:]] == inverse[order[:-1]]
    first, second = order[:-1][paired], order[1:][paired]
    neighbors = np.full(triangles.shape, -1, dtype=np.int64)
    neighbors.reshape(-1)[first] = second // 3
    neighbors.reshape(-1)[second] = first // 3
    # Cache immutable bytes, not ndarray metadata writable by a caller.
    return neighbors.tobytes(order='C')


_cached_edge_neighbors = lru_cache(maxsize=4)(_edge_neighbors)


def reviewed_edge_neighbors(triangles):
    """Reuse bounded exact connectivity only, never a deformed geometry.

    Call only after reviewed_triangle_topology. A changed ordering or index
    changes the byte key. Every returned array has independent metadata over
    immutable storage. Large inputs use the same algorithm without retention.
    """
    selected = np.asarray(triangles, dtype=np.int64)
    payload = selected.tobytes(order='C')
    build = _cached_edge_neighbors if len(payload) <= 4*1024*1024 else _edge_neighbors
    return np.frombuffer(build(payload), dtype=np.int64).reshape(selected.shape)


def reviewed_triangle_topology(
    vertex_count, triangles, *, require_connected_vertex_links=False,
):
    """Return immutable counts; never cache coordinates, normals or validity.

    Byte equality, not an object ID or an unchecked hash, gates reuse. At most
    four <=4 MiB connectivity payloads are retained. Large inputs are checked
    without retention. The caller owns dtype/shape conversion and geometry.
    """
    payload = np.asarray(triangles, dtype=np.int64).tobytes(order="C")
    checker = _cached_check if len(payload) <= 4*1024*1024 else _check
    return checker(
        int(vertex_count), payload, bool(require_connected_vertex_links),
    )
