# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Smooth-interior sign screen for reviewed embedded closed Q1 surfaces.

A global closest point strictly inside a smooth outward patch determines the
distance sign from its current normal. This is not an edge pseudonormal or a
self-intersection certificate. Unresolved cases retain the two-ray oracle.
"""
import numpy as np


def connected_shell(edges, face_count):
    """Called only after the owner's closed manifold topology validation."""
    adjacency = [set() for _ in range(face_count)]
    for uses in edges.values():
        a, b = uses[0][0], uses[1][0]
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, stack = set(), [0]
    while stack:
        face = stack.pop()
        if face not in seen:
            seen.add(face)
            stack.extend(adjacency[face] - seen)
    return len(seen) == face_count


def outward_volume(vertices):
    """Exact bilinear flux formula, with a conservative roundoff screen.

For x=a+b*r+c*s+d*r*s, integral(x dot (x_r cross x_s))
over [-1,1]^2 is 4*a dot (b cross c). This is not a triangular
surrogate. Positive volume identifies outward orientation only for the
reviewed embedded connected shell, never for arbitrary intersecting sheets.
"""
    origin = 0.5 * (vertices.min(axis=(0, 1)) + vertices.max(axis=(0, 1)))
    face = vertices - origin
    centers = face.mean(axis=1)
    area = 0.5 * np.cross(face[:, 2] - face[:, 0], face[:, 3] - face[:, 1])
    value = float(np.sum(np.sum(centers * area, axis=1)) / 3)
    scale = float(np.sum(np.linalg.norm(centers, axis=1)
                         * np.linalg.norm(area, axis=1)) / 3)
    return value, value > 1024 * np.finfo(float).eps * scale


def interior_sign(vertices, found, resolved, delta, distance, tolerance):
    """Return admission mask and signs; no force/projection is approximated.

Reconstruct the normal from current Q1 geometry rather than trusting an
extension's unit normal (transport validation alone does not prove orientation).
"""
    weights = found["weights"][resolved]
    face = vertices[found["facet_indices"][resolved]]
    u = weights[:, 1] + weights[:, 2]
    v = weights[:, 2] + weights[:, 3]
    dr = ((1-v)[:, None] * (face[:, 1] - face[:, 0])
          + v[:, None] * (face[:, 2] - face[:, 3]))
    ds = ((1-u)[:, None] * (face[:, 3] - face[:, 0])
          + u[:, None] * (face[:, 2] - face[:, 1]))
    normal = np.cross(dr, ds)
    length = np.linalg.norm(normal, axis=1)
    dot = np.divide(np.einsum("ij,ij->i", delta[resolved], normal), length,
                    out=np.zeros(len(resolved)), where=length > 0)
    use = (np.all(weights > 1e-6, axis=1)
           & (np.abs(dot) > 64 * tolerance)
           & (np.abs(dot) >= (1 - 256 * np.sqrt(np.finfo(float).eps))
              * distance[resolved]))
    return use, dot < 0
