# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private true-bilinear ray crossings; ambiguous rays request another direction.

Elimination follows the two-plane formulation of Ramsey, Potter & Hansen
(2004), DOI 10.1080/10867651.2004.10504896. No triangle intersection substitutes
for Q1 geometry. This is a bounded reference kernel, not continuous collision
detection or a self-intersection certificate.
"""

import numpy as np


class AmbiguousRay(ValueError):
    """A seam, tangent or degenerate elimination needs a different ray."""


def q1_ray_crossings(vertices, origin, direction):
    """Return (facet, u, v, ray_distance, oriented_crossing) for forward hits.

    Direction must be unit. Facets are already validated by the geometry owner.
    Near-boundary intersections are never counted twice or silently dropped.
    """
    if any(np.iscomplexobj(v) for v in (vertices, origin, direction)):
        raise ValueError("Q1 ray geometry must be real.")
    x = np.asarray(vertices, dtype=float)
    p, r = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    if (
        x.ndim != 3
        or x.shape[1:] != (4, 3)
        or p.shape != (3,)
        or r.shape != (3,)
        or not np.isfinite(x).all()
        or not np.isfinite(p).all()
        or not np.isfinite(r).all()
        or not np.isclose(r @ r, 1.0, atol=1e-13, rtol=0)
    ):
        raise ValueError("Finite Q1 facets, origin and unit ray are required.")
    if not len(x):
        return np.empty((0, 5))
    scales = np.linalg.norm(np.ptp(x, axis=1), axis=1)
    if np.any(scales <= 0):
        raise ValueError("Degenerate ray facet.")
    # Two perpendicular equations avoid division by a ray component.
    e = np.cross(r, np.eye(3)[np.argmin(np.abs(r))])
    e /= np.linalg.norm(e)
    projection = np.stack((e, np.cross(r, e)), axis=1)
    a = (x[:, 0] - p) / scales[:, None]
    b = (x[:, 1] - x[:, 0]) / scales[:, None]
    c = (x[:, 3] - x[:, 0]) / scales[:, None]
    d = (x[:, 2] - x[:, 1] - x[:, 3] + x[:, 0]) / scales[:, None]
    A, B, C, D = (v @ projection for v in (a, b, c, d))
    det = lambda v, w: v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]
    coefficients = np.stack((det(A, B), det(A, D) + det(C, B), det(C, D)), axis=1)
    magnitudes = np.max(np.abs(coefficients), axis=1)
    # Degenerate projection (including a coplanar ray) cannot certify a miss.
    if np.any(magnitudes < 256 * np.finfo(float).eps):
        raise AmbiguousRay("Degenerate Q1 ray elimination.")
    coefficients /= magnitudes[:, None]
    hits = []
    tol = 256 * np.finfo(float).eps
    for facet, (constant, linear, quadratic) in enumerate(coefficients):
        if abs(quadratic) <= tol:
            if abs(linear) <= tol:
                continue  # Nonzero constant: no intersection.
            roots = [-constant / linear]
        else:
            discriminant = linear * linear - 4 * quadratic * constant
            if discriminant < -tol:
                continue
            if discriminant <= tol:
                raise AmbiguousRay("Near-tangent Q1 ray root.")
            stable = -0.5 * (linear + np.copysign(np.sqrt(discriminant), linear))
            roots = [stable / quadratic, constant / stable]
        for v in roots:
            if v < -1e-10 or v > 1 + 1e-10:
                continue
            denominator = B[facet] + D[facet] * v
            component = int(np.argmax(np.abs(denominator)))
            if abs(denominator[component]) <= tol:
                raise AmbiguousRay("Unresolved Q1 ray parameter.")
            u = (
                -(A[facet, component] + C[facet, component] * v)
                / denominator[component]
            )
            if u < -1e-10 or u > 1 + 1e-10:
                continue
            offset = a[facet] + b[facet] * u + c[facet] * v + d[facet] * u * v
            t = float(offset @ r)
            if t < -1e-10:
                continue
            if np.linalg.norm(offset - t * r) > 1e-10:
                raise AmbiguousRay("Unresolved Q1 ray reconstruction.")
            if min(u, v, 1 - u, 1 - v, t) <= 1e-10:
                raise AmbiguousRay("Q1 ray touches a seam, vertex or its origin.")
            crossing = np.cross(b[facet] + d[facet] * v, c[facet] + d[facet] * u) @ r
            if abs(crossing) <= tol:
                raise AmbiguousRay("Q1 ray is tangent.")
            hits.append((facet, u, v, t * scales[facet], np.sign(crossing)))
    return np.asarray(hits, dtype=float).reshape(-1, 5)
