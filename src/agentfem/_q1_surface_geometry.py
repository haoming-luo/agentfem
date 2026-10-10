# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private Q1 inverse maps; no interface integration admission.

Basix owns interpolation. This adapter batches geometric inversion for boundary
queries, preserving all four parent nodes. Warped maps admit on-surface points
only; nearest-point projection and curved intersection integration are separate.
"""

from functools import lru_cache

import basix
import numpy as np


@lru_cache(maxsize=1)
def _element():
    return basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        1,
        basix.LagrangeVariant.equispaced,
    )


def _real(value, shape, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real.")
    value = np.asarray(raw, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return value


def planar_q1_pull_back(
    vertices, points, *, relative_tolerance=1e-11, max_iterations=30
):
    """Strict planar adapter retained for the existing planar overlap route."""
    return _pull_back(
        vertices,
        points,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        planar_only=True,
    )


def q1_surface_pull_back(
    vertices, points, *, relative_tolerance=1e-11, max_iterations=30
):
    """Invert a bilinear surface with convex projection onto its corner plane.

    Keeps real warpage. This is an on-surface inverse, not a closest-point
    projection. Convex projected corners ensure a one-to-one planar Q1 map;
    the bilinear surface is a graph over that projection. Folded projections
    are rejected rather than resolved through an arbitrary inverse branch.
    """
    return _pull_back(
        vertices,
        points,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        planar_only=False,
    )


def _pull_back(vertices, points, *, relative_tolerance, max_iterations, planar_only):
    """Return local coordinates and cyclic Q1 weights for one query per facet.

    Input shapes are (queries, 4, 3) and (queries, 3). Tolerance is relative
    to facet diameter, never a geometric coincidence allowance. Newton steps
    minimize the in-plane map residual; both reference containment and physical
    reconstruction are checked. No nearest-surface projection is performed.
    """
    raw = np.asarray(vertices)
    if raw.ndim != 3 or raw.shape[1:] != (4, 3):
        raise ValueError("vertices must have shape (queries, 4, 3).")
    count = len(raw)
    x = _real(vertices, (count, 4, 3), "vertices")
    query = _real(points, (count, 3), "points")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance <= 1e-8:
        raise ValueError("relative_tolerance must lie in (0, 1e-8].")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if not count:
        return np.empty((0, 2)), np.empty((0, 4))
    # Normalize to avoid units-dependent singularity thresholds and cancellation
    # in basis sums at large coordinate origins.
    diameter = np.linalg.norm(np.ptp(x, axis=1), axis=1)
    if np.any(diameter <= 0):
        raise ValueError("Degenerate Q1 facet.")
    q = (query - x[:, 0]) / diameter[:, None]
    x = (x - x[:, :1]) / diameter[:, None, None]
    edge = np.roll(x, -1, axis=1) - x
    normal = np.cross(edge[:, 0], -edge[:, 3])
    length = np.linalg.norm(normal, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("Degenerate Q1 corner.")
    normal /= length[:, None]
    if planar_only and np.any(
        np.abs(np.einsum("nki,ni->nk", x, normal)) > relative_tolerance
    ):
        raise ValueError("Warped Q1 facets are not admitted by the planar map.")
    turns = np.einsum("nki,ni->nk", np.cross(edge, np.roll(edge, -1, axis=1)), normal)
    if np.any(turns <= 1e-12):
        raise ValueError("Q1 facets must be strictly convex with cyclic nodes.")
    local = np.full((count, 2), 0.5)
    for _ in range(max_iterations):
        basis = _element().tabulate(1, local)[..., 0][:, :, [0, 1, 3, 2]]
        residual = np.einsum("nk,nki->ni", basis[0], x) - q
        if np.all(np.linalg.norm(residual, axis=1) <= relative_tolerance):
            break
        jacobian = np.einsum("ank,nki->nia", basis[1:3], x)
        gram = np.einsum("nia,nib->nab", jacobian, jacobian)
        rhs = np.einsum("nia,ni->na", jacobian, residual)
        try:
            step = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular Q1 inverse map.") from exc
        local -= step
        if not np.all(np.isfinite(local)):
            raise ValueError("Nonfinite Q1 inverse map.")
    weights = _element().tabulate(0, local)[0, :, :, 0][:, [0, 1, 3, 2]]
    reconstruction = np.einsum("nk,nki->ni", weights, x)
    if np.any(np.linalg.norm(reconstruction - q, axis=1) > relative_tolerance):
        raise ValueError("Q1 pull-back did not converge to a point on the facet.")
    if np.any(local < -relative_tolerance) or np.any(local > 1 + relative_tolerance):
        raise ValueError("Q1 query lies outside the reference facet.")
    return local, weights
