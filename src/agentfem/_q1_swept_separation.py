# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded proof of separation of linear point paths from moving Q1 patches.

The relative map is trilinear in (time, u, v). Its eight control vectors
enclose its image. Exact rational separating planes and dyadic subdivision
can exclude zero; unresolved/touching/crossing paths return False. This does
not locate impact times or certify self-intersection-free solid motion.
"""
from fractions import Fraction as Q

import numpy as np


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _separating_plane(control):
    points = control.reshape(-1, 3)
    # Exact coordinate planes are the cheapest convex-hull exclusion.
    for axis in range(3):
        values = points[:, axis]
        if min(values) > 0 or max(values) < 0:
            return True
    # Proposal axes need not be hull faces: strict common sign on every
    # control vector suffices. Use each parameter face's diagonal normal.
    for dimension in range(3):
        for side in (0, 1):
            face = np.take(control, side, axis=dimension)
            normal = _cross(face[1, 1]-face[0, 0], face[0, 1]-face[1, 0])
            values = [sum(x*n for x, n in zip(p, normal, strict=True))
                      for p in points]
            if min(values) > 0 or max(values) < 0:
                return True
    return False


def _patch_path_separated(old, new, point_old, point_new, *, max_depth):
    control = np.empty((2, 2, 2, 3), dtype=object)
    for t, (vertices, point) in enumerate(((old, point_old), (new, point_new))):
        for index, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            control[t, u, v] = [Q(float(x))-Q(float(p))
                                for x, p in zip(vertices[index], point, strict=True)]
    pending = [(control, 0)]
    visited = 0
    while pending:
        box, depth = pending.pop()
        visited += 1
        if _separating_plane(box):
            continue
        if depth >= max_depth or visited >= 4096:
            return False
        # Split the direction with greatest control variation. The choice
        # affects work only; exact dyadic restriction owns the proof.
        spans = [max(abs(v) for v in np.diff(box, axis=a).flat) for a in range(3)]
        axis = max(range(3), key=lambda a: spans[a])
        if spans[axis] == 0:
            return False
        start, end = np.take(box, 0, axis=axis), np.take(box, 1, axis=axis)
        middle = (start+end)/2
        pending.append((np.stack((middle, end), axis=axis), depth+1))
        pending.append((np.stack((start, middle), axis=axis), depth+1))
    return True


def separated_q1_paths(vertices_old, vertices_new, quadrilaterals,
                       points_old, points_new, *, max_depth=18):
    """Return a proof mask; False means unresolved, not a detected collision."""
    raw = [np.asarray(v) for v in (vertices_old, vertices_new, points_old, points_new)]
    if any(np.iscomplexobj(v) or not np.isfinite(v).all() for v in raw):
        raise ValueError('Swept Q1 separation requires finite real geometry.')
    old, new, p0, p1 = raw
    cells = np.asarray(quadrilaterals)
    if (old.ndim != 2 or old.shape[1:] != (3,) or new.shape != old.shape
            or p0.ndim != 2 or p0.shape[1:] != (3,) or p1.shape != p0.shape
            or cells.ndim != 2 or cells.shape[1:] != (4,)
            or cells.dtype.kind not in 'iu' or not len(cells)
            or np.any(cells < 0) or np.any(cells >= len(old))
            or isinstance(max_depth, bool) or not isinstance(max_depth, int)
            or not 0 <= max_depth <= 30):
        raise ValueError('Swept Q1 separation layout or depth differs.')
    x0, x1 = old[cells], new[cells]
    lower = np.minimum(x0.min(axis=1), x1.min(axis=1))
    upper = np.maximum(x0.max(axis=1), x1.max(axis=1))
    result = np.ones(len(p0), dtype=bool)
    for i, (a, b) in enumerate(zip(p0, p1, strict=True)):
        # Min/max selects original binary64 coordinates exactly. No rounded
        # difference or triangle surrogate can shrink these swept boxes.
        candidates = np.flatnonzero(np.all((lower <= np.maximum(a, b))
                                          & (upper >= np.minimum(a, b)), axis=1))
        result[i] = all(_patch_path_separated(x0[f], x1[f], a, b, max_depth=max_depth)
                        for f in candidates)
    return result
