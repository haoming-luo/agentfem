# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact finite-coordinate identity, never tolerance-based geometric merging.

Use columnar stable sorting rather than NumPy's generic structured-row sort.
Signed zeros compare equal; adjacent representable nonzero coordinates do not.
Input order, not lexicographic position, owns the representative of each class.
"""
import numpy as np


def _ordered_points(points):
    if np.iscomplexobj(points):
        raise ValueError('Coordinate identity requires finite real 3D points.')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Coordinate identity requires finite real 3D points.')
    # lexsort is stable and its last key is primary: x, then y, then z.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    distinct = np.ones(len(points), dtype=bool)
    distinct[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    return points, order, distinct


def has_duplicate_coordinates(points):
    """Whether any finite point rows compare exactly equal."""
    _, _, distinct = _ordered_points(points)
    return not bool(distinct.all())


def weld_coordinates_first(points):
    """Return first-occurrence coordinates and the exact inverse map."""
    points, order, distinct = _ordered_points(points)
    first = order[distinct]
    first_order = np.argsort(first, kind='stable')
    renumber = np.empty(len(first), dtype=np.int64)
    renumber[first_order] = np.arange(len(first))
    inverse = np.empty(len(points), dtype=np.int64)
    inverse[order] = renumber[np.cumsum(distinct, dtype=np.int64) - 1]
    return points[first[first_order]], inverse
