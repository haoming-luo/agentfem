# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Bernstein admission for fixed, trilinear Hex8 geometry.

Uses the determinant convex-hull construction of Johnen, Weill and Remacle
(2017), with 27 Basix evaluations rather than their optimized 20-volume rule.
Floating-point margins are conservative guards, not interval arithmetic.
"""

from functools import lru_cache
from itertools import product

import basix
import numpy as np


@lru_cache(maxsize=1)
def _derivatives():
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        1,
        basix.LagrangeVariant.equispaced,
    )
    points = np.array(list(product((0.0, 0.5, 1.0), repeat=3)))
    return element.tabulate(1, points)[1:4, :, :, 0].transpose(1, 2, 0)


def _bernstein(values):
    result = np.array(values, copy=True)
    for axis in range(result.ndim - 3, result.ndim):
        view = np.moveaxis(result, axis, -1)
        view[..., 1] = 2 * view[..., 1] - (view[..., 0] + view[..., 2]) / 2
    return result


def _children(coefficients):
    children = [coefficients]
    for axis in range(3):
        split = []
        for child in children:
            b = np.moveaxis(child, axis, -1)
            middle = (b[..., 0] + 2 * b[..., 1] + b[..., 2]) / 4
            left = np.stack((b[..., 0], (b[..., 0] + b[..., 1]) / 2, middle), -1)
            right = np.stack((middle, (b[..., 1] + b[..., 2]) / 2, b[..., 2]), -1)
            split.extend((np.moveaxis(left, -1, axis), np.moveaxis(right, -1, axis)))
        children = split
    return children


def _admit(coefficients, margin, *, max_depth=5, max_boxes=4096):
    """Return a lower bound, or reject inversion / unresolved geometry distinctly."""
    pending = [(coefficients, 0)]
    lower = np.inf
    visited = 0
    while pending:
        b, depth = pending.pop()
        visited += 1
        if np.min(b) > margin:
            lower = min(lower, float(np.min(b)))
            continue
        if np.min(b[::2, ::2, ::2]) <= margin:
            raise ValueError("Hex8 has an inverted or near-singular Jacobian.")
        if depth >= max_depth or visited >= max_boxes:
            raise ValueError(
                "Hex8 Jacobian positivity is unresolved within the bounded "
                "geometry check; improve the mesh before solving."
            )
        pending.extend((child, depth + 1) for child in _children(b))
    return lower


def require_positive_hex_jacobian(coordinates):
    """Check a caller-sized batch in Basix order; return determinant lower bounds.

    Fast path is vectorized. Ambiguous cells are subdivided one at a time with
    bounded depth/work; a negative control coefficient alone is not inversion.
    This checks local Jacobians, not collisions between separate mesh cells.
    """
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    jacobian = np.einsum("cai,qaj->cqij", centered, _derivatives())
    # Scalar triple products avoid one tiny LAPACK factorization per sample.
    # The existing conservative column-norm margin still guards cancellation;
    # the Bernstein admission and subdivision are unchanged.
    values = np.einsum(
        "...i,...i->...", jacobian[..., 0],
        np.cross(jacobian[..., 1], jacobian[..., 2]),
    )
    scale = np.prod(np.max(np.linalg.norm(jacobian, axis=2), axis=1), axis=1)
    coefficients = _bernstein(values.reshape(-1, 3, 3, 3))
    margin = 512 * np.finfo(float).eps * scale
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(margin)):
        raise ValueError("Hex8 Jacobian validity computation overflowed.")
    lower = coefficients.min(axis=(1, 2, 3))
    for index in np.flatnonzero(lower <= margin):
        try:
            lower[index] = _admit(coefficients[index], margin[index])
        except ValueError as error:
            raise ValueError(f"Geometry batch cell {index}: {error}") from error
    return lower
