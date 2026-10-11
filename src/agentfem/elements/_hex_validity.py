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


@lru_cache(maxsize=1)
def _derivative_matrix():
    """One fixed dense contraction, retaining all 27 validity samples."""
    matrix = np.ascontiguousarray(_derivatives().transpose(1, 0, 2).reshape(8, 81))
    matrix.setflags(write=False)
    return matrix


def _jacobians(centered_coordinates):
    count = len(centered_coordinates)
    values = centered_coordinates.transpose(0, 2, 1) @ _derivative_matrix()
    return values.reshape(count, 3, 27, 3).transpose(0, 2, 1, 3)


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
    jacobian = _jacobians(centered)
    # Scalar triple products avoid one tiny LAPACK factorization per sample.
    # The existing conservative column-norm margin still guards cancellation;
    # the Bernstein admission and subdivision are unchanged.
    values = np.einsum(
        "...i,...i->...",
        jacobian[..., 0],
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


@lru_cache(maxsize=1)
def _coordinate_edges():
    corners = basix.cell.geometry(basix.CellType.hexahedron)
    edges = []
    for axis in range(3):
        pairs = []
        for start, point in enumerate(corners):
            if point[axis] != 0:
                continue
            end = point.copy()
            end[axis] = 1
            pairs.append(
                (start, int(np.flatnonzero(np.all(corners == end, axis=1))[0]))
            )
        edges.append(pairs)
    result = np.asarray(edges)
    result.setflags(write=False)
    return result


def _jacobian_frobenius_bound(values):
    """Uniform bound from the four convex edge differences per Q1 derivative."""
    edges = _coordinate_edges()
    differences = values[:, edges[:, :, 1]] - values[:, edges[:, :, 0]]
    return np.sqrt(
        np.sum(np.max(np.sum(differences * differences, axis=3), axis=2), axis=1)
    )


def reference_perturbation_radius(coordinates):
    """Conservative sufficient perturbation radius, not a small-strain tolerance.

    For every reference point, sigma_min(J) >= 2*det(J)/||J||_F**2.
    The Bernstein lower determinant and convex edge upper norm give a uniform
    bound. A strict quarter of it leaves room for floating-point guards. This
    remains floating-point admission, not an interval-arithmetic certificate.
    """
    lower = require_positive_hex_jacobian(coordinates)
    norm = _jacobian_frobenius_bound(coordinates)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        radius = 0.5 * lower / (norm * norm)
    return np.where(np.isfinite(radius) & (radius > 0), radius, 0.0)


def admits_reference_perturbation(coordinates, displacement, radius):
    """Sufficient no-singularity homotopy check; False requests full admission.

    No accepted-state cache or displacement threshold is used. Large rigid
    rotation can fail this sufficient test and still pass the Bernstein path.
    """
    change = _jacobian_frobenius_bound(displacement)
    scale = np.max(np.abs(coordinates), axis=(1, 2)) + np.max(
        np.abs(displacement), axis=(1, 2)
    )
    guard = 128 * np.finfo(float).eps * scale
    return np.isfinite(change + guard) & (change + guard < radius)


def admits_affine_perturbation(coordinates, displacement, gradient, radius):
    """Sufficient validity after removing an orientation-preserving affine map.

    Write x = F X + r + translation, so J_current = F J_reference + J_r.
    The reference radius times sigma_min(F) is a conservative perturbation
    budget. Pure finite rotation/shear need not trigger 27-sample subdivision.
    False still requests the full Bernstein check; no inversion is repaired.
    """
    from .._orientation import positive_finite_determinants

    admitted = np.zeros(len(coordinates), dtype=bool)
    selected = positive_finite_determinants(gradient)
    if not np.any(selected):
        return admitted
    x, u, f = coordinates[selected], displacement[selected], gradient[selected]
    relative = x - x[:, :1]
    remainder = (u - u[:, :1]) - relative @ (f - np.eye(3)).swapaxes(1, 2)
    change = _jacobian_frobenius_bound(remainder)
    x_scale = np.max(np.abs(x), axis=(1, 2))
    u_scale = np.max(np.abs(u), axis=(1, 2))
    selected_radius = radius[selected]
    # A strict Gram-matrix ball bounds both extreme singular values without
    # decomposing every almost-rigid cell.  This is only a sufficient screen:
    # inconclusive points retain the original SVD and Bernstein path.
    with np.errstate(over="ignore", invalid="ignore"):
        gram = f.swapaxes(1, 2) @ f
        gram_norm = np.max(np.sum(np.abs(gram), axis=2), axis=1)
        delta = np.max(np.sum(np.abs(gram - np.eye(3)), axis=2), axis=1)
        delta += 128 * np.finfo(float).eps * (1 + gram_norm)
    fast = np.isfinite(delta) & (delta < 0.5)
    local = np.zeros(len(f), dtype=bool)
    if np.any(fast):
        lower = np.sqrt(1 - delta[fast])
        upper = np.sqrt(1 + delta[fast])
        guard = 256 * np.finfo(float).eps * (
            x_scale[fast] * (1 + upper) + u_scale[fast])
        total = change[fast] + guard
        local[fast] = np.isfinite(total) & (total < selected_radius[fast] * lower)
    remaining = ~local
    if np.any(remaining):
        singular = np.linalg.svd(f[remaining], compute_uv=False)
        scale = x_scale[remaining] * (1 + singular[:, 0]) + u_scale[remaining]
        guard = 256 * np.finfo(float).eps * scale
        total = change[remaining] + guard
        budget = selected_radius[remaining] * singular[:, -1]
        local[remaining] = np.isfinite(total) & (total < budget)
    admitted[selected] = local
    return admitted
