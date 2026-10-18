# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Positive reference-triangle quadrature over one quadratic sublevel set.

This is a geometric integration primitive, not a contact law.  It locates
conic tangencies and triangle-edge crossings before integrating each active
interval.  The positive rule approximates a smooth integrand numerically;
it does not prove an integration-error bound or a contact-feature tangent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadraticReferenceLevelSet:
    """q(s,t) = a*t² + (b0+b1*s)*t + c0+c1*s+c2*s²."""

    a: float
    b0: float
    b1: float
    c0: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        values = np.asarray((self.a, self.b0, self.b1, self.c0, self.c1, self.c2))
        if not np.all(np.isfinite(values)) or not np.any(values):
            raise ValueError("Quadratic region needs a finite, nonzero level set.")


def _real_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    """Stable real roots of a degree-at-most-two polynomial."""

    scale = max(abs(a), abs(b), abs(c), np.finfo(float).tiny)
    tol = 64.0 * np.finfo(float).eps * scale
    if abs(a) <= tol:
        return () if abs(b) <= tol else (-c / b,)
    discriminant = b * b - 4.0 * a * c
    discriminant_tol = (
        128.0 * np.finfo(float).eps * max(b * b, abs(4.0 * a * c), np.finfo(float).tiny)
    )
    if discriminant < -discriminant_tol:
        return ()
    if abs(discriminant) <= discriminant_tol:
        return (-b / (2.0 * a),)
    root = np.sqrt(discriminant)
    stable = -0.5 * (b + np.copysign(root, b))
    if stable == 0.0:  # pragma: no cover - positive discriminant guards this
        return (-b / (2.0 * a),)
    return (stable / a, c / stable)


def _unique_unit_interval(values) -> tuple[float, ...]:
    selected = sorted(
        min(1.0, max(0.0, float(value)))
        for value in values
        if -1.0e-13 < value < 1.0 + 1.0e-13
    )
    unique: list[float] = []
    for value in selected:
        if not unique or value - unique[-1] > 1.0e-13:
            unique.append(value)
    return tuple(unique)


def _outer_quadrature(level_set: QuadraticReferenceLevelSet, order: int):
    """Integrate s between conic tangencies and triangle-edge crossings."""

    a, b0, b1 = level_set.a, level_set.b0, level_set.b1
    c0, c1, c2 = level_set.c0, level_set.c1, level_set.c2
    discriminant = (
        b1 * b1 - 4.0 * a * c2,
        2.0 * b0 * b1 - 4.0 * a * c1,
        b0 * b0 - 4.0 * a * c0,
    )
    upper = (a - b1 + c2, -2.0 * a + b1 - b0 + c1, a + b0 + c0)
    tangent_breaks = _unique_unit_interval(_real_roots(*discriminant))
    outer_breaks = _unique_unit_interval(
        (0.0, 1.0, *_real_roots(c2, c1, c0), *_real_roots(*upper), *tangent_breaks)
    )
    nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    for first, last in zip(outer_breaks[:-1], outer_breaks[1:], strict=True):
        width = last - first
        if width <= 1.0e-13:
            continue
        left_tangent = any(abs(first - value) <= 1.0e-13 for value in tangent_breaks)
        right_tangent = any(abs(last - value) <= 1.0e-13 for value in tangent_breaks)
        for node, outer_weight in zip(nodes, gauss_weights, strict=True):
            coordinate = 0.5 * (node + 1.0)
            if left_tangent and right_tangent:
                angle = 0.5 * np.pi * coordinate
                mapped = np.sin(angle) ** 2
                jacobian = np.pi * np.sin(angle) * np.cos(angle)
            elif left_tangent:
                mapped = coordinate**2
                jacobian = 2.0 * coordinate
            elif right_tangent:
                mapped = 1.0 - (1.0 - coordinate) ** 2
                jacobian = 2.0 * (1.0 - coordinate)
            else:
                mapped = coordinate
                jacobian = 1.0
            yield first + width * mapped, 0.5 * width * outer_weight * jacobian


def quadratic_region_triangle_quadrature(
    level_set: QuadraticReferenceLevelSet,
    *,
    negative: bool,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate q<0 or q>0 on ``s,t>=0; s+t<=1`` with positive weights.

    Returns barycentric shape rows and unit-reference-triangle weights.
    The equality boundary has zero area; an identically zero polynomial is
    rejected because it cannot define a unique branch.  This routine does
    not certify branch admissibility or the accuracy of a physical response.
    """

    if not isinstance(level_set, QuadraticReferenceLevelSet):
        raise TypeError("Expected a QuadraticReferenceLevelSet.")
    if type(negative) is not bool:
        raise TypeError("Quadratic region side must be a boolean.")
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("Quadratic region Gauss order must be an integer in [2, 64].")
    a, b0, b1 = level_set.a, level_set.b0, level_set.b1
    c0, c1, c2 = level_set.c0, level_set.c1, level_set.c2
    nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    active_shapes = []
    active_weights = []
    for s, outer_weight in _outer_quadrature(level_set, order):
        upper_t = 1.0 - s
        b = b0 + b1 * s
        c = c0 + c1 * s + c2 * s * s
        inner_breaks = _unique_unit_interval(
            (
                0.0,
                upper_t,
                *(root for root in _real_roots(a, b, c) if 0.0 < root < upper_t),
            )
        )
        for start_t, end_t in zip(inner_breaks[:-1], inner_breaks[1:], strict=True):
            midpoint = 0.5 * (start_t + end_t)
            value = (a * midpoint + b) * midpoint + c
            if (value < 0.0) != negative:
                continue
            half_interval = 0.5 * (end_t - start_t)
            for inner_node, inner_weight in zip(nodes, gauss_weights, strict=True):
                t = midpoint + half_interval * inner_node
                active_shapes.append((1.0 - s - t, s, t))
                active_weights.append(outer_weight * half_interval * inner_weight)
    if not active_shapes:
        return np.empty((0, 3)), np.empty(0)
    shapes = np.asarray(active_shapes, dtype=float)
    weights = np.asarray(active_weights, dtype=float)
    if (
        not np.all(np.isfinite(shapes))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or np.any(shapes < -1.0e-12)
        or not np.allclose(shapes.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
        or float(weights.sum()) > 0.5 + 1.0e-12
    ):
        raise ValueError("Quadratic region produced invalid triangle quadrature.")
    return shapes, weights


def quadratic_level_set_boundary_quadrature(
    level_set: QuadraticReferenceLevelSet,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive coarea rule for ``∫_triangle f(s,t) δ(q(s,t)) ds dt``.

    The regular branch must be representable as one or more graphs ``t(s)``.
    A vertical component or unresolved ``dq/dt`` is refused. These weights
    are *not* physical arc lengths: multiplying by an integrand derivative
    gives the moving-region term for a separately specified piecewise law.
    No feature-specific contact tangent is inferred by this primitive.
    """

    if not isinstance(level_set, QuadraticReferenceLevelSet):
        raise TypeError("Expected a QuadraticReferenceLevelSet.")
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("Quadratic boundary Gauss order must be in [2, 64].")
    a, b0, b1 = level_set.a, level_set.b0, level_set.b1
    c0, c1, c2 = level_set.c0, level_set.c1, level_set.c2
    coefficient_scale = max(abs(value) for value in (a, b0, b1, c0, c1, c2))
    derivative_floor = 128.0 * np.finfo(float).eps * coefficient_scale
    if abs(a) <= derivative_floor:
        if abs(b0) <= derivative_floor and abs(b1) <= derivative_floor:
            raise NotImplementedError(
                "Quadratic boundary is vertical in this reference orientation."
            )
        if abs(b1) > derivative_floor:
            vertical_s = -b0 / b1
            if (
                0.0 <= vertical_s <= 1.0
                and abs(c0 + c1 * vertical_s + c2 * vertical_s**2) <= derivative_floor
            ):
                raise NotImplementedError(
                    "Quadratic boundary contains an unresolved vertical component."
                )
    shapes, weights = [], []
    for s, outer_weight in _outer_quadrature(level_set, order):
        upper_t = 1.0 - s
        b = b0 + b1 * s
        c = c0 + c1 * s + c2 * s * s
        for t in _real_roots(a, b, c):
            if not 0.0 < t < upper_t:
                continue
            derivative = 2.0 * a * t + b
            if abs(derivative) <= derivative_floor:
                raise NotImplementedError(
                    "Quadratic boundary has an unresolved reference tangent."
                )
            shapes.append((1.0 - s - t, s, t))
            weights.append(outer_weight / abs(derivative))
    if not shapes:
        return np.empty((0, 3)), np.empty(0)
    selected_shapes = np.asarray(shapes, dtype=float)
    selected_weights = np.asarray(weights, dtype=float)
    if (
        not np.all(np.isfinite(selected_shapes))
        or not np.all(np.isfinite(selected_weights))
        or np.any(selected_weights <= 0.0)
        or np.any(selected_shapes < -1.0e-12)
        or not np.allclose(
            selected_shapes.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Quadratic boundary produced invalid coarea quadrature.")
    return selected_shapes, selected_weights


__all__ = [
    "QuadraticReferenceLevelSet",
    "quadratic_region_triangle_quadrature",
    "quadratic_level_set_boundary_quadrature",
]
