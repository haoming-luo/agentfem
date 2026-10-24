# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Research-only P1 contact integral across a *curved* feature boundary.

Two local squared-distance candidates compete on one reference triangle:
distance to the interior of a horizontal edge and distance to a separate
vertex. The equality locus is a parabola. This is a local feature model,
not a signed-distance law for a complete polygon or a production Step.

The reference-area residual and tangent are integrated with split Gauss
rules. The tangent has a bulk term and a negative moving-boundary term.
All in-plane nodal directional differences use independently re-split current
feature regions. Omitting the boundary term is intentionally testable.
"""

from __future__ import annotations

import json

import numpy as np


VERTICES = np.asarray(((1.0, 0.0), (2.0, 0.0), (2.0, 3.0)))
_GAUSS, _WEIGHTS = np.polynomial.legendre.leggauss(20)
_H_EDGE = np.diag((0.0, 1.0))
_H_VERTEX = np.eye(2)


def feature_height(x):
    """Squared edge/vertex distances agree at y = feature_height(x)."""

    return 0.5 * ((np.asarray(x) - 3.0) ** 2 + 1.0)


def quadratic_bernstein_bounds(vertices, evaluate):
    """Bound one quadratic branch difference over an affine triangle.

    Degree-two Bernstein basis functions on a triangle are nonnegative and
    sum to one. Values at the three corners and three edge midpoints recover
    its six coefficients exactly in real arithmetic. The coefficient extrema
    bound the *whole* triangle, including an interior switch missed by all
    three vertices. A release implementation must add interval arithmetic
    and verify the candidate feature's own admissibility inequalities.
    """

    selected = np.asarray(vertices, dtype=float)
    if selected.shape != (3, 2) or not np.all(np.isfinite(selected)):
        raise ValueError("Expected one finite planar triangle.")
    midpoints = 0.5 * np.asarray((
        selected[0] + selected[1],
        selected[1] + selected[2],
        selected[2] + selected[0],
    ))
    values = np.asarray(evaluate(np.vstack((selected, midpoints))), dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected six finite quadratic branch evaluations.")
    edge_coefficients = np.asarray((
        2.0 * values[3] - 0.5 * (values[0] + values[1]),
        2.0 * values[4] - 0.5 * (values[1] + values[2]),
        2.0 * values[5] - 0.5 * (values[2] + values[0]),
    ))
    coefficients = np.concatenate((values[:3], edge_coefficients))
    return float(np.min(coefficients)), float(np.max(coefficients))


def _shapes(x, y) -> np.ndarray:
    return np.asarray((2.0 - x, x - 1.0 - y / 3.0, y / 3.0))


def _rule(start: float, end: float):
    return 0.5 * (end - start) * _GAUSS + 0.5 * (end + start), 0.5 * (end - start) * _WEIGHTS


def _displacements(value) -> np.ndarray:
    selected = np.zeros((3, 2)) if value is None else np.asarray(value, dtype=float)
    if selected.shape == (3,):
        selected = np.column_stack((np.zeros(3), selected))
    if selected.shape != (3, 2) or not np.all(np.isfinite(selected)):
        raise ValueError("Expected three finite in-plane nodal displacements.")
    return selected


def _current(x: float, y: float, displacement: np.ndarray) -> np.ndarray:
    return np.asarray((x, y)) + _shapes(x, y) @ displacement


def _entry(displacement: np.ndarray) -> float:
    """Locate where the moving feature curve enters the sloped triangle side."""

    def upper_difference(x):
        y = 3.0 * (x - 1.0)
        current = _current(x, y, displacement)
        return float(current[1] - feature_height(current[0]))

    left, right = 1.0, 2.0
    if upper_difference(left) >= 0.0 or upper_difference(right) <= 0.0:
        raise ValueError("Feature must enter through the sloped reference side.")
    first = _current(1.0, 0.0, displacement)
    last = _current(2.0, 3.0, displacement)
    slope = last - first
    if min(
        slope[1] - (first[0] - 3.0) * slope[0],
        slope[1] - (last[0] - 3.0) * slope[0],
    ) <= 0.0:
        raise ValueError("The tested edge must cross this conic exactly once.")
    for _ in range(60):
        midpoint = 0.5 * (left + right)
        if upper_difference(midpoint) < 0.0:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def _branch_height(x: float, displacement: np.ndarray) -> float:
    """Re-solve the current conic crossing on one reference vertical line."""

    low, high = 0.0, 3.0 * (x - 1.0)
    current_low = _current(x, low, displacement)
    current_high = _current(x, high, displacement)
    derivative_x = (displacement[2, 0] - displacement[1, 0]) / 3.0
    derivative_y = 1.0 + (displacement[2, 1] - displacement[1, 1]) / 3.0
    if min(
        derivative_y - (current_low[0] - 3.0) * derivative_x,
        derivative_y - (current_high[0] - 3.0) * derivative_x,
    ) <= 0.0 or (
        current_low[1] - feature_height(current_low[0]) >= 0.0
        or current_high[1] - feature_height(current_high[0]) <= 0.0
    ):
        raise ValueError("The tested vertical line must have one bracketed crossing.")
    for _ in range(55):
        middle = 0.5 * (low + high)
        current = _current(x, middle, displacement)
        difference = float(current[1] - feature_height(current[0]))
        if difference < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def split_residual_and_bulk_tangent(nodal_displacement=None):
    """Integrate the piecewise force and within-branch Hessian.

    The current integration regions are recomputed for every trial, but the
    area and shape functions remain those of the original P1 triangle.
    """

    displacement = _displacements(nodal_displacement)
    entry = _entry(displacement)
    residual = np.zeros((3, 2))
    tangent = np.zeros((3, 2, 3, 2))
    potential = 0.0
    for x_start, x_end, has_vertex_region in ((1.0, entry, False), (entry, 2.0, True)):
        x_points, x_weights = _rule(x_start, x_end)
        for x, x_weight in zip(x_points, x_weights, strict=True):
            upper = 3.0 * (x - 1.0)
            regions = ((0.0, upper, False),) if not has_vertex_region else (
                (0.0, _branch_height(x, displacement), False),
                (_branch_height(x, displacement), upper, True),
            )
            for bottom, top, vertex_branch in regions:
                y_points, y_weights = _rule(bottom, top)
                for y, y_weight in zip(y_points, y_weights, strict=True):
                    shape = _shapes(x, y)
                    current_x, current_y = _current(x, y, displacement)
                    if vertex_branch:
                        difference = np.asarray((current_x - 3.0, current_y - 1.0))
                        hessian = _H_VERTEX
                    else:
                        difference = np.asarray((0.0, current_y))
                        hessian = _H_EDGE
                    weight = x_weight * y_weight
                    potential += weight * 0.5 * float(difference @ difference)
                    residual += weight * shape[:, None] * difference[None, :]
                    tangent += weight * np.einsum(
                        "i,j,ab->iajb", shape, shape, hessian,
                    )
    return potential, residual, tangent, entry


def moving_feature_tangent(entry: float, nodal_displacement=None) -> np.ndarray:
    """Distributional derivative of the split residual at the current conic."""

    displacement = _displacements(nodal_displacement)
    moving = np.zeros((3, 2, 3, 2))
    points, weights = _rule(entry, 2.0)
    for x, weight in zip(points, weights, strict=True):
        y = _branch_height(x, displacement)
        shape = _shapes(x, y)
        current_x, _ = _current(x, y, displacement)
        difference = np.asarray((3.0 - current_x, 1.0))
        # phi = e_edge - e_vertex = current_y - feature_height(current_x).
        # The coarea denominator ds/|grad_ref phi| = dx/|d phi/dy|.
        derivative_y = (
            1.0 + (displacement[2, 1] - displacement[1, 1]) / 3.0
            - (current_x - 3.0)
            * (displacement[2, 0] - displacement[1, 0]) / 3.0
        )
        if derivative_y <= 0.0:
            raise ValueError("The tested feature curve must have one vertical crossing.")
        moving -= weight * np.einsum(
            "i,j,a,b->iajb", shape, shape, difference, difference,
        ) / derivative_y
    return moving


def _difference_audit(displacement):
    potential, residual, bulk, entry = split_residual_and_bulk_tangent(displacement)
    moving = moving_feature_tangent(entry, displacement)
    step = 1e-5
    numerical = np.zeros((3, 2, 3, 2))
    numerical_gradient = np.zeros((3, 2))
    for node in range(3):
        for component in range(2):
            direction = np.zeros((3, 2))
            direction[node, component] = step
            plus = split_residual_and_bulk_tangent(displacement + direction)
            minus = split_residual_and_bulk_tangent(displacement - direction)
            numerical[:, :, node, component] = (plus[1] - minus[1]) / (2.0 * step)
            numerical_gradient[node, component] = (plus[0] - minus[0]) / (2.0 * step)
    return {
        "potential": potential,
        "residual": residual,
        "entry": entry,
        "moving": moving,
        "first_variation_error": float(np.max(np.abs(numerical_gradient - residual))),
        "tangent_error_with_boundary": float(np.max(np.abs(numerical - bulk - moving))),
        "tangent_error_without_boundary": float(np.max(np.abs(numerical - bulk))),
    }


def main() -> None:
    base = _difference_audit(np.zeros((3, 2)))
    perturbed = _difference_audit(np.asarray((
        (0.012, -0.006), (-0.008, 0.014), (0.004, -0.01),
    )))
    result = {
        "scope": "research_only_two_quadratic_feature_branches_not_a_closed_tool",
        "feature_curve": "y=((x-3)^2+1)/2",
        "feature_entry_x": base["entry"],
        "straight_chord_midpoint_error": float(
            0.5 * (feature_height(base["entry"]) + feature_height(2.0))
            - feature_height(0.5 * (base["entry"] + 2.0))
        ),
        "potential": base["potential"],
        "reference_p1_nodal_residual": base["residual"].tolist(),
        "full_nodal_potential_gradient_error": base["first_variation_error"],
        "full_nodal_tangent_max_error_with_moving_boundary": base["tangent_error_with_boundary"],
        "full_nodal_tangent_max_error_without_moving_boundary": base["tangent_error_without_boundary"],
        "perturbed_full_nodal_potential_gradient_error": perturbed["first_variation_error"],
        "perturbed_full_nodal_tangent_max_error_with_moving_boundary": perturbed["tangent_error_with_boundary"],
        "perturbed_full_nodal_tangent_max_error_without_moving_boundary": perturbed["tangent_error_without_boundary"],
        "moving_boundary_term_is_negative_semidefinite": bool(
            np.max(np.linalg.eigvalsh(base["moving"].reshape(6, 6))) < 1e-12
        ),
        "public_contact_step": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
