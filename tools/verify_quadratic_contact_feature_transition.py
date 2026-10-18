# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Research-only P1 contact integral across a *curved* feature boundary.

Two local squared-distance candidates compete on one reference triangle:
distance to the interior of a horizontal edge and distance to a separate
vertex. The equality locus is a parabola. This is a local feature model,
not a signed-distance law for a complete polygon or a production Step.

The reference-area residual and tangent are integrated with split Gauss
rules. The tangent has a bulk term and a negative moving-boundary term.
Vertical nodal directional differences use independently re-split current
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


def _entry(vertical_displacement: np.ndarray) -> float:
    """Locate where the moving feature curve enters the sloped triangle side."""

    def upper_difference(x):
        y = 3.0 * (x - 1.0)
        return y + float(_shapes(x, y) @ vertical_displacement) - feature_height(x)

    left, right = 1.0, 2.0
    if upper_difference(left) >= 0.0 or upper_difference(right) <= 0.0:
        raise ValueError("Feature must enter through the sloped reference side.")
    for _ in range(60):
        midpoint = 0.5 * (left + right)
        if upper_difference(midpoint) < 0.0:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def _branch_height(x: float, vertical_displacement: np.ndarray) -> float:
    # Current y = (1 + (u_C-u_B)/3) * reference_y
    #             + (2-x)*u_A + (x-1)*u_B.
    factor = 1.0 + (vertical_displacement[2] - vertical_displacement[1]) / 3.0
    if factor <= 0.0:
        raise ValueError("The local P1 map must preserve vertical orientation.")
    offset = (2.0 - x) * vertical_displacement[0] + (x - 1.0) * vertical_displacement[1]
    return float((feature_height(x) - offset) / factor)


def split_residual_and_bulk_tangent(vertical_displacement=None):
    """Integrate the piecewise force and within-branch Hessian.

    The current integration regions are recomputed for every trial, but the
    area and shape functions remain those of the original P1 triangle.
    """

    displacement = np.zeros(3) if vertical_displacement is None else np.asarray(
        vertical_displacement, dtype=float,
    )
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise ValueError("Expected three finite vertical nodal displacements.")
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
                    current_y = y + float(shape @ displacement)
                    if vertex_branch:
                        difference = np.asarray((x - 3.0, current_y - 1.0))
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


def moving_feature_tangent(entry: float) -> np.ndarray:
    """Distributional derivative of the split residual at zero displacement."""

    moving = np.zeros((3, 2, 3, 2))
    points, weights = _rule(entry, 2.0)
    for x, weight in zip(points, weights, strict=True):
        shape = _shapes(x, feature_height(x))
        difference = np.asarray((3.0 - x, 1.0))
        # phi = e_edge - e_vertex = y - feature_height(x). Thus the
        # coarea denominator in this (x,y) parameterization is one.
        moving -= weight * np.einsum(
            "i,j,a,b->iajb", shape, shape, difference, difference,
        )
    return moving


def main() -> None:
    potential, residual, bulk, entry = split_residual_and_bulk_tangent()
    moving = moving_feature_tangent(entry)
    step = 1e-5
    numerical_vertical = np.zeros((3, 3))
    for node in range(3):
        direction = np.zeros(3)
        direction[node] = step
        plus = split_residual_and_bulk_tangent(direction)[1]
        minus = split_residual_and_bulk_tangent(-direction)[1]
        numerical_vertical[:, node] = (plus[:, 1] - minus[:, 1]) / (2.0 * step)
    result = {
        "scope": "research_only_two_quadratic_feature_branches_not_a_closed_tool",
        "feature_curve": "y=((x-3)^2+1)/2",
        "feature_entry_x": entry,
        "straight_chord_midpoint_error": float(
            0.5 * (feature_height(entry) + feature_height(2.0))
            - feature_height(0.5 * (entry + 2.0))
        ),
        "potential": potential,
        "reference_p1_nodal_residual": residual.tolist(),
        "vertical_tangent_max_error_with_moving_boundary": float(np.max(
            np.abs(numerical_vertical - (bulk + moving)[:, 1, :, 1]),
        )),
        "vertical_tangent_max_error_without_moving_boundary": float(np.max(
            np.abs(numerical_vertical - bulk[:, 1, :, 1]),
        )),
        "moving_boundary_term_is_negative_semidefinite": bool(
            np.max(np.linalg.eigvalsh(moving.reshape(6, 6))) < 1e-12
        ),
        "public_contact_step": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
