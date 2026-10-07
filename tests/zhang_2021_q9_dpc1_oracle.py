"""Independent interpolation oracle for the Zhang Q9/DPC1 element pair.

The oracle uses hand-written tensor-product Q2 Lagrange polynomials and the
mathematical span ``{1, xi, eta}`` of discontinuous P1.  Basix is queried only
as the implementation under test; no AgentFEM assembly or solver code is
reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import basix
import numpy as np


SCHEMA = "agentfem.q9-dpc1-independent-element-oracle.v1"
TOLERANCE = 2.0e-13


def _lagrange_q2_1d(values):
    x = np.asarray(values, dtype=float)
    basis = np.column_stack(
        (2.0 * (x - 0.5) * (x - 1.0), 4.0 * x * (1.0 - x), 2.0 * x * (x - 0.5))
    )
    derivative = np.column_stack((4.0 * x - 3.0, 4.0 - 8.0 * x, 4.0 * x - 1.0))
    return basis, derivative


def _manual_q9(points, node_points):
    points = np.asarray(points, dtype=float)
    node_points = np.asarray(node_points, dtype=float)
    lx, dlx = _lagrange_q2_1d(points[:, 0])
    ly, dly = _lagrange_q2_1d(points[:, 1])
    levels = np.asarray((0.0, 0.5, 1.0))
    node_indices = []
    for point in node_points:
        ix = int(np.argmin(np.abs(levels - point[0])))
        iy = int(np.argmin(np.abs(levels - point[1])))
        if abs(levels[ix] - point[0]) > 1.0e-14 or abs(levels[iy] - point[1]) > 1.0e-14:
            raise ValueError(
                "Q9 interpolation points are not the expected tensor nodes."
            )
        node_indices.append((ix, iy))
    values = np.column_stack(tuple(lx[:, ix] * ly[:, iy] for ix, iy in node_indices))
    derivatives = np.stack(
        (
            np.column_stack(tuple(dlx[:, ix] * ly[:, iy] for ix, iy in node_indices)),
            np.column_stack(tuple(lx[:, ix] * dly[:, iy] for ix, iy in node_indices)),
        ),
        axis=2,
    )
    return values, derivatives


def _tensor_gauss(order: int):
    points, weights = np.polynomial.legendre.leggauss(order)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    xy = np.asarray([(x, y) for x in points for y in points], dtype=float)
    ww = np.asarray([wx * wy for wx in weights for wy in weights], dtype=float)
    return xy, ww


def run_oracle() -> dict[str, object]:
    q9 = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        2,
        lagrange_variant=basix.LagrangeVariant.gll_warped,
    )
    sample = np.asarray(
        ((0.07, 0.11), (0.23, 0.81), (0.51, 0.37), (0.88, 0.73)),
        dtype=float,
    )
    manual, manual_derivative = _manual_q9(sample, q9.points)
    tabulated = q9.tabulate(1, sample)
    q9_value_error = float(np.max(np.abs(manual - tabulated[0, :, :, 0])))
    q9_dx_error = float(
        np.max(np.abs(manual_derivative[:, :, 0] - tabulated[1, :, :, 0]))
    )
    q9_dy_error = float(
        np.max(np.abs(manual_derivative[:, :, 1] - tabulated[2, :, :, 0]))
    )
    nodal, _ = _manual_q9(q9.points, q9.points)
    kronecker_error = float(np.max(np.abs(nodal - np.eye(9))))
    partition_error = float(np.max(np.abs(np.sum(manual, axis=1) - 1.0)))

    dpc1 = basix.create_element(
        basix.ElementFamily.DPC,
        basix.CellType.quadrilateral,
        1,
        dpc_variant=basix.DPCVariant.legendre,
        discontinuous=True,
    )
    dpc_values = dpc1.tabulate(0, sample)[0, :, :, 0]
    polynomial_span = np.column_stack((np.ones(len(sample)), sample))
    coefficients, *_ = np.linalg.lstsq(polynomial_span, dpc_values, rcond=None)
    dpc_span_error = float(np.max(np.abs(polynomial_span @ coefficients - dpc_values)))
    dpc_rank = int(np.linalg.matrix_rank(dpc_values))

    # A genuinely curved biquadratic geometry and an affine physical
    # displacement patch.  Q9 must recover its constant physical gradient
    # despite the non-affine reference-to-physical map.
    xi = q9.points[:, 0]
    eta = q9.points[:, 1]
    geometry_nodes = np.column_stack(
        (
            1.2 + 0.7 * xi + 0.1 * eta + 0.08 * xi * eta + 0.04 * xi**2,
            -0.3 + 0.2 * xi + 0.9 * eta - 0.06 * xi * eta + 0.03 * eta**2,
        )
    )
    expected_gradient = np.asarray(((0.13, -0.27), (0.41, 0.08)), dtype=float)
    displacement_nodes = geometry_nodes @ expected_gradient.T + np.asarray((0.2, -0.1))

    quadrature_points, weights = _tensor_gauss(5)
    shape, derivative = _manual_q9(quadrature_points, q9.points)
    gradient_errors = []
    determinants = []
    for index in range(len(quadrature_points)):
        jacobian = geometry_nodes.T @ derivative[index]
        displacement_derivative = displacement_nodes.T @ derivative[index]
        recovered = displacement_derivative @ np.linalg.inv(jacobian)
        gradient_errors.append(float(np.max(np.abs(recovered - expected_gradient))))
        determinants.append(float(np.linalg.det(jacobian)))
    determinants = np.asarray(determinants)
    if np.min(determinants) <= 0.0:
        raise RuntimeError("Independent curved Q9 oracle has an invalid Jacobian.")
    area = float(np.dot(weights, determinants))

    reference_points, reference_weights = _tensor_gauss(12)
    _, reference_derivative = _manual_q9(reference_points, q9.points)
    reference_determinants = np.asarray(
        [
            np.linalg.det(geometry_nodes.T @ reference_derivative[index])
            for index in range(len(reference_points))
        ]
    )
    reference_area = float(np.dot(reference_weights, reference_determinants))
    area_error = abs(area - reference_area)

    checks = {
        "q9_values_match_hand_polynomials": q9_value_error <= TOLERANCE,
        "q9_dx_matches_hand_polynomials": q9_dx_error <= TOLERANCE,
        "q9_dy_matches_hand_polynomials": q9_dy_error <= TOLERANCE,
        "q9_kronecker_property": kronecker_error <= TOLERANCE,
        "q9_partition_of_unity": partition_error <= TOLERANCE,
        "dpc1_is_complete_linear_polynomial_span": (
            dpc1.dim == 3 and dpc_rank == 3 and dpc_span_error <= TOLERANCE
        ),
        "curved_q9_affine_physical_patch": max(gradient_errors) <= TOLERANCE,
        "curved_q9_area_quadrature": area_error <= TOLERANCE,
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "independent_element_oracle": passed,
        "checks": checks,
        "metrics": {
            "q9_value_max_error": q9_value_error,
            "q9_dx_max_error": q9_dx_error,
            "q9_dy_max_error": q9_dy_error,
            "q9_kronecker_max_error": kronecker_error,
            "q9_partition_of_unity_max_error": partition_error,
            "dpc1_span_max_error": dpc_span_error,
            "dpc1_sample_rank": dpc_rank,
            "curved_patch_gradient_max_error": max(gradient_errors),
            "minimum_curved_jacobian": float(np.min(determinants)),
            "curved_area": area,
            "curved_area_reference": reference_area,
            "curved_area_absolute_error": area_error,
        },
        "tolerance": TOLERANCE,
        "decision_scope": (
            "independent Q9 interpolation, DPC1 polynomial completeness, "
            "curved isoparametric patch, and quadrature; no constitutive or "
            "external Table 5 promotion authority"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    evidence = run_oracle()
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
