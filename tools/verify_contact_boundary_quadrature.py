# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Research oracle for a clipped P1 triangular frictionless contact face.

This is deliberately outside the production Step. It proves the elementary
affine-gap integration target before introducing changing quadrature-point
identities, material/contact State transfer, or Newton trial partitions.
Only a fixed planar rigid gap, linear slave trace and reference-area penalty
are represented. It is not a curved/sharp-tool or mortar contact algorithm.
"""

from __future__ import annotations

import json

import numpy as np


# Positive degree-two rule on the reference triangle, whose area is 1/2.
_POINTS = np.asarray(((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)))
_WEIGHTS = np.full(3, 1 / 6)


def clipped_affine_gap_integral(nodal_gaps, *, penalty: float = 1.0) -> dict:
    """Integrate 0.5*k*min(g,0)^2 and its nodal derivatives exactly.

    The nodal gaps define an affine function on the unit reference triangle.
    Polygon clipping changes the integration domain, but its boundary term
    vanishes because both the potential and its first derivative contain g=0
    there. Tangents are one-sided at a changing contact topology.
    """

    gaps = np.asarray(nodal_gaps, dtype=float)
    if gaps.shape != (3,) or not np.all(np.isfinite(gaps)):
        raise ValueError("nodal_gaps must contain three finite P1 values")
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be positive and finite")
    vertices = [
        np.asarray(value, dtype=float)
        for value in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    clipped: list[np.ndarray] = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        g_start = float(start @ gaps)
        g_end = float(end @ gaps)
        if (g_start < 0.0) != (g_end < 0.0):
            clipped.append((g_end * start - g_start * end) / (g_end - g_start))
        if g_end < 0.0:
            clipped.append(end)
    energy = 0.0
    area = 0.0
    residual = np.zeros(3)
    tangent = np.zeros((3, 3))
    if len(clipped) >= 3:
        origin = clipped[0][1:]
        for index in range(1, len(clipped) - 1):
            first = clipped[index][1:]
            second = clipped[index + 1][1:]
            jacobian = abs(np.linalg.det(np.column_stack((first - origin, second - origin))))
            if jacobian <= 0.0:
                continue
            points = origin + _POINTS[:, :1] * (first - origin) + _POINTS[:, 1:] * (second - origin)
            shape = np.column_stack((1.0 - points[:, 0] - points[:, 1], points))
            local_gap = shape @ gaps
            weight = _WEIGHTS * jacobian
            area += float(np.sum(weight))
            energy += float(np.sum(0.5 * penalty * weight * local_gap**2))
            residual += penalty * np.einsum("q,q,qi->i", weight, local_gap, shape)
            tangent += penalty * np.einsum("q,qi,qj->ij", weight, shape, shape)
    return {
        "active_reference_area": area,
        "potential": energy,
        "nodal_residual": residual,
        "nodal_tangent": tangent,
    }


def main() -> None:
    import basix

    cases = []
    for width in (0.05, 0.2, 0.5):
        gaps = np.asarray((-width, 1 - width, -width))
        clipped = clipped_affine_gap_integral(gaps)
        exact = width**3 / 6 - width**4 / 24
        uncut = {}
        for degree in (2, 5, 7, 13):
            points, weights = basix.make_quadrature(basix.CellType.triangle, degree)
            energy = float(np.sum(
                weights * 0.5 * np.minimum(points[:, 0] - width, 0.0)**2
            ))
            uncut[str(degree)] = energy
        cases.append({
            "contact_strip_width": width,
            "analytical_potential": exact,
            "clipped_potential": clipped["potential"],
            "clipped_active_area": clipped["active_reference_area"],
            "uncut_gauss_potential": uncut,
        })
    print(json.dumps({
        "scope": "research_only_affine_P1_planar_contact_boundary",
        "production_step_integration": False,
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
