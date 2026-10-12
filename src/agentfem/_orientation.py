# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Orientation admission, not a determinant approximation or cell validity test."""

import numpy as np


def positive_finite_determinants(gradients):
    """Return pointwise admission; use LU outside a strict near-identity ball.

    If ||F-I||_infinity < 1, I+t(F-I) is nonsingular for 0<=t<=1;
    its determinant therefore remains positive. A deliberately smaller radius
    1/2 leaves a large floating-point margin. This avoids an LU factorization
    for orientation-only checks; callers needing J still compute J. It does
    not establish pointwise validity throughout a curved finite element.
    """
    f = np.asarray(gradients, dtype=float)
    if f.ndim != 3 or f.shape[1:] != (3, 3):
        raise ValueError("Orientation requires (points,3,3) gradients.")
    finite = np.isfinite(f).all(axis=(1, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        # Three column views avoid allocating/reducing an entire (n,3,3)
        # temporary. Keep left-associated row sums and the same strict ball.
        radius = np.zeros(len(f))
        for i in range(3):
            row = (np.abs(f[:, i, 0] - (i == 0))
                   + np.abs(f[:, i, 1] - (i == 1))) + np.abs(f[:, i, 2] - (i == 2))
            np.maximum(radius, row, out=radius)
    admitted = finite & (radius < 0.5)
    remaining = finite & ~admitted
    if np.any(remaining):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            jacobian = np.linalg.det(f[remaining])
        admitted[remaining] = (jacobian > 0) & np.isfinite(jacobian)
    return admitted
