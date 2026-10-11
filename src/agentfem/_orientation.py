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
        radius = np.max(np.sum(np.abs(f - np.eye(3)), axis=2), axis=1)
    admitted = finite & (radius < 0.5)
    remaining = finite & ~admitted
    if np.any(remaining):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            jacobian = np.linalg.det(f[remaining])
        admitted[remaining] = (jacobian > 0) & np.isfinite(jacobian)
    return admitted
