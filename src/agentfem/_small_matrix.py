# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded-size arithmetic helpers; not finite-element validity certificates."""
import numpy as np


def determinants_3x3(matrices):
    """Batch determinants, retaining LU for cancellation/overflow-sensitive rows.

    Direct six-product arithmetic avoids a tiny LAPACK factorization per row.
    Its absolute product sum screens relative cancellation. This computes J;
    it neither proves orientation exactly nor validates a curved cell map.
    """
    raw = np.asarray(matrices)
    if raw.ndim != 3 or raw.shape[1:] != (3, 3) or np.iscomplexobj(raw):
        raise ValueError("Determinants require real (count,3,3) matrices.")
    a = np.asarray(raw, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Determinants require finite matrices.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        terms = (
            a[:, 0, 0]*a[:, 1, 1]*a[:, 2, 2],
            a[:, 0, 1]*a[:, 1, 2]*a[:, 2, 0],
            a[:, 0, 2]*a[:, 1, 0]*a[:, 2, 1],
            a[:, 0, 2]*a[:, 1, 1]*a[:, 2, 0],
            a[:, 0, 1]*a[:, 1, 0]*a[:, 2, 2],
            a[:, 0, 0]*a[:, 1, 2]*a[:, 2, 1],
        )
        determinant = terms[0]+terms[1]+terms[2]-terms[3]-terms[4]-terms[5]
        absolute = sum(np.abs(t) for t in terms)
        uncertain = (~np.isfinite(absolute) | ~np.isfinite(determinant)
                     | (np.abs(determinant) <= .01*absolute)
                     | (absolute < np.finfo(float).tiny*128))
        if np.any(uncertain):
            determinant[uncertain] = np.linalg.det(a[uncertain])
    return determinant
