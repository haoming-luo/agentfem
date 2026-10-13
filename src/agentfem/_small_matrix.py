# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded-size arithmetic helpers; not finite-element validity certificates."""
import numpy as np

try:
    from ._hex_native import determinants as _native_determinants
except ImportError:
    _native_determinants = None


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
    if _native_determinants is not None and a.flags.c_contiguous and a.flags.aligned:
        values, flags = _native_determinants(a)
        status = np.frombuffer(flags,dtype=np.uint8)
        if np.any(status==2):
            raise ValueError('Determinants require finite matrices.')
        result = np.frombuffer(values,dtype=float).copy()
        pending = status==0
        if np.any(pending):
            with np.errstate(over='ignore',invalid='ignore',under='ignore'):
                result[pending] = np.linalg.det(a[pending])
        return result
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
        # An intermediate pair product can underflow and subsequently be
        # multiplied by a large third factor. A normal final sum alone does
        # not detect that lost term. Route extreme nonzero inputs to LU.
        tiny_inputs = np.any((a != 0) & (np.abs(a) < np.sqrt(np.finfo(float).tiny)),
                             axis=(1, 2))
        uncertain = (~np.isfinite(absolute) | ~np.isfinite(determinant)
                     | tiny_inputs
                     | (np.abs(determinant) <= .01*absolute)
                     | (absolute < np.finfo(float).tiny*128))
        if np.any(uncertain):
            determinant[uncertain] = np.linalg.det(a[uncertain])
    return determinant
