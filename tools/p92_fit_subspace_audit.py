"""Exact reachable-coordinate audit of an already completed bounded fit.

This is geometry of the retained box, not statistical identifiability or a
claim that enlarging/changing it would improve prediction. No solve or fit.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_measured_fit import write_new


def replacement_basis_from_svd(values, vectors, *, replacement_indices=(7, 8), retained=10):
    """Prospective geometry only: retain replacements plus strongest other axes.

    A stored full right SVD suffices for J.T@J. No response or left vectors are
    reconstructed, and no optimization is performed by this diagnostic.
    """
    values, vectors = np.asarray(values, dtype=float), np.asarray(vectors, dtype=float)
    if (values.ndim != 1 or vectors.shape != (len(values), len(values))
            or not np.isfinite(values).all() or not np.isfinite(vectors).all()
            or np.any(values < 0) or np.any(np.diff(values) > 0)
            or not np.allclose(vectors@vectors.T, np.eye(len(values)), rtol=0, atol=1e-10)):
        raise ValueError("Require a finite complete right SVD")
    ids = tuple(replacement_indices)
    if (not ids or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < len(values) for i in ids)
            or type(retained) is not int or not len(ids) < retained <= len(values)):
        raise ValueError("Invalid replacement and retained dimensions")
    gram = (vectors.T*values**2)@vectors
    other = [i for i in range(len(values)) if i not in ids]
    _, directions = np.linalg.eigh(gram[np.ix_(other, other)])
    basis = np.zeros((len(values), retained))
    basis[list(ids), np.arange(len(ids))] = 1.
    basis[np.ix_(other, range(len(ids), retained))] = directions[:, ::-1][:, :retained-len(ids)]
    # Compute restricted singular values directly from reconstructed S*V,
    # avoiding cancellation in a second Gram eigendecomposition.
    restricted = np.linalg.svd((values[:, None]*vectors)@basis, compute_uv=False)
    condition = float(restricted[0]/restricted[-1]) if restricted[-1] > 0 else None
    return basis, dict(singular_values=restricted.tolist(), condition=condition,
        passed=bool(restricted[-1] >= 1e-5 and condition is not None and condition <= 1e4),
        scope="Local restricted response sensitivity, not statistical identification or response admission")


def audit(report, *, replacement_indices=(7, 8)):
    basis = np.asarray(report["basis"], dtype=float)
    names = report["coordinates"]
    radius = report["maximum_log_coordinate_norm"]
    u = np.asarray(report["fit"]["coordinates"], dtype=float)
    z = np.asarray(report["fit"]["log_multipliers"], dtype=float)
    if (basis.ndim != 2 or min(basis.shape) < 1 or basis.shape[0] != len(names)
            or basis.shape[1] > basis.shape[0] or len(set(names)) != len(names)
            or not np.isfinite(basis).all() or not np.isfinite(radius) or radius <= 0
            or u.shape != (basis.shape[1],) or z.shape != (basis.shape[0],)
            or not np.isfinite(u).all() or not np.isfinite(z).all()
            or not np.allclose(basis.T@basis, np.eye(basis.shape[1]), rtol=0, atol=1e-10)):
        raise ValueError("Require finite orthonormal fit basis and declared coordinates")
    width = radius/np.sqrt(basis.shape[1])
    if (np.any(abs(u) > width+1e-12) or np.linalg.norm(z) > radius+1e-12
            or not np.allclose(basis@u, z, rtol=0, atol=1e-12)):
        raise ValueError("Fit coordinates do not match the declared bounded subspace")
    ids = tuple(replacement_indices)
    if (not ids or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < basis.shape[0] for i in ids)):
        raise ValueError("Require unique admitted replacement indices")
    rows = basis[list(ids)]
    values, vectors = np.linalg.eigh(rows@rows.T)
    # The sign of an eigenvector has no physical significance. Canonicalize
    # only for reproducible reporting; degeneracy still means non-uniqueness.
    direction = vectors[:, 0]
    if direction[np.argmax(abs(direction))] < 0:
        direction *= -1
    support = width*np.sum(abs(basis), axis=1)
    pair_support = width*np.sum(abs(direction@rows))
    return dict(schema="agentfem.research.fit-subspace-audit.v1",
        model_id=report["model_id"], candidate_count=basis.shape[0],
        retained_count=basis.shape[1], maximum_coordinate_norm=radius,
        retained_box_half_width=float(width),
        coordinates=[dict(name=name, reachable_symmetric_half_width=float(limit),
                          fitted_coordinate=float(value))
                     for name, limit, value in zip(names, support, z, strict=True)],
        replacement=dict(indices=ids, names=[names[i] for i in ids],
            projection_gram_eigenvalues=values.tolist(),
            weakest_coordinate_combination=direction.tolist(),
            weakest_combination_reachable_half_width=float(pair_support),
            weakest_combination_fitted_value=float(direction@z[list(ids)]),
            degenerate=bool(len(values) > 1 and abs(values[1]-values[0]) <= 1e-10)),
        fitted=False, acceptance_thresholds_changed=False,
        engineering_qualified=False,
        scope="Exact linear-image box support; not confidence limits, feasibility or mechanism rejection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.fit.read_bytes()
    result = audit(json.loads(raw))
    result["fit_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)
