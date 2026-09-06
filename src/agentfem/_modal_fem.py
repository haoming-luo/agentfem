"""PETSc/MPI evidence helpers for structural modal procedures."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def orient_mode_deterministically(mode, free_local, free_global) -> int:
    """Make the largest globally owned component positive.

    This resolves only the sign ambiguity of a real singleton mode.  A basis
    inside a repeated eigenspace remains intentionally free to rotate and must
    be compared through invariant-subspace evidence.
    """

    comm = mode.function_space.mesh.comm
    values = np.real(np.asarray(mode.x.array[free_local]))
    local_maximum = float(np.max(np.abs(values))) if values.size else 0.0
    maximum = float(comm.allreduce(local_maximum, op=MPI.MAX))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("Modal eigenvector has no finite nonzero component.")
    tied = np.abs(values) >= maximum * (1.0 - 64.0 * np.finfo(float).eps)
    sentinel = int(np.iinfo(np.int64).max)
    local_anchor = (
        int(np.min(np.asarray(free_global, dtype=np.int64)[tied]))
        if np.any(tied)
        else sentinel
    )
    anchor = int(comm.allreduce(local_anchor, op=MPI.MIN))
    local_value = float(
        np.sum(values[np.asarray(free_global, dtype=np.int64) == anchor])
    )
    anchor_value = float(comm.allreduce(local_value, op=MPI.SUM))
    if not np.isfinite(anchor_value) or anchor_value == 0.0:
        raise RuntimeError("Modal sign anchor could not be resolved.")
    if anchor_value < 0.0:
        mode.x.array[:] *= -1.0
        mode.x.scatter_forward()
    return anchor


def orthogonality_evidence(stiffness, mass, modes, eigenvalues):
    """Return global mass-orthogonality and stiffness-diagonalization errors."""

    count = len(modes)
    if count == 0:
        return float("inf"), float("inf")
    mass_gram = np.empty((count, count), dtype=float)
    stiffness_gram = np.empty((count, count), dtype=float)
    mass_action = mass.createVecLeft()
    stiffness_action = stiffness.createVecLeft()
    try:
        for column, right in enumerate(modes):
            mass.mult(right.x.petsc_vec, mass_action)
            stiffness.mult(right.x.petsc_vec, stiffness_action)
            for row, left in enumerate(modes):
                mass_gram[row, column] = float(
                    np.real(left.x.petsc_vec.dot(mass_action))
                )
                stiffness_gram[row, column] = float(
                    np.real(left.x.petsc_vec.dot(stiffness_action))
                )
    finally:
        mass_action.destroy()
        stiffness_action.destroy()
    mass_error = float(np.max(np.abs(mass_gram - np.eye(count))))
    expected_stiffness = np.diag(np.asarray(eigenvalues, dtype=float))
    stiffness_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    stiffness_error = float(
        np.max(np.abs(stiffness_gram - expected_stiffness)) / stiffness_scale
    )
    return mass_error, stiffness_error


def require_symmetric_operator(
    matrix,
    *,
    name: str,
    relative_tolerance: float,
) -> float:
    """Verify one real symmetric operator and return its absolute tolerance."""

    scale = float(matrix.norm(PETSc.NormType.INFINITY))
    if not np.isfinite(scale):
        raise ValueError(f"Modal {name} operator contains nonfinite values.")
    absolute_tolerance = max(1.0, scale) * float(relative_tolerance)
    if not matrix.isSymmetric(tol=absolute_tolerance):
        raise ValueError(
            f"Modal {name} operator must be symmetric for the generalized "
            "Hermitian eigenproblem."
        )
    return absolute_tolerance


__all__ = [
    "orient_mode_deterministically",
    "orthogonality_evidence",
    "require_symmetric_operator",
]
