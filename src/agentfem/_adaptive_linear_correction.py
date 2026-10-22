# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Attempt-local owner: current operator, optional earlier preconditioner.

The caller owns matrices/vectors; this attempt-local owner owns only its KSP.
"""

from dataclasses import dataclass, replace

import numpy as np
from petsc4py import PETSc


@dataclass(frozen=True)
class CorrectionEvidence:
    reused: bool
    rebuilt: bool
    true_residual: float
    threshold: float
    total_iterations: int


class _AdaptiveLinearCorrection:
    """Bounded serial GMRES with verified residual and one fresh-PC retry."""

    def __init__(self, options, *, reuse_iteration_limit=8):
        if options.ksp_type != "gmres" or options.pc_type not in {"lu", "ilu"}:
            raise ValueError("Adaptive preconditioning requires GMRES with LU or ILU.")
        if options.rtol is None or options.atol is None or options.max_it is None:
            raise ValueError("Adaptive preconditioning requires explicit linear tolerances and limit.")
        if (isinstance(reuse_iteration_limit, (bool, np.bool_))
                or not isinstance(reuse_iteration_limit, (int, np.integer))
                or reuse_iteration_limit < 1):
            raise ValueError("reuse_iteration_limit must be a positive integer.")
        # Only this owner may lower the managed policy to an ordinary KSP.
        # Raw adapters reject it instead of silently dropping verification.
        self.options = replace(options, preconditioner_reuse="none")
        self.reuse_iteration_limit = int(reuse_iteration_limit)
        self.ksp = None
        self.layout = None
        self.can_reuse = False
        self.closed = False
        self.last_evidence = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Adaptive linear correction is closed.")
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True
        self.can_reuse = False
        self.layout = None
        if self.ksp is not None:
            try:
                self.ksp.destroy()
            finally:
                self.ksp = None

    @staticmethod
    def _true_residual(matrix, rhs, solution):
        residual = rhs.duplicate()
        try:
            matrix.mult(solution, residual)
            residual.axpy(-1., rhs)
            return float(residual.norm())
        finally:
            residual.destroy()

    def solve(self, matrix, rhs, solution):
        from . import solvers

        if self.closed:
            raise RuntimeError("Adaptive linear correction is closed.")
        if matrix.comm.size != 1:
            raise NotImplementedError("Adaptive linear correction is validated only in serial.")
        layout = (matrix.getSize(), matrix.getType(), matrix.getBlockSize())
        self.last_evidence = None
        if self.ksp is not None and layout != self.layout:
            self.ksp.destroy()
            self.ksp = None
            self.can_reuse = False
        if self.ksp is None:
            self.ksp = solvers.create_ksp(matrix.comm, self.options)
            try:
                self.ksp.setPCSide(PETSc.PC.Side.RIGHT)
                self.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            except BaseException:
                self.ksp.destroy()
                self.ksp = None
                raise
            self.layout = layout
        reused = self.can_reuse
        self.can_reuse = False  # failures must not leave a reusable lease
        rhs_norm = float(rhs.norm())
        if not np.isfinite(rhs_norm):
            raise ValueError("AFM-LINEAR-NONFINITE: nonfinite right-hand-side norm.")
        threshold = max(self.options.atol, self.options.rtol * rhs_norm)
        if not np.isfinite(threshold):
            raise ValueError("AFM-LINEAR-NONFINITE: nonfinite linear residual threshold.")
        total_iterations = 0
        rebuilt = False
        for use_old in ((True, False) if reused else (False,)):
            self.ksp.getPC().setReusePreconditioner(use_old)
            solution.set(0.)
            try:
                info = solvers.solve_matrix_system(
                    matrix, rhs, solution, self.options, ksp=self.ksp,
                    raise_on_failure=False,
                )
            except PETSc.Error:
                if use_old:
                    rebuilt = True
                    continue
                raise
            total_iterations += info.iterations
            actual = self._true_residual(matrix, rhs, solution)
            valid = info.converged and np.isfinite(actual) and actual <= threshold
            if valid:
                self.can_reuse = info.iterations <= self.reuse_iteration_limit
                break
            if use_old:
                rebuilt = True
            elif info.converged:
                # A recursive/preconditioned norm is not a true-residual proof.
                info = replace(info, converged_reason=int(PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN),
                               residual_norm=actual)
        self.last_evidence = CorrectionEvidence(
            reused=reused, rebuilt=rebuilt, true_residual=actual,
            threshold=threshold, total_iterations=total_iterations,
        )
        return replace(info, iterations=total_iterations, residual_norm=actual)
