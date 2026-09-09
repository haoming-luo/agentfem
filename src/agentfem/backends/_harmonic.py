"""Private FEniCSx/PETSc execution for real-block harmonic systems.

The scientific operator and procedure layers decide what is assembled and
which frequency is solved. This module owns the reusable DOLFINx
``LinearProblem`` allocation, PETSc solve, and unpreconditioned algebraic
evidence. It is intentionally private while the FEniCSx-first backend seam is
still experimental.
"""

from __future__ import annotations

from dataclasses import dataclass

import dolfinx.fem.petsc as fem_petsc
import numpy as np

from ..solvers import LinearSolveInfo


@dataclass(frozen=True)
class HarmonicLinearSolveEvidence:
    """One direct harmonic linear solve and its physical residual evidence."""

    solve: LinearSolveInfo
    residual_norm: float
    relative_residual_norm: float
    relative_real_block_residual_norm: float
    relative_imaginary_block_residual_norm: float
    input_energy_per_cycle: float

    @property
    def converged(self) -> bool:
        return self.solve.converged

    def equilibrium(self) -> dict[str, float]:
        return {
            "residual_norm": self.residual_norm,
            "relative_residual_norm": self.relative_residual_norm,
            "relative_real_block_residual_norm": (
                self.relative_real_block_residual_norm
            ),
            "relative_imaginary_block_residual_norm": (
                self.relative_imaginary_block_residual_norm
            ),
        }


class PreparedHarmonicLinearProblem:
    """A reusable real-block DOLFINx linear problem.

    Repeated calls reassemble matrix values and the right-hand side into the
    same PETSc objects. The KSP object and allocations are reused; numerical
    factorization or preconditioner reuse is deliberately not claimed.
    """

    def __init__(
        self,
        bilinear_forms,
        linear_forms,
        *,
        solution_real,
        solution_imaginary,
        bcs,
        solver_options,
        petsc_options_prefix: str,
    ) -> None:
        self._problem = fem_petsc.LinearProblem(
            bilinear_forms,
            linear_forms,
            u=[solution_real, solution_imaginary],
            bcs=list(bcs),
            kind="nest",
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=solver_options.petsc_options(),
        )
        self._solve_count = 0

    @property
    def solve_count(self) -> int:
        return self._solve_count

    def solve(self) -> HarmonicLinearSolveEvidence:
        """Reassemble, solve, and return KSP plus ``A*x-b`` evidence."""

        problem = self._problem
        problem.solve()
        self._solve_count += 1
        solver = problem.solver
        solve = LinearSolveInfo(
            converged_reason=int(solver.getConvergedReason()),
            iterations=int(solver.getIterationNumber()),
            residual_norm=float(solver.getResidualNorm()),
        )

        action = problem.b.duplicate()
        residual = problem.b.duplicate()
        try:
            problem.A.mult(problem.x, action)
            action.copy(residual)
            residual.axpy(-1.0, problem.b)
            action_blocks = action.getNestSubVecs()
            residual_blocks = residual.getNestSubVecs()
            rhs_blocks = problem.b.getNestSubVecs()
            solution_blocks = problem.x.getNestSubVecs()
            if not all(
                len(blocks) == 2
                for blocks in (
                    action_blocks,
                    residual_blocks,
                    rhs_blocks,
                    solution_blocks,
                )
            ):
                raise RuntimeError(
                    "Harmonic real-block evidence requires exactly two PETSc blocks."
                )

            action_norm = float(action.norm())
            rhs_norm = float(problem.b.norm())
            residual_norm = float(residual.norm())
            system_scale = max(action_norm, rhs_norm, np.finfo(float).tiny)
            input_energy = float(
                np.pi
                * (
                    rhs_blocks[1].dot(solution_blocks[0])
                    - rhs_blocks[0].dot(solution_blocks[1])
                )
            )
            return HarmonicLinearSolveEvidence(
                solve=solve,
                residual_norm=residual_norm,
                relative_residual_norm=residual_norm / system_scale,
                relative_real_block_residual_norm=(
                    float(residual_blocks[0].norm()) / system_scale
                ),
                relative_imaginary_block_residual_norm=(
                    float(residual_blocks[1].norm()) / system_scale
                ),
                input_energy_per_cycle=input_energy,
            )
        finally:
            residual.destroy()
            action.destroy()

    def summary(self) -> dict[str, object]:
        return {
            "backend": "fenicsx_petsc_real_block",
            "problem_allocation_count": 1,
            "matrix_allocation_count": 1,
            "matrix_values_reassembled_each_solve": True,
            "ksp_object_reused": True,
            "factorization_reuse_claimed": False,
            "solve_count": self.solve_count,
        }


__all__ = ()
