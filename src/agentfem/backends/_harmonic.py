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
from dolfinx import fem
import numpy as np
import ufl
from petsc4py import PETSc

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

    @classmethod
    def from_system(
        cls,
        system,
        *,
        solution_real,
        solution_imaginary,
        bcs,
        solver_options,
        angular_frequency: float,
        load_phase: float,
        petsc_options_prefix: str,
    ) -> "PreparedHarmonicLinearProblem":
        """Lower one operator-level harmonic system to reusable real blocks."""

        system.check()
        domain = solution_real.function_space.mesh
        omega = fem.Constant(domain, PETSc.ScalarType(float(angular_frequency)))
        phase = float(load_phase)
        real_factor = fem.Constant(domain, PETSc.ScalarType(np.cos(phase)))
        imaginary_factor = fem.Constant(domain, PETSc.ScalarType(np.sin(phase)))
        real_space = solution_real.function_space
        imaginary_space = solution_imaginary.function_space
        real_trial = ufl.TrialFunction(real_space)
        real_test = ufl.TestFunction(real_space)
        imaginary_trial = ufl.TrialFunction(imaginary_space)
        imaginary_test = ufl.TestFunction(imaginary_space)

        storage_rr = _bind_bilinear(system.storage, real_trial, real_test)
        storage_ii = _bind_bilinear(
            system.storage,
            imaginary_trial,
            imaginary_test,
        )
        dynamic_rr = storage_rr
        dynamic_ii = storage_ii
        if system.mass is not None:
            dynamic_rr -= omega**2 * _bind_bilinear(
                system.mass,
                real_trial,
                real_test,
            )
            dynamic_ii -= omega**2 * _bind_bilinear(
                system.mass,
                imaginary_trial,
                imaginary_test,
            )

        coupling_ri = 0.0 * _bind_bilinear(
            system.storage,
            imaginary_trial,
            real_test,
        )
        coupling_ir = 0.0 * _bind_bilinear(
            system.storage,
            real_trial,
            imaginary_test,
        )
        if system.loss is not None:
            coupling_ri += _bind_bilinear(
                system.loss,
                imaginary_trial,
                real_test,
            )
            coupling_ir += _bind_bilinear(
                system.loss,
                real_trial,
                imaginary_test,
            )
        if system.damping is not None:
            coupling_ri += omega * _bind_bilinear(
                system.damping,
                imaginary_trial,
                real_test,
            )
            coupling_ir += omega * _bind_bilinear(
                system.damping,
                real_trial,
                imaginary_test,
            )

        real_load = real_factor * _bind_linear(system.force, real_test)
        imaginary_load = imaginary_factor * _bind_linear(
            system.force,
            imaginary_test,
        )
        prepared = cls(
            [[dynamic_rr, -coupling_ri], [coupling_ir, dynamic_ii]],
            [real_load, imaginary_load],
            solution_real=solution_real,
            solution_imaginary=solution_imaginary,
            bcs=bcs,
            solver_options=solver_options,
            petsc_options_prefix=petsc_options_prefix,
        )
        prepared._angular_frequency = omega
        prepared._system = system
        return prepared

    @property
    def angular_frequency(self) -> float | None:
        coefficient = getattr(self, "_angular_frequency", None)
        return None if coefficient is None else float(coefficient.value)

    def set_angular_frequency(self, value: float) -> None:
        """Update the frequency coefficient without reallocating the problem."""

        coefficient = getattr(self, "_angular_frequency", None)
        if coefficient is None:
            raise RuntimeError(
                "This prepared problem was built from fixed legacy block forms."
            )
        selected = float(value)
        if not np.isfinite(selected) or selected < 0.0:
            raise ValueError("Angular frequency must be finite and nonnegative.")
        coefficient.value = PETSc.ScalarType(selected)

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
        summary = {
            "backend": "fenicsx_petsc_real_block",
            "problem_allocation_count": 1,
            "matrix_allocation_count": 1,
            "matrix_values_reassembled_each_solve": True,
            "ksp_object_reused": True,
            "factorization_reuse_claimed": False,
            "solve_count": self.solve_count,
        }
        if self.angular_frequency is not None:
            summary["angular_frequency"] = self.angular_frequency
        return summary


def _expression(operator):
    return operator.expression if hasattr(operator, "expression") else operator


def _bind_bilinear(operator, trial, test):
    expression = _expression(operator)
    arguments = tuple(expression.arguments())
    if len(arguments) != 2:
        raise ValueError("A harmonic matrix operator must be bilinear.")
    replacements = {}
    for argument in arguments:
        replacements[argument] = trial if argument.number() == 1 else test
    return ufl.replace(expression, replacements)


def _bind_linear(operator, test):
    expression = _expression(operator)
    arguments = tuple(expression.arguments())
    if len(arguments) != 1:
        raise ValueError("A harmonic force operator must be linear.")
    return ufl.replace(expression, {arguments[0]: test})


__all__ = ()
