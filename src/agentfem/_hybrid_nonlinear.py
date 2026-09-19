# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal runtime for local-UFL plus matrix-free nonlinear equilibrium."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from agentfem.backends import (
    create_additive_tangent_matrix,
    fenicsx_tangent_action,
)


def _owned_constrained_dofs(bcs) -> tuple[int, ...]:
    selected = []
    for bc in bcs:
        dofs, owned = bc.dof_indices()
        selected.extend(int(value) for value in dofs[:owned])
    return tuple(sorted(set(selected)))


class PreparedHybridNonlinearProblem:
    """Internal Newton runtime for additive local and nonlocal operators.

    This class is deliberately not yet part of the public modelling language.
    It proves the algebra, MPI, and boundary contracts required before a
    hybrid ``SolutionProcedure`` can be promoted.
    """

    def __init__(
        self,
        residual_form,
        solution,
        contributions: Iterable[object],
        *,
        bcs=None,
        jacobian_form=None,
        options=None,
        petsc_options_prefix: str = "agentfem_hybrid_nonlinear_",
    ) -> None:
        from agentfem.solvers import NonlinearSolverOptions, NewtonSolverOptions

        if not isinstance(solution, fem.Function):
            raise TypeError("solution must be a dolfinx.fem.Function.")
        if not petsc_options_prefix:
            raise ValueError("petsc_options_prefix must not be empty.")
        self.solution = solution
        self.contributions = tuple(contributions)
        for contribution in self.contributions:
            for method in ("residual", "tangent_action"):
                if not callable(getattr(contribution, method, None)):
                    raise TypeError(
                        f"Every contribution must provide callable {method}()."
                    )
        self.bcs = [] if bcs is None else list(bcs)
        self.options = options or NonlinearSolverOptions(
            ksp_type="gmres",
            pc_type="lu",
        )
        if isinstance(self.options, NewtonSolverOptions):
            self.options = self.options.for_snes()
        if self.contributions and self.options.ksp_type.lower() == "preonly":
            raise ValueError(
                "A hybrid matrix-free Jacobian requires an iterative KSP; "
                "use ksp_type='gmres' or another compatible Krylov method."
            )
        self.residual_form = (
            residual_form
            if hasattr(residual_form, "_cpp_object")
            else fem.form(residual_form)
        )
        if jacobian_form is None:
            trial = ufl.TrialFunction(solution.function_space)
            jacobian_form = ufl.derivative(
                residual_form,
                solution,
                trial,
            )
        self.jacobian_form = (
            jacobian_form
            if hasattr(jacobian_form, "_cpp_object")
            else fem.form(jacobian_form)
        )
        self.local_matrix = fem_petsc.create_matrix(self.jacobian_form)
        self._assemble_local_matrix(solution.x.petsc_vec)
        self.constrained_local_dofs = _owned_constrained_dofs(self.bcs)
        self.actions = tuple(
            fenicsx_tangent_action(contribution, solution)
            for contribution in self.contributions
        )
        self.additive = create_additive_tangent_matrix(
            self.local_matrix,
            self.actions,
            constrained_local_dofs=self.constrained_local_dofs,
        )
        self.residual_vector = fem_petsc.create_vector(solution.function_space)
        self.solver = PETSc.SNES().create(self.local_matrix.comm)
        self.solver.setFunction(self._assemble_residual, self.residual_vector)
        self.solver.setJacobian(
            self._assemble_jacobian,
            self.additive.operator,
            self.local_matrix,
        )
        self.solver.setOptionsPrefix(petsc_options_prefix)
        selected_options = self.options.petsc_options()
        option_database = PETSc.Options()
        option_database.prefixPush(petsc_options_prefix)
        try:
            for key, value in selected_options.items():
                option_database[key] = value
            self.solver.setFromOptions()
        finally:
            for key in selected_options:
                del option_database[key]
            option_database.prefixPop()
        self._closed = False

    def _assign_state(self, source: PETSc.Vec) -> None:
        source.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        source.copy(self.solution.x.petsc_vec)
        self.solution.x.scatter_forward()

    def _assemble_local_matrix(self, source: PETSc.Vec) -> None:
        self._assign_state(source)
        self.local_matrix.zeroEntries()
        fem_petsc.assemble_matrix(
            self.local_matrix,
            self.jacobian_form,
            bcs=self.bcs,
            diag=1.0,
        )
        self.local_matrix.assemble()

    def _assemble_residual(
        self,
        snes: PETSc.SNES,
        source: PETSc.Vec,
        target: PETSc.Vec,
    ) -> None:
        fem_petsc.assemble_residual(
            snes,
            source,
            target,
            u=self.solution,
            residual=self.residual_form,
            jacobian=self.jacobian_form,
            bcs=self.bcs,
        )
        for contribution in self.contributions:
            residual = contribution.residual(self.solution)
            if not isinstance(residual, PETSc.Vec):
                raise TypeError("contribution.residual() must return a PETSc Vec.")
            try:
                if residual.getSizes() != target.getSizes():
                    raise ValueError(
                        "contribution residual does not match the solution space."
                    )
                if self.constrained_local_dofs:
                    residual.array[np.asarray(self.constrained_local_dofs)] = 0.0
                target.axpy(1.0, residual)
            finally:
                residual.destroy()

    def _assemble_jacobian(
        self,
        snes: PETSc.SNES,
        source: PETSc.Vec,
        operator: PETSc.Mat,
        preconditioner: PETSc.Mat,
    ) -> None:
        del snes, operator, preconditioner
        self._assemble_local_matrix(source)

    def solve(self):
        """Solve in place and return the field plus structured convergence."""

        from agentfem.solvers import NonlinearSolveInfo

        if self._closed:
            raise RuntimeError("PreparedHybridNonlinearProblem is closed.")
        self.solver.solve(None, self.solution.x.petsc_vec)
        self.solution.x.scatter_forward()
        info = NonlinearSolveInfo(
            converged_reason=int(self.solver.getConvergedReason()),
            iterations=int(self.solver.getIterationNumber()),
            function_norm=float(self.solver.getFunctionNorm()),
        )
        if not info.converged and self.options.error_if_not_converged:
            raise RuntimeError(
                "PETSc SNES did not converge for the hybrid nonlinear problem: "
                f"reason={info.converged_reason}, iterations={info.iterations}, "
                f"function_norm={info.function_norm:.6g}."
            )
        return self.solution, info

    def summary(self) -> dict[str, object]:
        return {
            "kind": "prepared_hybrid_nonlinear_problem",
            "status": "internal_promotion_gate",
            "local_operator": "assembled_ufl_jacobian",
            "nonlocal_contributions": len(self.contributions),
            "constraints": len(self.constrained_local_dofs),
            "tangent": self.additive.summary(),
            "solver": self.options.summary(),
        }

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release owned PETSc objects; keep the caller-owned field alive."""

        if self._closed:
            return
        solver = self.solver
        residual = self.residual_vector
        local_matrix = self.local_matrix
        additive = self.additive
        self.solver = None
        self.residual_vector = None
        self.local_matrix = None
        self.additive = None
        self._closed = True
        first_error = None
        for resource in (solver, residual, additive, local_matrix):
            try:
                resource.close() if resource is additive else resource.destroy()
            except Exception as exc:  # pragma: no cover - PETSc failure path
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        if self._closed:
            raise RuntimeError("PreparedHybridNonlinearProblem is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


__all__ = ["PreparedHybridNonlinearProblem"]
