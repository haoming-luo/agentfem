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
        preconditioner_form=None,
        options,
        solve_info_factory,
        petsc_options_prefix: str = "agentfem_hybrid_nonlinear_",
    ) -> None:
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
        if options is None or not callable(getattr(options, "petsc_options", None)):
            raise TypeError(
                "PreparedHybridNonlinearProblem requires a normalized nonlinear "
                "solver policy exposing petsc_options()."
            )
        if not callable(solve_info_factory):
            raise TypeError("solve_info_factory must be callable.")
        self.options = options
        self._solve_info_factory = solve_info_factory
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
        self.preconditioner_form = (
            self.jacobian_form
            if preconditioner_form is None
            else preconditioner_form
            if hasattr(preconditioner_form, "_cpp_object")
            else fem.form(preconditioner_form)
        )
        self.local_matrix = fem_petsc.create_matrix(self.jacobian_form)
        self.preconditioner_matrix = (
            self.local_matrix
            if self.preconditioner_form is self.jacobian_form
            else fem_petsc.create_matrix(self.preconditioner_form)
        )
        self._assemble_matrices(solution.x.petsc_vec)
        self.constrained_local_dofs = _owned_constrained_dofs(self.bcs)
        self.actions = tuple(
            fenicsx_tangent_action(contribution, solution)
            for contribution in self.contributions
        )
        self.additive = create_additive_tangent_matrix(
            self.local_matrix,
            self.actions,
            preconditioner_matrix=self.preconditioner_matrix,
            constrained_local_dofs=self.constrained_local_dofs,
        )
        self.residual_vector = fem_petsc.create_vector(solution.function_space)
        self.solver = PETSc.SNES().create(self.local_matrix.comm)
        self.solver.setFunction(self._assemble_residual, self.residual_vector)
        self.solver.setJacobian(
            self._assemble_jacobian,
            self.additive.operator,
            self.preconditioner_matrix,
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
        self.attempt_count = 0
        self.accepted_solve_count = 0
        self.last_solve_info = None
        self.last_linear_solve = None
        self._closed = False

    def _capture_linear_solve(self) -> dict[str, object]:
        """Capture the final Krylov state without owning another PETSc object."""

        ksp = self.solver.getKSP()
        reason = int(ksp.getConvergedReason())
        evidence = {
            "kind": "linear_solve_info",
            "converged": reason > 0,
            "converged_reason": reason,
            "iterations": int(ksp.getIterationNumber()),
            "residual_norm": float(ksp.getResidualNorm()),
        }
        total = getattr(self.solver, "getLinearSolveIterations", None)
        if callable(total):
            evidence["nonlinear_attempt_total_iterations"] = int(total())
        self.last_linear_solve = evidence
        return evidence

    def _assign_state(self, source: PETSc.Vec) -> None:
        source.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        source.copy(self.solution.x.petsc_vec)
        self.solution.x.scatter_forward()

    def _assemble_matrices(self, source: PETSc.Vec) -> None:
        self._assign_state(source)
        self._assemble_matrix(self.local_matrix, self.jacobian_form)
        if self.preconditioner_matrix is not self.local_matrix:
            self._assemble_matrix(
                self.preconditioner_matrix,
                self.preconditioner_form,
            )

    def _assemble_matrix(self, matrix: PETSc.Mat, form) -> None:
        matrix.zeroEntries()
        fem_petsc.assemble_matrix(
            matrix,
            form,
            bcs=self.bcs,
            diag=1.0,
        )
        matrix.assemble()

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
        self._assemble_matrices(source)

    def solve(self):
        """Solve in place and return the field plus structured convergence."""

        if self._closed:
            raise RuntimeError("PreparedHybridNonlinearProblem is closed.")
        accepted = self.solution.x.array.copy()
        self.attempt_count += 1
        try:
            self.solver.solve(None, self.solution.x.petsc_vec)
        except Exception:
            self._capture_linear_solve()
            self.solution.x.array[:] = accepted
            self.solution.x.scatter_forward()
            raise
        linear_info = self._capture_linear_solve()
        self.solution.x.scatter_forward()
        info = self._solve_info_factory(
            converged_reason=int(self.solver.getConvergedReason()),
            iterations=int(self.solver.getIterationNumber()),
            function_norm=float(self.solver.getFunctionNorm()),
        )
        self.last_solve_info = info
        if not info.converged and self.options.error_if_not_converged:
            self.solution.x.array[:] = accepted
            self.solution.x.scatter_forward()
            raise RuntimeError(
                "PETSc SNES did not converge for the hybrid nonlinear problem: "
                f"reason={info.converged_reason}, iterations={info.iterations}, "
                f"function_norm={info.function_norm:.6g}; "
                f"linear_reason={linear_info['converged_reason']}, "
                f"linear_iterations={linear_info['iterations']}, "
                f"linear_residual_norm={linear_info['residual_norm']:.6g}."
            )
        if info.converged:
            self.accepted_solve_count += 1
        return self.solution, info

    def assemble_physical_residual(self) -> PETSc.Vec:
        """Return internal-minus-external residual before strong BC rows.

        The caller owns the returned vector.  Constrained entries are retained
        so a future Procedure can extract reactions without reconstructing or
        double-counting nonlocal contributions.
        """

        if self._closed:
            raise RuntimeError("PreparedHybridNonlinearProblem is closed.")
        target = fem_petsc.create_vector(self.solution.function_space)
        target.set(0.0)
        fem_petsc.assemble_vector(target, self.residual_form)
        target.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        target.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        for contribution in self.contributions:
            residual = contribution.residual(self.solution)
            if not isinstance(residual, PETSc.Vec):
                target.destroy()
                raise TypeError("contribution.residual() must return a PETSc Vec.")
            try:
                if residual.getSizes() != target.getSizes():
                    raise ValueError(
                        "contribution residual does not match the solution space."
                    )
                target.axpy(1.0, residual)
            except Exception:
                target.destroy()
                raise
            finally:
                residual.destroy()
        return target

    def free_residual_norm(self) -> float:
        """Return the norm after suppressing retained constrained reactions."""

        residual = self.assemble_physical_residual()
        try:
            if self.constrained_local_dofs:
                residual.array[np.asarray(self.constrained_local_dofs)] = 0.0
            return float(residual.norm())
        finally:
            residual.destroy()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "prepared_hybrid_nonlinear_problem",
            "status": "internal_promotion_gate",
            "local_operator": "assembled_ufl_jacobian",
            "preconditioner": self.additive.summary()["preconditioner"],
            "nonlocal_contributions": len(self.contributions),
            "constraints": len(self.constrained_local_dofs),
            "tangent": None if self._closed else self.additive.summary(),
            "solver": self.options.summary(),
            "attempt_count": int(self.attempt_count),
            "accepted_solve_count": int(self.accepted_solve_count),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "last_linear_solve": (
                None
                if self.last_linear_solve is None
                else dict(self.last_linear_solve)
            ),
            "failure_state": "restored_to_pre_attempt_solution",
            "physical_residual": "retained_before_strong_boundary_rows",
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
        preconditioner_matrix = self.preconditioner_matrix
        additive = self.additive
        self.solver = None
        self.residual_vector = None
        self.local_matrix = None
        self.preconditioner_matrix = None
        self.additive = None
        self._closed = True
        first_error = None
        resources = [solver, residual, additive, local_matrix]
        if preconditioner_matrix is not local_matrix:
            resources.append(preconditioner_matrix)
        for resource in resources:
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
