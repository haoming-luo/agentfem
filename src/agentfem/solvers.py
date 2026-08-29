"""Linear algebra solver helpers for finite-element problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from petsc4py import PETSc

from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

from . import steps as step_controls


@dataclass(frozen=True)
class LinearSolverOptions:
    """PETSc KSP options for a linear solve."""

    ksp_type: str = "preonly"
    pc_type: str = "lu"
    rtol: float | None = None
    atol: float | None = None
    max_it: int | None = None
    factor_solver_type: str | None = None
    error_if_not_converged: bool = True

    def __post_init__(self) -> None:
        if self.rtol is not None and self.rtol <= 0.0:
            raise ValueError("LinearSolverOptions.rtol must be positive.")
        if self.atol is not None and self.atol <= 0.0:
            raise ValueError("LinearSolverOptions.atol must be positive.")
        if self.max_it is not None and self.max_it <= 0:
            raise ValueError("LinearSolverOptions.max_it must be positive.")
        factor_pc_types = {"lu", "cholesky", "ilu", "icc"}
        if (
            self.factor_solver_type is not None
            and self.pc_type.lower() not in factor_pc_types
        ):
            raise ValueError(
                "factor_solver_type requires a factorization preconditioner: "
                f"pc_type must be one of {sorted(factor_pc_types)}."
            )

    def summary(self) -> dict[str, object]:
        """Return an inspectable solver-policy record."""

        return {
            "kind": "linear_solver_options",
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_it": self.max_it,
            "factor_solver_type": self.factor_solver_type,
            "error_if_not_converged": self.error_if_not_converged,
        }


def direct_solver(*, package: str | None = None) -> LinearSolverOptions:
    """Create a direct linear-solver policy without PETSc option names."""

    return LinearSolverOptions(
        ksp_type="preonly",
        pc_type="lu",
        factor_solver_type=package,
    )


@dataclass(frozen=True)
class NonlinearSolverOptions:
    """PETSc SNES/KSP policy for nonlinear finite-element solves."""

    snes_type: str = "newtonls"
    rtol: float = 1.0e-8
    atol: float = 1.0e-10
    max_it: int = 50
    line_search_type: str | None = "bt"
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    factor_solver_type: str | None = None
    error_if_not_converged: bool = True

    def __post_init__(self) -> None:
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("Nonlinear solver tolerances must be positive.")
        if self.max_it <= 0:
            raise ValueError("NonlinearSolverOptions.max_it must be positive.")

    def petsc_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "snes_type": self.snes_type,
            "snes_rtol": self.rtol,
            "snes_atol": self.atol,
            "snes_max_it": self.max_it,
            "snes_error_if_not_converged": self.error_if_not_converged,
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
        }
        if self.line_search_type is not None:
            options["snes_linesearch_type"] = self.line_search_type
        if self.factor_solver_type is not None:
            options["pc_factor_mat_solver_type"] = self.factor_solver_type
        return options

    def summary(self) -> dict[str, object]:
        return {
            "kind": "nonlinear_solver_options",
            "snes_type": self.snes_type,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_it": self.max_it,
            "line_search_type": self.line_search_type,
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "factor_solver_type": self.factor_solver_type,
            "error_if_not_converged": self.error_if_not_converged,
        }


@dataclass(frozen=True)
class NewtonSolverOptions:
    """Backend-neutral Newton policy for nonlinear equilibrium."""

    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-9
    maximum_iterations: int = 30
    line_search: str | None = "backtracking"
    line_search_reduction: float = 0.5
    minimum_step_length: float = 1.0 / 128.0
    linear_solver: LinearSolverOptions = field(default_factory=direct_solver)
    error_if_not_converged: bool = True

    def __post_init__(self) -> None:
        if self.relative_tolerance <= 0.0 or self.absolute_tolerance <= 0.0:
            raise ValueError("Newton tolerances must be positive.")
        if self.maximum_iterations <= 0:
            raise ValueError("Newton maximum_iterations must be positive.")
        selected = (
            None
            if self.line_search is None
            else str(self.line_search).lower().replace("-", "_").strip()
        )
        selected = {
            "bt": "backtracking",
            "backtrack": "backtracking",
            "none": None,
            "off": None,
        }.get(selected, selected)
        if selected not in {None, "backtracking", "basic"}:
            raise ValueError(
                "Newton line_search must be 'backtracking', 'basic', or None."
            )
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("Newton line_search_reduction must lie in (0, 1).")
        if not 0.0 < self.minimum_step_length <= 1.0:
            raise ValueError("Newton minimum_step_length must lie in (0, 1].")
        object.__setattr__(self, "line_search", selected)

    def for_snes(self) -> NonlinearSolverOptions:
        """Translate this policy to the DOLFINx/PETSc SNES path."""

        linear = self.linear_solver
        return NonlinearSolverOptions(
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            max_it=self.maximum_iterations,
            line_search_type={
                None: None,
                "backtracking": "bt",
                "basic": "basic",
            }[self.line_search],
            ksp_type=linear.ksp_type,
            pc_type=linear.pc_type,
            factor_solver_type=linear.factor_solver_type,
            error_if_not_converged=self.error_if_not_converged,
        )

    def for_affine_reduction(self) -> "AffineNewtonOptions":
        """Translate this policy to the affine-reduction Newton path."""

        linear = self.linear_solver
        return AffineNewtonOptions(
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            max_it=self.maximum_iterations,
            line_search_reduction=self.line_search_reduction,
            line_search_minimum=(
                self.minimum_step_length
                if self.line_search == "backtracking"
                else 1.0
            ),
            ksp_type=linear.ksp_type,
            pc_type=linear.pc_type,
            ksp_rtol=1.0e-10 if linear.rtol is None else float(linear.rtol),
            ksp_max_it=1000 if linear.max_it is None else int(linear.max_it),
            factor_solver_type=linear.factor_solver_type,
            error_if_not_converged=self.error_if_not_converged,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "newton_solver",
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_iterations": self.maximum_iterations,
            "line_search": self.line_search,
            "line_search_reduction": self.line_search_reduction,
            "minimum_step_length": self.minimum_step_length,
            "linear_solver": self.linear_solver.summary(),
            "error_if_not_converged": self.error_if_not_converged,
        }


def newton(
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-9,
    maximum_iterations: int = 30,
    line_search: str | None = "backtracking",
    linear_solver: LinearSolverOptions | None = None,
    error_if_not_converged: bool = True,
) -> NewtonSolverOptions:
    """Create one Newton policy for ordinary and affine-constrained steps."""

    return NewtonSolverOptions(
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_iterations=maximum_iterations,
        line_search=line_search,
        linear_solver=(
            direct_solver() if linear_solver is None else linear_solver
        ),
        error_if_not_converged=error_if_not_converged,
    )


@dataclass(frozen=True)
class NonlinearSolveInfo:
    """Convergence evidence returned by a PETSc SNES solve."""

    converged_reason: int
    iterations: int
    function_norm: float

    @property
    def converged(self) -> bool:
        return self.converged_reason > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "nonlinear_solve_info",
            "converged": self.converged,
            "converged_reason": self.converged_reason,
            "iterations": self.iterations,
            "function_norm": self.function_norm,
        }


@dataclass(frozen=True)
class AffineNewtonOptions:
    """Newton policy for an affine-reduced nonlinear equilibrium path."""

    rtol: float = 1.0e-8
    atol: float = 1.0e-9
    max_it: int = 30
    line_search_reduction: float = 0.5
    line_search_minimum: float = 1.0 / 128.0
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    ksp_rtol: float = 1.0e-10
    ksp_max_it: int = 1000
    factor_solver_type: str | None = "mumps"
    error_if_not_converged: bool = True

    def __post_init__(self) -> None:
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("Affine Newton tolerances must be positive.")
        if self.max_it <= 0:
            raise ValueError("AffineNewtonOptions.max_it must be positive.")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("line_search_reduction must lie between zero and one.")
        if not 0.0 < self.line_search_minimum <= 1.0:
            raise ValueError("line_search_minimum must lie in (0, 1].")
        if self.ksp_rtol <= 0.0 or self.ksp_max_it <= 0:
            raise ValueError("Affine Newton KSP tolerances must be positive.")

    @property
    def linear_options(self) -> LinearSolverOptions:
        return LinearSolverOptions(
            ksp_type=self.ksp_type,
            pc_type=self.pc_type,
            rtol=self.ksp_rtol,
            max_it=self.ksp_max_it,
            factor_solver_type=self.factor_solver_type,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "affine_newton_options",
            "rtol": self.rtol,
            "atol": self.atol,
            "max_it": self.max_it,
            "line_search_reduction": self.line_search_reduction,
            "line_search_minimum": self.line_search_minimum,
            "linear_solver": self.linear_options.summary(),
            "error_if_not_converged": self.error_if_not_converged,
        }


@dataclass(frozen=True)
class AffineLoadIncrementInfo:
    """Convergence evidence for one macroscopic load increment."""

    load_factor: float
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    accepted_step_lengths: tuple[float, ...]
    reduced_dofs: int
    equation_mismatch: float
    increment: int = 0
    attempt: int = 1
    start_load_factor: float = 0.0
    message: str = ""
    checks: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "increment": self.increment,
            "attempt": self.attempt,
            "start_load_factor": self.start_load_factor,
            "load_factor": self.load_factor,
            "increment_size": self.load_factor - self.start_load_factor,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": _finite_or_none(self.initial_residual_norm),
            "residual_norm": _finite_or_none(self.residual_norm),
            "accepted_step_lengths": self.accepted_step_lengths,
            "reduced_dofs": self.reduced_dofs,
            "equation_mismatch": self.equation_mismatch,
            "message": self.message,
            "checks": dict(self.checks),
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "AffineLoadIncrementInfo":
        """Restore one accepted or attempted increment from checkpoint evidence."""

        return cls(
            load_factor=float(record["load_factor"]),
            converged=bool(record["converged"]),
            iterations=int(record["iterations"]),
            initial_residual_norm=_none_as_nan(record.get("initial_residual_norm")),
            residual_norm=_none_as_nan(record.get("residual_norm")),
            accepted_step_lengths=tuple(
                float(value) for value in record.get("accepted_step_lengths", ())
            ),
            reduced_dofs=int(record["reduced_dofs"]),
            equation_mismatch=float(record["equation_mismatch"]),
            increment=int(record.get("increment", 0)),
            attempt=int(record.get("attempt", 1)),
            start_load_factor=float(record.get("start_load_factor", 0.0)),
            message=str(record.get("message", "")),
            checks=dict(record.get("checks", {})),
        )


@dataclass(frozen=True)
class AffineLoadPathInfo:
    """Convergence evidence for an incrementally applied affine constraint."""

    increments: tuple[AffineLoadIncrementInfo, ...]
    attempts: tuple[AffineLoadIncrementInfo, ...] = ()
    incrementation: object | None = None
    target_factor: float = 1.0
    next_increment_size: float | None = None

    @property
    def converged(self) -> bool:
        return (
            bool(self.increments)
            and all(step.converged for step in self.increments)
            and abs(self.increments[-1].load_factor - self.target_factor) <= 1.0e-12
        )

    @property
    def completed_step(self) -> bool:
        """Whether the complete normalized path, rather than a pause, finished."""

        return self.converged and abs(self.target_factor - 1.0) <= 1.0e-12

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "affine_nonlinear_load_path",
            "converged": self.converged,
            "completed_step": self.completed_step,
            "target_factor": self.target_factor,
            "accepted_factor": (
                0.0 if not self.increments else self.increments[-1].load_factor
            ),
            "next_increment_size": self.next_increment_size,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "increments": [step.as_dict() for step in self.increments],
            "attempts": [step.as_dict() for step in self.attempts],
        }


@dataclass(frozen=True)
class SolveEvent:
    """One structured event emitted by an analysis procedure.

    Events are the common evidence stream for console progress, status files,
    result manifests, and agent monitoring.  ``display`` controls only the
    human progress view; recorders must retain the event regardless.
    """

    kind: str
    step_name: str
    step_number: int = 1
    increment: int = 0
    attempt: int = 0
    start_factor: float = 0.0
    target_factor: float = 0.0
    iteration: int = 0
    residual_norm: float | None = None
    step_length: float | None = None
    next_increment: float | None = None
    incrementation: str = ""
    message: str = ""
    time: float | None = None
    total_increments: int = 0
    display: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe, stable execution-event record."""

        def finite_or_none(value):
            if value is None:
                return None
            selected = float(value)
            return selected if isfinite(selected) else None

        return {
            "kind": self.kind,
            "step_name": self.step_name,
            "step_number": int(self.step_number),
            "increment": int(self.increment),
            "attempt": int(self.attempt),
            "start_factor": float(self.start_factor),
            "target_factor": float(self.target_factor),
            "iteration": int(self.iteration),
            "residual_norm": finite_or_none(self.residual_norm),
            "step_length": finite_or_none(self.step_length),
            "next_increment": finite_or_none(self.next_increment),
            "incrementation": self.incrementation,
            "message": self.message,
            "time": finite_or_none(self.time),
            "total_increments": int(self.total_increments),
            "display": bool(self.display),
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "SolveEvent":
        """Restore a recorded event from a result or checkpoint manifest."""

        return cls(
            kind=str(record["kind"]),
            step_name=str(record["step_name"]),
            step_number=int(record.get("step_number", 1)),
            increment=int(record.get("increment", 0)),
            attempt=int(record.get("attempt", 0)),
            start_factor=float(record.get("start_factor", 0.0)),
            target_factor=float(record.get("target_factor", 0.0)),
            iteration=int(record.get("iteration", 0)),
            residual_norm=record.get("residual_norm"),
            step_length=record.get("step_length"),
            next_increment=record.get("next_increment"),
            incrementation=str(record.get("incrementation", "")),
            message=str(record.get("message", "")),
            time=record.get("time"),
            total_increments=int(record.get("total_increments", 0)),
            display=bool(record.get("display", True)),
        )


def create_ksp(comm, options: LinearSolverOptions | None = None):
    """Create and configure a PETSc KSP object."""

    options = options or LinearSolverOptions()
    ksp = PETSc.KSP().create(comm)
    ksp.setType(options.ksp_type)
    pc = ksp.getPC()
    pc.setType(options.pc_type)
    if options.factor_solver_type is not None:
        pc.setFactorSolverType(options.factor_solver_type)
    if options.rtol is not None or options.atol is not None or options.max_it is not None:
        ksp.setTolerances(
            rtol=options.rtol,
            atol=options.atol,
            max_it=options.max_it,
        )
    return ksp


@dataclass(frozen=True)
class LinearSolveInfo:
    """PETSc KSP convergence evidence for one linear system solve."""

    converged_reason: int
    iterations: int
    residual_norm: float

    @property
    def converged(self) -> bool:
        return self.converged_reason > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "linear_solve_info",
            "converged": self.converged,
            "converged_reason": self.converged_reason,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
        }


class PreparedLinearProblem:
    """A linear problem whose constant matrix and KSP are assembled once.

    This lifecycle is intended for first-order transient and parameter-update
    loops where the bilinear form, constrained dof set, and solver policy stay
    fixed while coefficients in the right-hand side change. Boundary values
    may change, but their constrained dof locations must remain unchanged.
    """

    def __init__(
        self,
        bilinear_form,
        linear_form,
        solution,
        *,
        bcs=None,
        options: LinearSolverOptions | None = None,
    ):
        self.bilinear_form = (
            bilinear_form
            if hasattr(bilinear_form, "_cpp_object")
            else fem.form(bilinear_form)
        )
        self.linear_form = (
            linear_form if hasattr(linear_form, "_cpp_object") else fem.form(linear_form)
        )
        self.solution = solution
        self.bcs = [] if bcs is None else list(bcs)
        self.options = options or LinearSolverOptions()
        self.matrix = fem_petsc.assemble_matrix(self.bilinear_form, bcs=self.bcs)
        self.matrix.assemble()
        self.ksp = create_ksp(self.matrix.comm, self.options)
        self.ksp.setOperators(self.matrix)
        self.last_solve_info: LinearSolveInfo | None = None
        self.solve_count = 0
        self._closed = False

    def solve(self):
        """Assemble the current right-hand side and reuse the prepared solve."""

        if self._closed:
            raise RuntimeError("PreparedLinearProblem is closed.")
        vector = fem_petsc.assemble_vector(self.linear_form)
        fem_petsc.apply_lifting(vector, [self.bilinear_form], [self.bcs])
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        fem_petsc.set_bc(vector, self.bcs)
        self.ksp.solve(vector, self.solution.x.petsc_vec)
        self.solution.x.scatter_forward()
        info = LinearSolveInfo(
            converged_reason=int(self.ksp.getConvergedReason()),
            iterations=int(self.ksp.getIterationNumber()),
            residual_norm=float(self.ksp.getResidualNorm()),
        )
        vector.destroy()
        self.last_solve_info = info
        self.solve_count += 1
        if not info.converged and self.options.error_if_not_converged:
            _raise_linear_failure(info)
        return self.solution

    def summary(self) -> dict[str, object]:
        return {
            "kind": "prepared_linear_problem",
            "matrix_reused": True,
            "solve_count": int(self.solve_count),
            "solver": self.options.summary(),
            "last_solve": (
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
        }

    def close(self) -> None:
        """Release PETSc resources deterministically."""

        if self._closed:
            return
        self.ksp.destroy()
        self.matrix.destroy()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def prepare_linear_problem(
    bilinear_form,
    linear_form,
    solution,
    *,
    bcs=None,
    options: LinearSolverOptions | None = None,
) -> PreparedLinearProblem:
    """Prepare one constant linear operator for repeated right-hand sides."""

    return PreparedLinearProblem(
        bilinear_form,
        linear_form,
        solution,
        bcs=bcs,
        options=options,
    )


def solve_matrix_system(
    A,
    b,
    x,
    options: LinearSolverOptions | None = None,
    *,
    raise_on_failure: bool | None = None,
) -> LinearSolveInfo:
    """Solve ``A x = b`` and return explicit PETSc convergence evidence."""

    selected = options or LinearSolverOptions()
    ksp = create_ksp(A.comm, selected)
    ksp.setOperators(A)
    ksp.solve(b, x)
    info = LinearSolveInfo(
        converged_reason=int(ksp.getConvergedReason()),
        iterations=int(ksp.getIterationNumber()),
        residual_norm=float(ksp.getResidualNorm()),
    )
    ksp.destroy()
    should_raise = (
        selected.error_if_not_converged
        if raise_on_failure is None
        else bool(raise_on_failure)
    )
    if not info.converged and should_raise:
        _raise_linear_failure(info)
    return info


def solve_linear_problem(
    bilinear_form,
    linear_form,
    solution,
    *,
    bcs=None,
    options: LinearSolverOptions | None = None,
    return_info: bool = False,
):
    """Assemble and solve a standard linear variational problem.

    Parameters
    ----------
    bilinear_form:
        Compiled form for the left-hand side.
    linear_form:
        Compiled form for the right-hand side.
    solution:
        DOLFINx function storing the solution.
    bcs:
        Optional list of strong Dirichlet boundary conditions.
    options:
        PETSc KSP configuration.
    """

    bcs = [] if bcs is None else list(bcs)
    A = fem_petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    b = fem_petsc.assemble_vector(linear_form)
    fem_petsc.apply_lifting(b, [bilinear_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    selected = options or LinearSolverOptions()
    info = solve_matrix_system(
        A,
        b,
        solution.x.petsc_vec,
        selected,
        raise_on_failure=False,
    )
    solution.x.scatter_forward()

    b.destroy()
    A.destroy()
    if not info.converged and selected.error_if_not_converged:
        _raise_linear_failure(info)
    return (solution, info) if return_info else solution


def _raise_linear_failure(info: LinearSolveInfo) -> None:
    raise RuntimeError(
        "PETSc KSP did not converge: "
        f"reason={info.converged_reason}, iterations={info.iterations}, "
        f"residual_norm={info.residual_norm:.6g}."
    )


def solve_nonlinear_problem(
    residual_form,
    solution,
    *,
    bcs=None,
    jacobian_form=None,
    options: NonlinearSolverOptions | NewtonSolverOptions | None = None,
    petsc_options_prefix: str = "agentfem_nonlinear_",
) -> tuple[object, NonlinearSolveInfo]:
    """Solve ``R(u; v) = 0`` with the current DOLFINx PETSc/SNES interface."""

    from dolfinx.fem.petsc import NonlinearProblem

    selected = options or NonlinearSolverOptions()
    if isinstance(selected, NewtonSolverOptions):
        selected = selected.for_snes()
    problem = NonlinearProblem(
        residual_form,
        solution,
        bcs=[] if bcs is None else list(bcs),
        J=jacobian_form,
        petsc_options_prefix=petsc_options_prefix,
        petsc_options=selected.petsc_options(),
    )
    solved = problem.solve()
    solved.x.scatter_forward()
    solver = problem.solver
    info = NonlinearSolveInfo(
        converged_reason=int(solver.getConvergedReason()),
        iterations=int(solver.getIterationNumber()),
        function_norm=float(solver.getFunctionNorm()),
    )
    return solved, info


def _validate_affine_state_transaction(transaction) -> None:
    if transaction is None:
        return
    required = (
        "refresh_trial",
        "commit_increment",
        "rollback_increment",
        "snapshot_accepted_boundary",
        "restore_accepted_boundary",
    )
    missing = [name for name in required if not callable(getattr(transaction, name, None))]
    if missing:
        raise TypeError(
            "Affine state_transaction must provide callable "
            + ", ".join(f"{name}()" for name in required)
            + f"; missing {', '.join(missing)}."
        )


def _refresh_affine_trial_state(
    transaction,
    *,
    start_factor: float,
    target_factor: float,
):
    if transaction is None:
        return None
    return transaction.refresh_trial(
        start_factor=float(start_factor),
        target_factor=float(target_factor),
    )


def _try_refresh_affine_trial_state(
    transaction,
    *,
    start_factor: float,
    target_factor: float,
) -> tuple[bool, str]:
    """Refresh one constitutive trial without bypassing cutback semantics."""

    if transaction is None:
        return True, ""
    try:
        _refresh_affine_trial_state(
            transaction,
            start_factor=start_factor,
            target_factor=target_factor,
        )
    except Exception as exc:
        return (
            False,
            "constitutive update failed: "
            f"{type(exc).__name__}: {exc}",
        )
    return True, ""


def _commit_affine_trial_state(
    transaction,
    *,
    start_factor: float,
    target_factor: float,
) -> None:
    if transaction is None:
        return
    transaction.commit_increment(
        start_factor=float(start_factor),
        target_factor=float(target_factor),
    )


def _rollback_affine_trial_state(
    transaction,
    *,
    accepted_factor: float,
) -> None:
    if transaction is None:
        return
    transaction.rollback_increment(accepted_factor=float(accepted_factor))


def _snapshot_affine_accepted_state(transaction):
    """Capture the last committed boundary before a provisional commit."""

    if transaction is None:
        return None
    return transaction.snapshot_accepted_boundary()


def _restore_affine_accepted_state(transaction, snapshot) -> None:
    """Restore a committed boundary after post-commit lifecycle failure."""

    if transaction is None:
        return
    transaction.restore_accepted_boundary(snapshot)


def _rollback_post_commit_failure(
    *,
    function,
    nodal_state,
    transaction,
    transaction_state,
    accepted_history,
    accepted_history_state,
    attempted_history,
    attempted_history_state,
    on_acceptance_failure,
) -> None:
    """Undo every mutation made after one nonlinear increment converged."""

    function.x.array[:] = nodal_state
    function.x.scatter_forward()
    _restore_affine_accepted_state(transaction, transaction_state)
    accepted_history[:] = accepted_history_state
    attempted_history[:] = attempted_history_state
    if on_acceptance_failure is not None:
        on_acceptance_failure()


def _run_affine_acceptance_stage(
    function,
    *,
    stage: str,
    callback,
    args=(),
    kwargs=None,
) -> None:
    """Run one post-convergence stage with rank-consistent failure semantics.

    Accepted-boundary callbacks may write output, update histories, or create a
    checkpoint.  A filesystem or observer failure can therefore occur on only
    one MPI rank.  Letting the remaining ranks continue would split the
    accepted lifecycle and can deadlock the next constitutive collective.  The
    callback itself is executed on every rank, then its outcome is gathered
    before any later acceptance stage begins.

    Callback implementations must keep their own internal collective ordering
    deterministic.  This guard covers the common and otherwise dangerous case
    in which a local operation fails after the shared snapshot/assembly work.
    """

    comm = function.function_space.mesh.comm
    local_error = None
    try:
        callback(*args, **({} if kwargs is None else kwargs))
    except BaseException as exc:
        if comm.size == 1:
            raise
        local_error = f"{type(exc).__name__}: {exc}"
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        rank = next(index for index, error in enumerate(errors) if error is not None)
        raise RuntimeError(
            f"Rank {rank}: affine accepted-boundary {stage} failed: "
            f"{errors[rank]}"
        )


def solve_affine_nonlinear_path(
    residual_form,
    jacobian_form,
    solution,
    constraint,
    *,
    load_factors=None,
    incrementation=None,
    output_factors=(),
    options: AffineNewtonOptions | NewtonSolverOptions | None = None,
    on_increment=None,
    on_accepted_boundary=None,
    on_acceptance_failure=None,
    acceptance_check=None,
    state_transaction=None,
    stop_factor: float = 1.0,
    accepted_history=(),
    attempted_history=(),
    next_increment_size: float | None = None,
    reporter=None,
    step_name: str = "affine_nonlinear",
    step_number: int = 1,
) -> tuple[object, AffineLoadPathInfo]:
    """Solve a nonlinear path under ``u = T q + u_bar`` constraints.

    The implementation assembles the ordinary full-space DOLFINx residual and
    tangent, then applies exact variational reduction.  This keeps constitutive
    forms independent of the constraint backend.  A stateful constitutive
    provider may supply ``state_transaction`` with ``refresh_trial()``,
    ``commit_increment()`` and ``rollback_increment()`` methods. Trial fields
    are refreshed after every reconstructed displacement, including line-search
    probes, before residual or tangent assembly.
    """

    selected = options or AffineNewtonOptions()
    if isinstance(selected, NewtonSolverOptions):
        selected = selected.for_affine_reduction()
    _validate_affine_state_transaction(state_transaction)
    control = step_controls.normalize(
        incrementation,
        load_factors=load_factors,
    )
    selected_stop = float(stop_factor)
    if not np.isfinite(selected_stop) or not 0.0 < selected_stop <= 1.0:
        raise ValueError("Affine stop_factor must be finite and in (0, 1].")
    history = list(accepted_history)
    attempt_history = list(attempted_history)
    accepted_factor = (
        float(history[-1].load_factor)
        if history
        else float(getattr(state_transaction, "accepted_factor", 0.0))
    )
    if accepted_factor >= selected_stop - 1.0e-12:
        raise ValueError(
            "Affine stop_factor must exceed the currently accepted load factor."
        )
    if history and any(not item.converged for item in history):
        raise ValueError("Affine accepted_history contains a rejected increment.")
    if isinstance(control, step_controls.FixedIncrementation):
        _validate_fixed_affine_resume(control, accepted_factor, selected_stop)
    constraint_factors = (
        tuple(constraint.required_load_factors())
        if hasattr(constraint, "required_load_factors")
        else ()
    )
    required_factors = _normalized_output_factors(
        tuple(sorted(set((*tuple(output_factors), *constraint_factors))))
    )
    if isinstance(control, step_controls.FixedIncrementation):
        missing_path_factors = tuple(
            factor
            for factor in constraint_factors
            if not any(
                abs(float(factor) - float(declared)) <= 1.0e-12
                for declared in control.load_factors
            )
        )
        if missing_path_factors:
            raise ValueError(
                "Fixed affine incrementation must include every physical "
                "deformation-path knot; missing "
                f"{missing_path_factors}."
            )

    residual = fem.form(residual_form)
    jacobian = fem.form(jacobian_form)
    function = solution.value if hasattr(solution, "value") else solution
    space = function.function_space
    block_size = int(space.dofmap.index_map_bs)
    coordinates = (
        None
        if hasattr(constraint, "initial_reduced_values")
        else np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    )
    if space.mesh.comm.size > 1:
        return _solve_distributed_affine_nonlinear_path(
            residual,
            jacobian,
            function,
            constraint,
            control=control,
            required_factors=required_factors,
            options=selected,
            on_increment=on_increment,
            on_accepted_boundary=on_accepted_boundary,
            on_acceptance_failure=on_acceptance_failure,
            acceptance_check=acceptance_check,
            state_transaction=state_transaction,
            stop_factor=selected_stop,
            accepted_history=history,
            attempted_history=attempt_history,
            next_increment_size=next_increment_size,
            reporter=reporter,
            step_name=step_name,
            step_number=step_number,
        )
    previous_reduced: np.ndarray | None = None
    previous_affine: np.ndarray | None = None
    total_attempts = len(attempt_history)
    cutbacks = 0
    proposed_size = (
        (
            control.initial
            if next_increment_size is None
            else float(next_increment_size)
        )
        if isinstance(control, step_controls.AutomaticIncrementation)
        else _next_fixed_affine_factor(control, accepted_factor)
    )
    if accepted_factor > 1.0e-12:
        start_reduction = constraint.reduction(accepted_factor)
        start_F = constraint.deformation_gradient_at(accepted_factor)
        previous_affine = constraint.initial_reduced_values(
            start_reduction,
            start_F,
        )
        previous_reduced = function.x.array[
            start_reduction.independent_full_dofs
        ].copy()
        reconstructed = start_reduction.reconstruct(previous_reduced)
        if reconstructed.shape != function.x.array.shape or not np.allclose(
            reconstructed,
            function.x.array,
            rtol=0.0,
            atol=max(1.0e-11, 10.0 * float(getattr(constraint, "tolerance", 1.0e-12))),
        ):
            raise ValueError(
                "Restored affine displacement does not satisfy the checkpointed "
                "constraint and prescribed-control state."
            )
    _emit(
        reporter,
        SolveEvent(
            "step_resumed" if accepted_factor > 1.0e-12 else "step_started",
            step_name,
            step_number=step_number,
            incrementation=control.summary()["kind"],
        ),
    )

    while accepted_factor < selected_stop - 1.0e-12:
        increment_number = len(history) + 1
        if isinstance(control, step_controls.AutomaticIncrementation):
            if len(history) >= control.max_increments:
                message = (
                    f"maximum accepted increments ({control.max_increments}) "
                    "reached before load factor 1.0"
                )
                return _failed_affine_path(
                    function,
                    history,
                    attempt_history,
                    control,
                    selected,
                    reporter,
                    step_name,
                    step_number,
                    increment_number,
                    total_attempts,
                    accepted_factor,
                    message,
                )
            factor = min(selected_stop, accepted_factor + proposed_size)
            next_output = _next_output_factor(required_factors, accepted_factor)
            if next_output is not None:
                factor = min(factor, next_output)
        else:
            factor = _next_fixed_affine_factor(control, accepted_factor)
        attempt_number = cutbacks + 1
        total_attempts += 1
        _emit(
            reporter,
            SolveEvent(
                "increment_started",
                step_name,
                step_number=step_number,
                increment=increment_number,
                attempt=attempt_number,
                start_factor=accepted_factor,
                target_factor=factor,
            ),
        )
        rollback = function.x.array.copy()
        accepted_transaction_state = _snapshot_affine_accepted_state(
            state_transaction
        )
        accepted_history_state = list(history)
        attempted_history_state = list(attempt_history)
        reduction = constraint.reduction(factor)
        T = reduction.matrix(function.function_space.mesh.comm)
        current_F = constraint.deformation_gradient_at(factor)
        if hasattr(constraint, "initial_reduced_values"):
            current_affine = constraint.initial_reduced_values(
                reduction,
                current_F,
            )
        else:
            current_affine = reduction.initial_reduced_values(
                coordinates,
                current_F,
                block_size=block_size,
            )
        if previous_reduced is None:
            reduced_values = current_affine.copy()
        else:
            if previous_reduced.size != reduction.reduced_size:
                raise RuntimeError("Affine reduction topology changed between load increments.")
            reduced_values = previous_reduced + current_affine - previous_affine
        _assign_reconstructed(function, reduction, reduced_values)
        trial_valid, constitutive_message = _try_refresh_affine_trial_state(
            state_transaction,
            start_factor=accepted_factor,
            target_factor=factor,
        )

        accepted_steps: list[float] = []
        initial_norm = (
            _reduced_residual_norm(residual, T) if trial_valid else float("inf")
        )
        current_norm = initial_norm
        threshold = (
            selected.atol + selected.rtol * initial_norm
            if np.isfinite(initial_norm)
            else selected.atol
        )
        converged = np.isfinite(current_norm) and current_norm <= threshold
        iteration = 0
        reduced_tangent = None
        while not converged and iteration < selected.max_it:
            if not np.isfinite(current_norm):
                break
            iteration += 1
            full_residual = fem_petsc.assemble_vector(residual)
            full_residual.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            full_tangent = fem_petsc.assemble_matrix(jacobian)
            full_tangent.assemble()
            reduced_residual = T.createVecRight()
            T.multTranspose(full_residual, reduced_residual)
            reduced_tangent = full_tangent.PtAP(T, result=reduced_tangent)
            right_hand_side = reduced_residual.copy()
            right_hand_side.scale(-1.0)
            increment = reduced_residual.duplicate()
            increment.set(0.0)
            linear_info = solve_matrix_system(
                reduced_tangent,
                right_hand_side,
                increment,
                selected.linear_options,
                raise_on_failure=False,
            )
            direction = increment.array_r.copy() if linear_info.converged else None

            full_residual.destroy()
            full_tangent.destroy()
            reduced_residual.destroy()
            right_hand_side.destroy()
            increment.destroy()

            if direction is None:
                _emit(
                    reporter,
                    SolveEvent(
                        "iteration",
                        step_name,
                        step_number=step_number,
                        increment=increment_number,
                        attempt=attempt_number,
                        start_factor=accepted_factor,
                        target_factor=factor,
                        iteration=iteration,
                        residual_norm=current_norm,
                        step_length=0.0,
                        message=(
                            "linear correction failed: "
                            f"KSP reason {linear_info.converged_reason}"
                        ),
                    ),
                )
                break

            alpha = 1.0
            accepted = False
            trial_failure_message = ""
            while alpha + 1.0e-15 >= selected.line_search_minimum:
                trial = reduced_values + alpha * direction
                _assign_reconstructed(function, reduction, trial)
                trial_valid, trial_message = _try_refresh_affine_trial_state(
                    state_transaction,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
                if trial_valid:
                    trial_norm = _reduced_residual_norm(residual, T)
                else:
                    trial_norm = float("inf")
                    trial_failure_message = trial_message
                if np.isfinite(trial_norm) and (
                    trial_norm < current_norm
                    or trial_norm <= current_norm * (1.0 - 1.0e-4 * alpha)
                ):
                    reduced_values = trial
                    current_norm = trial_norm
                    accepted_steps.append(alpha)
                    accepted = True
                    break
                alpha *= selected.line_search_reduction
            if not accepted:
                _assign_reconstructed(function, reduction, reduced_values)
                restored, restore_message = _try_refresh_affine_trial_state(
                    state_transaction,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
                if not restored:
                    trial_failure_message = restore_message
                if trial_failure_message:
                    constitutive_message = trial_failure_message
                _emit(
                    reporter,
                    SolveEvent(
                        "iteration",
                        step_name,
                        step_number=step_number,
                        increment=increment_number,
                        attempt=attempt_number,
                        start_factor=accepted_factor,
                        target_factor=factor,
                        iteration=iteration,
                        residual_norm=current_norm,
                        step_length=0.0,
                        message=trial_failure_message,
                    ),
                )
                break
            _emit(
                reporter,
                SolveEvent(
                    "iteration",
                    step_name,
                    step_number=step_number,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=accepted_factor,
                    target_factor=factor,
                    iteration=iteration,
                    residual_norm=current_norm,
                    step_length=accepted_steps[-1],
                ),
            )
            converged = current_norm <= threshold

        checks = {}
        acceptance_message = constitutive_message
        if converged and acceptance_check is not None:
            checks = dict(acceptance_check())
            if not bool(checks.get("accepted", True)):
                converged = False
                acceptance_message = str(
                    checks.get(
                        "message",
                        "increment failed its physical acceptance check",
                    )
                )
        if reduced_tangent is not None:
            reduced_tangent.destroy()
        mismatch = float(constraint.mismatch(factor))
        increment_info = AffineLoadIncrementInfo(
            load_factor=factor,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=initial_norm,
            residual_norm=current_norm,
            accepted_step_lengths=tuple(accepted_steps),
            reduced_dofs=reduction.reduced_size,
            equation_mismatch=mismatch,
            increment=increment_number,
            attempt=attempt_number,
            start_load_factor=accepted_factor,
            message=acceptance_message,
            checks=checks,
        )
        attempt_history.append(increment_info)
        T.destroy()
        if converged:
            try:
                _commit_affine_trial_state(
                    state_transaction,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
                previous_reduced = reduced_values.copy()
                previous_affine = current_affine.copy()
                history.append(increment_info)
                accepted_size = factor - accepted_factor
                accepted_factor = factor
                cutbacks = 0
                if isinstance(control, step_controls.AutomaticIncrementation):
                    proposed_size = control.after_convergence(
                        accepted_size,
                        iteration,
                    )
                if on_increment is not None:
                    _run_affine_acceptance_stage(
                        function,
                        stage="output/observer callback",
                        callback=on_increment,
                        args=(
                            len(history), factor, function, increment_info
                        ),
                    )
                converged_event = SolveEvent(
                    "increment_converged",
                    step_name,
                    step_number=step_number,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=increment_info.start_load_factor,
                    target_factor=factor,
                    iteration=iteration,
                    residual_norm=current_norm,
                )
                _run_affine_acceptance_stage(
                    function,
                    stage="progress callback",
                    callback=_emit,
                    args=(reporter, converged_event),
                )
                if on_accepted_boundary is not None:
                    _run_affine_acceptance_stage(
                        function,
                        stage="lifecycle/checkpoint callback",
                        callback=_notify_affine_accepted_boundary,
                        args=(
                            on_accepted_boundary,
                            function,
                            history,
                            attempt_history,
                            control,
                            proposed_size,
                            factor,
                        ),
                    )
            except BaseException:
                _rollback_post_commit_failure(
                    function=function,
                    nodal_state=rollback,
                    transaction=state_transaction,
                    transaction_state=accepted_transaction_state,
                    accepted_history=history,
                    accepted_history_state=accepted_history_state,
                    attempted_history=attempt_history,
                    attempted_history_state=attempted_history_state,
                    on_acceptance_failure=on_acceptance_failure,
                )
                raise
            continue

        function.x.array[:] = rollback
        function.x.scatter_forward()
        _rollback_affine_trial_state(
            state_transaction,
            accepted_factor=accepted_factor,
        )
        if isinstance(control, step_controls.FixedIncrementation):
            message = (
                f"fixed increment failed at load factor {factor:.6g}: "
                f"residual={_residual_text(current_norm)}, "
                f"threshold={threshold:.6e}"
                + ("; " + acceptance_message if acceptance_message else "")
            )
            return _failed_affine_path(
                function,
                history,
                attempt_history,
                control,
                selected,
                reporter,
                step_name,
                step_number,
                increment_number,
                total_attempts,
                factor,
                message,
                iteration=iteration,
                residual_norm=current_norm,
            )

        cutbacks += 1
        next_size = control.after_failure(factor - accepted_factor)
        if cutbacks > control.max_cutbacks or next_size < control.minimum - 1.0e-15:
            reason = (
                f"maximum cutbacks ({control.max_cutbacks}) exceeded"
                if cutbacks > control.max_cutbacks
                else f"required increment {next_size:.3g} is below minimum {control.minimum:.3g}"
            )
            message = (
                f"automatic increment failed near load factor {factor:.6g}: "
                f"{reason}"
            )
            return _failed_affine_path(
                function,
                history,
                attempt_history,
                control,
                selected,
                reporter,
                step_name,
                step_number,
                increment_number,
                total_attempts,
                factor,
                message,
                iteration=iteration,
                residual_norm=current_norm,
            )
        proposed_size = max(control.minimum, next_size)
        _emit(
            reporter,
            SolveEvent(
                "increment_cutback",
                step_name,
                step_number=step_number,
                increment=increment_number,
                attempt=attempt_number,
                start_factor=accepted_factor,
                target_factor=factor,
                iteration=iteration,
                residual_norm=current_norm,
                next_increment=proposed_size,
                message=acceptance_message,
            ),
        )

    function.x.scatter_forward()
    _emit(
        reporter,
        SolveEvent(
            "step_completed" if selected_stop >= 1.0 - 1.0e-12 else "step_paused",
            step_name,
            step_number=step_number,
            increment=len(history),
            attempt=total_attempts,
            target_factor=selected_stop,
        ),
    )
    return function, AffineLoadPathInfo(
        tuple(history),
        tuple(attempt_history),
        control,
        selected_stop,
        (
            proposed_size
            if isinstance(control, step_controls.AutomaticIncrementation)
            and selected_stop < 1.0 - 1.0e-12
            else None
        ),
    )


def _notify_affine_accepted_boundary(
    callback,
    function,
    accepted_history,
    attempted_history,
    control,
    proposed_size,
    accepted_factor,
) -> None:
    """Publish one fully committed boundary to the owning lifecycle.

    Checkpoint writers need the complete accepted and attempted path after the
    constitutive transaction has committed.  Keeping this callback distinct
    from spatial output avoids making checkpoint cadence depend on field-output
    cadence.
    """

    if callback is None:
        return
    next_size = (
        float(proposed_size)
        if isinstance(control, step_controls.AutomaticIncrementation)
        and float(accepted_factor) < 1.0 - 1.0e-12
        else None
    )
    callback(
        function,
        tuple(accepted_history),
        tuple(attempted_history),
        next_size,
    )


def _solve_distributed_affine_nonlinear_path(
    residual,
    jacobian,
    function,
    constraint,
    *,
    control,
    required_factors,
    options,
    on_increment,
    on_accepted_boundary,
    on_acceptance_failure,
    acceptance_check,
    state_transaction,
    stop_factor,
    accepted_history,
    attempted_history,
    next_increment_size,
    reporter,
    step_name,
    step_number,
):
    """Distributed Newton path using homogeneous ``dolfinx_mpc`` corrections."""

    import dolfinx_mpc

    reduction = constraint.distributed_reduction()
    correction = reduction.correction()
    reduction.validate_prefix_layout(correction)
    history = list(accepted_history)
    attempt_history = list(attempted_history)
    accepted_factor = (
        float(history[-1].load_factor)
        if history
        else float(getattr(state_transaction, "accepted_factor", 0.0))
    )
    total_attempts = len(attempt_history)
    cutbacks = 0
    proposed_size = (
        (
            control.initial
            if next_increment_size is None
            else float(next_increment_size)
        )
        if isinstance(control, step_controls.AutomaticIncrementation)
        else _next_fixed_affine_factor(control, accepted_factor)
    )
    if accepted_factor <= 1.0e-12:
        function.x.array[:] = 0.0
        function.x.scatter_forward()
    elif float(constraint.mismatch(accepted_factor)) > max(
        1.0e-10,
        10.0 * float(getattr(constraint, "tolerance", 1.0e-12)),
    ):
        raise ValueError(
            "Restored distributed affine displacement violates its equation state."
        )
    _emit(
        reporter,
        SolveEvent(
            "step_resumed" if accepted_factor > 1.0e-12 else "step_started",
            step_name,
            step_number=step_number,
            incrementation=control.summary()["kind"],
            message="distributed affine solve with dolfinx_mpc",
        ),
    )

    while accepted_factor < stop_factor - 1.0e-12:
        increment_number = len(history) + 1
        if isinstance(control, step_controls.AutomaticIncrementation):
            if len(history) >= control.max_increments:
                message = (
                    f"maximum accepted increments ({control.max_increments}) "
                    "reached before load factor 1.0"
                )
                return _failed_affine_path(
                    function,
                    history,
                    attempt_history,
                    control,
                    options,
                    reporter,
                    step_name,
                    step_number,
                    increment_number,
                    total_attempts,
                    accepted_factor,
                    message,
                )
            factor = min(stop_factor, accepted_factor + proposed_size)
            next_output = _next_output_factor(required_factors, accepted_factor)
            if next_output is not None:
                factor = min(factor, next_output)
        else:
            factor = _next_fixed_affine_factor(control, accepted_factor)

        attempt_number = cutbacks + 1
        total_attempts += 1
        _emit(
            reporter,
            SolveEvent(
                "increment_started",
                step_name,
                step_number=step_number,
                increment=increment_number,
                attempt=attempt_number,
                start_factor=accepted_factor,
                target_factor=factor,
            ),
        )
        rollback = function.x.array.copy()
        accepted_transaction_state = _snapshot_affine_accepted_state(
            state_transaction
        )
        accepted_history_state = list(history)
        attempted_history_state = list(attempt_history)
        constraint.apply_affine_increment(accepted_factor, factor)
        trial_valid, constitutive_message = _try_refresh_affine_trial_state(
            state_transaction,
            start_factor=accepted_factor,
            target_factor=factor,
        )

        accepted_steps: list[float] = []
        initial_norm = (
            _distributed_reduced_residual_norm(
                residual,
                reduction,
                dolfinx_mpc,
            )
            if trial_valid
            else float("inf")
        )
        current_norm = initial_norm
        threshold = (
            options.atol + options.rtol * initial_norm
            if np.isfinite(initial_norm)
            else options.atol
        )
        converged = np.isfinite(current_norm) and current_norm <= threshold
        iteration = 0
        while not converged and iteration < options.max_it:
            if not np.isfinite(current_norm):
                break
            iteration += 1
            direction, linear_info = _distributed_newton_direction(
                residual,
                jacobian,
                correction,
                reduction,
                dolfinx_mpc,
                options,
            )
            if direction is None:
                _emit(
                    reporter,
                    SolveEvent(
                        "iteration",
                        step_name,
                        step_number=step_number,
                        increment=increment_number,
                        attempt=attempt_number,
                        start_factor=accepted_factor,
                        target_factor=factor,
                        iteration=iteration,
                        residual_norm=current_norm,
                        step_length=0.0,
                        message=(
                            "linear correction failed: "
                            f"KSP reason {linear_info.converged_reason}"
                        ),
                    ),
                )
                break
            base = function.x.array.copy()
            alpha = 1.0
            accepted = False
            trial_failure_message = ""
            while alpha + 1.0e-15 >= options.line_search_minimum:
                function.x.array[:] = base + alpha * direction
                function.x.scatter_forward()
                trial_valid, trial_message = _try_refresh_affine_trial_state(
                    state_transaction,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
                if trial_valid:
                    trial_norm = _distributed_reduced_residual_norm(
                        residual,
                        reduction,
                        dolfinx_mpc,
                    )
                else:
                    trial_norm = float("inf")
                    trial_failure_message = trial_message
                if np.isfinite(trial_norm) and (
                    trial_norm < current_norm
                    or trial_norm <= current_norm * (1.0 - 1.0e-4 * alpha)
                ):
                    current_norm = trial_norm
                    accepted_steps.append(alpha)
                    accepted = True
                    break
                alpha *= options.line_search_reduction
            if not accepted:
                function.x.array[:] = base
                function.x.scatter_forward()
                restored, restore_message = _try_refresh_affine_trial_state(
                    state_transaction,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
                if not restored:
                    trial_failure_message = restore_message
                if trial_failure_message:
                    constitutive_message = trial_failure_message
                _emit(
                    reporter,
                    SolveEvent(
                        "iteration",
                        step_name,
                        step_number=step_number,
                        increment=increment_number,
                        attempt=attempt_number,
                        start_factor=accepted_factor,
                        target_factor=factor,
                        iteration=iteration,
                        residual_norm=current_norm,
                        step_length=0.0,
                        message=trial_failure_message,
                    ),
                )
                break
            _emit(
                reporter,
                SolveEvent(
                    "iteration",
                    step_name,
                    step_number=step_number,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=accepted_factor,
                    target_factor=factor,
                    iteration=iteration,
                    residual_norm=current_norm,
                    step_length=accepted_steps[-1],
                ),
            )
            converged = current_norm <= threshold

        checks = {}
        acceptance_message = constitutive_message
        if converged and acceptance_check is not None:
            checks = dict(acceptance_check())
            if not bool(checks.get("accepted", True)):
                converged = False
                acceptance_message = str(
                    checks.get(
                        "message",
                        "increment failed its physical acceptance check",
                    )
                )
        mismatch = float(constraint.mismatch(factor))
        increment_info = AffineLoadIncrementInfo(
            load_factor=factor,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=initial_norm,
            residual_norm=current_norm,
            accepted_step_lengths=tuple(accepted_steps),
            reduced_dofs=reduction.reduced_size,
            equation_mismatch=mismatch,
            increment=increment_number,
            attempt=attempt_number,
            start_load_factor=accepted_factor,
            message=acceptance_message,
            checks=checks,
        )
        attempt_history.append(increment_info)
        if converged:
            try:
                _commit_affine_trial_state(
                    state_transaction,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
                history.append(increment_info)
                accepted_size = factor - accepted_factor
                accepted_factor = factor
                cutbacks = 0
                if isinstance(control, step_controls.AutomaticIncrementation):
                    proposed_size = control.after_convergence(
                        accepted_size,
                        iteration,
                    )
                if on_increment is not None:
                    _run_affine_acceptance_stage(
                        function,
                        stage="output/observer callback",
                        callback=on_increment,
                        args=(
                            len(history), factor, function, increment_info
                        ),
                    )
                converged_event = SolveEvent(
                    "increment_converged",
                    step_name,
                    step_number=step_number,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=increment_info.start_load_factor,
                    target_factor=factor,
                    iteration=iteration,
                    residual_norm=current_norm,
                )
                _run_affine_acceptance_stage(
                    function,
                    stage="progress callback",
                    callback=_emit,
                    args=(reporter, converged_event),
                )
                if on_accepted_boundary is not None:
                    _run_affine_acceptance_stage(
                        function,
                        stage="lifecycle/checkpoint callback",
                        callback=_notify_affine_accepted_boundary,
                        args=(
                            on_accepted_boundary,
                            function,
                            history,
                            attempt_history,
                            control,
                            proposed_size,
                            factor,
                        ),
                    )
            except BaseException:
                _rollback_post_commit_failure(
                    function=function,
                    nodal_state=rollback,
                    transaction=state_transaction,
                    transaction_state=accepted_transaction_state,
                    accepted_history=history,
                    accepted_history_state=accepted_history_state,
                    attempted_history=attempt_history,
                    attempted_history_state=attempted_history_state,
                    on_acceptance_failure=on_acceptance_failure,
                )
                raise
            continue

        function.x.array[:] = rollback
        function.x.scatter_forward()
        _rollback_affine_trial_state(
            state_transaction,
            accepted_factor=accepted_factor,
        )
        if isinstance(control, step_controls.FixedIncrementation):
            message = (
                f"fixed increment failed at load factor {factor:.6g}: "
                f"residual={_residual_text(current_norm)}, "
                f"threshold={threshold:.6e}"
                + ("; " + acceptance_message if acceptance_message else "")
            )
            return _failed_affine_path(
                function,
                history,
                attempt_history,
                control,
                options,
                reporter,
                step_name,
                step_number,
                increment_number,
                total_attempts,
                factor,
                message,
                iteration=iteration,
                residual_norm=current_norm,
            )

        cutbacks += 1
        next_size = control.after_failure(factor - accepted_factor)
        if cutbacks > control.max_cutbacks or next_size < control.minimum - 1.0e-15:
            reason = (
                f"maximum cutbacks ({control.max_cutbacks}) exceeded"
                if cutbacks > control.max_cutbacks
                else f"required increment {next_size:.3g} is below minimum {control.minimum:.3g}"
            )
            message = (
                f"automatic increment failed near load factor {factor:.6g}: "
                f"{reason}"
            )
            return _failed_affine_path(
                function,
                history,
                attempt_history,
                control,
                options,
                reporter,
                step_name,
                step_number,
                increment_number,
                total_attempts,
                factor,
                message,
                iteration=iteration,
                residual_norm=current_norm,
            )
        proposed_size = max(control.minimum, next_size)
        _emit(
            reporter,
            SolveEvent(
                "increment_cutback",
                step_name,
                step_number=step_number,
                increment=increment_number,
                attempt=attempt_number,
                start_factor=accepted_factor,
                target_factor=factor,
                iteration=iteration,
                residual_norm=current_norm,
                next_increment=proposed_size,
                message=acceptance_message,
            ),
        )

    function.x.scatter_forward()
    _emit(
        reporter,
        SolveEvent(
            "step_completed" if stop_factor >= 1.0 - 1.0e-12 else "step_paused",
            step_name,
            step_number=step_number,
            increment=len(history),
            attempt=total_attempts,
            target_factor=stop_factor,
        ),
    )
    return function, AffineLoadPathInfo(
        tuple(history),
        tuple(attempt_history),
        control,
        stop_factor,
        (
            proposed_size
            if isinstance(control, step_controls.AutomaticIncrementation)
            and stop_factor < 1.0 - 1.0e-12
            else None
        ),
    )


def _distributed_reduced_residual_norm(
    residual,
    reduction,
    dolfinx_mpc,
) -> float:
    vector = dolfinx_mpc.assemble_vector(residual, reduction.mpc)
    vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    _homogenize_mpc_vector(vector, reduction.mpc)
    fem_petsc.set_bc(vector, reduction.bcs)
    value = float(vector.norm())
    vector.destroy()
    return value


def _distributed_newton_direction(
    residual,
    jacobian,
    correction,
    reduction,
    dolfinx_mpc,
    options,
) -> tuple[np.ndarray | None, LinearSolveInfo]:
    full_residual = dolfinx_mpc.assemble_vector(residual, reduction.mpc)
    full_residual.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    _homogenize_mpc_vector(full_residual, reduction.mpc)
    fem_petsc.set_bc(full_residual, reduction.bcs)
    tangent = dolfinx_mpc.assemble_matrix(
        jacobian,
        reduction.mpc,
        bcs=reduction.bcs,
    )
    tangent.assemble()
    right_hand_side = full_residual.copy()
    right_hand_side.scale(-1.0)
    correction.x.array[:] = 0.0
    info = solve_matrix_system(
        tangent,
        right_hand_side,
        correction.x.petsc_vec,
        options.linear_options,
        raise_on_failure=False,
    )
    direction = None
    if info.converged:
        correction.x.scatter_forward()
        reduction.mpc.homogenize(correction)
        reduction.mpc.backsubstitution(correction)
        local_size = reduction.original_space.dofmap.index_map.size_local
        ghost_count = reduction.original_space.dofmap.index_map.num_ghosts
        block_size = int(reduction.original_space.dofmap.index_map_bs)
        original_array_size = int((local_size + ghost_count) * block_size)
        direction = correction.x.array[:original_array_size].copy()
    full_residual.destroy()
    right_hand_side.destroy()
    tangent.destroy()
    return direction, info


def _homogenize_mpc_vector(vector, mpc) -> None:
    """Zero slave rows in an augmented PETSc vector."""

    with vector.localForm() as local:
        local.array_w[np.asarray(mpc.slaves, dtype=np.int32)] = 0.0
    vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )


def _failed_affine_path(
    function,
    history,
    attempts,
    control,
    options,
    reporter,
    step_name,
    step_number,
    increment,
    total_attempts,
    target_factor,
    message,
    *,
    iteration=0,
    residual_norm=None,
):
    _emit(
        reporter,
        SolveEvent(
            "step_failed",
            step_name,
            step_number=step_number,
            increment=increment,
            attempt=total_attempts,
            target_factor=target_factor,
            iteration=iteration,
            residual_norm=residual_norm,
            message=message,
        ),
    )
    info = AffineLoadPathInfo(tuple(history), tuple(attempts), control)
    if options.error_if_not_converged:
        raise RuntimeError(message)
    function.x.scatter_forward()
    return function, info


def _normalized_output_factors(values) -> tuple[float, ...]:
    selected = tuple(float(value) for value in values)
    if any(not isfinite(value) or value <= 0.0 or value > 1.0 for value in selected):
        raise ValueError("Output factors must lie in the normalized interval (0, 1].")
    if any(right <= left for left, right in zip(selected, selected[1:])):
        raise ValueError("Output factors must be strictly increasing.")
    return selected


def _validate_fixed_affine_resume(control, accepted_factor: float, stop_factor: float) -> None:
    nodes = (0.0, *tuple(float(value) for value in control.load_factors))
    tolerance = 1.0e-12
    if not any(abs(accepted_factor - value) <= tolerance for value in nodes):
        raise ValueError(
            "A fixed affine restart must begin at one of the declared load factors."
        )
    if not any(abs(stop_factor - value) <= tolerance for value in nodes[1:]):
        raise ValueError(
            "Affine solve(until=...) with fixed incrementation must stop at a "
            "declared load factor."
        )


def _next_fixed_affine_factor(control, accepted_factor: float) -> float:
    for value in control.load_factors:
        if float(value) > float(accepted_factor) + 1.0e-12:
            return float(value)
    raise RuntimeError("Fixed affine load path has no remaining increment.")


def _next_output_factor(values, current: float) -> float | None:
    for value in values:
        if value > current + 1.0e-12:
            return value
    return None


def _emit(reporter, event: SolveEvent) -> None:
    if reporter is None:
        return
    if hasattr(reporter, "emit"):
        reporter.emit(event)
    else:
        reporter(event)


def _finite_or_none(value):
    selected = float(value)
    return selected if np.isfinite(selected) else None


def _none_as_nan(value) -> float:
    return float("nan") if value is None else float(value)


def _residual_text(value) -> str:
    return f"{float(value):.6e}" if np.isfinite(value) else "non-finite"


def _assign_reconstructed(function, reduction, reduced_values) -> None:
    values = reduction.reconstruct(reduced_values)
    if values.size != function.x.array.size:
        raise RuntimeError("Affine reconstruction size does not match the solution vector.")
    function.x.array[:] = values
    function.x.scatter_forward()


def _reduced_residual_norm(residual_form, transformation) -> float:
    full = fem_petsc.assemble_vector(residual_form)
    full.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    reduced = transformation.createVecRight()
    transformation.multTranspose(full, reduced)
    value = float(reduced.norm())
    full.destroy()
    reduced.destroy()
    return value
