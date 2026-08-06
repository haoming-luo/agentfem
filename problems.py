"""Problem/state containers for standard finite-element workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from . import assembly
from . import fields
from . import spaces
from . import time
from .kernel import dofs
from .solvers import (
    AffineNewtonOptions,
    LinearSolverOptions,
    NewtonSolverOptions,
    NonlinearSolverOptions,
    SolveEvent,
    solve_affine_nonlinear_path,
    solve_linear_problem,
    solve_nonlinear_problem,
)


@dataclass
class FEMProblem:
    """Lightweight finite-element problem description.

    This object is intentionally descriptive: it helps humans and agents inspect
    a model without hiding weak forms, assembly, or solver choices.
    """

    name: str
    domain: object
    spaces: dict[str, object] = field(default_factory=dict)
    fields: dict[str, object] = field(default_factory=dict)
    materials: list[object] = field(default_factory=list)
    constraints: list[object] = field(default_factory=list)
    loads: list[object] = field(default_factory=list)
    boundary_models: list[object] = field(default_factory=list)
    forms: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """Return a compact problem summary for logs or agent inspection."""

        return {
            "name": self.name,
            "topological_dim": self.domain.topology.dim,
            "geometric_dim": self.domain.geometry.dim,
            "spaces": tuple(self.spaces.keys()),
            "fields": tuple(self.fields.keys()),
            "materials": tuple(_describe_asset(material) for material in self.materials),
            "constraints": tuple(_describe_asset(item) for item in self.constraints),
            "loads": tuple(_describe_asset(item) for item in self.loads),
            "boundary_models": tuple(_describe_asset(item) for item in self.boundary_models),
            "forms": tuple(self.forms.keys()),
        }


@dataclass
class LinearVariationalProblem:
    """A standard linear variational problem, ``a(u, v) = L(v)``."""

    bilinear_form: object
    linear_form: object
    solution: object
    bcs: list = field(default_factory=list)
    solver_options: LinearSolverOptions | None = None
    last_solve_info: object | None = field(default=None, init=False)

    def solve(self):
        """Assemble and solve the problem into ``solution``."""

        solution, info = solve_linear_problem(
            self.bilinear_form,
            self.linear_form,
            self.solution,
            bcs=self.bcs,
            options=self.solver_options,
            return_info=True,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self, *, name: str = "linear_variational_result"):
        """Solve and wrap the solution in a scientific result object."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=name,
            metadata={"solve": self.last_solve_info.as_dict()},
        )


@dataclass
class LinearSystemProblem:
    """Engineering-level linear system problem, usually ``K x = F``."""

    system: object
    solution: object | None = None
    unknown: object | None = None
    bcs: list = field(default_factory=list)
    solver_options: LinearSolverOptions | None = None
    last_solve_info: object | None = field(default=None, init=False)

    @classmethod
    def from_operators(
        cls,
        K,
        F,
        *,
        unknown=None,
        solution=None,
        constraints=None,
        bcs=None,
        solver_options: LinearSolverOptions | None = None,
        name: str = "Kx_eq_F",
    ):
        """Create a linear-system problem from engineering notation."""

        from . import operators

        return cls(
            system=operators.linear_system(K, F, name=name),
            unknown=unknown,
            solution=solution,
            bcs=_collect_bcs(constraints=constraints, bcs=bcs),
            solver_options=solver_options,
        )

    def solve(self):
        """Compile the system operators and solve into ``solution``."""

        solution = self._solution()
        solution, info = solve_linear_problem(
            fem.form(self.system.lhs_form()),
            fem.form(self.system.rhs_form()),
            solution,
            bcs=self.bcs,
            options=self.solver_options,
            return_info=True,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self, *, name: str | None = None):
        """Solve and return a :class:`SimulationResult`."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=name or getattr(self.system, "name", "linear_system_result"),
            metadata={"problem": self.summary()},
        )

    def summary(self) -> dict[str, object]:
        """Return an inspectable K/F problem summary."""

        return {
            "kind": "linear_system_problem",
            "system": self.system.summary() if hasattr(self.system, "summary") else repr(self.system),
            "solution": getattr(self._solution(), "name", repr(self._solution())),
            "num_bcs": len(self.bcs),
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else LinearSolverOptions().summary()
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
        }

    def reaction_field(self, *, name: str = "RF"):
        """Return the unconstrained algebraic residual ``K u - F``.

        At converged free dofs this is zero up to solver tolerance; values at
        strongly constrained dofs are the nodal reactions. Weak, affine-MPC,
        and contact reactions require their own verified definitions.
        """

        import dolfinx.fem.petsc as fem_petsc
        from petsc4py import PETSc

        solution = self._solution()
        lhs = fem.form(self.system.lhs_form())
        rhs = fem.form(self.system.rhs_form())
        matrix = fem_petsc.assemble_matrix(lhs)
        matrix.assemble()
        external = fem_petsc.assemble_vector(rhs)
        external.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        residual = matrix.createVecLeft()
        matrix.mult(solution.x.petsc_vec, residual)
        residual.axpy(-1.0, external)
        reaction = fem.Function(solution.function_space, name=name)
        values = residual.array_r
        reaction.x.array[: len(values)] = values
        reaction.x.scatter_forward()
        residual.destroy()
        external.destroy()
        matrix.destroy()
        return reaction

    def _solution(self):
        if self.solution is not None:
            return self.solution
        if self.unknown is not None and hasattr(self.unknown, "value"):
            return self.unknown.value
        raise ValueError("LinearSystemProblem requires solution or unknown.")


@dataclass
class NonlinearVariationalProblem:
    """Nonlinear residual problem ``R(u; v) = 0`` solved by PETSc SNES."""

    residual_form: object
    solution: object
    bcs: list = field(default_factory=list)
    jacobian_form: object | None = None
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None
    name: str = "nonlinear_problem"
    petsc_options_prefix: str = "agentfem_nonlinear_"
    procedure: object | None = None
    last_solve_info: object | None = field(default=None, init=False)

    def solve(self):
        """Solve and return the live DOLFINx solution field."""

        solution, info = solve_nonlinear_problem(
            self.residual_form,
            self.solution,
            bcs=self.bcs,
            jacobian_form=self.jacobian_form,
            options=self.solver_options,
            petsc_options_prefix=self.petsc_options_prefix,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self):
        """Solve and return a result with SNES convergence evidence."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "nonlinear_variational_problem",
            "name": self.name,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "num_bcs": len(self.bcs),
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else NonlinearSolverOptions().summary()
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }

    def reaction_field(self, *, name: str = "RF"):
        """Return the assembled nonlinear residual at the current solution."""

        return _reaction_field(self.residual_form, self.solution, name=name)


@dataclass(frozen=True)
class NonlinearLoadIncrementInfo:
    """Convergence evidence for one ordinary nonlinear load increment."""

    increment: int
    attempt: int
    start_load_factor: float
    load_factor: float
    converged: bool
    iterations: int
    residual_norm: float
    converged_reason: int
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
            "residual_norm": (
                float(self.residual_norm)
                if np.isfinite(self.residual_norm)
                else None
            ),
            "converged_reason": self.converged_reason,
            "message": self.message,
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class NonlinearLoadPathInfo:
    """Accepted and attempted increments for an ordinary nonlinear step."""

    increments: tuple[NonlinearLoadIncrementInfo, ...]
    attempts: tuple[NonlinearLoadIncrementInfo, ...]
    incrementation: object

    @property
    def converged(self) -> bool:
        return (
            bool(self.increments)
            and all(item.converged for item in self.increments)
            and abs(self.increments[-1].load_factor - 1.0) <= 1.0e-12
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "nonlinear_load_path",
            "converged": self.converged,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": tuple(item.as_dict() for item in self.increments),
            "attempts": tuple(item.as_dict() for item in self.attempts),
        }


@dataclass
class IncrementalNonlinearVariationalProblem:
    """Ordinary nonlinear equilibrium with automatic load incrementation.

    Unlike :class:`AffineNonlinearVariationalProblem`, this path uses standard
    Dirichlet constraints. Natural loads and prescribed end-of-step values are
    driven by one normalized factor. Failed attempts restore the accepted
    field and boundary state before cutback.
    """

    residual_form: object
    solution: object
    factor: object
    value_path: object
    update_load: object | None = None
    acceptance_check: object | None = None
    bcs: list = field(default_factory=list)
    jacobian_form: object | None = None
    incrementation: object | None = None
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None
    output_every: int | None = 1
    progress: object = True
    status_file: object | None = None
    name: str = "incremental_nonlinear"
    petsc_options_prefix: str = "agentfem_incremental_nonlinear_"
    procedure: object | None = None
    result_field_factory: object | None = None
    last_solve_info: NonlinearLoadPathInfo | None = field(default=None, init=False)
    snapshots: list = field(default_factory=list, init=False)
    execution_events: list = field(default_factory=list, init=False)

    def solve(self):
        from . import steps as step_controls
        from .diagnostics import (
            SolveEventRecorder,
            StandardRunReporter,
            comm_of,
            compose_reporters,
        )

        if self.output_every is not None and self.output_every <= 0:
            raise ValueError("Incremental nonlinear output_every must be positive.")
        control = step_controls.normalize(self.incrementation)
        selected_options = self.solver_options or NewtonSolverOptions()
        snes_options = (
            selected_options.for_snes()
            if isinstance(selected_options, NewtonSolverOptions)
            else selected_options
        )
        attempt_options = replace(snes_options, error_if_not_converged=False)
        recorder = SolveEventRecorder(self.execution_events)
        recorder.clear()
        if self.progress is True:
            reporter = compose_reporters(
                recorder,
                StandardRunReporter(
                    comm_of(self.solution),
                    status_file=self.status_file,
                    show_iterations=False,
                ),
            )
        elif self.progress in (False, None):
            reporter = recorder
        else:
            reporter = compose_reporters(recorder, self.progress)

        def emit(event) -> None:
            if reporter is not None:
                reporter.emit(event)

        self.snapshots.clear()
        self.snapshots.append(_load_snapshot(0, 0.0, self.solution))
        history: list[NonlinearLoadIncrementInfo] = []
        attempts: list[NonlinearLoadIncrementInfo] = []
        accepted_factor = 0.0
        total_attempts = 0
        cutbacks = 0
        proposed_size = (
            control.initial
            if isinstance(control, step_controls.AutomaticIncrementation)
            else control.load_factors[0]
        )
        self._update_factor(0.0)
        emit(
            SolveEvent(
                "step_started",
                self.name,
                incrementation=control.summary()["kind"],
            )
        )

        while accepted_factor < 1.0 - 1.0e-12:
            increment_number = len(history) + 1
            if isinstance(control, step_controls.AutomaticIncrementation):
                if len(history) >= control.max_increments:
                    self._fail(
                        history,
                        attempts,
                        control,
                        emit,
                        increment_number,
                        total_attempts,
                        accepted_factor,
                        "maximum accepted increments reached before load factor 1.0",
                    )
                factor = min(1.0, accepted_factor + proposed_size)
            else:
                factor = control.load_factors[len(history)]
            attempt_number = cutbacks + 1
            total_attempts += 1
            emit(
                SolveEvent(
                    "increment_started",
                    self.name,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
            )
            rollback = self.solution.x.array.copy()
            self._update_factor(factor)
            solve_info = None
            message = ""
            try:
                _, solve_info = solve_nonlinear_problem(
                    self.residual_form,
                    self.solution,
                    bcs=self.bcs,
                    jacobian_form=self.jacobian_form,
                    options=attempt_options,
                    petsc_options_prefix=(
                        f"{self.petsc_options_prefix}{increment_number}_{attempt_number}_"
                    ),
                )
                converged = solve_info.converged and bool(
                    np.all(np.isfinite(self.solution.x.array))
                )
            except (RuntimeError, ValueError) as exc:
                converged = False
                message = f"{type(exc).__name__}: {exc}"
            checks = {}
            if converged and self.acceptance_check is not None:
                checks = dict(self.acceptance_check())
                if not bool(checks.get("accepted", True)):
                    converged = False
                    message = str(
                        checks.get(
                            "message",
                            "increment failed its physical acceptance check",
                        )
                    )
            info = NonlinearLoadIncrementInfo(
                increment=increment_number,
                attempt=attempt_number,
                start_load_factor=accepted_factor,
                load_factor=factor,
                converged=converged,
                iterations=0 if solve_info is None else solve_info.iterations,
                residual_norm=(
                    float("nan") if solve_info is None else solve_info.function_norm
                ),
                converged_reason=(
                    0 if solve_info is None else solve_info.converged_reason
                ),
                message=message,
                checks=checks,
            )
            attempts.append(info)
            if converged:
                history.append(info)
                accepted_size = factor - accepted_factor
                accepted_factor = factor
                cutbacks = 0
                if (
                    self.output_every is not None
                    and (
                        len(history) % self.output_every == 0
                        or abs(factor - 1.0) <= 1.0e-12
                    )
                ):
                    self.snapshots.append(
                        _load_snapshot(
                            len(history),
                            factor,
                            self.solution,
                            solve_info=solve_info,
                        )
                    )
                emit(
                    SolveEvent(
                        "increment_converged",
                        self.name,
                        increment=increment_number,
                        attempt=attempt_number,
                        start_factor=info.start_load_factor,
                        target_factor=factor,
                        iteration=info.iterations,
                        residual_norm=info.residual_norm,
                    )
                )
                if isinstance(control, step_controls.AutomaticIncrementation):
                    proposed_size = control.after_convergence(
                        accepted_size,
                        info.iterations,
                    )
                continue

            self.solution.x.array[:] = rollback
            self.solution.x.scatter_forward()
            self._update_factor(accepted_factor)
            if isinstance(control, step_controls.FixedIncrementation):
                self._fail(
                    history,
                    attempts,
                    control,
                    emit,
                    increment_number,
                    total_attempts,
                    factor,
                    f"fixed increment failed at load factor {factor:.6g}",
                    info=info,
                )
            cutbacks += 1
            next_size = control.after_failure(factor - accepted_factor)
            if cutbacks > control.max_cutbacks or next_size < control.minimum - 1.0e-15:
                self._fail(
                    history,
                    attempts,
                    control,
                    emit,
                    increment_number,
                    total_attempts,
                    factor,
                    "automatic increment exhausted its cutback policy",
                    info=info,
                )
            proposed_size = max(control.minimum, next_size)
            emit(
                SolveEvent(
                    "increment_cutback",
                    self.name,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=accepted_factor,
                    target_factor=factor,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    next_increment=proposed_size,
                )
            )

        self.last_solve_info = NonlinearLoadPathInfo(
            tuple(history), tuple(attempts), control
        )
        emit(
            SolveEvent(
                "step_completed",
                self.name,
                increment=len(history),
                attempt=total_attempts,
                target_factor=1.0,
            )
        )
        return self.solution

    def _update_factor(self, factor: float) -> None:
        self.factor.value = PETSc.ScalarType(factor)
        self.value_path.update(factor)
        if self.update_load is not None:
            self.update_load(float(factor))

    def _fail(
        self,
        history,
        attempts,
        control,
        emit,
        increment,
        total_attempts,
        factor,
        message,
        *,
        info=None,
    ) -> None:
        self.last_solve_info = NonlinearLoadPathInfo(
            tuple(history), tuple(attempts), control
        )
        emit(
            SolveEvent(
                "step_failed",
                self.name,
                increment=increment,
                attempt=total_attempts,
                target_factor=factor,
                iteration=0 if info is None else info.iterations,
                residual_norm=None if info is None else info.residual_norm,
                message=message,
            )
        )
        raise RuntimeError(message)

    def solve_result(self):
        from .results import add_execution_trace, from_solution

        solution = self.solve()
        generated = (
            () if self.result_field_factory is None
            else tuple(self.result_field_factory())
        )
        primary = solution if not generated else generated[0]
        result = from_solution(
            primary,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
            },
        )
        for field in generated[1:]:
            result.add_field(
                getattr(field, "name", type(field).__name__),
                field,
                processing={
                    "method": "primary_mixed_finite_element_subfield",
                    "representation": "collapsed_finite_element_dofs",
                    "postprocessed": False,
                },
            )
        add_execution_trace(result, self.execution_events)
        return result

    def reaction_field(self, *, name: str = "RF"):
        return _reaction_field(self.residual_form, self.solution, name=name)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "incremental_nonlinear_variational_problem",
            "name": self.name,
            "num_bcs": len(self.bcs),
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "snapshot_count": len(self.snapshots),
            "primary_result_fields": (
                None if self.result_field_factory is None else tuple(self.primary_fields)
                if hasattr(self, "primary_fields") else "generated"
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class AffineNonlinearVariationalProblem:
    """Nonlinear equilibrium under an exact affine dof reduction."""

    residual_form: object
    jacobian_form: object
    solution: object
    constraint: object
    load_factors: tuple[float, ...] | None = None
    incrementation: object | None = None
    solver_options: AffineNewtonOptions | NewtonSolverOptions | None = None
    output_every: int | None = 1
    output_factors: tuple[float, ...] = ()
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    name: str = "affine_nonlinear_problem"
    procedure: object | None = None
    last_solve_info: object | None = field(default=None, init=False)
    snapshots: list = field(default_factory=list, init=False)

    def solve(self):
        if self.output_every is not None and self.output_every <= 0:
            raise ValueError("Affine nonlinear output_every must be positive.")
        self.snapshots.clear()
        self.snapshots.append(_load_snapshot(0, 0.0, self.solution, zero=True))

        def capture(index, factor, solution, solve_info):
            save_by_increment = (
                self.output_every is not None
                and index % self.output_every == 0
            )
            save_by_factor = any(
                abs(factor - value) <= 1.0e-12
                for value in self.output_factors
            )
            if save_by_increment or save_by_factor or abs(factor - 1.0) <= 1.0e-12:
                self.snapshots.append(
                    _load_snapshot(
                        index,
                        factor,
                        solution,
                        solve_info=solve_info,
                    )
                )

        from .diagnostics import StandardRunReporter, comm_of

        if self.progress is True:
            reporter = StandardRunReporter(
                comm_of(self.solution),
                status_file=self.status_file,
            )
        elif self.progress in (False, None):
            reporter = None
        else:
            reporter = self.progress
        solution, info = solve_affine_nonlinear_path(
            self.residual_form,
            self.jacobian_form,
            self.solution,
            self.constraint,
            load_factors=self.load_factors,
            incrementation=self.incrementation,
            output_factors=self.output_factors,
            options=self.solver_options,
            on_increment=capture,
            reporter=reporter,
            step_name=self.name,
            step_number=self.step_number,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self):
        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "affine_nonlinear_variational_problem",
            "name": self.name,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "constraint": self.constraint.summary(),
            "load_factors": self.load_factors,
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "output_every": self.output_every,
            "output_factors": self.output_factors,
            "snapshot_count": len(self.snapshots),
            "step_number": self.step_number,
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else AffineNewtonOptions().summary()
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class AnalysisStep:
    """Inspectable analysis step that owns one algebraic solve.

    The step is the public workflow layer between a ``Study`` and an algebraic
    problem. It records the analysis intent and method, but keeps K/C/F-style
    operators visible through ``system`` and ``problem``.
    """

    name: str
    problem: LinearSystemProblem
    study: object | None = None
    method: str = "direct_linear_solve"
    dt: float | None = None
    procedure: object | None = None
    result_field_factory: object | None = None

    @property
    def system(self):
        """Return the engineering algebraic system used by this step."""

        return self.problem.system

    @property
    def bcs(self):
        """Return boundary conditions collected for this step."""

        return self.problem.bcs

    def solve(self):
        """Solve this analysis step."""

        return self.problem.solve()

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        field_variables=None,
        strict_output: bool = False,
    ):
        """Solve while retaining the existing ``solve()`` return contract.

        ``solve()`` continues to return the live DOLFINx solution field for
        backwards compatibility.  This method returns the higher-level result
        container used by post-processing, campaigns, and datasets.  Model-
        generated static-solid steps add their standard derived fields after
        convergence; ``output=...`` writes the final field set in one call.
        ``field_variables`` overrides the engineering default without exposing
        solver or projection plumbing in the top-level model.
        """

        from . import io
        from .results import from_solution, static_force_balance

        solution = self.problem.solve()
        if fields and field_variables is not None:
            raise ValueError(
                "Choose explicit live fields or field_variables, not both."
            )
        if field_variables is not None and self.result_field_factory is None:
            raise ValueError(
                "This analysis step does not provide declarative derived fields."
            )
        generated = ()
        if not fields and self.result_field_factory is not None:
            generated = tuple(self.result_field_factory(field_variables))
        generated_ids = {id(item) for item in generated}
        selected_fields = tuple(fields) if fields else (solution, *generated)
        if not selected_fields:
            selected_fields = (solution,)
        result = from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "study": (
                    _describe_asset(self.study) if self.study is not None else None
                ),
            },
        )
        for item in selected_fields:
            function = _unwrap_result_field(item)
            if function is solution:
                continue
            result.add_field(
                getattr(function, "name", type(function).__name__),
                function,
                location=_field_location(function),
                description=(
                    "Constitutive result projected to a discontinuous finite-"
                    "element space; no nodal extrapolation or interelement "
                    "smoothing is applied."
                    if id(item) in generated_ids
                    else ""
                ),
                processing=(
                    _projected_field_processing(function)
                    if id(item) in generated_ids
                    else None
                ),
            )
        is_static_solid = bool(
            self.study is not None
            and getattr(self.study, "is_solid_mechanics", False)
            and len(tuple(getattr(solution, "ufl_shape", ()))) == 1
        )
        if is_static_solid:
            equilibrium = static_force_balance(self.problem)
            result.add_quantities(
                {
                    "external_force_resultant": equilibrium.external,
                    "reaction_force_resultant": equilibrium.reaction,
                    "force_balance_residual": equilibrium.residual,
                    "relative_force_balance_error": equilibrium.relative_error,
                },
                kind="diagnostic",
                descriptions={
                    "external_force_resultant": (
                        "Resultant of the assembled linear-system right-hand side."
                    ),
                    "reaction_force_resultant": (
                        "Resultant of strong-constraint algebraic reactions."
                    ),
                    "force_balance_residual": "Reaction plus external-force resultant.",
                    "relative_force_balance_error": (
                        "Norm of the force-balance residual divided by the larger "
                        "external or reaction resultant norm."
                    ),
                },
            )
            result.metadata["static_equilibrium"] = equilibrium.as_dict()
        if output is not None:
            path = Path(output)
            domain = solution.function_space.mesh
            try:
                if domain.comm.size == 1:
                    from .results.output import write_unified_xdmf_series

                    auxiliary = tuple(
                        _unwrap_result_field(item)
                        for item in selected_fields
                        if _unwrap_result_field(item) is not solution
                    )
                    write_unified_xdmf_series(
                        path,
                        (SimpleNamespace(solution=solution, load_factor=0.0),),
                        (auxiliary,),
                        deformation_scale=0.0,
                    )
                    layout = "single_uniform_grid"
                    backend = "agentfem_unified_xdmf"
                else:
                    writable = []
                    for item in selected_fields:
                        function = _unwrap_result_field(item)
                        if function is solution:
                            coordinate_maps = getattr(domain.geometry, "cmaps", ())
                            degree = int(
                                getattr(coordinate_maps[0], "degree", 1)
                                if coordinate_maps
                                else 1
                            )
                            function = io.interpolate_for_xdmf(
                                function,
                                degree=degree,
                                name=getattr(function, "name", "U"),
                            )
                        writable.append(function)
                    with io.XDMFTimeSeries(path, domain) as writer:
                        writer.write_fields(0.0, *writable)
                    layout = "dolfinx_multigrid"
                    backend = "dolfinx_xdmf_collective"
                result.metadata["field_output"] = {
                    "status": "completed",
                    "backend": backend,
                    "layout": layout,
                    "geometry": "reference",
                    "warp_field": (
                        "U"
                        if len(tuple(getattr(solution, "ufl_shape", ()))) == 1
                        else None
                    ),
                }
                result.add_artifact("fields_xdmf", path)
                heavy = path.with_suffix(".h5")
                if heavy.exists():
                    result.add_artifact("fields_hdf5", heavy)
            except Exception as exc:
                result.status = "completed_with_output_errors"
                result.metadata["field_output"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "requested_path": str(path),
                }
                if strict_output:
                    raise
                warnings.warn(
                    f"Simulation completed, but field output failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return result

    def summary(self) -> dict[str, object]:
        """Return a compact, agent-readable step summary."""

        return {
            "kind": "analysis_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "method": self.method,
            "dt": self.dt,
            "problem": self.problem.summary(),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class ExplicitDynamicsStep:
    """Inspectable second-order explicit dynamics step.

    This is the workflow layer for the standard loop
    ``predict -> apply constraints -> residual -> acceleration -> advance``.
    The integrator and residual operator remain explicit so advanced users can
    inspect or replace them.
    """

    name: str
    state: object
    integrator: object
    residual: object
    dt: float
    steps: int
    study: object | None = None
    prescribed: tuple[object, ...] = ()
    constraints: tuple[object, ...] = ()
    update_load: object | None = None
    save_every: int | None = None
    print_every: int | None = None
    procedure: object | None = None
    history_monitor: object | None = None
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)

    def run(
        self,
        *,
        output=None,
        domain=None,
        fields=(),
        progress=None,
        comm=None,
        until_step: int | None = None,
        history=(),
    ):
        """Run the explicit dynamics step with optional output and progress text."""

        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.state.u)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        if self.completed_steps == 0:
            self.execution_events.clear()
            self.accepted_times.clear()
            self.history_records.clear()
        reporter = _transient_reporter(
            selected_progress,
            selected_comm,
            self.execution_events,
            self.status_file,
        )
        stop_step = _transient_stop_step(self, until_step)
        save_every = _save_interval(self.save_every, output=output, steps=self.steps)
        print_every = _print_interval(self.print_every, self.steps)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=save_every,
            print_every=print_every,
            start_step=self.completed_steps + 1,
            stop_step=stop_step,
        )
        output_fields = tuple(fields) or (
            self.state.u.value,
            self.state.v.value,
            self.state.a.value,
        )
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = output_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )

        _emit_transient_started(reporter, self)
        _record_transient_history(self, self.completed_steps * self.dt)

        if output is None:
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self,
                    info,
                    reporter,
                    selected_progress,
                    self.state,
                    selected_comm,
                )
            _emit_transient_completed(reporter, self)
            return self

        if domain is None:
            domain = self.state.u.function_space.mesh
        with io.XDMFTimeSeries(self.last_output, domain) as xdmf:
            xdmf.write_fields(self.completed_steps * self.dt, *output_fields)
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self,
                    info,
                    reporter,
                    selected_progress,
                    self.state,
                    selected_comm,
                )
                if info.should_save:
                    xdmf.write_fields(info.time, *output_fields)
        _emit_transient_completed(reporter, self)
        return self

    def solve(self):
        self.run()
        return self.state.u.value

    def solve_result(
        self, *, output=None, fields=(), history=(), progress=None, comm=None
    ):
        from .results import add_execution_trace, from_solution

        _configure_transient_history(self, history)
        if output is not None and self.completed_steps >= self.steps:
            raise RuntimeError(
                "Transient field output must be requested before completion; "
                "completed steps cannot be reconstructed from final state alone."
            )
        if output is not None or self.completed_steps != self.steps:
            self.run(
                output=output,
                fields=fields,
                history=history,
                progress=progress,
                comm=comm,
            )
        solution = self.state.u.value
        result = from_solution(solution, name=self.name, metadata={"step": self.summary()})
        add_execution_trace(result, self.execution_events)
        _attach_transient_output(
            result,
            self,
            tuple(fields)
            or self.last_output_fields
            or (self.state.u.value, self.state.v.value, self.state.a.value),
        )
        return result

    def save_checkpoint(self, path) -> Path:
        """Save a partition-bound explicit dynamics restart."""

        return _save_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
        )

    def load_checkpoint(self, path) -> None:
        """Restore explicit state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
        )

    def _advance_one(self, t: float) -> None:
        if self.update_load is not None:
            self.update_load(t)
        self.integrator.step(
            self.dt,
            time=t,
            residual_operator=self.residual,
            prescribed=self.prescribed,
            constraints=self.constraints,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact, agent-readable explicit step summary."""

        return {
            "kind": "explicit_dynamics_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "dt": self.dt,
            "steps": self.steps,
            "completed_steps": self.completed_steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "integrator": (
                self.integrator.summary()
                if hasattr(self.integrator, "summary")
                else repr(self.integrator)
            ),
            "residual": (
                self.residual.summary() if hasattr(self.residual, "summary") else repr(self.residual)
            ),
            "num_prescribed": len(self.prescribed),
            "num_constraints": len(self.constraints),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class ImplicitDynamicsStep:
    """Linear Newmark/generalized-alpha structural-dynamics step."""

    name: str
    state: object
    problem: LinearVariationalProblem
    parameters: object
    dt: float
    steps: int
    displacement_predictor: object
    velocity_predictor: object
    displacement_alpha_predictor: object
    velocity_alpha_predictor: object
    study: object | None = None
    update_load: object | None = None
    save_every: int | None = None
    print_every: int | None = None
    procedure: object | None = None
    history_monitor: object | None = None
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)

    def run(
        self,
        *,
        output=None,
        fields=(),
        progress=None,
        comm=None,
        until_step: int | None = None,
        history=(),
    ):
        """Advance the implicit dynamics step with standard progress output."""

        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.state.u)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        if self.completed_steps == 0:
            self.execution_events.clear()
            self.accepted_times.clear()
            self.history_records.clear()
        reporter = _transient_reporter(
            selected_progress,
            selected_comm,
            self.execution_events,
            self.status_file,
        )
        stop_step = _transient_stop_step(self, until_step)
        save_every = _save_interval(self.save_every, output=output, steps=self.steps)
        print_every = _print_interval(self.print_every, self.steps)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=save_every,
            print_every=print_every,
            start_step=self.completed_steps + 1,
            stop_step=stop_step,
        )
        output_fields = tuple(fields) or (
            self.state.u.value,
            self.state.v.value,
            self.state.a.value,
        )
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = output_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )
        _emit_transient_started(reporter, self)
        _record_transient_history(self, self.completed_steps * self.dt)
        if output is None:
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self, info, reporter, selected_progress, self.state,
                    selected_comm,
                )
            _emit_transient_completed(reporter, self)
            return self
        domain = self.state.u.function_space.mesh
        with io.XDMFTimeSeries(self.last_output, domain) as xdmf:
            xdmf.write_fields(self.completed_steps * self.dt, *output_fields)
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self, info, reporter, selected_progress, self.state,
                    selected_comm,
                )
                if info.should_save:
                    xdmf.write_fields(info.time, *output_fields)
        _emit_transient_completed(reporter, self)
        return self

    def solve(self):
        self.run()
        return self.state.u.value

    def solve_result(
        self, *, output=None, fields=(), history=(), progress=None, comm=None
    ):
        from .results import add_execution_trace, from_solution

        _configure_transient_history(self, history)
        if output is not None and self.completed_steps >= self.steps:
            raise RuntimeError(
                "Transient field output must be requested before completion; "
                "completed steps cannot be reconstructed from final state alone."
            )
        if output is not None or self.completed_steps != self.steps:
            self.run(
                output=output,
                fields=fields,
                history=history,
                progress=progress,
                comm=comm,
            )
        solution = self.state.u.value
        result = from_solution(
            solution,
            name=self.name,
            metadata={"step": self.summary()},
        )
        add_execution_trace(result, self.execution_events)
        _attach_transient_output(
            result,
            self,
            tuple(fields)
            or self.last_output_fields
            or (self.state.u.value, self.state.v.value, self.state.a.value),
        )
        return result

    def save_checkpoint(self, path) -> Path:
        """Save a partition-bound implicit dynamics restart."""

        return _save_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
        )

    def load_checkpoint(self, path) -> None:
        """Restore implicit state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
        )

    def _advance_one(self, time_value: float) -> None:
        p = self.parameters
        dt = self.dt
        u = self.state.u.value
        v = self.state.v.value
        a = self.state.a.value
        u_predictor = self.displacement_predictor
        v_predictor = self.velocity_predictor
        u_predictor.x.array[:] = (
            u.x.array
            + dt * v.x.array
            + dt**2 * (0.5 - p.beta) * a.x.array
        )
        v_predictor.x.array[:] = (
            v.x.array + dt * (1.0 - p.gamma) * a.x.array
        )
        self.displacement_alpha_predictor.x.array[:] = (
            (1.0 - p.alpha_f) * u_predictor.x.array
            + p.alpha_f * u.x.array
        )
        self.velocity_alpha_predictor.x.array[:] = (
            (1.0 - p.alpha_f) * v_predictor.x.array
            + p.alpha_f * v.x.array
        )
        for function in (
            u_predictor,
            v_predictor,
            self.displacement_alpha_predictor,
            self.velocity_alpha_predictor,
        ):
            function.x.scatter_forward()
        if self.update_load is not None:
            evaluation_time = (
                (1.0 - p.alpha_f) * time_value
                + p.alpha_f * (time_value - dt)
            )
            self.update_load(evaluation_time)
        self.problem.solve()
        self.state.u_next.value.x.array[:] = (
            u_predictor.x.array
            + p.beta * dt**2 * self.state.a_next.value.x.array
        )
        self.state.v_next.value.x.array[:] = (
            v_predictor.x.array
            + p.gamma * dt * self.state.a_next.value.x.array
        )
        self.state.u_next.value.x.scatter_forward()
        self.state.v_next.value.x.scatter_forward()
        self.state.advance_state()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "implicit_dynamics_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "integration": self.parameters.summary(),
            "dt": self.dt,
            "steps": self.steps,
            "completed_steps": self.completed_steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "problem": {
                "num_bcs": len(self.problem.bcs),
                "solver": (
                    self.problem.solver_options.summary()
                    if self.problem.solver_options is not None
                    else LinearSolverOptions().summary()
                ),
            },
        }


@dataclass
class FirstOrderTransientStep:
    """Reusable implicit-Euler step loop for heat/diffusion problems."""

    name: str
    problem: AnalysisStep
    current: object
    previous: object
    dt: float
    steps: int
    study: object | None = None
    update_load: object | None = None
    save_every: int | None = None
    print_every: int | None = None
    progress: object = True
    procedure: object | None = None
    history_monitor: object | None = None
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)

    def run(
        self,
        *,
        output=None,
        fields=(),
        progress=None,
        comm=None,
        until_step: int | None = None,
        history=(),
    ):
        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.current)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        if self.completed_steps == 0:
            self.execution_events.clear()
            self.accepted_times.clear()
            self.history_records.clear()
        reporter = _transient_reporter(
            selected_progress,
            selected_comm,
            self.execution_events,
            self.status_file,
        )
        stop_step = _transient_stop_step(self, until_step)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=_save_interval(
                self.save_every,
                output=output,
                steps=self.steps,
            ),
            print_every=_print_interval(self.print_every, self.steps),
            start_step=self.completed_steps + 1,
            stop_step=stop_step,
        )
        selected_fields = tuple(fields) or (self.current,)
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = selected_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )

        _emit_transient_started(reporter, self)
        _record_transient_history(self, self.completed_steps * self.dt)

        def advance(info):
            if self.update_load is not None:
                self.update_load(info.time)
            self.problem.solve()
            self.previous.x.array[:] = self.current.x.array
            self.previous.x.scatter_forward()
            _accept_transient_increment(
                self, info, reporter, selected_progress, self.current,
                selected_comm,
            )

        if output is None:
            for info in stepper:
                advance(info)
            _emit_transient_completed(reporter, self)
            return self
        domain = self.current.function_space.mesh
        with io.XDMFTimeSeries(self.last_output, domain) as xdmf:
            xdmf.write_fields(self.completed_steps * self.dt, *selected_fields)
            for info in stepper:
                advance(info)
                if info.should_save:
                    xdmf.write_fields(info.time, *selected_fields)
        _emit_transient_completed(reporter, self)
        return self

    def solve(self):
        self.run()
        return self.current

    def solve_result(
        self, *, output=None, fields=(), history=(), progress=None, comm=None
    ):
        from .results import add_execution_trace, from_solution

        _configure_transient_history(self, history)
        if output is not None and self.completed_steps >= self.steps:
            raise RuntimeError(
                "Transient field output must be requested before completion; "
                "completed steps cannot be reconstructed from final state alone."
            )
        if output is not None or self.completed_steps != self.steps:
            self.run(
                output=output,
                fields=fields,
                history=history,
                progress=progress,
                comm=comm,
            )
        solution = self.current
        result = from_solution(
            solution,
            name=self.name,
            metadata={"step": self.summary()},
        )
        add_execution_trace(result, self.execution_events)
        _attach_transient_output(
            result,
            self,
            tuple(fields) or self.last_output_fields or (self.current,),
        )
        return result

    def save_checkpoint(self, path) -> Path:
        """Save a partition-bound first-order transient restart."""

        return _save_transient_checkpoint(
            self,
            path,
            {"current": self.current, "previous": self.previous},
        )

    def load_checkpoint(self, path) -> None:
        """Restore first-order state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {"current": self.current, "previous": self.previous},
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "first_order_transient_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "dt": self.dt,
            "steps": self.steps,
            "completed_steps": self.completed_steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "problem": self.problem.summary(),
        }


def _attach_transient_output(result, step, output_fields) -> None:
    """Attach the accepted time axis and one logical XDMF/HDF5 dataset."""

    result.metadata["accepted_times"] = tuple(float(item) for item in step.accepted_times)
    output_start = step.last_output_start_time
    result.metadata["transient"] = {
        "completed_steps": int(step.completed_steps),
        "total_steps": int(step.steps),
        "output_start_time": output_start,
        "output_scope": (
            None
            if step.last_output is None
            else "complete"
            if output_start == 0.0
            else "continuation_segment"
        ),
    }
    if step.history_requests:
        result.metadata["transient"]["history_requests"] = [
            request.summary() for request in step.history_requests
        ]
    if step.history_records:
        coordinates = [item["time"] for item in step.history_records]
        names = tuple(
            name
            for name in step.history_records[0]
            if name != "time"
        )
        requests = {
            request.name: request
            for request in getattr(step, "history_requests", ())
        }
        result.add_histories(
            coordinates,
            {
                name: [item[name] for item in step.history_records]
                for name in names
            },
            abscissa_name="time",
            abscissa_unit="s",
            units={
                name: getattr(requests.get(name), "unit", None)
                for name in names
            },
            descriptions={
                name: (
                    getattr(requests.get(name), "description", "")
                    or _TRANSIENT_HISTORY_DESCRIPTIONS.get(name, "")
                )
                for name in names
            },
        )
    for checkpoint in step.checkpoints:
        result.add_checkpoint(checkpoint)
    path = step.last_output
    if path is None:
        return
    result.add_artifact("fields_xdmf", path)
    heavy_data = path.with_suffix(".h5")
    if heavy_data.is_file():
        result.add_artifact("fields_hdf5", heavy_data)
    for item in output_fields:
        function = getattr(item, "value", item)
        name = getattr(function, "name", type(function).__name__)
        result.add_field(
            name,
            function,
            artifact=path,
            description=(
                "Transient field in the shared XDMF/HDF5 series; "
                f"this output segment starts at time {output_start:g}."
            ),
        )


_TRANSIENT_HISTORY_DESCRIPTIONS = {
    "kinetic_energy": "Discrete kinetic energy, one half v-transpose M v.",
    "strain_energy": "Recoverable linear strain energy, one half u-transpose K u.",
    "total_mechanical_energy": "Sum of discrete kinetic and recoverable strain energy.",
    "thermal_content": (
        "Discrete thermal content, one-transpose C T, relative to the model's "
        "temperature zero."
    ),
    "applied_heat_rate": "Applied volumetric, flux, and Robin-source heat rate, one-transpose Q.",
    "outward_heat_rate": "Net discrete conduction/Robin rate, one-transpose K T.",
    "heat_balance_residual": (
        "Implicit-Euler energy residual: delta thermal content plus dt times "
        "outward rate minus applied rate. Strong-temperature reactions appear "
        "in this residual until reported separately."
    ),
}


def _transient_reporter(progress, comm, events, status_file=None):
    from .diagnostics import SolveEventRecorder, compose_reporters

    recorder = SolveEventRecorder(events)
    if progress is True:
        from .diagnostics import StandardRunReporter

        return compose_reporters(
            recorder,
            StandardRunReporter(
                comm,
                status_file=status_file,
                show_iterations=False,
            ),
        )
    if hasattr(progress, "emit"):
        return compose_reporters(recorder, progress)
    return recorder


def _emit_transient_started(reporter, step) -> None:
    if reporter is None:
        return
    procedure = getattr(step, "procedure", None)
    algorithm = getattr(procedure, "algorithm", "time_integration")
    reporter.emit(
        SolveEvent(
            "transient_started" if step.completed_steps == 0 else "transient_resumed",
            step.name,
            incrementation=algorithm,
            total_increments=step.steps,
        )
    )


def _report_transient_increment(
    reporter,
    selected_progress,
    info,
    step,
    state,
    comm,
) -> None:
    if reporter is not None:
        reporter.emit(
            SolveEvent(
                "time_increment",
                step.name,
                increment=info.index,
                time=float(info.time),
                total_increments=step.steps,
                display=bool(info.should_print),
            )
        )
    if info.should_print and callable(selected_progress):
        from .diagnostics import print_on_root

        message = selected_progress(info, state)
        if message:
            print_on_root(comm, message)


def _accept_transient_increment(
    step,
    info,
    reporter,
    selected_progress,
    report_state,
    comm,
) -> None:
    """Commit one accepted time increment through the shared lifecycle."""

    step.completed_steps = int(info.index)
    step.accepted_times.append(float(info.time))
    _record_transient_history(step, info.time)
    _report_transient_increment(
        reporter,
        selected_progress,
        info,
        step,
        report_state,
        comm,
    )
    _write_scheduled_checkpoint(step)


def _write_scheduled_checkpoint(step) -> Path | None:
    policy = getattr(step, "checkpoint_policy", None)
    if policy is None or not policy.due(step.completed_steps, step.steps):
        return None
    written = step.save_checkpoint(
        policy.path(
            step_name=step.name,
            increment=step.completed_steps,
        )
    )
    _apply_checkpoint_retention(step, policy)
    return written


def _emit_transient_completed(reporter, step) -> None:
    if reporter is None:
        return
    times = getattr(step, "accepted_times", ())
    reporter.emit(
        SolveEvent(
            (
                "transient_completed"
                if step.completed_steps >= step.steps
                else "transient_paused"
            ),
            step.name,
            increment=step.completed_steps,
            time=float(step.completed_steps) * float(step.dt),
            total_increments=step.steps,
        )
    )


def _transient_stop_step(step, until_step: int | None) -> int:
    stop = step.steps if until_step is None else int(until_step)
    if not step.completed_steps <= stop <= step.steps:
        raise ValueError(
            "until_step must lie between the completed step and total steps."
        )
    return stop


def _record_transient_history(step, time_value: float) -> None:
    monitor = getattr(step, "history_monitor", None)
    requests = tuple(getattr(step, "history_requests", ()))
    if monitor is None and not requests:
        return
    selected_time = float(time_value)
    if step.history_records and np.isclose(
        step.history_records[-1]["time"],
        selected_time,
    ):
        if hasattr(monitor, "restore"):
            monitor.restore(step.history_records[-1])
        return
    values = {}
    if monitor is not None:
        if isinstance(step, FirstOrderTransientStep):
            values.update(monitor.evaluate(step.current))
        elif hasattr(monitor, "evaluate"):
            values.update(
                monitor.evaluate(
                    displacement=step.state.u,
                    velocity=step.state.v,
                )
            )
        else:
            values.update(monitor(step, selected_time))
    for request in requests:
        if not hasattr(request, "evaluate_transient"):
            raise TypeError(
                f"History request {type(request).__name__} cannot be evaluated "
                "during a transient Step."
            )
        name = str(request.name)
        if name == "time" or name in values:
            raise ValueError(
                f"Transient history name {name!r} conflicts with an existing "
                "history channel."
            )
        values[name] = request.evaluate_transient(step, selected_time)
    frame = {"time": selected_time}
    for name, value in values.items():
        selected = float(value)
        if not np.isfinite(selected):
            raise ValueError(f"Transient history {name!r} is not finite.")
        frame[str(name)] = selected
    if step.history_records:
        expected = set(step.history_records[0])
        actual = set(frame)
        if actual != expected:
            raise RuntimeError(
                "Transient history channels changed after the analysis began: "
                f"expected {tuple(sorted(expected))}, received "
                f"{tuple(sorted(actual))}. When restarting, pass "
                "the same history requests used by the original Step."
            )
    if len(frame) > 1:
        step.history_records.append(frame)


def _configure_transient_history(step, requests) -> None:
    """Bind immutable accepted-increment requests to one transient Step."""

    selected = tuple(requests)
    if not selected:
        return
    names = [str(getattr(request, "name", "")) for request in selected]
    if any(not name for name in names):
        raise ValueError("Every transient history request requires a name.")
    if len(set(names)) != len(names):
        raise ValueError("Transient history request names must be unique.")
    current = tuple(getattr(step, "history_requests", ()))
    if current and current != selected:
        raise RuntimeError(
            "Transient history requests are fixed after execution begins; "
            "continue with the same request objects."
        )
    step.history_requests = selected


def _apply_checkpoint_retention(step, policy) -> None:
    """Prune published scheduled checkpoints beyond an explicit policy."""

    keep_last = getattr(policy, "keep_last", None)
    scheduled = [
        record
        for record in step.checkpoints
        if record.metadata.get("role") != "restart_source"
    ]
    if keep_last is None or len(scheduled) <= keep_last:
        return
    from . import checkpointing

    obsolete = scheduled[:-keep_last]
    state = step.current if isinstance(step, FirstOrderTransientStep) else step.state.u
    function = getattr(state, "value", state)
    for record in obsolete:
        checkpointing._remove_transient_checkpoint(
            record.path,
            comm=function.function_space.mesh.comm,
        )
    removed = {id(record) for record in obsolete}
    step.checkpoints[:] = [
        record for record in step.checkpoints if id(record) not in removed
    ]


def _save_transient_checkpoint(step, path, state) -> Path:
    from . import checkpointing
    from .results import CheckpointRecord

    manifest = checkpointing.save_transient_checkpoint(
        path,
        step_kind=step.summary()["kind"],
        step_name=step.name,
        procedure=step.procedure,
        dt=step.dt,
        total_steps=step.steps,
        completed_steps=step.completed_steps,
        state=state,
        accepted_times=step.accepted_times,
        execution_events=step.execution_events,
        history_records=step.history_records,
    )
    record = CheckpointRecord(
        name=f"{step.name}_{step.completed_steps}",
        path=manifest,
        schema=checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,
        step_name=step.name,
        coordinate_name="time",
        coordinate_value=float(step.completed_steps) * float(step.dt),
        portable=False,
        metadata={
            "completed_steps": step.completed_steps,
            "total_steps": step.steps,
            "portability": "same mesh partition and MPI size",
        },
    )
    step.checkpoints.append(record)
    return manifest


def _load_transient_checkpoint(step, path, state) -> None:
    from . import checkpointing
    from .results import CheckpointRecord

    metadata = checkpointing.load_transient_checkpoint(
        path,
        step_kind=step.summary()["kind"],
        step_name=step.name,
        procedure=step.procedure,
        dt=step.dt,
        total_steps=step.steps,
        state=state,
    )
    step.completed_steps = int(metadata["completed_steps"])
    step.accepted_times[:] = [float(value) for value in metadata["accepted_times"]]
    step.execution_events[:] = [
        SolveEvent.from_dict(item) for item in metadata["execution_events"]
    ]
    step.history_records[:] = [
        {name: float(value) for name, value in item.items()}
        for item in metadata["history_records"]
    ]
    step.checkpoints.append(
        CheckpointRecord(
            name=f"{step.name}_{step.completed_steps}_restart",
            path=Path(metadata["manifest_path"]),
            schema=checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,
            step_name=step.name,
            coordinate_name="time",
            coordinate_value=float(step.completed_steps) * float(step.dt),
            portable=False,
            metadata={
                "role": "restart_source",
                "portability": metadata["portability"],
            },
        )
    )


@dataclass
class TransientState:
    """Current/next fields for a first-order transient unknown."""

    current: object
    next: object

    @classmethod
    def create(cls, V, *, name: str = "Field"):
        """Create a zero-initialized transient state on a function space."""

        return cls(
            current=spaces.named_function(V, name),
            next=spaces.named_function(V, name),
        )

    def accept_step(self) -> None:
        """Copy ``next`` into ``current``."""

        dofs.copy_function(self.current, self.next)


@dataclass
class SecondOrderDynamicsState:
    """Displacement/velocity/acceleration fields for second-order dynamics."""

    u: object
    v: object
    a: object
    v_mid: object
    u_next: object
    v_next: object
    a_next: object

    @classmethod
    def create(
        cls,
        V,
        *,
        displacement_name: str = "Displacement",
        velocity_name: str = "Velocity",
        acceleration_name: str = "Acceleration",
    ):
        """Create zero-initialized explicit dynamics fields for a space."""

        return cls(
            u=fields.wrap(spaces.named_function(V, displacement_name)),
            v=fields.wrap(spaces.named_function(V, velocity_name)),
            a=fields.wrap(spaces.named_function(V, acceleration_name)),
            v_mid=fields.wrap(spaces.named_function(V, f"{velocity_name}_Midstep")),
            u_next=fields.wrap(spaces.named_function(V, displacement_name)),
            v_next=fields.wrap(spaces.named_function(V, velocity_name)),
            a_next=fields.wrap(spaces.named_function(V, acceleration_name)),
        )

    def predict_displacement(self, dt: float) -> None:
        """Predict ``u_next`` from the current state."""

        time.central_difference_predict_displacement(self.u_next, self.u, self.v, self.a, dt)

    def set_acceleration_from_residual(self, residual, inv_mass: np.ndarray) -> None:
        """Set ``a_next`` from a residual vector and inverse lumped mass."""

        time.acceleration_from_residual(self.a_next, residual, inv_mass)

    def update_midstep_velocity(self, dt: float) -> None:
        """Update ``v_mid`` with the central-difference half-step formula."""

        time.central_difference_update_midstep_velocity(self.v_mid, self.v, self.a, dt)

    def correct_velocity(self, dt: float) -> None:
        """Correct ``v_next`` using ``a`` and ``a_next``."""

        time.central_difference_correct_velocity(self.v_next, self.v, self.a, self.a_next, dt)

    def update_velocity(self, dt: float) -> None:
        """Update ``v_next`` from ``v_mid`` and ``a_next``."""

        time.central_difference_update_velocity(self.v_next, self.v_mid, self.a_next, dt)

    def update_displacement(self) -> None:
        """Copy ``u_next`` into the current displacement state."""

        dofs.copy_function(self.u, self.u_next)

    def advance_state(self) -> None:
        """Copy next-step fields into current fields."""

        dofs.copy_function(self.u, self.u_next)
        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)

    def accept_step(self) -> None:
        """Compatibility alias for ``advance_state``."""

        self.advance_state()

    def accept_displacement(self) -> None:
        """Compatibility alias for ``update_displacement``."""

        self.update_displacement()

    def accept_velocity_acceleration(self) -> None:
        """Compatibility alias for advancing velocity and acceleration."""

        self.advance_velocity_acceleration()

    def advance_velocity_acceleration(self) -> None:
        """Advance velocity and acceleration to the next time level."""

        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)


@dataclass
class LumpedMassOperator:
    """Diagonal mass operator for explicit dynamics."""

    mass: np.ndarray
    inv_mass: np.ndarray

    @classmethod
    def assemble(cls, V, density=1.0, measure=None):
        """Assemble a lumped mass operator for a function space."""

        if measure is None:
            mass = assembly.assemble_lumped_mass(V, density)
        else:
            mass = assembly.assemble_lumped_mass(V, density, measure=measure)
        return cls(mass=mass, inv_mass=assembly.inverse_diagonal(mass))


ExplicitDynamicsState = SecondOrderDynamicsState


def second_order_state(field_or_space, **kwargs) -> SecondOrderDynamicsState:
    """Create a second-order dynamics state from a field or function space."""

    source = None
    if hasattr(field_or_space, "space"):
        V = field_or_space.space
        source = getattr(field_or_space, "value", None)
    elif hasattr(field_or_space, "function_space"):
        V = field_or_space.function_space
        source = field_or_space
    else:
        V = field_or_space
    state = SecondOrderDynamicsState.create(V, **kwargs)
    if source is not None:
        state.u = fields.wrap(source)
        state.u_next.assign(source)
    return state


def linear_system(
    K,
    F,
    *,
    unknown=None,
    solution=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    name: str = "Kx_eq_F",
) -> LinearSystemProblem:
    """Create a ``K x = F`` problem without exposing variational boilerplate."""

    return LinearSystemProblem.from_operators(
        K,
        F,
        unknown=unknown,
        solution=solution,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )


def linear_static(
    K,
    F,
    *,
    study=None,
    unknown=None,
    solution=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    result_field_factory=None,
    name: str = "linear_static",
) -> AnalysisStep:
    """Create a linear static analysis step in ``K x = F`` notation."""

    _require_study_analysis(study, "linear_static")
    problem = linear_system(
        K,
        F,
        unknown=unknown,
        solution=solution,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )
    from . import procedures

    return AnalysisStep(
        name=name,
        study=study,
        problem=problem,
        method="linear_static",
        procedure=procedures.linear_static(),
        result_field_factory=result_field_factory,
    )


def nonlinear(
    residual,
    solution,
    *,
    jacobian=None,
    constraints=None,
    bcs=None,
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None,
    name: str = "nonlinear",
    petsc_options_prefix: str = "agentfem_nonlinear_",
) -> NonlinearVariationalProblem:
    """Create a general nonlinear residual problem."""

    from . import procedures

    return NonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        bcs=_collect_bcs(constraints=constraints, bcs=bcs),
        solver_options=solver_options,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
        procedure=procedures.nonlinear_static(),
    )


def incremental_nonlinear(
    residual,
    solution,
    *,
    factor,
    value_path,
    update_load=None,
    acceptance_check=None,
    jacobian=None,
    incrementation=None,
    constraints=None,
    bcs=None,
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None,
    output_every: int | None = 1,
    progress=True,
    status_file=None,
    name: str = "incremental_nonlinear",
    petsc_options_prefix: str = "agentfem_incremental_nonlinear_",
) -> IncrementalNonlinearVariationalProblem:
    """Create standard-BC nonlinear equilibrium over a normalized load path."""

    from . import procedures
    from . import steps as step_controls

    return IncrementalNonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        factor=factor,
        value_path=value_path,
        update_load=update_load,
        acceptance_check=acceptance_check,
        bcs=_collect_bcs(constraints=constraints, bcs=bcs),
        incrementation=step_controls.normalize(incrementation),
        solver_options=solver_options,
        output_every=output_every,
        progress=progress,
        status_file=status_file,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
        procedure=procedures.nonlinear_static(),
    )


def affine_nonlinear(
    residual,
    solution,
    *,
    jacobian,
    constraint,
    load_factors=None,
    incrementation=None,
    solver_options: AffineNewtonOptions | NewtonSolverOptions | None = None,
    output_every: int | None = 1,
    output_factors=(),
    progress=True,
    status_file=None,
    name: str = "affine_nonlinear",
) -> AffineNonlinearVariationalProblem:
    """Create a nonlinear problem reduced by an affine constraint map."""

    from . import procedures

    return AffineNonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        constraint=constraint,
        load_factors=(
            None
            if load_factors is None
            else tuple(float(value) for value in load_factors)
        ),
        incrementation=incrementation,
        solver_options=solver_options,
        output_every=(
            None if output_every is None else int(output_every)
        ),
        output_factors=tuple(float(value) for value in output_factors),
        progress=progress,
        status_file=status_file,
        name=name,
        procedure=procedures.nonlinear_static(),
    )


@dataclass(frozen=True)
class LoadIncrementSnapshot:
    """A copied solution state at one nonlinear load factor."""

    index: int
    load_factor: float
    solution: object
    solve_info: object | None = None

    def summary(self) -> dict[str, object]:
        return {
            "index": self.index,
            "load_factor": self.load_factor,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "solve": (
                None
                if self.solve_info is None
                else self.solve_info.as_dict()
            ),
        }


def _load_snapshot(
    index: int,
    load_factor: float,
    solution,
    *,
    solve_info=None,
    zero: bool = False,
) -> LoadIncrementSnapshot:
    selected = solution.value if hasattr(solution, "value") else solution
    copied = fem.Function(
        selected.function_space,
        name=getattr(selected, "name", "Solution"),
    )
    if not zero:
        copied.x.array[:] = selected.x.array
        copied.x.scatter_forward()
    return LoadIncrementSnapshot(
        index=int(index),
        load_factor=float(load_factor),
        solution=copied,
        solve_info=solve_info,
    )


def first_order_transient(
    *,
    capacity,
    stiffness,
    history,
    source=None,
    dt: float,
    study=None,
    unknown=None,
    solution=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    name: str = "first_order_transient_step",
    method: str = "implicit_euler",
) -> AnalysisStep:
    """Create a first-order transient step.

    This builds the common implicit Euler system
    ``(C / dt + K) x_next = C x_previous / dt + Q`` while keeping ``C``, ``K``,
    history, and source as explicit operator-level inputs.
    """

    from . import operators

    _require_study_analysis(study, "first_order_transient")
    if dt <= 0.0:
        raise ValueError("first_order_transient requires dt > 0.")

    C_over_dt = operators.scale(
        capacity,
        1.0 / dt,
        name="C_over_dt",
        kind=f"{method}_capacity_over_dt",
    )
    history_over_dt = operators.scale(
        history,
        1.0 / dt,
        name="F_history_over_dt",
        kind=f"{method}_history_over_dt",
    )
    lhs = operators.combine(
        C_over_dt,
        stiffness,
        name="K_effective",
        kind=f"{method}_lhs",
    )
    rhs_terms = (history_over_dt,) if source is None else (history_over_dt, source)
    rhs = operators.combine(
        *rhs_terms,
        name="F_effective",
        kind=f"{method}_rhs",
    )
    problem = linear_system(
        lhs,
        rhs,
        unknown=unknown,
        solution=solution,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )
    from . import procedures

    return AnalysisStep(
        name=name,
        study=study,
        problem=problem,
        method=method,
        dt=dt,
        procedure=procedures.implicit_euler(),
    )


def first_order_transient_run(
    *,
    capacity,
    stiffness,
    history,
    current,
    previous,
    dt: float,
    steps: int,
    source=None,
    study=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    name: str = "first_order_transient",
) -> FirstOrderTransientStep:
    """Create an executable implicit-Euler time step and loop."""

    from . import procedures
    from .diagnostics import ThermalBalanceMonitor

    if steps <= 0:
        raise ValueError("first_order_transient_run requires steps > 0.")
    problem = first_order_transient(
        capacity=capacity,
        stiffness=stiffness,
        history=history,
        source=source,
        dt=dt,
        study=study,
        solution=current,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )
    return FirstOrderTransientStep(
        name=name,
        problem=problem,
        current=current,
        previous=previous,
        dt=float(dt),
        steps=int(steps),
        study=study,
        update_load=update_load,
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        procedure=procedures.implicit_euler(),
        history_monitor=ThermalBalanceMonitor(
            capacity=capacity,
            stiffness=stiffness,
            source=source,
            dt=float(dt),
        ),
    )


def explicit_dynamics(
    *,
    state,
    integrator,
    residual,
    stiffness=None,
    dt: float,
    steps: int,
    study=None,
    prescribed=(),
    constraints=(),
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    name: str = "explicit_dynamics",
) -> ExplicitDynamicsStep:
    """Create a second-order explicit dynamics step."""

    _require_study_analysis(study, "second_order_dynamics")
    if dt <= 0.0:
        raise ValueError("explicit_dynamics requires dt > 0.")
    if steps <= 0:
        raise ValueError("explicit_dynamics requires steps > 0.")
    from . import procedures
    from .diagnostics import MechanicalEnergyMonitor

    return ExplicitDynamicsStep(
        name=name,
        study=study,
        state=state,
        integrator=integrator,
        residual=residual,
        history_monitor=MechanicalEnergyMonitor(
            mass=integrator.mass,
            stiffness=stiffness,
        ),
        prescribed=tuple(_as_list(prescribed)),
        constraints=tuple(_as_list(constraints)),
        update_load=update_load,
        dt=dt,
        steps=int(steps),
        save_every=None if save_every is None else int(save_every),
        print_every=None if print_every is None else int(print_every),
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        procedure=procedures.central_difference(),
    )


def implicit_dynamics(
    *,
    state,
    mass,
    stiffness,
    force,
    damping=None,
    dt: float,
    steps: int,
    parameters=None,
    study=None,
    constraints=(),
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    update_load=None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    save_every: int | None = None,
    print_every: int | None = None,
    name: str = "implicit_dynamics",
) -> ImplicitDynamicsStep:
    """Create a linear Newmark or generalized-alpha dynamics step.

    Strong constraints are imposed as zero acceleration. This is correct for
    time-invariant prescribed displacements; moving supports require a
    prescribed kinematic-history object and are rejected by the future
    validation layer rather than silently approximated here.
    """

    from . import procedures
    from .diagnostics import MechanicalEnergyMonitor
    from .time import implicit as implicit_time

    _require_study_analysis(study, "second_order_dynamics")
    if dt <= 0.0 or steps <= 0:
        raise ValueError("implicit_dynamics requires dt > 0 and steps > 0.")
    selected = implicit_time.newmark() if parameters is None else parameters
    am = selected.alpha_m
    af = selected.alpha_f
    beta = selected.beta
    gamma = selected.gamma
    effective_expression = (
        (1.0 - am) * mass.expression
        + (1.0 - af) * gamma * dt * (
            0 if damping is None else damping.expression
        )
        + (1.0 - af) * beta * dt**2 * stiffness.expression
    )
    V = state.a.value.function_space
    u_predictor = fem.Function(V, name="DisplacementPredictor")
    v_predictor = fem.Function(V, name="VelocityPredictor")
    u_alpha_predictor = fem.Function(V, name="DisplacementAlphaPredictor")
    v_alpha_predictor = fem.Function(V, name="VelocityAlphaPredictor")
    rhs = force.expression - am * ufl.action(
        mass.expression,
        state.a.value,
    )
    if damping is not None:
        rhs -= ufl.action(damping.expression, v_alpha_predictor)
    rhs -= ufl.action(stiffness.expression, u_alpha_predictor)
    source_bcs = _collect_bcs(constraints=constraints, bcs=bcs)
    acceleration_bcs = _zero_kinematic_bcs(source_bcs, V)
    problem = LinearVariationalProblem(
        bilinear_form=fem.form(effective_expression),
        linear_form=fem.form(rhs),
        solution=state.a_next.value,
        bcs=acceleration_bcs,
        solver_options=solver_options,
    )
    return ImplicitDynamicsStep(
        name=name,
        state=state,
        problem=problem,
        parameters=selected,
        dt=float(dt),
        steps=int(steps),
        displacement_predictor=u_predictor,
        velocity_predictor=v_predictor,
        displacement_alpha_predictor=u_alpha_predictor,
        velocity_alpha_predictor=v_alpha_predictor,
        study=study,
        update_load=update_load,
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        history_monitor=MechanicalEnergyMonitor(
            mass=mass,
            stiffness=stiffness,
        ),
        procedure=(
            procedures.newmark()
            if selected.method == "newmark"
            else procedures.generalized_alpha()
        ),
    )


def _zero_kinematic_bcs(source_bcs, V) -> list:
    result = []
    shape = V.element.value_shape
    value = (
        PETSc.ScalarType(0.0)
        if len(shape) == 0
        else np.zeros(shape, dtype=PETSc.ScalarType)
    )
    for bc in source_bcs:
        dof_indices = bc.dof_indices()
        dofs = dof_indices[0] if isinstance(dof_indices, tuple) else dof_indices
        result.append(fem.dirichletbc(value, dofs, V))
    return result


def _collect_bcs(*, constraints=None, bcs=None) -> list:
    result = []
    if bcs is not None:
        result.extend(_as_list(bcs))
    if constraints is not None:
        for item in _as_list(constraints):
            if hasattr(item, "bcs"):
                result.extend(item.bcs)
            elif hasattr(item, "bc"):
                result.append(item.bc)
            else:
                result.append(item)
    return result


def _reaction_field(residual_form, solution, *, name: str):
    import dolfinx.fem.petsc as fem_petsc
    from petsc4py import PETSc

    residual = fem_petsc.assemble_vector(fem.form(residual_form))
    residual.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    reaction = fem.Function(solution.function_space, name=name)
    values = residual.array_r
    reaction.x.array[: len(values)] = values
    reaction.x.scatter_forward()
    residual.destroy()
    return reaction


def _print_interval(print_every: int | None, steps: int) -> int:
    if print_every is None:
        return max(1, int(steps) // 10)
    return int(print_every)


def _save_interval(save_every: int | None, *, output, steps: int) -> int:
    if save_every is not None:
        return int(save_every)
    if output is None:
        return 0
    return int(steps)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _unwrap_result_field(field):
    return fields.unwrap(field)


def _field_location(field) -> str:
    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    discontinuous = bool(
        getattr(element, "discontinuous", False)
        or getattr(basix_element, "discontinuous", False)
    )
    return "cells" if discontinuous else "nodes"


def _projected_field_processing(field) -> dict[str, object]:
    """Describe the post-processing contract of a projected result field."""

    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    degree = getattr(basix_element, "degree", None)
    family = getattr(basix_element, "family", None)
    selected_degree = None if degree is None else int(degree)
    return {
        "source_position": "constitutive_expression",
        "method": "global_l2_projection",
        "representation": (
            "cell_average" if selected_degree == 0 else "discontinuous_field"
        ),
        "space_family": (
            None if family is None else str(getattr(family, "name", family))
        ),
        "space_degree": selected_degree,
        "nodal_extrapolation": False,
        "interelement_smoothing": False,
        "material_boundary_averaging": False,
    }


def _describe_asset(asset) -> object:
    if hasattr(asset, "as_dict"):
        return asset.as_dict()
    if hasattr(asset, "summary"):
        return asset.summary()
    return getattr(asset, "name", repr(asset))


def _require_study_analysis(study, analysis: str) -> None:
    if study is not None and hasattr(study, "require"):
        study.require(analysis=analysis)
