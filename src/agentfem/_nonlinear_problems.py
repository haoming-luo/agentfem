# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Incremental and affine nonlinear procedure implementations.

The stable factories remain in `agentfem.problems`. This module owns
load-path advancement, accepted/trial state transactions, cutback, nonlinear
checkpointing, and accepted-increment snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from ._problem_fields import reaction_field as _reaction_field
from .constraints.affine import AffineConstraintDualHistory
from .solvers import (
    AffineNewtonOptions,
    NewtonSolverOptions,
    NonlinearSolverOptions,
    SolveEvent,
    solve_affine_nonlinear_path,
    solve_nonlinear_problem,
)


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
                float(self.residual_norm) if np.isfinite(self.residual_norm) else None
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
    result_field_recovery: object | None = None
    result_field_role: str = "primary_subfield"
    snapshot_field_factory: object | None = None
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
        self.snapshots.append(
            _load_snapshot(
                0,
                0.0,
                self.solution,
                field_factory=self.snapshot_field_factory,
            )
        )
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
                if self.output_every is not None and (
                    len(history) % self.output_every == 0
                    or abs(factor - 1.0) <= 1.0e-12
                ):
                    self.snapshots.append(
                        _load_snapshot(
                            len(history),
                            factor,
                            self.solution,
                            solve_info=solve_info,
                            field_factory=self.snapshot_field_factory,
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

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve and complete one model-owned nonlinear result lifecycle."""

        from .results._nonlinear_step import from_incremental_nonlinear_step
        from .results.performance import attach_performance

        started = perf_counter()
        solution = self.solve()
        solve_seconds = perf_counter() - started
        result_started = perf_counter()
        result = from_incremental_nonlinear_step(
            self,
            solution,
            output=output,
            fields=fields,
            strict_output=strict_output,
            metadata=metadata,
        )
        return attach_performance(
            result,
            stages={
                "solve": solve_seconds,
                "result_assembly_and_output": perf_counter() - result_started,
                "total": perf_counter() - started,
            },
            solution=solution,
            source=self,
        )

    def reaction_field(self, *, name: str = "RF"):
        return _reaction_field(self.residual_form, self.solution, name=name)

    def summary(self) -> dict[str, object]:
        summary = {
            "kind": "incremental_nonlinear_variational_problem",
            "name": self.name,
            "num_bcs": len(self.bcs),
            "incrementation": (
                None if self.incrementation is None else self.incrementation.summary()
            ),
            "snapshot_count": len(self.snapshots),
            "primary_result_fields": (
                None
                if self.result_field_factory is None
                else tuple(self.primary_fields)
                if hasattr(self, "primary_fields")
                else "generated"
            ),
            "result_field_recovery": (
                "provider" if self.result_field_recovery is not None else "default"
            ),
            "result_field_role": self.result_field_role,
            "last_solve": (
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
            "procedure": (None if self.procedure is None else self.procedure.summary()),
        }
        if hasattr(self, "result_field_manifest"):
            summary["result_field_manifest"] = tuple(self.result_field_manifest)
        return summary


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
    result_field_factory: object | None = None
    result_field_recovery: object | None = None
    result_field_role: str = "primary_subfield"
    snapshot_field_factory: object | None = None
    state_transaction: object | None = None
    checkpoint_policy: object | None = None
    acceptance_check: object | None = None
    accepted_observers: tuple[object, ...] = ()
    last_solve_info: object | None = field(default=None, init=False)
    snapshots: list = field(default_factory=list, init=False)
    accepted_history_recorders: dict[str, object] = field(
        default_factory=dict,
        init=False,
    )
    execution_events: list[object] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    accepted_load_factor: float = field(default=0.0, init=False)
    accepted_increments: list[object] = field(default_factory=list, init=False)
    attempted_increments: list[object] = field(default_factory=list, init=False)
    next_increment_size: float | None = field(default=None, init=False)
    constraint_dual_history: AffineConstraintDualHistory = field(
        default_factory=AffineConstraintDualHistory,
        init=False,
    )

    def _capture_constraint_dual(self, load_factor: float) -> None:
        provider = getattr(self.constraint, "dual_evidence", None)
        if provider is None:
            return
        evidence = provider(self, load_factor=float(load_factor))
        if evidence is None:
            return
        outgoing_evidence = provider(
            self,
            load_factor=float(load_factor),
            path_side="right",
        )
        self.constraint_dual_history.append(
            load_factor,
            evidence,
            outgoing_evidence=outgoing_evidence,
        )

    def solve(self, *, until: float = 1.0):
        """Advance to an accepted load factor without discarding prior history."""

        selected_until = float(until)
        if not np.isfinite(selected_until) or not 0.0 < selected_until <= 1.0:
            raise ValueError("Affine solve until must be finite and in (0, 1].")
        if selected_until <= self.accepted_load_factor + 1.0e-12:
            raise ValueError(
                "Affine solve until must exceed the currently accepted load factor."
            )
        if self.output_every is not None and self.output_every <= 0:
            raise ValueError("Affine nonlinear output_every must be positive.")
        # Validate rollback support before initialization or any accepted
        # observer is allowed to mutate its history.
        initial_observer_state = _snapshot_accepted_observers(
            self.accepted_observers,
            comm=self.solution.function_space.mesh.comm,
        )
        initial_snapshots = list(self.snapshots)
        initial_events = list(self.execution_events)
        initial_solution = self.solution.x.array.copy()
        initial_transaction_state = (
            None
            if self.state_transaction is None
            else self.state_transaction.snapshot_accepted_boundary()
        )
        self.snapshots.clear()
        fresh = self.accepted_load_factor <= 1.0e-12
        try:
            if fresh:
                self.execution_events.clear()
                self.constraint_dual_history.clear()
                if self.state_transaction is not None and hasattr(
                    self.state_transaction, "initialize"
                ):
                    self.state_transaction.initialize()
            elif self.state_transaction is not None:
                transaction_factor = float(
                    getattr(self.state_transaction, "accepted_factor", -1.0)
                )
                if abs(transaction_factor - self.accepted_load_factor) > 1.0e-12:
                    raise RuntimeError(
                        "Affine problem and material transaction disagree on the "
                        "accepted load factor."
                    )
                if hasattr(self.state_transaction, "prepare_resume"):
                    self.state_transaction.prepare_resume()
            initial_snapshot = _load_snapshot(
                len(self.accepted_increments),
                self.accepted_load_factor,
                self.solution,
                zero=fresh,
                zero_fields=self.state_transaction is None,
                field_factory=self.snapshot_field_factory,
            )
            self.snapshots.append(initial_snapshot)
            for observer in self.accepted_observers:
                if not fresh and hasattr(observer, "prepare_resume"):
                    observer.prepare_resume(initial_snapshot)
                else:
                    observer.reset(initial_snapshot)
            if fresh:
                self._capture_constraint_dual(self.accepted_load_factor)
        except BaseException:
            self.solution.x.array[:] = initial_solution
            self.solution.x.scatter_forward()
            if self.state_transaction is not None:
                self.state_transaction.restore_accepted_boundary(
                    initial_transaction_state
                )
            self.snapshots[:] = initial_snapshots
            self.execution_events[:] = initial_events
            _restore_accepted_observers(initial_observer_state)
            raise

        pending_acceptance = None

        def acceptance_backup():
            return {
                "snapshots": list(self.snapshots),
                "observers": _snapshot_accepted_observers(
                    self.accepted_observers,
                    comm=self.solution.function_space.mesh.comm,
                ),
                "accepted_load_factor": self.accepted_load_factor,
                "accepted_increments": list(self.accepted_increments),
                "attempted_increments": list(self.attempted_increments),
                "next_increment_size": self.next_increment_size,
                "execution_events": list(self.execution_events),
                "checkpoints": list(self.checkpoints),
                "constraint_dual_history": (
                    self.constraint_dual_history.snapshot_runtime_state()
                ),
            }

        def restore_pending_acceptance():
            nonlocal pending_acceptance
            if pending_acceptance is None:
                return
            state = pending_acceptance
            self.snapshots[:] = state["snapshots"]
            _restore_accepted_observers(state["observers"])
            self.accepted_load_factor = state["accepted_load_factor"]
            self.accepted_increments[:] = state["accepted_increments"]
            self.attempted_increments[:] = state["attempted_increments"]
            self.next_increment_size = state["next_increment_size"]
            self.execution_events[:] = state["execution_events"]
            self.checkpoints[:] = state["checkpoints"]
            self.constraint_dual_history.restore_runtime_state(
                state["constraint_dual_history"]
            )
            pending_acceptance = None

        def capture(index, factor, solution, solve_info):
            nonlocal pending_acceptance
            if pending_acceptance is not None:
                raise RuntimeError(
                    "An accepted-boundary callback transaction is already active."
                )
            pending_acceptance = acceptance_backup()
            save_by_increment = (
                self.output_every is not None and index % self.output_every == 0
            )
            save_by_factor = any(
                abs(factor - value) <= 1.0e-12 for value in self.output_factors
            )
            should_save = (
                save_by_increment or save_by_factor or abs(factor - 1.0) <= 1.0e-12
            )
            try:
                accepted_snapshot = None
                if self.accepted_observers or should_save:
                    accepted_snapshot = _load_snapshot(
                        index,
                        factor,
                        solution,
                        solve_info=solve_info,
                        field_factory=self.snapshot_field_factory,
                    )
                for observer in self.accepted_observers:
                    observer.accept(accepted_snapshot)
                self._capture_constraint_dual(factor)
                if should_save:
                    self.snapshots.append(accepted_snapshot)
            except BaseException:
                restore_pending_acceptance()
                raise

        def accept_boundary(
            _solution,
            accepted_history,
            attempted_history,
            next_increment_size,
        ):
            """Synchronize and checkpoint one fully committed boundary."""

            nonlocal pending_acceptance
            if pending_acceptance is None:
                pending_acceptance = acceptance_backup()
            try:
                self.accepted_increments[:] = list(accepted_history)
                self.attempted_increments[:] = list(attempted_history)
                self.accepted_load_factor = float(accepted_history[-1].load_factor)
                self.next_increment_size = next_increment_size
                policy = self.checkpoint_policy
                if policy is not None:
                    increment = len(self.accepted_increments)
                    due = increment % int(policy.every) == 0
                    due = due or (
                        bool(policy.final)
                        and self.accepted_load_factor >= 1.0 - 1.0e-12
                    )
                    if due:
                        self.save_checkpoint(
                            policy.path(
                                step_name=self.name,
                                increment=increment,
                            ),
                            portable=bool(policy.portable),
                        )
                        _prune_affine_checkpoints(self)
            except BaseException:
                restore_pending_acceptance()
                raise
            pending_acceptance = None

        from .diagnostics import (
            SolveEventRecorder,
            StandardRunReporter,
            comm_of,
            compose_reporters,
        )

        recorder = SolveEventRecorder(self.execution_events)
        if self.progress is True:
            visible_reporter = StandardRunReporter(
                comm_of(self.solution),
                status_file=self.status_file,
            )
        elif self.progress in (False, None):
            visible_reporter = None
        else:
            visible_reporter = self.progress
        reporter = compose_reporters(recorder, visible_reporter)
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
            on_accepted_boundary=accept_boundary,
            on_acceptance_failure=restore_pending_acceptance,
            acceptance_check=self.acceptance_check,
            state_transaction=self.state_transaction,
            stop_factor=selected_until,
            accepted_history=self.accepted_increments,
            attempted_history=self.attempted_increments,
            next_increment_size=self.next_increment_size,
            reporter=reporter,
            step_name=self.name,
            step_number=self.step_number,
        )
        self.last_solve_info = info
        self.accepted_increments[:] = list(info.increments)
        self.attempted_increments[:] = list(info.attempts)
        self.accepted_load_factor = (
            self.accepted_load_factor
            if not info.increments
            else float(info.increments[-1].load_factor)
        )
        self.next_increment_size = info.next_increment_size
        return solution

    def _checkpoint_identity(self) -> dict[str, object]:
        """Return the partition-independent scientific identity of this path."""

        from .checkpointing import function_portable_identity

        transaction = self.state_transaction
        material = getattr(transaction, "material", getattr(self, "material", None))
        response = getattr(transaction, "response", getattr(self, "response", None))
        constraint_identity = (
            self.constraint.scientific_identity()
            if hasattr(self.constraint, "scientific_identity")
            else self.constraint.summary()
        )
        portable_nodal_state = (
            None
            if transaction is None or not hasattr(transaction, "portable_nodal_state")
            else transaction.portable_nodal_state()
        )
        solution_identity = (
            function_portable_identity(self.solution)
            if portable_nodal_state is None
            else {
                name: function_portable_identity(function)
                for name, function in sorted(portable_nodal_state.items())
            }
        )
        return {
            "step_name": self.name,
            "procedure": (
                self.procedure.summary()
                if hasattr(self.procedure, "summary")
                else self.procedure
            ),
            "material": (
                material.summary() if hasattr(material, "summary") else material
            ),
            "state_schema": (
                response.state.state_schema.summary() if response is not None else None
            ),
            "quadrature_degree": (
                None if response is None else int(response.state.degree)
            ),
            "incrementation": (
                self.incrementation.summary()
                if hasattr(self.incrementation, "summary")
                else self.incrementation
            ),
            "solver_options": (
                self.solver_options.summary()
                if hasattr(self.solver_options, "summary")
                else self.solver_options
            ),
            "solution": solution_identity,
            "constraint": constraint_identity,
            "accepted_history_recorders": {
                name: {"kind": type(recorder).__name__}
                for name, recorder in sorted(self.accepted_history_recorders.items())
            },
        }

    @staticmethod
    def _checkpoint_manifest_path(path) -> Path:
        selected = Path(path)
        if selected.name.endswith(".checkpoint.json"):
            return selected
        if selected.suffix:
            selected = selected.with_suffix("")
        return selected.with_name(selected.name + ".checkpoint.json")

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        """Save one accepted affine state and its complete path evidence.

        The stable stateful route always writes coordinate/cell-keyed state.
        This is deliberately stronger than a rank-local default: the same
        checkpoint can be resumed with a different compatible MPI partition.
        """

        if self.state_transaction is None:
            raise TypeError("Affine checkpointing requires a state transaction.")
        comm = self.solution.function_space.mesh.comm
        local_problem = None
        if (
            abs(
                float(self.state_transaction.accepted_factor)
                - self.accepted_load_factor
            )
            > 1.0e-12
        ):
            local_problem = (
                "Checkpointing is permitted only at a fully accepted material state."
            )
        elif not np.allclose(
            self.solution.x.array,
            self.state_transaction.accepted_solution.x.array,
            rtol=0.0,
            atol=1.0e-12,
        ):
            local_problem = "Checkpointing is permitted only when U equals U_ACCEPTED."
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(f"Rank {rank}: {problems[rank]}")
        # ``portable=False`` is retained for compatibility with the common
        # policy API; the public affine format is intentionally always portable.
        del portable
        from .checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
        )
        from .results import CheckpointRecord

        manifest = self._checkpoint_manifest_path(path)
        custom_nodal_state = (
            None
            if not hasattr(self.state_transaction, "portable_nodal_state")
            else self.state_transaction.portable_nodal_state()
        )
        nodal_state = (
            {
                "U": self.solution,
                "U_ACCEPTED": self.state_transaction.accepted_solution,
            }
            if custom_nodal_state is None
            else custom_nodal_state
        )
        bundle = save_portable_state_bundle(
            manifest,
            state=nodal_state,
        )
        quadrature = self.state_transaction.response.state.save(
            manifest.with_name(
                f"{manifest.name.removesuffix('.checkpoint.json')}."
                f"{bundle['generation']}.quadrature"
            ),
            material=self.state_transaction.material,
        )
        payload = {
            "schema": "agentfem.affine-stateful-checkpoint.v1",
            "identity": self._checkpoint_identity(),
            "coordinate_name": "load_factor",
            "coordinate": self.accepted_load_factor,
            "writer_rank_count": int(self.solution.function_space.mesh.comm.size),
            "portable": True,
            "nodal_state": bundle["record"],
            "nodal_identity": bundle["identities"],
            "quadrature_state": checkpoint_file_record(quadrature),
            "accepted_increments": [
                item.as_dict() for item in self.accepted_increments
            ],
            "attempted_increments": [
                item.as_dict() for item in self.attempted_increments
            ],
            "next_increment_size": self.next_increment_size,
            "execution_events": [
                item.as_dict() if hasattr(item, "as_dict") else dict(item)
                for item in self.execution_events
            ],
            "accepted_observer_state": {
                name: recorder.checkpoint_state()
                for name, recorder in sorted(self.accepted_history_recorders.items())
                if hasattr(recorder, "checkpoint_state")
            },
            "constraint_dual_history": (
                self.constraint_dual_history.checkpoint_state()
            ),
        }
        error = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest,
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"Affine checkpoint manifest write failed: {error}")
        comm.barrier()
        record = CheckpointRecord(
            name=f"{self.name}_increment_{len(self.accepted_increments)}",
            path=manifest,
            schema=payload["schema"],
            step_name=self.name,
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=True,
            metadata={
                "accepted_increment_count": len(self.accepted_increments),
                "writer_rank_count": int(comm.size),
                "role": "accepted_state",
            },
        )
        self.checkpoints[:] = [
            item for item in self.checkpoints if item.path != manifest
        ]
        self.checkpoints.append(record)
        return manifest

    def load_checkpoint(self, path) -> None:
        """Atomically restore an accepted affine state and resume metadata."""

        if self.state_transaction is None:
            raise TypeError("Affine checkpointing requires a state transaction.")
        from .checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )
        from .results import CheckpointRecord
        from .solvers import AffineLoadIncrementInfo

        manifest = self._checkpoint_manifest_path(path)
        comm = self.solution.function_space.mesh.comm
        envelope = None
        if comm.rank == 0:
            try:
                envelope = {
                    "payload": json.loads(manifest.read_text(encoding="utf-8")),
                    "error": None,
                }
            except Exception as exc:
                envelope = {
                    "payload": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        envelope = comm.bcast(envelope, root=0)
        if envelope["error"] is not None:
            raise RuntimeError(
                f"Affine checkpoint manifest read failed: {envelope['error']}"
            )
        payload = envelope["payload"]
        if payload.get("schema") != "agentfem.affine-stateful-checkpoint.v1":
            raise ValueError("Unsupported affine stateful checkpoint schema.")
        current_identity = json.loads(
            json.dumps(self._checkpoint_identity(), sort_keys=True)
        )
        if payload.get("identity") != current_identity:
            raise ValueError(
                "Affine checkpoint material, state, mesh, increment/solver "
                "control, constraint equations, or deformation history differ "
                "from the current analysis."
            )
        validation = None
        if comm.rank == 0:
            try:
                validate_checkpoint_record(
                    manifest.parent,
                    payload["nodal_state"],
                )
                quadrature_path = validate_checkpoint_record(
                    manifest.parent,
                    payload["quadrature_state"],
                )
                validation = {
                    "quadrature_path": str(quadrature_path),
                    "error": None,
                }
            except Exception as exc:
                validation = {
                    "quadrature_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        validation = comm.bcast(validation, root=0)
        if validation["error"] is not None:
            raise RuntimeError(
                f"Affine checkpoint payload validation failed: {validation['error']}"
            )
        quadrature_path = Path(validation["quadrature_path"])
        solution_backup = self.solution.x.array.copy()
        accepted_backup = self.state_transaction.accepted_solution.x.array.copy()
        transaction_backup = self.state_transaction.snapshot_runtime_state()
        lifecycle_backup = {
            "accepted_load_factor": self.accepted_load_factor,
            "accepted_increments": list(self.accepted_increments),
            "attempted_increments": list(self.attempted_increments),
            "next_increment_size": self.next_increment_size,
            "execution_events": list(self.execution_events),
            "last_solve_info": self.last_solve_info,
            "snapshots": list(self.snapshots),
            "checkpoints": list(self.checkpoints),
            "constraint_dual_history": (
                self.constraint_dual_history.snapshot_runtime_state()
            ),
        }
        observer_backup = {
            name: recorder.snapshot_runtime_state()
            for name, recorder in self.accepted_history_recorders.items()
            if hasattr(recorder, "snapshot_runtime_state")
        }
        try:
            custom_nodal_state = (
                None
                if not hasattr(self.state_transaction, "portable_nodal_state")
                else self.state_transaction.portable_nodal_state()
            )
            nodal_state = (
                {
                    "U": self.solution,
                    "U_ACCEPTED": self.state_transaction.accepted_solution,
                }
                if custom_nodal_state is None
                else custom_nodal_state
            )
            load_portable_state_bundle(
                manifest,
                state=nodal_state,
                record=payload["nodal_state"],
                identities=payload["nodal_identity"],
            )
            if custom_nodal_state is not None:
                restore_nodal_state = getattr(
                    self.state_transaction,
                    "restore_portable_nodal_state",
                    None,
                )
                if restore_nodal_state is None:
                    raise TypeError(
                        "A custom portable nodal-state provider must also "
                        "implement restore_portable_nodal_state()."
                    )
                restore_nodal_state(nodal_state)
            self.state_transaction.response.state.load(
                quadrature_path,
                material=self.state_transaction.material,
            )
            local_equal = np.allclose(
                self.solution.x.array,
                self.state_transaction.accepted_solution.x.array,
                rtol=0.0,
                atol=1.0e-12,
            )
            if not comm.allreduce(bool(local_equal), op=MPI.LAND):
                raise ValueError("Affine checkpoint U and U_ACCEPTED differ.")
            accepted = [
                AffineLoadIncrementInfo.from_dict(item)
                for item in payload["accepted_increments"]
            ]
            attempted = [
                AffineLoadIncrementInfo.from_dict(item)
                for item in payload["attempted_increments"]
            ]
            coordinate = float(payload["coordinate"])
            if not accepted or abs(accepted[-1].load_factor - coordinate) > 1.0e-12:
                raise ValueError(
                    "Affine checkpoint coordinate and accepted history disagree."
                )
            self.accepted_increments[:] = accepted
            self.attempted_increments[:] = attempted
            self.accepted_load_factor = coordinate
            self.next_increment_size = payload.get("next_increment_size")
            self.execution_events[:] = [
                SolveEvent.from_dict(item)
                for item in payload.get("execution_events", ())
            ]
            self.constraint_dual_history.restore_checkpoint_state(
                payload.get("constraint_dual_history"),
                accepted_factor=coordinate,
            )
            self.state_transaction.accepted_factor = coordinate
            self.state_transaction.prepare_resume()
            observer_state = payload.get("accepted_observer_state", {})
            if set(observer_state) != set(self.accepted_history_recorders):
                raise ValueError(
                    "Affine checkpoint observer history differs from the "
                    "current output lifecycle."
                )
            current_snapshot = _load_snapshot(
                len(accepted),
                coordinate,
                self.solution,
                solve_info=accepted[-1],
                field_factory=self.snapshot_field_factory,
            )
            for name, record in observer_state.items():
                recorder = self.accepted_history_recorders[name]
                if not hasattr(recorder, "restore_checkpoint_state"):
                    raise TypeError(f"Accepted observer {name!r} is not restartable.")
                recorder.restore_checkpoint_state(
                    record,
                    current_snapshot=current_snapshot,
                )
        except Exception:
            self.solution.x.array[:] = solution_backup
            self.solution.x.scatter_forward()
            self.state_transaction.accepted_solution.x.array[:] = accepted_backup
            self.state_transaction.accepted_solution.x.scatter_forward()
            self.state_transaction.restore_runtime_state(transaction_backup)
            self.accepted_load_factor = lifecycle_backup["accepted_load_factor"]
            self.accepted_increments[:] = lifecycle_backup["accepted_increments"]
            self.attempted_increments[:] = lifecycle_backup["attempted_increments"]
            self.next_increment_size = lifecycle_backup["next_increment_size"]
            self.execution_events[:] = lifecycle_backup["execution_events"]
            self.last_solve_info = lifecycle_backup["last_solve_info"]
            self.snapshots[:] = lifecycle_backup["snapshots"]
            self.checkpoints[:] = lifecycle_backup["checkpoints"]
            self.constraint_dual_history.restore_runtime_state(
                lifecycle_backup["constraint_dual_history"]
            )
            for name, state in observer_backup.items():
                recorder = self.accepted_history_recorders.get(name)
                if recorder is not None and hasattr(
                    recorder,
                    "restore_runtime_state",
                ):
                    recorder.restore_runtime_state(state)
            raise
        record = CheckpointRecord(
            name=f"{self.name}_restart_{len(self.accepted_increments)}",
            path=manifest,
            schema=payload["schema"],
            step_name=self.name,
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=True,
            metadata={
                "writer_rank_count": payload.get("writer_rank_count"),
                "reader_rank_count": int(comm.size),
                "restart_mode": "portable_coordinate_and_cell_keyed_state",
                "role": "restart_source",
            },
        )
        self.checkpoints.append(record)

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve and complete one affine nonlinear result lifecycle."""

        from .results._nonlinear_step import from_affine_nonlinear_step
        from .results.performance import attach_performance

        started = perf_counter()
        solution = self.solve()
        solve_seconds = perf_counter() - started
        result_started = perf_counter()
        result = from_affine_nonlinear_step(
            self,
            solution,
            output=output,
            fields=fields,
            strict_output=strict_output,
            metadata=metadata,
        )
        return attach_performance(
            result,
            stages={
                "solve": solve_seconds,
                "result_assembly_and_output": perf_counter() - result_started,
                "total": perf_counter() - started,
            },
            solution=solution,
            source=self,
        )

    def summary(self) -> dict[str, object]:
        summary = {
            "kind": "affine_nonlinear_variational_problem",
            "name": self.name,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "constraint": self.constraint.summary(),
            "load_factors": self.load_factors,
            "incrementation": (
                None if self.incrementation is None else self.incrementation.summary()
            ),
            "output_every": self.output_every,
            "output_factors": self.output_factors,
            "snapshot_count": len(self.snapshots),
            "accepted_observers": tuple(
                observer.summary() for observer in self.accepted_observers
            ),
            "execution_event_count": len(self.execution_events),
            "accepted_load_factor": self.accepted_load_factor,
            "accepted_increment_count": len(self.accepted_increments),
            "attempted_increment_count": len(self.attempted_increments),
            "next_increment_size": self.next_increment_size,
            "constraint_dual_sample_count": len(self.constraint_dual_history.records),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "checkpoint_count": len(self.checkpoints),
            "state_transaction": (
                None
                if self.state_transaction is None
                else (
                    self.state_transaction.summary()
                    if hasattr(self.state_transaction, "summary")
                    else {"kind": type(self.state_transaction).__name__}
                )
            ),
            "step_number": self.step_number,
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else AffineNewtonOptions().summary()
            ),
            "last_solve": (
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
            "procedure": (None if self.procedure is None else self.procedure.summary()),
            "result_field_recovery": (
                "provider" if self.result_field_recovery is not None else "default"
            ),
            "result_field_role": self.result_field_role,
        }
        if hasattr(self, "mixed_formulation"):
            summary["numerical_formulation"] = dict(self.mixed_formulation)
        return summary


def _prune_affine_checkpoints(step) -> None:
    """Apply the common retention policy to portable affine checkpoints."""

    policy = getattr(step, "checkpoint_policy", None)
    keep_last = None if policy is None else getattr(policy, "keep_last", None)
    scheduled = [
        record
        for record in step.checkpoints
        if record.metadata.get("role") != "restart_source"
    ]
    if keep_last is None or len(scheduled) <= int(keep_last):
        return
    from .checkpointing import remove_stateful_checkpoint

    obsolete = scheduled[: -int(keep_last)]
    comm = step.solution.function_space.mesh.comm
    for record in obsolete:
        remove_stateful_checkpoint(record.path, comm=comm)
    removed = {id(record) for record in obsolete}
    step.checkpoints[:] = [
        record for record in step.checkpoints if id(record) not in removed
    ]


def _snapshot_accepted_observers(
    observers,
    *,
    comm=None,
) -> tuple[tuple[object, object], ...]:
    """Capture every accepted-history observer before a boundary mutation."""

    snapshots = []
    local_problem = None
    try:
        for observer in observers:
            capture = getattr(observer, "snapshot_runtime_state", None)
            restore = getattr(observer, "restore_runtime_state", None)
            if not callable(capture) or not callable(restore):
                raise TypeError(
                    "Accepted-increment observers must provide "
                    "snapshot_runtime_state() and restore_runtime_state() so "
                    "output/checkpoint failures cannot split the solve lifecycle; "
                    f"got {type(observer).__name__}."
                )
            snapshots.append((observer, capture()))
    except Exception as exc:
        if comm is None or comm.size == 1:
            raise
        local_problem = f"{type(exc).__name__}: {exc}"
    if comm is not None and comm.size > 1:
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(
                f"Rank {rank}: accepted-observer state snapshot failed: "
                f"{problems[rank]}"
            )
    return tuple(snapshots)


def _restore_accepted_observers(snapshots) -> None:
    """Restore observers captured by ``_snapshot_accepted_observers``."""

    for observer, state in snapshots:
        observer.restore_runtime_state(state)


@dataclass(frozen=True)
class LoadIncrementSnapshot:
    """A copied solution state at one nonlinear load factor."""

    index: int
    load_factor: float
    solution: object
    solve_info: object | None = None
    fields: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "index": self.index,
            "load_factor": self.load_factor,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "fields": tuple(self.fields),
            "solve": (None if self.solve_info is None else self.solve_info.as_dict()),
        }


def _load_snapshot(
    index: int,
    load_factor: float,
    solution,
    *,
    solve_info=None,
    zero: bool = False,
    zero_fields: bool | None = None,
    field_factory=None,
) -> LoadIncrementSnapshot:
    auxiliary = {}
    if field_factory is None:
        selected = solution.value if hasattr(solution, "value") else solution
    else:
        selected, auxiliary = field_factory()
    copied = fem.Function(
        selected.function_space,
        name=getattr(selected, "name", "Solution"),
    )
    if not zero:
        copied.x.array[:] = selected.x.array
        copied.x.scatter_forward()
    selected_zero_fields = bool(zero) if zero_fields is None else bool(zero_fields)
    copied_fields = {}
    for name, value in dict(auxiliary).items():
        source = getattr(value, "function", value)
        field_copy = fem.Function(
            source.function_space,
            name=getattr(source, "name", str(name)),
        )
        if not selected_zero_fields:
            field_copy.x.array[:] = source.x.array
            field_copy.x.scatter_forward()
        if all(hasattr(value, item) for item in ("points", "weights", "value_shape")):
            from .constitutive.quadrature import QuadratureField

            copied_fields[str(name)] = QuadratureField(
                function=field_copy,
                points=np.asarray(value.points, dtype=float).copy(),
                weights=np.asarray(value.weights, dtype=float).copy(),
                value_shape=tuple(value.value_shape),
            )
        else:
            copied_fields[str(name)] = field_copy
    return LoadIncrementSnapshot(
        index=int(index),
        load_factor=float(load_factor),
        solution=copied,
        solve_info=solve_info,
        fields=copied_fields,
    )


__all__ = (
    "AffineNonlinearVariationalProblem",
    "IncrementalNonlinearVariationalProblem",
    "LoadIncrementSnapshot",
    "NonlinearLoadIncrementInfo",
    "NonlinearLoadPathInfo",
)
