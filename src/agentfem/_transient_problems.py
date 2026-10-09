# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Transient procedure implementations behind `agentfem.problems`.

The public compatibility module retains factory functions. This module owns
time advancement, accepted-state cadence, progress, checkpointing, and restart;
scientific result assembly remains in `agentfem.results`.
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

from . import assembly
from . import checkpointing
from . import fields
from . import time
from .diagnostics import PerformanceLedger
from .events import SolveEvent
from ._operator_lifecycle import OperatorLifecycleLedger, boundary_dof_identity
from .solvers import LinearSolverOptions


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
    history_every: int = 1
    procedure: object | None = None
    history_monitor: object | None = None
    stability: object | None = None
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    last_output_backend: str | None = field(default=None, init=False)
    last_output_layout: str | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    performance: PerformanceLedger = field(
        default_factory=PerformanceLedger,
        init=False,
    )

    def checkpoint_capabilities(self) -> checkpointing.CheckpointCapabilities:
        """Declare the durable state and MPI restart contract."""

        return _transient_checkpoint_capabilities(
            "displacement",
            "velocity",
            "acceleration",
            "accepted time/history ledger",
        )

    def operator_lifecycle_summary(self) -> dict[str, object]:
        """Describe how typed inputs enter the matrix-free Explicit route.

        Explicit central difference does not retain a prepared global tangent,
        so every accepted increment reevaluates the residual after applying the
        time-input plan.  A preflight stability estimate is still a separate
        scientific promise: operator- or state-changing inputs require the
        declared bound to remain valid for the complete path.
        """

        input_summary = time.input_summary(self.update_load)
        stability_scope = "not_declared"
        if self.stability is not None:
            stability_scope = (
                "caller_must_bound_complete_path"
                if input_summary["changes_operator"]
                else "fixed_preflight_bound"
            )
        return {
            "kind": "explicit_residual_lifecycle",
            "time_inputs": input_summary,
            "operator_policy": "evaluate_residual_each_increment",
            "prepared_operator_reused": False,
            "state_acceptance": "commit_after_integrator_acceptance",
            "stability_scope": stability_scope,
        }

    def initialize_from_preload(
        self,
        displacement,
        *,
        source_step=None,
        initial_velocity=None,
        mode: str = "equilibrium",
        force_tolerance: float = 1.0e-8,
        source_energy: float | None = None,
    ):
        """Transfer a quasi-static configuration into this Explicit step.

        Reactions at held strong constraints are excluded from the free-force
        equilibrium norm.  The transfer uses the step's own mass, residual,
        constraint projection, and energy monitor so the public workflow does
        not need to reconstruct solver plumbing.
        """

        from . import fracture
        from .time import explicit as explicit_time

        monitor = getattr(self.history_monitor, "energy", None)

        def project(field) -> None:
            explicit_time.project_homogeneous_kinematics(
                field,
                prescribed=self.prescribed,
                constraints=self.constraints,
            )

        return fracture.transfer_preload_to_explicit(
            displacement,
            state=self.state,
            mass=self.integrator.mass,
            residual=self.residual,
            initial_velocity=initial_velocity,
            mode=mode,
            force_tolerance=force_tolerance,
            acceleration_projection=project,
            energy_monitor=monitor,
            source_energy=source_energy,
            source_step=getattr(source_step, "name", source_step),
            destination_step=self.name,
        )

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
        _validate_checkpoint_policy(self)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        run_started = perf_counter()
        self.integrator.performance = self.performance
        _bind_performance_ledger(self.residual, self.performance)
        if self.completed_steps == 0:
            self.performance.reset()
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
        selected_fields = fields or (
            self.state.u.value,
            self.state.v.value,
            self.state.a.value,
        )
        output_fields, live_field_sets = _transient_output_fields(selected_fields)
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = output_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )

        _emit_transient_started(reporter, self)
        if self.completed_steps == 0 and hasattr(
            self.residual, "initialize_accepted_state"
        ):
            self.residual.initialize_accepted_state(time=0.0)
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
            self.performance.add("run_wall", perf_counter() - run_started)
            return self

        if domain is None:
            domain = self.state.u.function_space.mesh
        series, actual_output, backend, layout = _transient_result_series(
            self.last_output,
            domain,
        )
        self.last_output = actual_output
        self.last_output_backend = backend
        self.last_output_layout = layout
        with series as xdmf:
            _refresh_transient_output_fields(live_field_sets)
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
                    _refresh_transient_output_fields(live_field_sets)
                    xdmf.write_fields(info.time, *output_fields)
        _emit_transient_completed(reporter, self)
        self.performance.add("run_wall", perf_counter() - run_started)
        return self

    def solve(self):
        self.run()
        return self.state.u.value

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        history=(),
        progress=None,
        comm=None,
        metadata=None,
    ):
        return _solve_transient_result(
            self,
            solution=self.state.u.value,
            default_fields=(
                self.state.u.value,
                self.state.v.value,
                self.state.a.value,
            ),
            output=output,
            fields=fields,
            history=history,
            progress=progress,
            comm=comm,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool = False) -> Path:
        """Save explicit state, optionally portable across MPI partitions."""

        return _save_transient_checkpoint(
            self,
            path,
            {
                "displacement": self.state.u,
                "velocity": self.state.v,
                "acceleration": self.state.a,
            },
            portable=portable,
        )

    def load_checkpoint(self, path) -> None:
        """Restore explicit state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {
                "displacement": self.state.u,
                "velocity": self.state.v,
                "acceleration": self.state.a,
            },
        )
        self.integrator.last_residual_owned = None

    def _advance_one(self, t: float) -> None:
        accepted = self.state.snapshot()
        previous_residual = getattr(self.integrator, "last_residual_owned", None)
        residual_state = (
            getattr(self.residual, "transaction_snapshot", self.residual.snapshot)()
            if hasattr(self.residual, "snapshot") and hasattr(self.residual, "restore")
            else None
        )
        try:
            if self.update_load is not None:
                from .provenance import collective_call

                collective_call(
                    lambda: self.update_load(t),
                    comm=fields.unwrap(self.state.u).function_space.mesh.comm,
                    label="Explicit time inputs",
                )
            if hasattr(self.residual, "update_time"):
                self.residual.update_time(t)
            self.integrator.step(
                self.dt,
                time=t,
                residual_operator=self.residual,
                prescribed=self.prescribed,
                constraints=self.constraints,
            )
            if hasattr(self.residual, "commit"):
                self.residual.commit()
        except Exception as failure:
            accepted_time = self.completed_steps * self.dt
            try:
                if residual_state is not None:
                    self.residual.restore(residual_state)
                elif hasattr(self.residual, "rollback"):
                    self.residual.rollback()
            except Exception as recovery_failure:
                failure.add_note(f"Residual rollback failed: {recovery_failure}")
            callbacks = [self.update_load, getattr(self.residual, "update_time", None)]
            callbacks.extend(getattr(item, "update", None) for item in self.prescribed)
            for callback in callbacks:
                if callable(callback):
                    try:
                        callback(accepted_time)
                    except Exception as recovery_failure:
                        failure.add_note(f"Could not restore accepted-time input: {recovery_failure}")
            self.state.restore(accepted)
            self.integrator.last_residual_owned = previous_residual
            raise

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
            "history_every": self.history_every,
            "history_evaluation_every": 1,
            "time_inputs": time.input_summary(self.update_load),
            "operator_lifecycle": self.operator_lifecycle_summary(),
            "performance": self.performance.summary(),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "checkpoint_capabilities": self.checkpoint_capabilities().summary(
                policy=self.checkpoint_policy
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "stability": (
                None
                if self.stability is None
                else (
                    self.stability.summary()
                    if hasattr(self.stability, "summary")
                    else self.stability
                )
            ),
            "integrator": (
                self.integrator.summary()
                if hasattr(self.integrator, "summary")
                else repr(self.integrator)
            ),
            "residual": (
                self.residual.summary()
                if hasattr(self.residual, "summary")
                else repr(self.residual)
            ),
            "num_prescribed": len(self.prescribed),
            "num_constraints": len(self.constraints),
            "procedure": (None if self.procedure is None else self.procedure.summary()),
        }


@dataclass
class ImplicitDynamicsStep:
    """Linear Newmark/generalized-alpha structural-dynamics step."""

    name: str
    state: object
    problem: object
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
    operator_policy: str = "auto"
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    last_output_backend: str | None = field(default=None, init=False)
    last_output_layout: str | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    performance: PerformanceLedger = field(
        default_factory=PerformanceLedger,
        init=False,
    )

    def checkpoint_capabilities(self) -> checkpointing.CheckpointCapabilities:
        """Declare the durable state and MPI restart contract."""

        return _transient_checkpoint_capabilities(
            "displacement",
            "velocity",
            "acceleration",
            "accepted time/history ledger",
        )
    _selected_operator_policy: str = field(default="", init=False, repr=False)
    _operator_policy_reason: str = field(default="", init=False, repr=False)
    _prepared_problem: object | None = field(default=None, init=False, repr=False)
    _operator_fingerprint_value: object | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _operator_ledger: OperatorLifecycleLedger = field(
        default_factory=OperatorLifecycleLedger, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        selected = str(self.operator_policy).strip().lower().replace("-", "_")
        allowed = {"auto", "reuse", "refresh_each_step"}
        if selected not in allowed:
            raise ValueError(
                "operator_policy must be 'auto', 'reuse', or 'refresh_each_step'."
            )
        self.operator_policy = selected
        time_effects = time.input_effects(self.update_load)
        changes_operator = bool(
            time_effects
            & {
                time.TimeInputEffect.OPERATOR,
                time.TimeInputEffect.STATE,
            }
        )
        if selected == "auto":
            if changes_operator:
                self._selected_operator_policy = "refresh_each_step"
                self._operator_policy_reason = (
                    "time-input contract changes operator or state"
                )
            else:
                self._selected_operator_policy = "reuse"
                self._operator_policy_reason = (
                    "fixed effective operator with RHS/output-only time inputs"
                )
        elif selected == "reuse" and changes_operator:
            effects = ", ".join(sorted(item.value for item in time_effects))
            raise ValueError(
                "AFM-DYNAMICS-OPERATOR-002: operator_policy='reuse' conflicts "
                f"with time-input effects [{effects}]. Use 'auto' or "
                "'refresh_each_step', or narrow a custom callback with "
                "agentfem.time.input_update(..., effects=...)."
            )
        else:
            self._selected_operator_policy = selected
            self._operator_policy_reason = "explicit user policy"

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
        _validate_checkpoint_policy(self)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        run_started = perf_counter()
        if self.completed_steps == 0:
            self.performance.reset()
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
            try:
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
            except BaseException:
                self.close()
                raise
            _emit_transient_completed(reporter, self)
            self.performance.add("run_wall", perf_counter() - run_started)
            if self.completed_steps >= self.steps:
                self.close()
            return self
        domain = self.state.u.function_space.mesh
        series, actual_output, backend, layout = _transient_result_series(
            self.last_output,
            domain,
        )
        self.last_output = actual_output
        self.last_output_backend = backend
        self.last_output_layout = layout
        try:
            with series as xdmf:
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
        except BaseException:
            self.close()
            raise
        _emit_transient_completed(reporter, self)
        self.performance.add("run_wall", perf_counter() - run_started)
        if self.completed_steps >= self.steps:
            self.close()
        return self

    def solve(self):
        self.run()
        return self.state.u.value

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        history=(),
        progress=None,
        comm=None,
        metadata=None,
    ):
        return _solve_transient_result(
            self,
            solution=self.state.u.value,
            default_fields=(
                self.state.u.value,
                self.state.v.value,
                self.state.a.value,
            ),
            output=output,
            fields=fields,
            history=history,
            progress=progress,
            comm=comm,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool = False) -> Path:
        """Save implicit state, optionally portable across MPI partitions."""

        return _save_transient_checkpoint(
            self,
            path,
            {
                "displacement": self.state.u,
                "velocity": self.state.v,
                "acceleration": self.state.a,
            },
            portable=portable,
        )

    def load_checkpoint(self, path) -> None:
        """Restore implicit state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {
                "displacement": self.state.u,
                "velocity": self.state.v,
                "acceleration": self.state.a,
            },
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
            u.x.array + dt * v.x.array + dt**2 * (0.5 - p.beta) * a.x.array
        )
        v_predictor.x.array[:] = v.x.array + dt * (1.0 - p.gamma) * a.x.array
        self.displacement_alpha_predictor.x.array[:] = (
            1.0 - p.alpha_f
        ) * u_predictor.x.array + p.alpha_f * u.x.array
        self.velocity_alpha_predictor.x.array[:] = (
            1.0 - p.alpha_f
        ) * v_predictor.x.array + p.alpha_f * v.x.array
        for function in (
            u_predictor,
            v_predictor,
            self.displacement_alpha_predictor,
            self.velocity_alpha_predictor,
        ):
            function.x.scatter_forward()
        if self.update_load is not None:
            evaluation_time = (1.0 - p.alpha_f) * time_value + p.alpha_f * (
                time_value - dt
            )
            self.update_load(evaluation_time)
        self._solve_problem()
        self.state.u_next.value.x.array[:] = (
            u_predictor.x.array + p.beta * dt**2 * self.state.a_next.value.x.array
        )
        self.state.v_next.value.x.array[:] = (
            v_predictor.x.array + p.gamma * dt * self.state.a_next.value.x.array
        )
        self.state.u_next.value.x.scatter_forward()
        self.state.v_next.value.x.scatter_forward()
        self.state.advance_state()

    def _solve_problem(self) -> None:
        """Solve one effective system under the declared operator policy."""

        if self._selected_operator_policy == "refresh_each_step":
            started = perf_counter()
            self.problem.solve()
            self.performance.add("linear_system_solve", perf_counter() - started)
            self._record_lifecycle(
                self.problem.last_lifecycle_summary,
                accumulate=True,
            )
            return

        fingerprint = self._operator_fingerprint()
        if self._operator_fingerprint_value is None:
            self._operator_fingerprint_value = fingerprint
        elif fingerprint != self._operator_fingerprint_value:
            self.close()
            raise RuntimeError(
                "AFM-DYNAMICS-OPERATOR-001: the effective operator identity "
                "changed during a reuse lifecycle. Build a new Step or use "
                "operator_policy='refresh_each_step'."
            )
        if self._prepared_problem is None:
            started = perf_counter()
            self._prepared_problem = self.problem.prepare()
            self._operator_ledger.begin_prepared()
            self.performance.add("matrix_preparation", perf_counter() - started)
        try:
            started = perf_counter()
            self.problem.solve_prepared(self._prepared_problem)
            self.performance.add("linear_system_solve", perf_counter() - started)
        except Exception:
            self.close()
            raise
        self._record_lifecycle(
            self.problem.last_lifecycle_summary,
            accumulate=False,
        )

    def _operator_fingerprint(self) -> tuple[object, ...]:
        """Return the runtime invariants required by fixed-operator reuse."""

        V = self.problem.solution.function_space
        index_map = V.dofmap.index_map
        parameters = (
            self.parameters.summary()
            if hasattr(self.parameters, "summary")
            else repr(self.parameters)
        )
        return (
            float(self.dt),
            json.dumps(parameters, sort_keys=True, default=str),
            int(index_map.size_global),
            int(V.dofmap.index_map_bs),
            boundary_dof_identity(self.problem.bcs),
        )

    def _record_lifecycle(self, summary, *, accumulate: bool) -> None:
        self._operator_ledger.record(
            summary, self.problem.last_solve_info, accumulate=accumulate,
        )

    def operator_lifecycle_summary(self) -> dict[str, object]:
        """Return inspectable evidence for operator reuse or refresh."""

        return {
            "kind": "transient_linear_operator_lifecycle",
            "requested_policy": self.operator_policy,
            "selected_policy": self._selected_operator_policy,
            "selection_reason": self._operator_policy_reason,
            "time_inputs": time.input_summary(self.update_load),
            "matrix_reused": self._selected_operator_policy == "reuse",
            **self._operator_ledger.summary(),
        }

    def close(self) -> None:
        """Release a live prepared operator without discarding its evidence."""

        prepared = self._prepared_problem
        self._prepared_problem = None
        if prepared is not None:
            prepared.close()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "implicit_dynamics_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "procedure": (None if self.procedure is None else self.procedure.summary()),
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
            "checkpoint_capabilities": self.checkpoint_capabilities().summary(
                policy=self.checkpoint_policy
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "operator_lifecycle": self.operator_lifecycle_summary(),
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
    problem: object
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
    operator_policy: str = "auto"
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    last_output_backend: str | None = field(default=None, init=False)
    last_output_layout: str | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    captured_histories: list[object] = field(default_factory=list, init=False)
    performance: PerformanceLedger = field(
        default_factory=PerformanceLedger,
        init=False,
    )

    def checkpoint_capabilities(self) -> checkpointing.CheckpointCapabilities:
        """Declare the durable state and MPI restart contract."""

        return _transient_checkpoint_capabilities(
            "current field",
            "previous field",
            "accepted time/history ledger",
        )
    _selected_operator_policy: str = field(default="", init=False, repr=False)
    _operator_policy_reason: str = field(default="", init=False, repr=False)
    _prepared_problem: object | None = field(default=None, init=False, repr=False)
    _operator_fingerprint_value: object | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _operator_ledger: OperatorLifecycleLedger = field(
        default_factory=OperatorLifecycleLedger, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        selected = str(self.operator_policy).strip().lower().replace("-", "_")
        allowed = {"auto", "reuse", "refresh_each_step"}
        if selected not in allowed:
            raise ValueError(
                "operator_policy must be 'auto', 'reuse', or 'refresh_each_step'."
            )
        self.operator_policy = selected
        linear_problem = getattr(self.problem, "problem", None)
        supports_reuse = callable(getattr(linear_problem, "prepare", None))
        input_effects = time.input_effects(self.update_load)
        changes_operator = bool(
            input_effects
            & {
                time.TimeInputEffect.OPERATOR,
                time.TimeInputEffect.STATE,
            }
        )
        if selected == "reuse" and not supports_reuse:
            raise ValueError(
                "AFM-TRANSIENT-OPERATOR-003: operator_policy='reuse' is not "
                "available for a nonlinear first-order residual. Use 'auto' "
                "or 'refresh_each_step'."
            )
        if selected == "reuse" and changes_operator:
            effects = ", ".join(sorted(item.value for item in input_effects))
            raise ValueError(
                "AFM-TRANSIENT-OPERATOR-002: operator_policy='reuse' conflicts "
                f"with time-input effects [{effects}]. Use 'auto' or "
                "'refresh_each_step', or narrow a custom callback with "
                "agentfem.time.input_update(..., effects=...)."
            )
        if selected == "auto" and not supports_reuse:
            self._selected_operator_policy = "refresh_each_step"
            self._operator_policy_reason = (
                "nonlinear residual requires per-step assembly"
            )
        elif selected == "auto" and changes_operator:
            self._selected_operator_policy = "refresh_each_step"
            self._operator_policy_reason = (
                "time-input contract changes operator or state"
            )
        elif selected == "auto":
            self._selected_operator_policy = "reuse"
            self._operator_policy_reason = (
                "fixed effective operator with RHS/output-only time inputs"
            )
        else:
            self._selected_operator_policy = selected
            self._operator_policy_reason = "explicit user policy"

    def capture_history(
        self,
        source=None,
        *,
        name: str | None = None,
        unit: str | None = None,
        every: int = 1,
        interpolation: str = "linear",
        outside: str = "error",
    ):
        """Capture a scalar field at accepted physical times.

        The returned :class:`agentfem.histories.FieldHistory` can be passed
        directly to a later thermo-mechanical or creep step.
        """

        from . import histories

        selected = self.current if source is None else source
        recorder = histories.FieldHistory(
            selected,
            name=name or getattr(getattr(selected, "value", selected), "name", "field"),
            unit=unit,
            every=every,
            interpolation=interpolation,
            outside=outside,
            metadata={
                "source_step": getattr(self, "name", type(self).__name__),
                "source_procedure": (
                    self.procedure.summary()
                    if hasattr(self.procedure, "summary")
                    else None
                ),
                "source_study": (
                    self.study.summary() if hasattr(self.study, "summary") else None
                ),
                "accepted_time_only": True,
                "transfer_role": "sequential_field_input",
            },
        )
        self.captured_histories.append(recorder)
        return recorder

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
        _validate_checkpoint_policy(self)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        run_started = perf_counter()
        if self.completed_steps == 0:
            self.performance.reset()
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
        self._record_captured_histories(force=True)

        def advance(info):
            current_rollback = self.current.x.array.copy()
            previous_rollback = self.previous.x.array.copy()
            accepted_time = self.completed_steps * self.dt
            try:
                if self.update_load is not None:
                    self.update_load(info.time)
                self._solve_problem()
            except Exception as failure:
                if self.update_load is not None:
                    try:
                        self.update_load(accepted_time)
                    except Exception as restore_failure:
                        failure.add_note(
                            f"Could not restore load at accepted time {accepted_time}: {restore_failure}"
                        )
                self.current.x.array[:] = current_rollback
                self.current.x.scatter_forward()
                self.previous.x.array[:] = previous_rollback
                self.previous.x.scatter_forward()
                raise
            self.previous.x.array[:] = self.current.x.array
            self.previous.x.scatter_forward()
            _accept_transient_increment(
                self,
                info,
                reporter,
                selected_progress,
                self.current,
                selected_comm,
            )
            self._record_captured_histories(force=self.completed_steps == self.steps)

        try:
            if output is None:
                for info in stepper:
                    advance(info)
                _emit_transient_completed(reporter, self)
                return self
            domain = self.current.function_space.mesh
            series, actual_output, backend, layout = _transient_result_series(
                self.last_output,
                domain,
            )
            self.last_output = actual_output
            self.last_output_backend = backend
            self.last_output_layout = layout
            with series as xdmf:
                xdmf.write_fields(self.completed_steps * self.dt, *selected_fields)
                for info in stepper:
                    advance(info)
                    if info.should_save:
                        xdmf.write_fields(info.time, *selected_fields)
            _emit_transient_completed(reporter, self)
            return self
        except BaseException:
            self.close()
            raise
        finally:
            self.performance.add("run_wall", perf_counter() - run_started)
            if self.completed_steps >= self.steps:
                self.close()

    def _solve_problem(self) -> None:
        """Solve one first-order increment under the declared lifecycle."""

        linear_problem = getattr(self.problem, "problem", None)
        if self._selected_operator_policy == "refresh_each_step":
            started = perf_counter()
            self.problem.solve()
            stage = (
                "linear_system_solve"
                if linear_problem is not None
                else "nonlinear_system_solve"
            )
            self.performance.add(stage, perf_counter() - started)
            if linear_problem is not None:
                self._record_lifecycle(
                    linear_problem.last_lifecycle_summary,
                    accumulate=True,
                )
            return

        fingerprint = self._operator_fingerprint()
        if self._operator_fingerprint_value is None:
            self._operator_fingerprint_value = fingerprint
        elif fingerprint != self._operator_fingerprint_value:
            self.close()
            raise RuntimeError(
                "AFM-TRANSIENT-OPERATOR-001: the effective first-order "
                "operator identity changed during a reuse lifecycle. Build a "
                "new Step or use operator_policy='refresh_each_step'."
            )
        if self._prepared_problem is None:
            started = perf_counter()
            self._prepared_problem = linear_problem.prepare()
            self._operator_ledger.begin_prepared()
            self.performance.add("matrix_preparation", perf_counter() - started)
        try:
            started = perf_counter()
            self.problem.solve(prepared=self._prepared_problem)
            self.performance.add("linear_system_solve", perf_counter() - started)
        except Exception:
            self.close()
            raise
        self._record_lifecycle(
            linear_problem.last_lifecycle_summary,
            accumulate=False,
        )

    def _operator_fingerprint(self) -> tuple[object, ...]:
        """Return runtime invariants required by fixed-operator reuse."""

        linear_problem = getattr(self.problem, "problem", None)
        solution = linear_problem._solution()
        V = solution.function_space
        index_map = V.dofmap.index_map
        return (
            float(self.dt),
            int(index_map.size_global),
            int(V.dofmap.index_map_bs),
            boundary_dof_identity(linear_problem.bcs),
        )

    def _record_lifecycle(self, summary, *, accumulate: bool) -> None:
        linear_problem = getattr(self.problem, "problem", None)
        self._operator_ledger.record(
            summary, getattr(linear_problem, "last_solve_info", None),
            accumulate=accumulate,
        )

    def operator_lifecycle_summary(self) -> dict[str, object]:
        """Return evidence for first-order operator reuse or refresh."""

        return {
            "kind": "first_order_operator_lifecycle",
            "requested_policy": self.operator_policy,
            "selected_policy": self._selected_operator_policy,
            "selection_reason": self._operator_policy_reason,
            "time_inputs": time.input_summary(self.update_load),
            "matrix_reused": self._selected_operator_policy == "reuse",
            **self._operator_ledger.summary(),
        }

    def close(self) -> None:
        """Release a live prepared operator without discarding evidence."""

        prepared = self._prepared_problem
        self._prepared_problem = None
        if prepared is not None:
            prepared.close()

    def _record_captured_histories(self, *, force: bool = False) -> None:
        selected_time = self.completed_steps * self.dt
        for recorder in self.captured_histories:
            if force or self.completed_steps % recorder.every == 0:
                recorder.record(selected_time)

    def solve(self):
        self.run()
        return self.current

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        history=(),
        progress=None,
        comm=None,
        metadata=None,
    ):
        return _solve_transient_result(
            self,
            solution=self.current,
            default_fields=(self.current,),
            output=output,
            fields=fields,
            history=history,
            progress=progress,
            comm=comm,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool = False) -> Path:
        """Save first-order state, optionally portable across MPI partitions."""

        return _save_transient_checkpoint(
            self,
            path,
            {"current": self.current, "previous": self.previous},
            portable=portable,
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
            "procedure": (None if self.procedure is None else self.procedure.summary()),
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
            "checkpoint_capabilities": self.checkpoint_capabilities().summary(
                policy=self.checkpoint_policy
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "captured_histories": [
                recorder.summary() for recorder in self.captured_histories
            ],
            "time_inputs": time.input_summary(self.update_load),
            "operator_lifecycle": self.operator_lifecycle_summary(),
            "problem": self.problem.summary(),
        }


def _solve_transient_result(
    step,
    *,
    solution,
    default_fields,
    output,
    fields,
    history,
    progress,
    comm,
    metadata,
):
    """Shared transient solve/output/result lifecycle for all equation orders."""

    context = getattr(step, "execution_context", None)
    selected_output = output
    if selected_output is None and context is not None:
        selected_output = context.configured_output
    selected_history = tuple(history)
    if not selected_history and context is not None:
        selected_history = context.configured_history
    if selected_output is not None:
        from .results import OutputPlan

        if isinstance(selected_output, OutputPlan):
            raise ValueError(
                "Declarative OutputPlan currently describes finite-strain "
                "static output. Pass an XDMF path to a transient Step."
            )
    _configure_transient_history(step, selected_history)
    if selected_output is not None and step.completed_steps >= step.steps:
        raise RuntimeError(
            "Transient field output must be requested before completion; "
            "completed steps cannot be reconstructed from final state alone."
        )
    if selected_output is not None or step.completed_steps != step.steps:
        step.run(
            output=selected_output,
            fields=fields,
            history=selected_history,
            progress=progress,
            comm=comm,
        )
    from .results._transient_step import from_transient_step

    output_fields, live_field_sets = _transient_output_fields(
        fields or step.last_output_fields or tuple(default_fields)
    )
    return from_transient_step(
        step,
        solution,
        output_fields=output_fields,
        live_field_sets=live_field_sets,
        metadata=metadata,
    )


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
    stability = getattr(step, "stability", None)
    stability_message = ""
    if stability is not None:
        selected = getattr(stability, "selected", None)
        controller = getattr(stability, "controller", None)
        if selected is not None:
            ratio = float(step.dt) / float(selected)
            stability_message = f"dt={float(step.dt):.6g}, dt/dt_limit={ratio:.3g}" + (
                "" if controller is None else f", controller={controller}"
            )
    reporter.emit(
        SolveEvent(
            "transient_started" if step.completed_steps == 0 else "transient_resumed",
            step.name,
            incrementation=algorithm,
            total_increments=step.steps,
            message=stability_message,
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
        channels = []
        metrics = {}
        records = getattr(step, "history_records", ())
        if records:
            latest = records[-1]
            if "relative_energy_balance_error" in latest:
                metrics["relative_energy_balance_error"] = float(
                    latest["relative_energy_balance_error"]
                )
                channels.append(
                    f"energy_err={float(latest['relative_energy_balance_error']):.3e}"
                )
            elif "energy_balance_error" in latest:
                metrics["energy_balance_error"] = float(
                    latest["energy_balance_error"]
                )
                channels.append(
                    f"energy_err={float(latest['energy_balance_error']):.3e}"
                )
        contact_progress = getattr(
            getattr(step, "residual", None),
            "contact_progress_evidence",
            None,
        )
        if callable(contact_progress):
            terms = tuple(contact_progress())
            if terms:
                contact_active = sum(
                    int(term["active_point_count"]) for term in terms
                )
                maximum_penetration = max(
                    float(term["maximum_penetration"]) for term in terms
                )
                contact_force = sum(
                    float(term["contact_force_norm"]) for term in terms
                )
                contact_work = sum(
                    float(term["contact_motion_work"]) for term in terms
                )
                contact_power = sum(
                    float(term["contact_motion_power"]) for term in terms
                )
                sticking = sum(
                    int(term.get("sticking_point_count", 0)) for term in terms
                )
                sliding = sum(
                    int(term.get("sliding_point_count", 0)) for term in terms
                )
                friction_dissipation = sum(
                    float(term.get("contact_friction_dissipation", 0.0))
                    for term in terms
                )
                metrics.update(
                    contact_pair_count=float(len(terms)),
                    contact_active_point_count=float(contact_active),
                    contact_maximum_penetration=maximum_penetration,
                    contact_force_norm_sum=contact_force,
                    contact_motion_work=contact_work,
                    contact_motion_power=contact_power,
                    contact_sticking_point_count=float(sticking),
                    contact_sliding_point_count=float(sliding),
                    contact_friction_dissipation=friction_dissipation,
                )
                channels.extend(
                    (
                        f"contact_pairs={len(terms)}",
                        f"contact_active={contact_active}",
                        f"max_pen={maximum_penetration:.3e}",
                        f"contact_force={contact_force:.3e}",
                        f"contact_work={contact_work:.3e}",
                        f"contact_power={contact_power:.3e}",
                        f"stick/slide={sticking}/{sliding}",
                        f"friction_diss={friction_dissipation:.3e}",
                    )
                )
                invalid = sum(
                    int(term["invalid_point_count"]) for term in terms
                )
                if invalid:
                    metrics["contact_invalid_point_count"] = float(invalid)
                    channels.append(f"contact_invalid={invalid}")
        message = " | ".join(channels)
        reporter.emit(
            SolveEvent(
                "time_increment",
                step.name,
                increment=info.index,
                time=float(info.time),
                total_increments=step.steps,
                display=bool(info.should_print),
                message=message,
                metrics=metrics,
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
    history_every = int(getattr(step, "history_every", 1))
    store_history = info.index % history_every == 0 or info.index == step.steps
    _record_transient_history(step, info.time, store=store_history)
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
        ),
        portable=bool(getattr(policy, "portable", False)),
    )
    _apply_checkpoint_retention(step, policy)
    return written


def _transient_checkpoint_capabilities(
    *state_components: str,
) -> checkpointing.CheckpointCapabilities:
    return checkpointing.CheckpointCapabilities(
        schemas=(checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,),
        boundary="accepted_step",
        payload_scope="full_restart_state",
        state_components=tuple(state_components),
        atomic_publication=True,
        rank_count_portability="requires_portable_policy",
        identity_scope=(
            "mesh coordinates and topology",
            "function-space element and value shape",
            "time grid and accepted position",
            "time-dependent input identities",
            "procedure scientific inputs",
        ),
        limitations=(
            "portable nodal state does not imply unregistered constitutive state",
        ),
        evidence=(
            "serial restart",
            "MPI same-partition restart",
            "MPI 1-to-2 and 2-to-1 portable restart",
        ),
    )


def _validate_checkpoint_policy(step) -> None:
    policy = getattr(step, "checkpoint_policy", None)
    if policy is not None:
        checkpointing.validate_policy(step, policy)


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


def _record_transient_history(
    step,
    time_value: float,
    *,
    store: bool = True,
) -> None:
    """Advance history monitors every increment and store at their cadence.

    Stateful scientific ledgers, notably external work, must consume every
    accepted increment. ``history_every`` controls retained records only; it
    must never change the computed balance merely by changing output cadence.
    """

    monitor = getattr(step, "history_monitor", None)
    requests = tuple(getattr(step, "history_requests", ()))
    if monitor is None and not requests:
        return
    selected_time = float(time_value)
    if (
        store
        and step.history_records
        and step.history_records[-1]["time"] == selected_time
    ):
        if hasattr(monitor, "restore"):
            monitor.restore(step.history_records[-1])
        return
    values = {}
    monitor_started = perf_counter()
    if monitor is not None:
        if isinstance(step, FirstOrderTransientStep):
            values.update(monitor.evaluate(step.current))
        else:
            monitor_kwargs = {
                "displacement": step.state.u,
                "velocity": step.state.v,
            }
            if getattr(monitor, "accepts_accepted_residual", False):
                monitor_kwargs["residual_owned"] = (
                    None
                    if step.completed_steps == 0
                    else getattr(
                        step.integrator,
                        "last_residual_owned",
                        None,
                    )
                )
            if not store and hasattr(monitor, "advance"):
                monitor.advance(**monitor_kwargs)
            elif hasattr(monitor, "evaluate"):
                values.update(monitor.evaluate(**monitor_kwargs))
            else:
                values.update(monitor(step, selected_time))
    performance = getattr(step, "performance", None)
    if monitor is not None and performance is not None:
        performance.add(
            "history_snapshot" if store else "history_advance",
            perf_counter() - monitor_started,
        )
    if not store:
        return
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


def _bind_performance_ledger(consumer, ledger, visited=None) -> None:
    """Attach one timing ledger to nested solver consumers without ownership."""

    if consumer is None:
        return
    selected_visited = set() if visited is None else visited
    identity = id(consumer)
    if identity in selected_visited:
        return
    selected_visited.add(identity)
    try:
        consumer.performance = ledger
    except (AttributeError, TypeError):
        pass
    for name in ("base", "bulk", "cohesive", "energy"):
        nested = getattr(consumer, name, None)
        if nested is not None:
            _bind_performance_ledger(nested, ledger, selected_visited)


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


def _save_transient_checkpoint(step, path, state, *, portable: bool = False) -> Path:
    from . import checkpointing
    from .results import CheckpointRecord

    # Path-dependent monitors must archive their accepted endpoint, including
    # when checkpoint cadence is independent of retained history cadence.
    _record_transient_history(step, float(step.completed_steps) * float(step.dt))
    manifest = checkpointing.save_transient_checkpoint(
        path,
        step_kind=step.summary()["kind"],
        step_name=step.name,
        procedure=step.procedure,
        dt=step.dt,
        total_steps=step.steps,
        completed_steps=step.completed_steps,
        state=state,
        time_inputs=time.input_summary(step.update_load),
        accepted_times=step.accepted_times,
        execution_events=step.execution_events,
        history_records=step.history_records,
        auxiliary_state=(
            {"residual": step.residual.snapshot()}
            if hasattr(getattr(step, "residual", None), "snapshot")
            else None
        ),
        portable=portable,
    )
    record = CheckpointRecord(
        name=f"{step.name}_{step.completed_steps}",
        path=manifest,
        schema=checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,
        step_name=step.name,
        coordinate_name="time",
        coordinate_value=float(step.completed_steps) * float(step.dt),
        portable=bool(portable),
        metadata={
            "completed_steps": step.completed_steps,
            "total_steps": step.steps,
            "portability": (
                "nodal state portable across MPI partitions and rank counts"
                if portable
                else "same mesh partition and MPI size"
            ),
        },
    )
    step.checkpoints.append(record)
    return manifest


def _load_transient_checkpoint(step, path, state) -> None:
    """Extend file-level atomicity across auxiliary state and input restoration."""
    arrays = {name: fields.unwrap(value).x.array.copy() for name, value in state.items()}
    state_owner = getattr(step, "state", None)
    full_state = state_owner.snapshot() if hasattr(state_owner, "snapshot") else None
    residual = getattr(step, "residual", None)
    old_residual = (residual.snapshot() if hasattr(residual, "snapshot")
                    and hasattr(residual, "restore") else None)
    old_completed = step.completed_steps
    lists = {name: list(getattr(step, name)) for name in (
        "accepted_times", "execution_events", "history_records", "checkpoints")}
    try:
        _load_transient_checkpoint_impl(step, path, state)
    except Exception as failure:
        step.completed_steps = old_completed
        for name, value in lists.items():
            getattr(step, name)[:] = value
        if old_residual is not None:
            try:
                residual.restore(old_residual)
            except Exception as recovery_failure:
                failure.add_note(f"Auxiliary checkpoint rollback failed: {recovery_failure}")
        callbacks = [getattr(step, "update_load", None), getattr(residual, "update_time", None)]
        callbacks.extend(getattr(item, "update", None) for item in getattr(step, "prescribed", ()))
        for callback in callbacks:
            if callable(callback):
                try:
                    callback(old_completed * step.dt)
                except Exception as recovery_failure:
                    failure.add_note(f"Checkpoint input rollback failed: {recovery_failure}")
        if full_state is not None:
            state_owner.restore(full_state)
        else:
            for name, value in state.items():
                target = fields.unwrap(value)
                target.x.array[:] = arrays[name]
                target.x.scatter_forward()
        raise


def _load_transient_checkpoint_impl(step, path, state) -> None:
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
        time_inputs=time.input_summary(step.update_load),
    )
    auxiliary = metadata.get("auxiliary_state")
    residual = getattr(step, "residual", None)
    if auxiliary is not None:
        if residual is None or not hasattr(residual, "restore"):
            raise ValueError(
                "Checkpoint contains auxiliary residual state, but the current "
                "step has no compatible residual-state consumer."
            )
        residual.restore(auxiliary["residual"])
    elif getattr(
        residual,
        "checkpoint_state_required",
        hasattr(residual, "restore"),
    ):
        raise ValueError(
            "The current step requires auxiliary residual state that is absent "
            "from this checkpoint."
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
    restart_time = float(step.completed_steps) * float(step.dt)
    monitor = getattr(step, "history_monitor", None)
    if callable(getattr(monitor, "restore", None)) and (
        not step.history_records or step.history_records[-1]["time"] != restart_time
    ):
        raise ValueError(
            "Checkpoint lacks the accepted endpoint for its path-dependent history monitor."
        )
    if getattr(step, "update_load", None) is not None:
        step.update_load(restart_time)
    if hasattr(residual, "update_time"):
        residual.update_time(restart_time)
    for item in tuple(getattr(step, "prescribed", ())):
        if hasattr(item, "update"):
            item.update(restart_time)
    step.checkpoints.append(
        CheckpointRecord(
            name=f"{step.name}_{step.completed_steps}_restart",
            path=Path(metadata["manifest_path"]),
            schema=checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,
            step_name=step.name,
            coordinate_name="time",
            coordinate_value=float(step.completed_steps) * float(step.dt),
            portable=bool(metadata.get("portable", False)),
            metadata={
                "role": "restart_source",
                "portability": metadata["portability"],
            },
        )
    )


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


def _transient_output_fields(selected):
    """Flatten Functions and live derived-field groups for one writer."""

    output = []
    live = []

    def visit(item):
        if hasattr(item, "update") and hasattr(item, "fields"):
            live.append(item)
            for field in tuple(item.fields):
                visit(field)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)
        else:
            output.append(fields.unwrap(item))

    visit(selected)
    if not output:
        raise ValueError("Transient output requires at least one field.")
    return tuple(output), tuple(live)


def _transient_result_series(path, domain):
    """Choose one ParaView-readable dataset layout for transient output."""

    selected = Path(path)
    if selected.suffix.lower() == ".pvd" or domain.comm.size > 1:
        from . import io

        actual = (
            selected
            if selected.suffix.lower() == ".pvd"
            else selected.with_suffix(".pvd")
        )
        return (
            io.ParaViewTimeSeries(actual, domain),
            actual,
            "dolfinx_vtk_collective",
            "single_unstructured_grid_per_time",
        )

    from .results.output import UnifiedXDMFTimeSeries

    if selected.suffix.lower() != ".xdmf":
        raise ValueError("Transient field output must use an .xdmf or .pvd path.")
    return (
        UnifiedXDMFTimeSeries(selected, deformation_scale=0.0),
        selected,
        "agentfem_unified_xdmf",
        "single_uniform_grid",
    )


def _refresh_transient_output_fields(live_field_sets) -> None:
    for selected in live_field_sets:
        selected.update()


def _describe_asset(asset) -> object:
    if hasattr(asset, "as_dict"):
        return asset.as_dict()
    if hasattr(asset, "summary"):
        return asset.summary()
    return getattr(asset, "name", repr(asset))


__all__ = (
    "ExplicitDynamicsStep",
    "FirstOrderTransientStep",
    "ImplicitDynamicsStep",
)
