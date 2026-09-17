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
from . import fields
from . import time
from .diagnostics import PerformanceLedger
from .solvers import LinearSolverOptions, SolveEvent


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
        if self.update_load is not None:
            self.update_load(t)
        try:
            self.integrator.step(
                self.dt,
                time=t,
                residual_operator=self.residual,
                prescribed=self.prescribed,
                constraints=self.constraints,
            )
        except Exception:
            if hasattr(self.residual, "rollback"):
                self.residual.rollback()
            raise
        else:
            if hasattr(self.residual, "commit"):
                self.residual.commit()

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
            "performance": self.performance.summary(),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
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
        domain = self.state.u.function_space.mesh
        series, actual_output, backend, layout = _transient_result_series(
            self.last_output,
            domain,
        )
        self.last_output = actual_output
        self.last_output_backend = backend
        self.last_output_layout = layout
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
        self.problem.solve()
        self.state.u_next.value.x.array[:] = (
            u_predictor.x.array + p.beta * dt**2 * self.state.a_next.value.x.array
        )
        self.state.v_next.value.x.array[:] = (
            v_predictor.x.array + p.gamma * dt * self.state.a_next.value.x.array
        )
        self.state.u_next.value.x.scatter_forward()
        self.state.v_next.value.x.scatter_forward()
        self.state.advance_state()

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
        linear_problem = getattr(self.problem, "problem", None)
        prepare = getattr(linear_problem, "prepare", None)
        prepared = prepare() if callable(prepare) else None

        def advance(info):
            if self.update_load is not None:
                self.update_load(info.time)
            current_rollback = self.current.x.array.copy()
            previous_rollback = self.previous.x.array.copy()
            try:
                if prepared is None:
                    self.problem.solve()
                else:
                    self.problem.solve(prepared=prepared)
            except Exception:
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
        finally:
            self.performance.add("run_wall", perf_counter() - run_started)
            close = getattr(prepared, "close", None)
            if callable(close):
                close()

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
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "captured_histories": [
                recorder.summary() for recorder in self.captured_histories
            ],
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

    return from_transient_step(
        step,
        solution,
        output_fields=(
            tuple(fields) or step.last_output_fields or tuple(default_fields)
        ),
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
        message = ""
        records = getattr(step, "history_records", ())
        if records:
            latest = records[-1]
            channels = []
            if "relative_energy_balance_error" in latest:
                channels.append(
                    f"energy_err={float(latest['relative_energy_balance_error']):.3e}"
                )
            elif "energy_balance_error" in latest:
                channels.append(
                    f"energy_err={float(latest['energy_balance_error']):.3e}"
                )
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
        and np.isclose(
            step.history_records[-1]["time"],
            selected_time,
        )
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
    auxiliary = metadata.get("auxiliary_state")
    residual = getattr(step, "residual", None)
    if auxiliary is not None:
        if residual is None or not hasattr(residual, "restore"):
            raise ValueError(
                "Checkpoint contains auxiliary residual state, but the current "
                "step has no compatible residual-state consumer."
            )
        residual.restore(auxiliary["residual"])
    elif hasattr(residual, "restore"):
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
    if getattr(step, "update_load", None) is not None:
        step.update_load(restart_time)
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
