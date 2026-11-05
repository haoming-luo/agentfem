# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Global equilibrium for provider-neutral small-strain materials."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc

from .. import amplitudes, checkpointing, procedures
from .. import steps as step_controls
from ..constitutive import elasticity
from ..constitutive.material_driver import SmallStrainMaterialQuadratureResponse
from ..constitutive.small_strain_user_material import SmallStrainUserMaterial
from ..diagnostics import (
    SolveEventRecorder,
    StandardRunReporter,
    comm_of,
    compose_reporters,
)
from ..events import SolveEvent
from ..solvers import NewtonSolverOptions, newton, solve_matrix_system
from ._incremental_runtime import (
    NewtonCorrection,
    NewtonEvaluation,
    run_newton_attempt,
)


@dataclass(frozen=True)
class SmallStrainMaterialIncrementInfo:
    """Evidence for one attempted equilibrium increment."""

    increment: int
    attempt: int
    start_load_factor: float
    load_factor: float
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    minimum_suggested_time_scale: float = 1.0
    applicability_counts: dict[str, int] = field(default_factory=dict)
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        finite = lambda value: float(value) if np.isfinite(value) else None
        return {
            "increment": self.increment,
            "attempt": self.attempt,
            "start_load_factor": self.start_load_factor,
            "load_factor": self.load_factor,
            "increment_size": self.load_factor - self.start_load_factor,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": finite(self.initial_residual_norm),
            "residual_norm": finite(self.residual_norm),
            "minimum_suggested_time_scale": finite(self.minimum_suggested_time_scale),
            "applicability_counts": dict(self.applicability_counts),
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, record):
        return cls(
            increment=int(record["increment"]),
            attempt=int(record["attempt"]),
            start_load_factor=float(record["start_load_factor"]),
            load_factor=float(record["load_factor"]),
            converged=bool(record["converged"]),
            iterations=int(record["iterations"]),
            initial_residual_norm=(
                float("inf")
                if record["initial_residual_norm"] is None
                else float(record["initial_residual_norm"])
            ),
            residual_norm=(
                float("inf")
                if record["residual_norm"] is None
                else float(record["residual_norm"])
            ),
            minimum_suggested_time_scale=float(
                record.get("minimum_suggested_time_scale") or 1.0
            ),
            applicability_counts={
                str(name): int(value)
                for name, value in record.get("applicability_counts", {}).items()
            },
            rejection_reason=record.get("rejection_reason"),
        )


@dataclass(frozen=True)
class SmallStrainMaterialPathInfo:
    increments: tuple[SmallStrainMaterialIncrementInfo, ...]
    attempts: tuple[SmallStrainMaterialIncrementInfo, ...]
    incrementation: object

    @property
    def converged(self) -> bool:
        return (
            bool(self.increments)
            and bool(self.attempts)
            and all(item.converged for item in self.increments)
            and self.attempts[-1].converged
        )

    @property
    def completed_step(self) -> bool:
        return self.converged and np.isclose(self.increments[-1].load_factor, 1.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "small_strain_material_nonlinear_load_path",
            "converged": self.converged,
            "completed_step": self.completed_step,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": [item.as_dict() for item in self.increments],
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass
class SmallStrainMaterialStep:
    """Implicit Newton Step driven by an ordinary small-strain material."""

    name: str
    solution: object
    material: SmallStrainUserMaterial
    response: SmallStrainMaterialQuadratureResponse
    residual_form: object
    tangent_form: object
    load_factor: object
    amplitude: amplitudes.Amplitude
    bcs: tuple[object, ...]
    prescribed_values: tuple[tuple[object, np.ndarray, object], ...]
    incrementation: object
    solver_options: NewtonSolverOptions
    duration: float | None = None
    time_unit: str = "s"
    temperature: object | None = None
    material_points: tuple = ()
    unit_system: object | None = None
    external_force: object | None = None
    study: object | None = None
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    procedure: object = field(
        default_factory=lambda: procedures.nonlinear_static(stateful=True)
    )
    accepted_load_factor: float = field(default=0.0, init=False)
    accepted_increments: list[SmallStrainMaterialIncrementInfo] = field(
        default_factory=list, init=False
    )
    attempted_increments: list[SmallStrainMaterialIncrementInfo] = field(
        default_factory=list, init=False
    )
    execution_events: list[object] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    last_solve_info: SmallStrainMaterialPathInfo | None = field(
        default=None, init=False
    )
    next_increment_size: float | None = field(default=None, init=False)
    _strain_evaluator: object = field(init=False, repr=False)
    _accepted_strain: np.ndarray = field(init=False, repr=False)
    _trial_result: object | None = field(default=None, init=False, repr=False)

    def checkpoint_capabilities(self) -> checkpointing.CheckpointCapabilities:
        return checkpointing.CheckpointCapabilities(
            schemas=(self._checkpoint_schema,),
            boundary="accepted_increment",
            payload_scope="full_restart_state",
            state_components=(
                "accepted displacement",
                "named user-material quadrature state",
                "load-path ledger",
                "solver continuation controls",
            )
            + (
                ("accepted stress, tangent and declared energy response",)
                if self.duration is not None
                else ()
            ),
            atomic_publication=True,
            rank_count_portability="supported"
            if self.duration is None
            else "unsupported",
            identity_scope=(
                "mesh and function space",
                "material parameter and state schemas",
                "load amplitude and procedure controls",
            ),
            limitations=(
                "provider state must implement the portable quadrature contract"
                if self.duration is None
                else "physical-time response restore requires the same serial quadrature layout",
            ),
            evidence=(
                "portable nodal and quadrature state round trip"
                if self.duration is None
                else "same-layout serial exact-response restart and failure rollback",
            ),
        )

    def __post_init__(self) -> None:
        from ._prescribed_temperature import freeze_uniform_temperature

        if self.temperature is not None and self.duration is None:
            raise ValueError("Prescribed temperature requires a physical duration.")
        self._temperature_input = freeze_uniform_temperature(
            self.temperature, self.duration
        )
        if self._temperature_input is not None and any(
            (variable.output_name or variable.name) == "TEMP"
            for variable in self.material.state_schema.variables
        ):
            raise ValueError("Material state alias TEMP is reserved for prescribed temperature.")
        self.material_points = tuple(self.material_points)
        self._point_history = None
        if self.material_points and self.duration is None:
            raise NotImplementedError(
                "Material-point histories require a physical duration."
            )
        expression = elasticity.strain(self.solution, study=self.study)
        self._strain_evaluator = self.response.state.compile_expression(
            expression, value_shape=(3, 3)
        )
        self._accepted_strain = self._evaluate_strain()
        if self.duration is None:
            self._update_material(time=0.0, time_increment=1.0)
            self.response.commit()
        else:
            self._apply_loading(0.0)
            self._physical_identity = json.dumps(
                self._checkpoint_identity(), sort_keys=True
            )
        if self.material_points:
            from ..results.material_points import _MaterialPointRecorder

            channels = {
                "S": {
                    "shape": [3, 3],
                    "unit": getattr(self.unit_system, "stress", None),
                    "measure": "Cauchy stress; Cartesian tensor",
                },
                "E": {
                    "shape": [3, 3],
                    "unit": "1",
                    "measure": "total infinitesimal strain; tensor shear, not mechanical strain",
                },
            }
            if self._temperature_input is not None:
                channels["TEMP"] = {
                    "shape": [],
                    "unit": "K",
                    "measure": "prescribed uniform temperature; no heat solve",
                }
            for variable in self.material.state_schema.variables:
                name = variable.output_name or variable.name
                if name in channels or name == "TEMP":
                    raise ValueError(
                        "Material history state aliases collide with reserved channels."
                    )
                channels[name] = {
                    "shape": list(variable.shape),
                    "unit": variable.unit,
                    "measure": variable.description + "; accepted constitutive state",
                }
            self._point_history = _MaterialPointRecorder(
                self.material_points,
                self.response.cauchy_stress,
                channels=channels,
                time_unit=self.time_unit,
                scientific_identity=self._checkpoint_identity(),
            )

    def _point_selection_identity(self):
        from ..results.material_points import MaterialPointHistoryRequest

        if not self.material_points:
            return {}
        if not all(
            isinstance(item, MaterialPointHistoryRequest)
            for item in self.material_points
        ):
            raise TypeError(
                "material_points must contain results.material_point requests."
            )
        return {
            "material_points": [
                {"name": item.name, "cell": item.cell, "point": item.point}
                for item in self.material_points
            ],
            "point_history_units": None
            if self.unit_system is None
            else self.unit_system.summary(),
        }

    @property
    def material_point_histories(self):
        return () if self._point_history is None else self._point_history.histories()

    def _prepare_point_frame(self, time, strain):
        if self._point_history is None:
            return None
        indices = self._point_history.indices
        values = {
            "S": self.response.cauchy_stress.values[indices],
            "E": strain[indices],
        }
        if self._temperature_input is not None:
            values["TEMP"] = np.full(len(indices), self._temperature_input.at(time))
        for variable in self.material.state_schema.variables:
            values[variable.output_name or variable.name] = self.state.trial[
                variable.name
            ].values[indices]
        return self._point_history.prepare(time, values, selected=True)

    def _check_physical_identity(self):
        if self._point_history is not None:
            self._point_history.validate_reference()
        if (
            hasattr(self, "_physical_identity")
            and json.dumps(self._checkpoint_identity(), sort_keys=True)
            != self._physical_identity
        ):
            raise ValueError(
                "Physical-time material scientific inputs changed; construct a new Step."
            )

    @property
    def _checkpoint_schema(self):
        version = 1 if self.duration is None else 2
        return f"agentfem.small-strain-material-step-checkpoint.v{version}"

    @property
    def time(self):
        return (
            None if self.duration is None else self.accepted_load_factor * self.duration
        )

    def _response_fields(self):
        fields = {
            "stress": self.response.cauchy_stress,
            "tangent": self.response.tangent,
            "stored_energy": self.response.stored_energy_density,
            "dissipation": self.response.dissipation_density_increment,
            **{
                f"component_{name}": value
                for name, value in self.response.stored_energy_density_components.items()
            },
        }
        return {name: value for name, value in fields.items() if value is not None}

    def _response_snapshot(self):
        return (
            {
                name: value.values.copy()
                for name, value in self._response_fields().items()
            },
            self.response.last_diagnostics,
            self.response.last_applicability,
            self._trial_result,
        )

    def _restore_response(self, snapshot):
        values, diagnostics, applicability, trial = snapshot
        for name, value in self._response_fields().items():
            value.assign(values[name])
        self.response.last_diagnostics = diagnostics
        self.response.last_applicability = applicability
        self._trial_result = trial

    @property
    def state(self):
        """Expose the ordinary state owner used by output/checkpoint tooling."""

        return self.response.state

    def _evaluate_strain(self) -> np.ndarray:
        return self.response.state.evaluate_expression(
            self._strain_evaluator, value_shape=(3, 3)
        )

    def _update_material(self, *, time: float, time_increment: float):
        scale = 1.0 if self.duration is None else self.duration
        thermal = {}
        if self._temperature_input is not None:
            endpoint = self._temperature_input.at(float(time) * scale)
            start = self._temperature_input.at(self.accepted_load_factor * scale)
            thermal = {
                "temperature": endpoint,
                "temperature_increment": endpoint - start,
            }
        self._trial_result = self.response.update(
            self.material,
            strain_old=self._accepted_strain,
            strain_new=self._evaluate_strain(),
            time=float(time) * scale,
            time_increment=(
                max(float(time_increment), np.finfo(float).eps)
                if self.duration is None
                else float(time_increment) * scale
            ),
            **thermal,
        )
        return self._trial_result

    def solve(self, *, until: float = 1.0):
        self._check_physical_identity()
        selected_until = float(until)
        if not self.accepted_load_factor < selected_until <= 1.0:
            raise ValueError(
                "until must be greater than the accepted factor and at most 1."
            )
        reporter = self._reporter()
        accepted_factor = self.accepted_load_factor
        proposed_size = (
            (
                self.incrementation.initial
                if self.next_increment_size is None
                else self.next_increment_size
            )
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation)
            else self.incrementation.load_factors[0]
        )
        consecutive_cutbacks = 0
        self._emit(
            reporter,
            SolveEvent(
                "step_started",
                self.name,
                step_number=self.step_number,
                incrementation=self.incrementation.summary()["kind"],
            ),
        )
        self._apply_loading(accepted_factor)
        while accepted_factor < selected_until - 1.0e-12:
            increment = len(self.accepted_increments) + 1
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                if len(self.accepted_increments) >= self.incrementation.max_increments:
                    raise RuntimeError(
                        f"{self.name}: max_increments reached before completion."
                    )
                target = min(selected_until, accepted_factor + proposed_size)
            else:
                remaining = [
                    value
                    for value in self.incrementation.load_factors
                    if value > accepted_factor + 1.0e-12
                ]
                if not remaining:
                    raise RuntimeError(
                        "Fixed incrementation has no factor beyond restored state."
                    )
                target = min(selected_until, remaining[0])
            if self.duration is not None and self.amplitude.kind == "tabular":
                knots = self.amplitude.summary()["metadata"]["times"]
                target = min(
                    target,
                    next((float(k) for k in knots if accepted_factor < k < 1), 1.0),
                )
            if self._temperature_input is not None:
                thermal_knots = (
                    t / self.duration for t in self._temperature_input.times
                )
                target = min(
                    target,
                    next(
                        (k for k in thermal_knots if accepted_factor + 1e-12 < k < 1),
                        1.0,
                    ),
                )
            attempt = consecutive_cutbacks + 1
            self._emit(
                reporter,
                SolveEvent(
                    "increment_started",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=accepted_factor,
                    target_factor=target,
                ),
            )
            displacement = self.solution.x.array.copy()
            state = self.response.snapshot()
            old_strain = self._accepted_strain.copy()
            old_response = (
                self._response_snapshot() if self.duration is not None else None
            )
            try:
                self._apply_loading(target)
                info = self._solve_increment(
                    increment=increment,
                    attempt=attempt,
                    start_factor=accepted_factor,
                    target_factor=target,
                    reporter=reporter,
                )
            except RuntimeError as exc:
                info = SmallStrainMaterialIncrementInfo(
                    increment,
                    attempt,
                    accepted_factor,
                    target,
                    False,
                    0,
                    float("inf"),
                    float("inf"),
                    rejection_reason=str(exc),
                )
            except BaseException:
                # An interrupt or assembly error is not an accepted material
                # increment. Do not leave trial displacement beside old history.
                if old_response is not None:
                    self.solution.x.array[:] = displacement
                    self.solution.x.scatter_forward()
                    self.response.restore(state)
                    self._accepted_strain = old_strain
                    self._apply_loading(accepted_factor)
                    self._restore_response(old_response)
                raise
            if info.converged and info.minimum_suggested_time_scale < 1.0 - 1.0e-12:
                info = replace(
                    info,
                    converged=False,
                    rejection_reason=(
                        "material provider requested increment scale "
                        f"{info.minimum_suggested_time_scale:.6g}"
                    ),
                )
            self.attempted_increments.append(info)
            if info.converged:
                try:
                    accepted_strain = self._evaluate_strain()
                    point_frame = self._prepare_point_frame(
                        target * self.duration if self.duration is not None else target,
                        accepted_strain,
                    )
                    self.response.commit()
                except BaseException:
                    if old_response is not None:
                        self.solution.x.array[:] = displacement
                        self.solution.x.scatter_forward()
                        self.response.restore(state)
                        self._accepted_strain = old_strain
                        self._apply_loading(accepted_factor)
                        self._restore_response(old_response)
                        self.attempted_increments.pop()
                    raise
                self._accepted_strain = accepted_strain
                if point_frame is not None:
                    self._point_history.frames.append(point_frame)
                self.accepted_increments.append(info)
                accepted_size = target - accepted_factor
                accepted_factor = target
                self.accepted_load_factor = target
                consecutive_cutbacks = 0
                if isinstance(
                    self.incrementation, step_controls.AutomaticIncrementation
                ):
                    proposed_size = self.incrementation.after_convergence(
                        accepted_size, info.iterations
                    )
                    self.next_increment_size = proposed_size
                self._emit(
                    reporter,
                    SolveEvent(
                        "increment_converged",
                        self.name,
                        step_number=self.step_number,
                        increment=increment,
                        attempt=attempt,
                        start_factor=info.start_load_factor,
                        target_factor=target,
                        iteration=info.iterations,
                        residual_norm=info.residual_norm,
                    ),
                )
                continue
            self.solution.x.array[:] = displacement
            self.solution.x.scatter_forward()
            self.response.restore(state)
            self._accepted_strain = old_strain
            self._apply_loading(accepted_factor)
            if old_response is not None:
                self._restore_response(old_response)
            else:
                self._update_material(
                    time=accepted_factor,
                    time_increment=max(target - accepted_factor, np.finfo(float).eps),
                )
            if not isinstance(
                self.incrementation, step_controls.AutomaticIncrementation
            ):
                self._fail(reporter, info, "fixed increment did not converge")
                return self.solution
            consecutive_cutbacks += 1
            proposed_size = self.incrementation.after_failure(target - accepted_factor)
            if info.minimum_suggested_time_scale < 1.0:
                proposed_size = min(
                    proposed_size,
                    (target - accepted_factor) * info.minimum_suggested_time_scale,
                )
            self.next_increment_size = proposed_size
            if (
                consecutive_cutbacks > self.incrementation.max_cutbacks
                or proposed_size < self.incrementation.minimum
            ):
                self._fail(
                    reporter,
                    info,
                    "automatic incrementation exhausted its cutback allowance",
                )
                return self.solution
            self._emit(
                reporter,
                SolveEvent(
                    "increment_cutback",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=accepted_factor,
                    target_factor=target,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    next_increment=proposed_size,
                    message=info.rejection_reason,
                ),
            )
        self.last_solve_info = SmallStrainMaterialPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        self._emit(
            reporter,
            SolveEvent(
                "step_completed" if accepted_factor >= 1.0 - 1.0e-12 else "step_paused",
                self.name,
                step_number=self.step_number,
                increment=len(self.accepted_increments),
                attempt=len(self.attempted_increments),
                target_factor=accepted_factor,
            ),
        )
        return self.solution

    def _solve_increment(
        self, *, increment, attempt, start_factor, target_factor, reporter
    ):
        def evaluate():
            result = self._update_material(
                time=target_factor,
                time_increment=target_factor - start_factor,
            )
            rhs, norm = self._correction_rhs()
            return NewtonEvaluation(result, rhs, norm)

        def correct(evaluation, _iteration):
            tangent = fem_petsc.assemble_matrix(self.tangent_form, bcs=self.bcs)
            tangent.assemble()
            correction = evaluation.residual.duplicate()
            correction.set(0.0)
            linear = solve_matrix_system(
                tangent,
                evaluation.residual,
                correction,
                self.solver_options.linear_solver,
                raise_on_failure=False,
            )
            tangent.destroy()
            if not linear.converged:
                correction.destroy()
                return NewtonCorrection(
                    False,
                    rejection_reason="linear correction solve did not converge",
                )
            base = self.solution.x.array.copy()
            direction = correction.array_r.copy()
            correction.destroy()
            alpha, accepted_trial = self._line_search(
                base,
                direction,
                evaluation.residual_norm,
                target_factor,
                target_factor - start_factor,
            )
            next_evaluation = (
                None if accepted_trial is None else NewtonEvaluation(*accepted_trial)
            )
            return NewtonCorrection(
                alpha > 0.0,
                step_length=alpha,
                next_evaluation=next_evaluation,
                rejection_reason=(
                    None if alpha > 0.0 else "line search rejected every trial"
                ),
            )

        def report_iteration(iteration_number, residual_norm, step_length):
            self._emit(
                reporter,
                SolveEvent(
                    "iteration",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=start_factor,
                    target_factor=target_factor,
                    iteration=iteration_number,
                    residual_norm=residual_norm,
                    step_length=step_length,
                ),
            )

        outcome = run_newton_attempt(
            self.solver_options,
            evaluate=evaluate,
            correct=correct,
            release=lambda residual: residual.destroy(),
            on_iteration=report_iteration,
        )
        summary = outcome.payload.summary() if outcome.payload is not None else {}
        return SmallStrainMaterialIncrementInfo(
            increment=increment,
            attempt=attempt,
            start_load_factor=start_factor,
            load_factor=target_factor,
            converged=outcome.converged,
            iterations=outcome.iterations,
            initial_residual_norm=outcome.initial_residual_norm,
            residual_norm=outcome.residual_norm,
            minimum_suggested_time_scale=float(
                summary.get("minimum_suggested_time_scale", 1.0)
            ),
            applicability_counts=dict(summary.get("applicability_counts", {})),
            rejection_reason=outcome.rejection_reason,
        )

    def _line_search(self, base, direction, base_norm, time, time_increment):
        alpha = 1.0
        if self.solver_options.line_search in {None, "basic"}:
            self._assign_trial(base, direction, alpha)
            return alpha, None
        while alpha + 1.0e-15 >= self.solver_options.minimum_step_length:
            self._assign_trial(base, direction, alpha)
            result = self._update_material(time=time, time_increment=time_increment)
            rhs, norm = self._correction_rhs()
            if np.isfinite(norm) and norm < base_norm:
                return alpha, (result, rhs, norm)
            rhs.destroy()
            alpha *= self.solver_options.line_search_reduction
        self.solution.x.array[:] = base
        self.solution.x.scatter_forward()
        self.response.rollback()
        return 0.0, None

    def _assign_trial(self, base, direction, alpha):
        self.solution.x.array[:] = base
        self.solution.x.array[: len(direction)] += alpha * direction
        self.solution.x.scatter_forward()

    def _correction_rhs(self):
        residual = fem_petsc.assemble_vector(self.residual_form)
        fem_petsc.apply_lifting(
            residual,
            [self.tangent_form],
            [self.bcs],
            x0=[self.solution.x.petsc_vec],
            alpha=-1.0,
        )
        residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        residual.scale(-1.0)
        fem_petsc.set_bc(residual, self.bcs, x0=self.solution.x.petsc_vec, alpha=1.0)
        return residual, float(residual.norm())

    def _reporter(self):
        recorder = SolveEventRecorder(self.execution_events)
        if self.progress is True:
            return compose_reporters(
                recorder,
                StandardRunReporter(
                    comm_of(self.solution), status_file=self.status_file
                ),
            )
        if self.progress in (False, None):
            return recorder
        return compose_reporters(recorder, self.progress)

    def _set_prescribed_factor(self, factor):
        for constant, target, _bc in self.prescribed_values:
            selected = float(factor) * target
            constant.value = (
                PETSc.ScalarType(selected.item())
                if selected.ndim == 0 or selected.size == 1
                else np.asarray(selected, dtype=PETSc.ScalarType)
            )

    def _apply_loading(self, coordinate):
        value = self.amplitude(coordinate)
        self.load_factor.value = PETSc.ScalarType(value)
        self._set_prescribed_factor(value)

    def _emit(self, reporter, event):
        if self.duration is not None:
            coordinate = (
                self.accepted_load_factor
                if event.kind == "step_started" or event.target_factor is None
                else event.target_factor
            )
            event = replace(
                event,
                time=coordinate * self.duration,
                coordinate_name="time",
                coordinate_value=coordinate * self.duration,
                coordinate_unit=self.time_unit,
            )
        reporter.emit(event) if hasattr(reporter, "emit") else reporter(event)

    def _fail(self, reporter, info, message):
        self._emit(
            reporter,
            SolveEvent(
                "step_failed",
                self.name,
                step_number=self.step_number,
                increment=info.increment,
                attempt=info.attempt,
                start_factor=info.start_load_factor,
                target_factor=info.load_factor,
                iteration=info.iterations,
                residual_norm=info.residual_norm,
                message=message,
            ),
        )
        self.last_solve_info = SmallStrainMaterialPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        if self.solver_options.error_if_not_converged:
            raise RuntimeError(f"{self.name}: {message}.")

    def reaction_field(self, *, name="RF"):
        residual = fem_petsc.assemble_vector(self.residual_form)
        residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        reaction = fem.Function(self.solution.function_space, name=name)
        reaction.x.array[: len(residual.array_r)] = residual.array_r
        reaction.x.scatter_forward()
        residual.destroy()
        return reaction

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        output_fields=(),
        strict_output=False,
        metadata=None,
    ):
        self._check_physical_identity()
        from ..results import (
            add_execution_trace,
            complete_result,
            from_solution,
            recover_integration_point_field,
        )

        if fields and output_fields:
            raise ValueError("Pass fields=... or output_fields=..., not both.")
        selected_fields = tuple(fields) or tuple(output_fields)
        solution = (
            self.solve() if self.accepted_load_factor < 1.0 - 1.0e-12 else self.solution
        )
        result = from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "state": self.response.summary(),
            },
        )
        add_execution_trace(result, self.execution_events)
        if self._temperature_input is not None:
            thermal_field = fem.Function(
                fem.functionspace(self.solution.function_space.mesh, ("DG", 0)),
                name="TEMP",
            )
            thermal_field.x.array[:] = self._temperature_input.at(self.time)
            thermal_field.x.scatter_forward()
            result.add_field(
                "TEMP",
                thermal_field,
                unit="K",
                location="cells",
                description="Accepted prescribed uniform temperature; no heat solve.",
                processing={"method": "prescribed_uniform_input", "committed": True},
            )
            result.metadata["prescribed_temperature"] = (
                self._temperature_input.summary()
            )
        sources = [
            (
                "S",
                self.response.cauchy_stress,
                "Cauchy stress at constitutive integration points.",
            )
        ]
        for variable in self.state.state_schema.variables:
            sources.append(
                (
                    variable.output_name or variable.name,
                    self.state.committed[variable.name],
                    variable.description,
                )
            )
        if self.response.stored_energy_density is not None:
            sources.append(
                ("SENER", self.response.stored_energy_density, "Stored energy density.")
            )
        if self.response.dissipation_density_increment is not None:
            sources.append(
                (
                    "DENER_INC",
                    self.response.dissipation_density_increment,
                    "Accepted-increment dissipation density.",
                )
            )
        for name, source, description in sources:
            result.add_field(
                name,
                source.function,
                location="quadrature_points",
                description=description,
                processing={
                    "source_position": "quadrature_points",
                    "method": "constitutive_update"
                    if name == "S"
                    else "constitutive_state",
                    "representation": "quadrature_values",
                    "postprocessed": False,
                    "committed": True,
                },
                sampling=source,
            )
            recovered = recover_integration_point_field(
                source,
                name=f"{name}_CELL",
                description=f"Cell average of {name} preserving material discontinuities.",
            )
            result.add_field(
                recovered.name,
                recovered.field,
                unit=recovered.unit,
                location=recovered.location,
                description=recovered.description,
                processing=recovered.processing,
            )
        result.add_field(
            "RF",
            self.reaction_field(),
            description="Full nodal equilibrium residual for reaction extraction.",
            processing={
                "method": "assembled_equilibrium_residual",
                "representation": "finite_element_dofs",
                "postprocessed": False,
            },
        )
        counts = self.response.summary()["applicability_counts"]
        result.add_quantities(
            {f"material_points_{name}": value for name, value in counts.items()},
            kind="diagnostic",
        )
        result.metadata["material"] = _material_summary(self.material)
        if self._point_history is not None:
            for history in self.material_point_histories:
                result.histories[history.name] = history
            result.metadata["material_point_history"] = json.loads(
                json.dumps(self._point_history.identity)
            )
        if self.duration is not None:
            result.metadata["physical_time"] = {
                "duration": self.duration,
                "accepted_time": self.time,
                "unit": self.time_unit,
                "initial_state": "stress_free_zero_history",
            }
            coordinates = np.asarray(
                [0.0, *(item.load_factor for item in self.accepted_increments)]
            )
            result.add_history(
                "load_amplitude",
                coordinates * self.duration,
                [self.amplitude(float(value)) for value in coordinates],
                unit="1",
                abscissa_unit=self.time_unit,
                description="Accepted common load amplitude against physical time.",
            )
        if hasattr(self.material, "evidence"):
            result.metadata["learned_constitutive"] = self.material.evidence(
                runtime=_runtime_evidence(self.material),
                diagnostics={
                    "applicability_counts": counts,
                    "accepted_increments": len(self.accepted_increments),
                    "attempted_increments": len(self.attempted_increments),
                },
            )
        for checkpoint in self.checkpoints:
            result.add_checkpoint(checkpoint)
        return complete_result(
            self,
            result,
            output=output,
            fields=selected_fields,
            strict_output=strict_output,
            time=0.0 if self.time is None else self.time,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        self._check_physical_identity()
        from ..checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
        )
        from ..results import CheckpointRecord

        if self.duration is not None:
            if not self.amplitude.serializable:
                raise NotImplementedError(
                    "Physical-time checkpoint requires a serializable amplitude identity."
                )
            if portable:
                raise NotImplementedError(
                    "Physical-time response checkpoints require the same serial quadrature layout."
                )
            if not self.accepted_increments:
                raise ValueError(
                    "A physical-time checkpoint requires at least one accepted increment."
                )
        selected = Path(path)
        if selected.suffix:
            selected = selected.with_suffix("")
        manifest = selected.with_name(selected.name + ".checkpoint.json")
        bundle = save_portable_state_bundle(manifest, state={"U": self.solution})
        quadrature = self.state.save(
            manifest.with_name(f"{selected.name}.{bundle['generation']}.quadrature"),
            material=self.material,
        )
        payload = {
            "schema": self._checkpoint_schema,
            "step_identity": self._checkpoint_identity(),
            "coordinate": self.accepted_load_factor,
            "nodal_state": bundle["record"],
            "nodal_identity": bundle["identities"],
            "quadrature_state": checkpoint_file_record(quadrature),
            "accepted_increments": [
                item.as_dict() for item in self.accepted_increments
            ],
            "attempted_increments": [
                item.as_dict() for item in self.attempted_increments
            ],
            "execution_events": [item.as_dict() for item in self.execution_events],
            "next_increment_size": self.next_increment_size,
        }
        if self._point_history is not None:
            payload["material_point_history"] = self._point_history.payload()
        if self.duration is not None:
            response_path = manifest.with_name(
                f"{selected.name}.{bundle['generation']}.response.npz"
            )
            checkpointing.atomic_savez(
                response_path,
                **{
                    name: value.values
                    for name, value in self._response_fields().items()
                },
                cell_keys=self.response.cauchy_stress.owned_cell_keys,
                diagnostics=json.dumps(
                    [dict(item) for item in self.response.last_diagnostics],
                    allow_nan=False,
                ),
                applicability=json.dumps(self.response.last_applicability),
            )
            payload["accepted_response"] = checkpoint_file_record(response_path)
        comm = self.solution.function_space.mesh.comm
        problem = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
            except Exception as exc:
                problem = f"{type(exc).__name__}: {exc}"
        problem = comm.bcast(problem, root=0)
        if problem:
            raise RuntimeError(
                f"Small-strain material checkpoint write failed: {problem}"
            )
        comm.barrier()
        record = CheckpointRecord(
            name=f"{self.name}_{self.accepted_load_factor:g}",
            path=manifest,
            schema=payload["schema"],
            step_name=self.name,
            coordinate_name="load_factor" if self.duration is None else "time",
            coordinate_value=self.accepted_load_factor
            if self.duration is None
            else self.time,
            portable=self.duration is None,
            metadata={
                "state_variables": ("U", *tuple(self.state.transaction.names)),
                "material": _material_summary(self.material),
                **(
                    {
                        "normalized_coordinate": self.accepted_load_factor,
                        "time_unit": self.time_unit,
                    }
                    if self.duration is not None
                    else {}
                ),
            },
        )
        self.checkpoints.append(record)
        return manifest

    def load_checkpoint(self, path) -> None:
        """Restore an accepted path without exposing a partially restored step."""

        self._check_physical_identity()
        response = self.response
        saved_fields = [
            *self.state.committed.values(),
            *self.state.trial.values(),
            response.cauchy_stress,
            response.tangent,
            response.stored_energy_density,
            response.dissipation_density_increment,
            *response.stored_energy_density_components.values(),
        ]
        arrays = [
            (field, field.values.copy()) for field in saved_fields if field is not None
        ]
        displacement = self.solution.x.array.copy()
        constants = [self.load_factor, *(item[0] for item in self.prescribed_values)]
        constant_values = [np.array(item.value, copy=True) for item in constants]
        attributes = {
            name: getattr(self, name)
            for name in (
                "accepted_load_factor",
                "next_increment_size",
                "last_solve_info",
                "_accepted_strain",
                "_trial_result",
            )
        }
        histories = {
            name: list(getattr(self, name))
            for name in (
                "accepted_increments",
                "attempted_increments",
                "execution_events",
            )
        }
        diagnostics = response.last_diagnostics
        applicability = response.last_applicability
        try:
            self._load_checkpoint(path)
        except BaseException:
            # Restore saved fields directly: calling the provider again could
            # fail for the same reason that interrupted checkpoint loading.
            self.solution.x.array[:] = displacement
            self.solution.x.scatter_forward()
            for field, values in arrays:
                field.assign(values)
            for constant, values in zip(constants, constant_values):
                constant.value = values
            for name, value in attributes.items():
                setattr(self, name, value)
            for name, values in histories.items():
                getattr(self, name)[:] = values
            response.last_diagnostics = diagnostics
            response.last_applicability = applicability
            raise

    def _load_checkpoint(self, path) -> None:
        from ..checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )

        manifest = Path(path)
        comm = self.solution.function_space.mesh.comm
        identity = json.loads(json.dumps(self._checkpoint_identity(), sort_keys=True))
        error = None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("schema") != self._checkpoint_schema:
                raise ValueError("Unsupported small-strain material checkpoint schema.")
            if payload.get("step_identity") != identity:
                raise ValueError(
                    "Small-strain material checkpoint scientific identity differs."
                )
            quadrature_path = validate_checkpoint_record(
                manifest.parent, payload["quadrature_state"]
            )
            response_snapshot = None
            if self.duration is not None:
                response_path = validate_checkpoint_record(
                    manifest.parent, payload["accepted_response"]
                )
                with np.load(response_path, allow_pickle=False) as archive:
                    if set(archive.files) != set(self._response_fields()) | {
                        "cell_keys",
                        "diagnostics",
                        "applicability",
                    }:
                        raise ValueError("Accepted response archive fields differ.")
                    if not np.array_equal(
                        archive["cell_keys"],
                        self.response.cauchy_stress.owned_cell_keys,
                    ):
                        raise ValueError(
                            "Physical-time response quadrature layout differs."
                        )
                    values = {}
                    for name, field in self._response_fields().items():
                        value = archive[name]
                        if value.shape != field.values.shape or not np.all(
                            np.isfinite(value)
                        ):
                            raise ValueError(
                                f"Invalid accepted response field: {name}."
                            )
                        values[name] = value.copy()
                    diagnostics = tuple(json.loads(str(archive["diagnostics"])))
                    applicability = tuple(json.loads(str(archive["applicability"])))
                    count = len(values["stress"])
                    if (
                        len(applicability) != count
                        or any(not isinstance(item, str) for item in applicability)
                        or (
                            diagnostics
                            and (
                                len(diagnostics) != count
                                or any(
                                    not isinstance(item, dict) for item in diagnostics
                                )
                            )
                        )
                    ):
                        raise ValueError("Invalid accepted response diagnostics.")
                    response_snapshot = (values, diagnostics, applicability, None)
            coordinate = float(payload["coordinate"])
            if not np.isfinite(coordinate) or not 0 <= coordinate <= 1:
                raise ValueError("Checkpoint load factor must be finite and in [0, 1].")
            accepted = [
                SmallStrainMaterialIncrementInfo.from_dict(item)
                for item in payload["accepted_increments"]
            ]
            if self.duration is not None and not accepted:
                raise ValueError("Physical-time checkpoint has no accepted increment.")
            attempted = [
                SmallStrainMaterialIncrementInfo.from_dict(item)
                for item in payload["attempted_increments"]
            ]
            previous = 0.0
            completed = []
            for record, info in zip(payload["attempted_increments"], attempted):
                if (
                    type(record["converged"]) is not bool
                    or type(record["increment"]) is not int
                    or info.increment != len(completed) + 1
                    or type(record["attempt"]) is not int
                    or info.attempt < 1
                    or type(record["iterations"]) is not int
                    or info.iterations < 0
                    or info.start_load_factor != previous
                    or not previous < info.load_factor <= 1
                    or (
                        info.converged
                        and (
                            not np.isfinite(info.residual_norm)
                            or info.residual_norm < 0
                        )
                    )
                ):
                    raise ValueError("Invalid checkpoint attempt ledger.")
                if info.converged:
                    completed.append(record)
                    previous = info.load_factor
            if completed != payload["accepted_increments"] or previous != coordinate:
                raise ValueError(
                    "Checkpoint ledger does not end at the accepted state."
                )
            point_frames = None
            if self._point_history is not None:
                point_frames = self._point_history.validated_frames(
                    payload.get("material_point_history"),
                    accepted_times=[
                        item.load_factor * self.duration for item in accepted
                    ],
                )
            events = [
                SolveEvent.from_dict(item)
                for item in payload.get("execution_events", ())
            ]
            nodal_state = payload["nodal_state"]
            nodal_identity = payload["nodal_identity"]
            next_size = payload.get("next_increment_size")
            if next_size is not None:
                next_size = float(next_size)
                if not np.isfinite(next_size) or next_size <= 0:
                    raise ValueError(
                        "Checkpoint next increment must be finite and positive."
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [item for item in comm.allgather(error) if item]
        if errors:
            raise ValueError("Small-strain checkpoint validation failed: " + errors[0])
        load_portable_state_bundle(
            manifest,
            state={"U": self.solution},
            record=nodal_state,
            identities=nodal_identity,
        )
        self.state.load(quadrature_path, material=self.material)
        self.accepted_load_factor = coordinate
        self.accepted_increments[:] = accepted
        self.attempted_increments[:] = attempted
        self.execution_events[:] = events
        self.next_increment_size = next_size
        self._apply_loading(self.accepted_load_factor)
        prescribed = self.solution.x.array.copy()
        for bc in self.bcs:
            bc.set(prescribed)
        boundary_matches = np.allclose(
            prescribed, self.solution.x.array, rtol=1e-12, atol=1e-13
        )
        if not all(comm.allgather(bool(boundary_matches))):
            raise ValueError(
                "Checkpoint displacement violates its accepted boundary state."
            )
        self._accepted_strain = self._evaluate_strain()
        if response_snapshot is not None:
            self._restore_response(response_snapshot)
        else:
            self._update_material(
                time=self.accepted_load_factor,
                time_increment=np.finfo(float).eps,
            )
        self.last_solve_info = SmallStrainMaterialPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        if self._point_history is not None:
            self._point_history.frames = point_frames

    def _checkpoint_identity(self):
        from ..checkpointing import function_portable_identity

        physical = {}
        if self.duration is not None:
            form = (
                None if self.external_force is None else self.external_force.expression
            )
            physical = {
                "physical_time": {
                    "duration": self.duration,
                    "unit": self.time_unit,
                    "initial_state": "stress_free_zero_history",
                },
                "constraints": [
                    {"dofs": bc.dof_indices()[0].tolist(), "value": value.tolist()}
                    for _, value, bc in self.prescribed_values
                ],
                "load": None
                if form is None
                else {
                    "signature": form.signature(),
                    "coefficients": [
                        sha256(np.asarray(c.x.array).tobytes()).hexdigest()
                        for c in form.coefficients()
                    ],
                    "constants": [
                        np.asarray(c.value).tolist() for c in form.constants()
                    ],
                },
            }
        if self._temperature_input is not None:
            physical["temperature"] = self._temperature_input.summary()
        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "material": _material_summary(self.material),
            "amplitude": self.amplitude.summary(),
            "incrementation": self.incrementation.summary(),
            "solution": function_portable_identity(self.solution),
            "quadrature": self.state.summary()["transaction"],
            **physical,
            **self._point_selection_identity(),
        }

    def summary(self):
        return {
            "kind": "small_strain_material_step",
            "name": self.name,
            "study": None if self.study is None else self.study.summary(),
            "procedure": self.procedure.summary(),
            "material": _material_summary(self.material),
            "state": self.response.summary(),
            "incrementation": self.incrementation.summary(),
            "solver": self.solver_options.summary(),
            "num_bcs": len(self.bcs),
            "accepted_load_factor": self.accepted_load_factor,
            "physical_time": self.time,
            "duration": self.duration,
            "time_unit": self.time_unit if self.duration is not None else None,
            "prescribed_temperature": None
            if self._temperature_input is None
            else self._temperature_input.summary(),
            "last_solve": None
            if self.last_solve_info is None
            else self.last_solve_info.as_dict(),
            "checkpoint_capabilities": self.checkpoint_capabilities().summary(),
        }


def small_strain_material_step(
    *,
    displacement,
    material,
    external_force,
    constraints=(),
    study=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree=2,
    progress=True,
    status_file=None,
    amplitude=None,
    duration=None,
    time_unit="s",
    temperature=None,
    material_points=(),
    unit_system=None,
    name="small_strain_material",
) -> SmallStrainMaterialStep:
    """Build implicit small-strain equilibrium, optionally on a physical-time path."""
    if not isinstance(material, SmallStrainUserMaterial):
        raise TypeError(
            "small_strain_material_step requires a SmallStrainUserMaterial."
        )
    if duration is None and not bool(getattr(material, "rate_independent", False)):
        raise NotImplementedError(
            "The first global small-strain material Step requires an explicit "
            "rate_independent=True declaration. Time/temperature-dependent "
            "providers need a physical-time procedure and restart contract."
        )
    domain = displacement.value.function_space.mesh
    if duration is not None:
        if (
            isinstance(duration, (bool, np.bool_))
            or not np.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("duration must be a finite positive physical time.")
        duration = float(duration)
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("time_unit must be a nonempty physical-time unit label.")
        if domain.comm.size != 1:
            raise NotImplementedError(
                "Physical-time small-strain materials currently require serial execution."
            )
        if getattr(material, "stress_free_initial_state", False) is not True:
            raise NotImplementedError(
                "Physical-time materials must declare stress_free_initial_state=True for zero strain and zero named history."
            )
        if any(
            np.any(variable.initial_values() != 0)
            for variable in material.state_schema.variables
        ):
            raise NotImplementedError(
                "Physical-time initialization currently requires zero named history values."
            )
        if np.any(displacement.value.x.array != 0):
            raise ValueError(
                "Physical-time material initialization requires zero displacement; use load_checkpoint for continuation."
            )
        policy = step_controls.normalize(incrementation)
        if getattr(policy, "maximum_inelastic_increment", None) is not None:
            raise NotImplementedError(
                "Generic materials have no universal inelastic scalar; use provider suggested_time_scale."
            )
        if (
            isinstance(policy, step_controls.FixedIncrementation)
            and policy.load_factors[-1] != 1.0
        ):
            raise ValueError(
                "A physical-time fixed path must end at normalized coordinate 1."
            )
        if solver_options is not None:
            if not isinstance(solver_options, NewtonSolverOptions):
                raise TypeError(
                    "Physical-time material equilibrium requires NewtonSolverOptions."
                )
            if not solver_options.error_if_not_converged:
                raise NotImplementedError(
                    "Physical-time material equilibrium must raise on exhausted increments."
                )
        if external_force is not None and any(
            c is displacement.value for c in external_force.expression.coefficients()
        ):
            raise NotImplementedError(
                "Displacement-dependent loads require a verified external tangent."
            )
    if domain.geometry.dim != 3:
        raise NotImplementedError(
            "The first generic small-strain material Step supports 3D solids; reduced kinematics require an explicit provider contract."
        )
    stored = bool(getattr(material, "provides_stored_energy_density", False))
    dissipation = bool(
        getattr(material, "provides_dissipation_density_increment", False)
    )
    components = tuple(getattr(material, "stored_energy_component_names", ()))
    response = SmallStrainMaterialQuadratureResponse.create(
        domain,
        material.state_schema,
        degree=quadrature_degree,
        stored_energy=stored,
        dissipation=dissipation,
        stored_energy_component_names=components,
    )
    selected_amplitude = (
        amplitudes.ramp()
        if amplitude is None
        else amplitudes.as_amplitude(
            amplitude, name="small_strain_material_load_amplitude"
        )
    )
    if not np.isclose(selected_amplitude(0.0), 0.0):
        raise ValueError("A material Step load amplitude must start at zero.")
    load_factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    strain_test = elasticity.strain(displacement.test, study=study)
    strain_trial = elasticity.strain(displacement.trial, study=study)
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        response.tangent.function[i, j, k, l] * strain_trial[k, l], (i, j)
    )
    residual = (
        ufl.inner(response.cauchy_stress.function, strain_test) * response.measure
    )
    if external_force is not None:
        residual -= load_factor * external_force.expression
    jacobian = ufl.inner(tangent_action, strain_test) * response.measure
    bcs, prescribed = [], []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            bcs.extend(item.bcs)
            candidates = getattr(item, "dirichlet", ())
        elif hasattr(item, "bc"):
            bcs.append(item.bc)
            candidates = (item,)
        else:
            bcs.append(item)
            candidates = ()
        for constraint in candidates:
            value = getattr(constraint, "value", None)
            if value is not None and hasattr(value, "value"):
                prescribed.append(
                    (value, np.asarray(value.value, dtype=float).copy(), constraint.bc)
                )
    if duration is not None and len(prescribed) != len(bcs):
        raise NotImplementedError(
            "Physical-time materials require inspectable strong displacement constraints."
        )
    return SmallStrainMaterialStep(
        name=name,
        solution=displacement.value,
        material=material,
        response=response,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        load_factor=load_factor,
        amplitude=selected_amplitude,
        bcs=tuple(bcs),
        prescribed_values=tuple(prescribed),
        incrementation=step_controls.normalize(incrementation),
        solver_options=newton() if solver_options is None else solver_options,
        duration=duration,
        time_unit=time_unit,
        temperature=temperature,
        material_points=material_points,
        unit_system=unit_system,
        external_force=external_force,
        study=study,
        progress=progress,
        status_file=status_file,
    )


def _material_summary(material):
    selected = getattr(material, "summary", None)
    return (
        selected()
        if callable(selected)
        else {
            "kind": "small_strain_user_material",
            "name": material.name,
            "state_schema": material.state_schema.summary(),
            "parameter_schema": material.parameter_schema.summary(),
            "tangent_convention": material.tangent_convention.summary(),
        }
    )


def _runtime_evidence(material):
    implementation = getattr(material, "implementation", material)
    selected = getattr(implementation, "runtime_evidence", None)
    if callable(selected):
        return selected()
    return {"implementation": type(implementation).__name__}


__all__ = [
    "SmallStrainMaterialIncrementInfo",
    "SmallStrainMaterialPathInfo",
    "SmallStrainMaterialStep",
    "small_strain_material_step",
]
