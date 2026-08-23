"""Global quasi-static small-strain creep with committed quadrature state."""

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

from .. import amplitudes
from .. import procedures
from .. import steps as step_controls
from ..constitutive import elasticity
from ..constitutive.creep import IsotropicPowerLawCreepMaterial
from ..constitutive.quadrature import (
    CreepQuadratureState,
    QuadratureField,
    QuadratureMaterialMap,
)
from ..diagnostics import (
    SolveEventRecorder,
    StandardRunReporter,
    comm_of,
    compose_reporters,
)
from ..solvers import NewtonSolverOptions, SolveEvent, newton, solve_matrix_system


@dataclass(frozen=True)
class CreepIncrementInfo:
    increment: int
    attempt: int
    start_factor: float
    end_factor: float
    start_time: float
    end_time: float
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    creeping_points: int
    maximum_creep_increment: float
    maximum_local_iterations: int
    minimum_temperature: float | None = None
    maximum_temperature: float | None = None
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        def finite_or_none(value):
            selected = float(value)
            return selected if np.isfinite(selected) else None

        return {
            "increment": self.increment,
            "attempt": self.attempt,
            "start_factor": self.start_factor,
            "end_factor": self.end_factor,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "time_increment": self.end_time - self.start_time,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": finite_or_none(self.initial_residual_norm),
            "residual_norm": finite_or_none(self.residual_norm),
            "creeping_points": self.creeping_points,
            "maximum_creep_increment": finite_or_none(
                self.maximum_creep_increment
            ),
            "maximum_local_iterations": self.maximum_local_iterations,
            "minimum_temperature": self.minimum_temperature,
            "maximum_temperature": self.maximum_temperature,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "CreepIncrementInfo":
        return cls(
            increment=int(record["increment"]),
            attempt=int(record["attempt"]),
            start_factor=float(record["start_factor"]),
            end_factor=float(record["end_factor"]),
            start_time=float(record["start_time"]),
            end_time=float(record["end_time"]),
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
            creeping_points=int(record["creeping_points"]),
            maximum_creep_increment=(
                float("inf")
                if record["maximum_creep_increment"] is None
                else float(record["maximum_creep_increment"])
            ),
            maximum_local_iterations=int(record["maximum_local_iterations"]),
            minimum_temperature=(
                None
                if record.get("minimum_temperature") is None
                else float(record["minimum_temperature"])
            ),
            maximum_temperature=(
                None
                if record.get("maximum_temperature") is None
                else float(record["maximum_temperature"])
            ),
            rejection_reason=record.get("rejection_reason"),
        )


@dataclass(frozen=True)
class CreepPathInfo:
    increments: tuple[CreepIncrementInfo, ...]
    attempts: tuple[CreepIncrementInfo, ...]
    duration: float
    incrementation: object

    @property
    def converged(self) -> bool:
        return (
            bool(self.increments)
            and all(item.converged for item in self.increments)
            and bool(self.attempts)
            and self.attempts[-1].converged
        )

    @property
    def completed_step(self) -> bool:
        return self.converged and abs(self.increments[-1].end_time - self.duration) <= (
            1.0e-12 * max(1.0, self.duration)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "implicit_creep_time_path",
            "converged": self.converged,
            "completed_step": self.completed_step,
            "duration": self.duration,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": [item.as_dict() for item in self.increments],
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass(frozen=True)
class CreepEnergyFrame:
    time: float
    elastic_strain_energy: float
    creep_dissipation_increment: float
    creep_dissipation: float

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "CreepEnergyFrame":
        return cls(**{name: float(record[name]) for name in cls.__dataclass_fields__})


@dataclass
class ImplicitCreepStep:
    """Adaptive backward-Euler creep step with global Newton equilibrium."""

    name: str
    solution: object
    material: IsotropicPowerLawCreepMaterial | QuadratureMaterialMap
    state: CreepQuadratureState
    residual_form: object
    tangent_form: object
    load_factor: object
    amplitude: amplitudes.Amplitude
    temperature: object | None
    bcs: tuple[object, ...]
    prescribed_values: tuple[tuple[object, np.ndarray, object], ...]
    time_dependent_constraints: tuple[object, ...]
    duration: float
    incrementation: object
    solver_options: NewtonSolverOptions
    study: object | None = None
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    procedure: object = field(default_factory=procedures.implicit_creep)
    accepted_time: float = field(default=0.0, init=False)
    last_solve_info: CreepPathInfo | None = field(default=None, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    accepted_increments: list[CreepIncrementInfo] = field(default_factory=list, init=False)
    attempted_increments: list[CreepIncrementInfo] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    energy_history: list[CreepEnergyFrame] = field(default_factory=list, init=False)
    next_increment_size: float | None = field(default=None, init=False)
    _strain_evaluator: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.duration = float(self.duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("Implicit creep duration must be finite and positive.")
        materials = (
            tuple(self.material.materials.values())
            if isinstance(self.material, QuadratureMaterialMap)
            else (self.material,)
        )
        temperature_modes = {
            item.temperature_dependence is not None for item in materials
        }
        if len(temperature_modes) != 1:
            raise ValueError(
                "One creep step cannot mix isothermal and Arrhenius material "
                "regions; split the procedure or give every region the same "
                "temperature-dependence contract."
            )
        requires_temperature = temperature_modes.pop()
        if requires_temperature and self.temperature is None:
            raise ValueError(
                "Temperature-dependent creep requires temperature=... in kelvin."
            )
        if not requires_temperature and self.temperature is not None:
            raise ValueError(
                "temperature=... was supplied to an isothermal creep material."
            )
        self._strain_evaluator = self.state.compile_strain(
            elasticity.strain(self.solution)
        )

    @property
    def accepted_factor(self) -> float:
        return self.accepted_time / self.duration

    def solve(self, *, until: float | None = None):
        """Advance to a physical time, or to the full step duration."""

        selected_until = self.duration if until is None else float(until)
        if not self.accepted_time < selected_until <= self.duration:
            raise ValueError(
                "until must be greater than accepted_time and no larger than duration."
            )
        until_factor = selected_until / self.duration
        reporter = self._reporter()
        accepted: list[CreepIncrementInfo] = []
        attempts: list[CreepIncrementInfo] = []
        accepted_factor = self.accepted_factor
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
                incrementation="implicit creep / "
                + self.incrementation.summary()["kind"],
                time=self.accepted_time,
            ),
        )
        self._apply_loading(self.accepted_time)
        while accepted_factor < until_factor - 1.0e-12:
            increment = len(self.accepted_increments) + len(accepted) + 1
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                if len(self.accepted_increments) + len(accepted) >= self.incrementation.max_increments:
                    raise RuntimeError(
                        "Implicit creep reached max_increments before the step end."
                    )
                target_factor = min(until_factor, accepted_factor + proposed_size)
            else:
                remaining = [
                    value
                    for value in self.incrementation.load_factors
                    if value > accepted_factor + 1.0e-12
                ]
                if not remaining:
                    raise RuntimeError(
                        "Fixed incrementation has no point beyond restored creep time."
                    )
                target_factor = min(until_factor, remaining[0])
            start_time = accepted_factor * self.duration
            end_time = target_factor * self.duration
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
                    target_factor=target_factor,
                    time=end_time,
                ),
            )
            displacement_snapshot = self.solution.x.array.copy()
            state_snapshot = self.state.snapshot()
            self._apply_loading(end_time)
            info = self._solve_increment(
                increment=increment,
                attempt=attempt,
                start_factor=accepted_factor,
                target_factor=target_factor,
                start_time=start_time,
                end_time=end_time,
                reporter=reporter,
            )
            if (
                info.converged
                and isinstance(self.incrementation, step_controls.AutomaticIncrementation)
                and self.incrementation.maximum_inelastic_increment is not None
                and info.maximum_creep_increment
                > self.incrementation.maximum_inelastic_increment
            ):
                info = replace(
                    info,
                    converged=False,
                    rejection_reason=(
                        "maximum equivalent creep-strain increment "
                        f"{info.maximum_creep_increment:.6g} exceeds "
                        f"{self.incrementation.maximum_inelastic_increment:.6g}"
                    ),
                )
            attempts.append(info)
            if info.converged:
                self.state.commit()
                accepted.append(info)
                accepted_size = target_factor - accepted_factor
                accepted_factor = target_factor
                self.accepted_time = end_time
                self._record_energy(end_time, state_snapshot)
                consecutive_cutbacks = 0
                if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                    proposed_size = self.incrementation.after_convergence(
                        accepted_size,
                        info.iterations,
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
                        start_factor=info.start_factor,
                        target_factor=target_factor,
                        iteration=info.iterations,
                        residual_norm=info.residual_norm,
                        time=end_time,
                    ),
                )
                continue

            self.solution.x.array[:] = displacement_snapshot
            self.solution.x.scatter_forward()
            self.state.restore(state_snapshot)
            self._apply_loading(start_time)
            if not isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                self.accepted_increments.extend(accepted)
                self.attempted_increments.extend(attempts)
                self._fail(reporter, info, "fixed creep increment did not converge")
                return self.solution
            consecutive_cutbacks += 1
            proposed_size = self.incrementation.after_failure(
                target_factor - accepted_factor
            )
            self.next_increment_size = proposed_size
            if (
                consecutive_cutbacks > self.incrementation.max_cutbacks
                or proposed_size < self.incrementation.minimum
            ):
                self.accepted_increments.extend(accepted)
                self.attempted_increments.extend(attempts)
                self._fail(
                    reporter,
                    info,
                    "automatic creep incrementation exhausted its cutback allowance",
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
                    target_factor=target_factor,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    next_increment=proposed_size,
                    message=info.rejection_reason or "global Newton did not converge",
                    time=end_time,
                ),
            )

        self.accepted_increments.extend(accepted)
        self.attempted_increments.extend(attempts)
        self.last_solve_info = CreepPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.duration,
            self.incrementation,
        )
        self._emit(
            reporter,
            SolveEvent(
                "step_completed" if self.accepted_time >= self.duration else "step_paused",
                self.name,
                step_number=self.step_number,
                increment=len(self.accepted_increments),
                attempt=len(self.attempted_increments),
                target_factor=self.accepted_factor,
                time=self.accepted_time,
            ),
        )
        return self.solution

    def _solve_increment(
        self,
        *,
        increment: int,
        attempt: int,
        start_factor: float,
        target_factor: float,
        start_time: float,
        end_time: float,
        reporter,
    ) -> CreepIncrementInfo:
        initial_norm = None
        norm = float("inf")
        update_info = {
            "creeping_points": 0,
            "maximum_creep_increment": 0.0,
            "maximum_local_iterations": 0,
            "minimum_temperature": None,
            "maximum_temperature": None,
        }
        converged = False
        rejection_reason = None
        iteration = 0
        for iteration in range(self.solver_options.maximum_iterations + 1):
            try:
                update_info = self.state.update(
                    self.state.evaluate_strain(self._strain_evaluator),
                    self.material,
                    time_start=start_time,
                    time_end=end_time,
                    temperature_values=self._temperature_values(),
                )
            except RuntimeError as exc:
                rejection_reason = str(exc)
                break
            rhs, norm = self._correction_rhs()
            if initial_norm is None:
                initial_norm = norm
            threshold = self.solver_options.absolute_tolerance + (
                self.solver_options.relative_tolerance * initial_norm
            )
            if np.isfinite(norm) and norm <= threshold:
                rhs.destroy()
                converged = True
                break
            if iteration == self.solver_options.maximum_iterations:
                rhs.destroy()
                rejection_reason = "global Newton iteration limit reached"
                break
            tangent = fem_petsc.assemble_matrix(self.tangent_form, bcs=self.bcs)
            tangent.assemble()
            correction = rhs.duplicate()
            correction.set(0.0)
            linear_info = solve_matrix_system(
                tangent,
                rhs,
                correction,
                self.solver_options.linear_solver,
                raise_on_failure=False,
            )
            tangent.destroy()
            rhs.destroy()
            if not linear_info.converged:
                correction.destroy()
                rejection_reason = (
                    "linear correction failed: KSP reason "
                    f"{linear_info.converged_reason}"
                )
                break
            base = self.solution.x.array.copy()
            direction = correction.array_r.copy()
            correction.destroy()
            alpha = self._line_search(
                base,
                direction,
                norm,
                time_start=start_time,
                time_end=end_time,
            )
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
                    iteration=iteration + 1,
                    residual_norm=norm,
                    step_length=alpha,
                    time=end_time,
                ),
            )
            if alpha == 0.0:
                rejection_reason = "line search could not reduce the residual"
                break
        return CreepIncrementInfo(
            increment=increment,
            attempt=attempt,
            start_factor=start_factor,
            end_factor=target_factor,
            start_time=start_time,
            end_time=end_time,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=float(initial_norm or 0.0),
            residual_norm=float(norm),
            creeping_points=int(update_info["creeping_points"]),
            maximum_creep_increment=float(update_info["maximum_creep_increment"]),
            maximum_local_iterations=int(update_info["maximum_local_iterations"]),
            minimum_temperature=update_info["minimum_temperature"],
            maximum_temperature=update_info["maximum_temperature"],
            rejection_reason=rejection_reason,
        )

    def _line_search(
        self,
        base,
        direction,
        base_norm: float,
        *,
        time_start: float,
        time_end: float,
    ) -> float:
        options = self.solver_options
        alpha = 1.0
        if options.line_search in {None, "basic"}:
            self._assign_trial(base, direction, alpha)
            return alpha
        while alpha + 1.0e-15 >= options.minimum_step_length:
            self._assign_trial(base, direction, alpha)
            try:
                self.state.update(
                    self.state.evaluate_strain(self._strain_evaluator),
                    self.material,
                    time_start=time_start,
                    time_end=time_end,
                    temperature_values=self._temperature_values(),
                )
            except RuntimeError:
                alpha *= options.line_search_reduction
                continue
            rhs, trial_norm = self._correction_rhs()
            rhs.destroy()
            if np.isfinite(trial_norm) and trial_norm < base_norm:
                return alpha
            alpha *= options.line_search_reduction
        self.solution.x.array[:] = base
        self.solution.x.scatter_forward()
        self.state.rollback()
        return 0.0

    def _assign_trial(self, base, direction, alpha: float) -> None:
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
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        residual.scale(-1.0)
        fem_petsc.set_bc(
            residual,
            self.bcs,
            x0=self.solution.x.petsc_vec,
            alpha=1.0,
        )
        return residual, float(residual.norm())

    def _apply_loading(self, physical_time: float) -> None:
        if self.temperature is not None and hasattr(self.temperature, "apply"):
            self.temperature.apply(physical_time)
        value = self.amplitude(physical_time)
        self.load_factor.value = PETSc.ScalarType(value)
        for constant, target, _bc in self.prescribed_values:
            selected = value * target
            constant.value = (
                PETSc.ScalarType(selected.item())
                if selected.ndim == 0 or selected.size == 1
                else np.asarray(selected, dtype=PETSc.ScalarType)
            )
        for constraint in self.time_dependent_constraints:
            constraint.update(physical_time)

    def _temperature_values(self):
        if self.temperature is None:
            return None
        selected = getattr(self.temperature, "value", self.temperature)
        if hasattr(selected, "function_space"):
            return self.state.evaluate_scalar(selected)
        scalar = np.asarray(selected, dtype=float)
        if scalar.size != 1:
            raise ValueError(
                "Creep temperature must be a scalar or scalar finite-element field."
            )
        value = float(scalar.reshape(-1)[0])
        return np.full(len(self.state.stress.values), value, dtype=float)

    def _temperature_summary(self) -> dict[str, object] | None:
        values = self._temperature_values()
        if values is None:
            return None
        canonical = np.ascontiguousarray(values, dtype=np.float64)
        selected = getattr(self.temperature, "value", self.temperature)
        result = {
            "kind": (
                "finite_element_field"
                if hasattr(selected, "function_space")
                else "constant"
            ),
            "unit": "K",
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "quadrature_value_count": int(canonical.size),
            "quadrature_values_sha256": sha256(canonical.tobytes()).hexdigest(),
            "field_name": getattr(selected, "name", None),
        }
        if hasattr(self.temperature, "scientific_identity"):
            result["history"] = self.temperature.scientific_identity()
            result["active_time"] = getattr(self.temperature, "active_time", None)
        return result

    def _record_energy(self, time: float, old_state) -> None:
        mechanical_strain = elasticity.strain(self.solution) - self.state.creep_strain.function
        elastic_density = 0.5 * ufl.inner(self.state.stress.function, mechanical_strain)
        elastic_local = fem.assemble_scalar(fem.form(elastic_density * self.state.measure))
        elastic = float(self.state.domain.comm.allreduce(elastic_local))

        increment = QuadratureField.create(
            self.state.domain,
            name="DCE",
            degree=self.state.degree,
            value_shape=(3, 3),
            scheme=self.state.scheme,
        )
        increment.assign(
            self.state.creep_strain.values - np.asarray(old_state["creep_strain"])
        )
        dissipation_density = ufl.inner(self.state.stress.function, increment.function)
        local = fem.assemble_scalar(fem.form(dissipation_density * self.state.measure))
        dissipation_increment = float(self.state.domain.comm.allreduce(local))
        cumulative = dissipation_increment
        if self.energy_history:
            cumulative += self.energy_history[-1].creep_dissipation
        self.energy_history.append(
            CreepEnergyFrame(
                time=float(time),
                elastic_strain_energy=elastic,
                creep_dissipation_increment=dissipation_increment,
                creep_dissipation=cumulative,
            )
        )

    def reaction_field(self, *, name: str = "RF"):
        residual = fem_petsc.assemble_vector(self.residual_form)
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        reaction = fem.Function(self.solution.function_space, name=name)
        values = residual.array_r
        reaction.x.array[: len(values)] = values
        reaction.x.scatter_forward()
        residual.destroy()
        return reaction

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        comm = self.solution.function_space.mesh.comm
        selected_portable = comm.size != 1 if portable is None else bool(portable)
        if selected_portable:
            return self._save_portable_checkpoint(path)
        if comm.size != 1:
            raise ValueError("Distributed creep checkpoints must use portable=True.")
        selected = Path(path)
        if selected.suffix != ".npz":
            selected = selected.with_suffix(".npz")
        selected.parent.mkdir(parents=True, exist_ok=True)
        state = self.state.snapshot()
        identity = self._checkpoint_identity()
        from ..checkpointing import atomic_savez

        atomic_savez(
            selected,
            schema="agentfem.implicit-creep-checkpoint.v1",
            step_identity=json.dumps(identity, sort_keys=True),
            displacement=self.solution.x.array,
            accepted_time=self.accepted_time,
            creep_strain=state["creep_strain"],
            equivalent_creep_strain=state["equivalent_creep_strain"],
            accepted_increments=json.dumps(
                [item.as_dict() for item in self.accepted_increments]
            ),
            attempted_increments=json.dumps(
                [item.as_dict() for item in self.attempted_increments]
            ),
            execution_events=json.dumps(
                [event.as_dict() for event in self.execution_events]
            ),
            energy_history=json.dumps(
                [frame.as_dict() for frame in self.energy_history]
            ),
            next_increment_size=(
                np.nan if self.next_increment_size is None else self.next_increment_size
            ),
        )
        from ..results import CheckpointRecord

        record = CheckpointRecord(
            name=f"{self.name}_{self.accepted_time:g}",
            path=selected,
            schema="agentfem.implicit-creep-checkpoint.v1",
            step_name=self.name,
            coordinate_name="time",
            coordinate_value=self.accepted_time,
            portable=False,
            metadata={
                "reason": "serial dof and quadrature layout checkpoint",
                "state_variables": ("U", "CE", "CEEQ"),
                "identity": identity,
            },
        )
        record.write_manifest()
        self.checkpoints.append(record)
        return selected

    def load_checkpoint(self, path) -> None:
        selected = Path(path)
        manifest = (
            selected
            if selected.name.endswith(".checkpoint.json")
            else selected.with_suffix("").with_name(
                selected.with_suffix("").name + ".checkpoint.json"
            )
        )
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("schema") == "agentfem.implicit-creep-checkpoint.v2":
                self._load_portable_checkpoint(manifest, payload)
                return
        if self.solution.function_space.mesh.comm.size != 1:
            raise ValueError(
                "This legacy creep checkpoint is partition-bound; use a v2 "
                "portable checkpoint for distributed restart."
            )
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != "agentfem.implicit-creep-checkpoint.v1":
                raise ValueError("Unsupported implicit creep checkpoint schema.")
            stored_identity = json.loads(str(data["step_identity"]))
            current_identity = json.loads(
                json.dumps(self._checkpoint_identity(), sort_keys=True)
            )
            if stored_identity != current_identity:
                if stored_identity.get("temperature") != current_identity.get(
                    "temperature"
                ):
                    raise ValueError(
                        "Creep checkpoint temperature field identity differs."
                    )
                raise ValueError(
                    "Creep checkpoint material, duration, increment control, "
                    "quadrature state, or mesh/function layout differs."
                )
            displacement = np.asarray(data["displacement"])
            if displacement.size != self.solution.x.array.size:
                raise ValueError("Creep checkpoint displacement layout does not match.")
            self.solution.x.array[:] = displacement
            self.solution.x.scatter_forward()
            self.state.restore(
                {
                    "creep_strain": data["creep_strain"],
                    "equivalent_creep_strain": data["equivalent_creep_strain"],
                }
            )
            self.accepted_time = float(data["accepted_time"])
            self.accepted_increments[:] = [
                CreepIncrementInfo.from_dict(item)
                for item in json.loads(str(data["accepted_increments"]))
            ]
            self.attempted_increments[:] = [
                CreepIncrementInfo.from_dict(item)
                for item in json.loads(str(data["attempted_increments"]))
            ]
            self.execution_events[:] = [
                SolveEvent.from_dict(item)
                for item in json.loads(str(data["execution_events"]))
            ]
            self.energy_history[:] = [
                CreepEnergyFrame.from_dict(item)
                for item in json.loads(str(data["energy_history"]))
            ]
            selected_size = float(data["next_increment_size"])
            self.next_increment_size = selected_size if np.isfinite(selected_size) else None
            self._apply_loading(self.accepted_time)
            self.last_solve_info = CreepPathInfo(
                tuple(self.accepted_increments),
                tuple(self.attempted_increments),
                self.duration,
                self.incrementation,
            )
            self.state.refresh_response(
                self.state.evaluate_strain(self._strain_evaluator),
                self.material,
            )

    def _save_portable_checkpoint(self, path) -> Path:
        from ..checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
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
            "schema": "agentfem.implicit-creep-checkpoint.v2",
            "step_identity": self._portable_checkpoint_identity(),
            "coordinate": float(self.accepted_time),
            "nodal_state": bundle["record"],
            "nodal_identity": bundle["identities"],
            "quadrature_state": checkpoint_file_record(quadrature),
            "accepted_increments": [item.as_dict() for item in self.accepted_increments],
            "attempted_increments": [item.as_dict() for item in self.attempted_increments],
            "execution_events": [item.as_dict() for item in self.execution_events],
            "energy_history": [item.as_dict() for item in self.energy_history],
            "next_increment_size": self.next_increment_size,
        }
        comm = self.solution.function_space.mesh.comm
        error = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"Creep checkpoint manifest write failed: {error}")
        comm.barrier()
        from ..results import CheckpointRecord

        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_{self.accepted_time:g}",
                path=manifest,
                schema="agentfem.implicit-creep-checkpoint.v2",
                step_name=self.name,
                coordinate_name="time",
                coordinate_value=self.accepted_time,
                portable=True,
                metadata={"state_variables": ("U", "CE", "CEEQ")},
            )
        )
        return manifest

    def _load_portable_checkpoint(self, manifest: Path, payload: dict) -> None:
        from ..checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )

        current = json.loads(
            json.dumps(self._portable_checkpoint_identity(), sort_keys=True)
        )
        if payload.get("step_identity") != current:
            raise ValueError("Portable creep checkpoint scientific identity differs.")
        load_portable_state_bundle(
            manifest,
            state={"U": self.solution},
            record=payload["nodal_state"],
            identities=payload["nodal_identity"],
        )
        self.state.load(
            validate_checkpoint_record(manifest.parent, payload["quadrature_state"]),
            material=self.material,
        )
        self.accepted_time = float(payload["coordinate"])
        self.accepted_increments[:] = [
            CreepIncrementInfo.from_dict(item) for item in payload["accepted_increments"]
        ]
        self.attempted_increments[:] = [
            CreepIncrementInfo.from_dict(item) for item in payload["attempted_increments"]
        ]
        self.execution_events[:] = [
            SolveEvent.from_dict(item) for item in payload["execution_events"]
        ]
        self.energy_history[:] = [
            CreepEnergyFrame.from_dict(item) for item in payload["energy_history"]
        ]
        self.next_increment_size = payload.get("next_increment_size")
        self._apply_loading(self.accepted_time)
        self.last_solve_info = CreepPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.duration,
            self.incrementation,
        )
        self.state.refresh_response(
            self.state.evaluate_strain(self._strain_evaluator), self.material
        )

    def _portable_checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "duration": self.duration,
            "material": self.material.as_dict(),
            "amplitude": self.amplitude.summary(),
            "incrementation": self.incrementation.summary(),
            "solution": function_portable_identity(self.solution),
            "quadrature": self.state.summary()["transaction"],
            "temperature": self._portable_temperature_summary(),
        }

    def _portable_temperature_summary(self) -> dict[str, object] | None:
        if hasattr(self.temperature, "portable_identity"):
            return self.temperature.portable_identity()
        if hasattr(self.temperature, "scientific_identity"):
            return self.temperature.scientific_identity()
        selected = getattr(self.temperature, "value", self.temperature)
        if selected is None:
            return None
        if hasattr(selected, "function_space"):
            from ..checkpointing import function_portable_identity

            return {
                "kind": "finite_element_field",
                "unit": "K",
                "field_name": getattr(selected, "name", None),
                "identity": function_portable_identity(selected),
            }
        scalar = np.asarray(selected, dtype=float)
        return {"kind": "constant", "unit": "K", "value": scalar.tolist()}

    def _checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_partition_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "duration": self.duration,
            "material": self.material.as_dict(),
            "amplitude": self.amplitude.summary(),
            "incrementation": self.incrementation.summary(),
            "solution": function_partition_identity(self.solution),
            "creep_strain": function_partition_identity(
                self.state.creep_strain.function
            ),
            "equivalent_creep_strain": function_partition_identity(
                self.state.equivalent_creep_strain.function
            ),
            "temperature": self._temperature_summary(),
        }

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        output_fields=(),
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve and publish creep state through the common result path."""
        from ..results import (
            add_execution_trace,
            complete_result,
            from_solution,
            recover_integration_point_field,
        )

        if fields and output_fields:
            raise ValueError("Pass fields=... or output_fields=..., not both.")
        selected_output_fields = tuple(fields) or tuple(output_fields)

        solution = self.solve() if self.accepted_time < self.duration else self.solution
        result = from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "state": self.state.summary(),
            },
        )
        add_execution_trace(result, self.execution_events)
        for name, field_value, description in (
            ("S", self.state.stress.function, "Stress at constitutive integration points."),
            ("CE", self.state.creep_strain.function, "Committed creep strain."),
            (
                "CEEQ",
                self.state.equivalent_creep_strain.function,
                "Committed equivalent creep strain.",
            ),
            (
                "MISES",
                self.state.equivalent_stress().function,
                "Von Mises stress at integration points.",
            ),
        ):
            result.add_field(
                name,
                field_value,
                location="quadrature_points",
                description=description,
                processing={
                    "source_position": "quadrature_points",
                    "method": "implicit_constitutive_update",
                    "representation": "quadrature_values",
                    "postprocessed": False,
                    "committed": name in {"CE", "CEEQ"},
                },
            )
        for source, recovered_name in (
            (self.state.stress, "S_CELL"),
            (self.state.creep_strain, "CE_CELL"),
            (self.state.equivalent_creep_strain, "CEEQ_CELL"),
            (self.state.equivalent_stress(), "MISES_CELL"),
        ):
            recovered = recover_integration_point_field(
                source,
                name=recovered_name,
            )
            result.add_field(
                recovered.name,
                recovered.field,
                unit=recovered.unit,
                location=recovered.location,
                description=recovered.description,
                processing=recovered.processing,
            )
        reaction = self.reaction_field()
        result.add_field(
            "RF",
            reaction,
            description="Full nodal residual for reaction extraction.",
            processing={"method": "assembled_equilibrium_residual"},
        )
        result.add_quantities(
            {
                "analysis_time": self.accepted_time,
                "maximum_equivalent_creep_strain": (
                    self.state.equivalent_creep_strain.global_max()
                ),
                "creeping_integration_points": (
                    self.state.equivalent_creep_strain.global_count_nonzero()
                ),
            },
            kind="diagnostic",
        )
        temperature_values = self._temperature_values()
        if temperature_values is not None:
            selected_temperature = getattr(
                self.temperature, "value", self.temperature
            )
            if hasattr(selected_temperature, "function_space"):
                result.add_field(
                    "TEMP",
                    selected_temperature,
                    unit="K",
                    description="Temperature field consumed by Arrhenius creep.",
                    processing={
                        "method": "quadrature_interpolation_for_constitutive_update",
                        "postprocessed": False,
                    },
                )
            result.add_quantities(
                {
                    "minimum_creep_temperature": float(np.min(temperature_values)),
                    "maximum_creep_temperature": float(np.max(temperature_values)),
                },
                units={
                    "minimum_creep_temperature": "K",
                    "maximum_creep_temperature": "K",
                },
                kind="diagnostic",
            )
        if self.accepted_increments:
            times = np.asarray([item.end_time for item in self.accepted_increments])
            increment_histories = {
                "newton_iterations": [
                    item.iterations for item in self.accepted_increments
                ],
                "maximum_creep_increment": [
                    item.maximum_creep_increment for item in self.accepted_increments
                ],
                "maximum_local_iterations": [
                    item.maximum_local_iterations for item in self.accepted_increments
                ],
            }
            materials = (
                tuple(self.material.materials.values())
                if isinstance(self.material, QuadratureMaterialMap)
                else (self.material,)
            )
            if all(item.temperature_dependence is not None for item in materials):
                increment_histories.update(
                    {
                        "minimum_creep_temperature": [
                            item.minimum_temperature for item in self.accepted_increments
                        ],
                        "maximum_creep_temperature": [
                            item.maximum_temperature for item in self.accepted_increments
                        ],
                    }
                )
            result.add_histories(
                times,
                increment_histories,
                abscissa_name="time",
                abscissa_unit="s",
            )
        if self.energy_history:
            times = np.asarray([item.time for item in self.energy_history])
            result.add_histories(
                times,
                {
                    "elastic_strain_energy": [
                        item.elastic_strain_energy for item in self.energy_history
                    ],
                    "creep_dissipation_increment": [
                        item.creep_dissipation_increment for item in self.energy_history
                    ],
                    "creep_dissipation": [
                        item.creep_dissipation for item in self.energy_history
                    ],
                },
                abscissa_name="time",
                abscissa_unit="s",
            )
        for checkpoint in self.checkpoints:
            result.add_checkpoint(checkpoint)
        return complete_result(
            self,
            result,
            output=output,
            fields=selected_output_fields,
            strict_output=strict_output,
            metadata=metadata,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "implicit_creep_step",
            "name": self.name,
            "study": None if self.study is None else self.study.summary(),
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "state": self.state.summary(),
            "duration": self.duration,
            "accepted_time": self.accepted_time,
            "incrementation": self.incrementation.summary(),
            "solver": self.solver_options.summary(),
            "amplitude": self.amplitude.summary(),
            "temperature": self._temperature_summary(),
            "last_solve": (
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
            "next_increment_size": self.next_increment_size,
        }

    def _reporter(self):
        recorder = SolveEventRecorder(self.execution_events)
        if self.progress is True:
            return compose_reporters(
                recorder,
                StandardRunReporter(comm_of(self.solution), status_file=self.status_file),
            )
        if self.progress in (False, None):
            return recorder
        return compose_reporters(recorder, self.progress)

    @staticmethod
    def _emit(reporter, event) -> None:
        if reporter is not None:
            reporter.emit(event) if hasattr(reporter, "emit") else reporter(event)

    def _fail(self, reporter, info: CreepIncrementInfo, message: str) -> None:
        self._emit(
            reporter,
            SolveEvent(
                "step_failed",
                self.name,
                step_number=self.step_number,
                increment=info.increment,
                attempt=info.attempt,
                start_factor=info.start_factor,
                target_factor=info.end_factor,
                iteration=info.iterations,
                residual_norm=info.residual_norm,
                message=message,
                time=info.end_time,
            ),
        )
        self.last_solve_info = CreepPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.duration,
            self.incrementation,
        )
        if self.solver_options.error_if_not_converged:
            raise RuntimeError(f"{self.name}: {message}.")


def implicit_creep_step(
    *,
    displacement,
    material,
    duration: float,
    external_force,
    constraints=(),
    study=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    progress=True,
    status_file=None,
    amplitude=None,
    temperature=None,
    name: str = "implicit_creep",
    _experimental_distributed: bool = False,
) -> ImplicitCreepStep:
    """Build the first global 3D implicit power-law creep step."""

    if not isinstance(
        material, (IsotropicPowerLawCreepMaterial, QuadratureMaterialMap)
    ):
        raise TypeError(
            "implicit_creep_step requires IsotropicPowerLawCreepMaterial or "
            "a QuadratureMaterialMap of that family."
        )
    domain = displacement.value.function_space.mesh
    if isinstance(material, QuadratureMaterialMap) and material.domain is not domain:
        raise ValueError("Creep quadrature material map belongs to a different mesh.")
    if domain.comm.size != 1 and not _experimental_distributed:
        raise NotImplementedError(
            "Distributed creep quadrature state and portable restart are "
            "supported, but the custom global Newton equilibrium path has not "
            "yet passed the MPI partition-interface patch test. Run this Step "
            "in serial."
        )
    if domain.geometry.dim != 3:
        raise NotImplementedError(
            "The first global creep driver supports 3D small-strain solids."
        )
    state = CreepQuadratureState.create(domain, degree=quadrature_degree)
    selected_amplitude = (
        amplitudes.constant(1.0, name="held_creep_load")
        if amplitude is None
        else amplitudes.as_amplitude(amplitude, name="creep_load_amplitude")
    )
    load_factor = fem.Constant(domain, PETSc.ScalarType(selected_amplitude(0.0)))
    strain_test = elasticity.strain(displacement.test)
    strain_trial = elasticity.strain(displacement.trial)
    stress = state.stress.function
    tangent = state.tangent.function
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        tangent[i, j, k, l] * strain_trial[k, l],
        (i, j),
    )
    residual = ufl.inner(stress, strain_test) * state.measure
    if external_force is not None:
        residual -= load_factor * external_force.expression
    jacobian = ufl.inner(tangent_action, strain_test) * state.measure
    selected_bcs = []
    prescribed_values = []
    time_dependent_constraints = []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected_bcs.extend(item.bcs)
            for constraint in getattr(item, "dirichlet", ()):
                value = getattr(constraint, "value", None)
                if value is not None and hasattr(value, "value"):
                    prescribed_values.append(
                        (
                            value,
                            np.asarray(value.value, dtype=float).copy(),
                            constraint.bc,
                        )
                    )
        elif hasattr(item, "bc"):
            selected_bcs.append(item.bc)
            if hasattr(item, "amplitude") and hasattr(item, "update"):
                time_dependent_constraints.append(item)
            else:
                value = getattr(item, "value", None)
                if value is not None and hasattr(value, "value"):
                    prescribed_values.append(
                        (value, np.asarray(value.value, dtype=float).copy(), item.bc)
                    )
        else:
            selected_bcs.append(item)
    return ImplicitCreepStep(
        name=name,
        solution=displacement.value,
        material=material,
        state=state,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        load_factor=load_factor,
        amplitude=selected_amplitude,
        temperature=temperature,
        bcs=tuple(selected_bcs),
        prescribed_values=tuple(prescribed_values),
        time_dependent_constraints=tuple(time_dependent_constraints),
        duration=duration,
        incrementation=step_controls.normalize(incrementation),
        solver_options=newton() if solver_options is None else solver_options,
        study=study,
        progress=progress,
        status_file=status_file,
    )


__all__ = [
    "CreepEnergyFrame",
    "CreepIncrementInfo",
    "CreepPathInfo",
    "ImplicitCreepStep",
    "implicit_creep_step",
]
