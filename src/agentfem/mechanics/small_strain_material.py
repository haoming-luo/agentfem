# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Global equilibrium for provider-neutral small-strain materials."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc

from .. import amplitudes, procedures
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
from ..solvers import NewtonSolverOptions, SolveEvent, newton, solve_matrix_system


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

    def __post_init__(self) -> None:
        expression = elasticity.strain(self.solution, study=self.study)
        self._strain_evaluator = self.response.state.compile_expression(
            expression, value_shape=(3, 3)
        )
        self._accepted_strain = self._evaluate_strain()
        self._update_material(time=0.0, time_increment=1.0)
        self.response.commit()

    @property
    def state(self):
        """Expose the ordinary state owner used by output/checkpoint tooling."""

        return self.response.state

    def _evaluate_strain(self) -> np.ndarray:
        return self.response.state.evaluate_expression(
            self._strain_evaluator, value_shape=(3, 3)
        )

    def _update_material(self, *, time: float, time_increment: float):
        self._trial_result = self.response.update(
            self.material,
            strain_old=self._accepted_strain,
            strain_new=self._evaluate_strain(),
            time=float(time),
            time_increment=max(float(time_increment), np.finfo(float).eps),
        )
        return self._trial_result

    def solve(self, *, until: float = 1.0):
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
            self._apply_loading(target)
            try:
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
                self.response.commit()
                self._accepted_strain = self._evaluate_strain()
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
        initial_norm = None
        norm = float("inf")
        result = None
        accepted_trial = None
        converged = False
        iteration = 0
        for iteration in range(self.solver_options.maximum_iterations + 1):
            if accepted_trial is None:
                result = self._update_material(
                    time=target_factor, time_increment=target_factor - start_factor
                )
                rhs, norm = self._correction_rhs()
            else:
                result, rhs, norm = accepted_trial
                accepted_trial = None
            if initial_norm is None:
                initial_norm = norm
            threshold = (
                self.solver_options.absolute_tolerance
                + self.solver_options.relative_tolerance * initial_norm
            )
            if np.isfinite(norm) and norm <= threshold:
                rhs.destroy()
                converged = True
                break
            if iteration == self.solver_options.maximum_iterations:
                rhs.destroy()
                break
            tangent = fem_petsc.assemble_matrix(self.tangent_form, bcs=self.bcs)
            tangent.assemble()
            correction = rhs.duplicate()
            correction.set(0.0)
            linear = solve_matrix_system(
                tangent,
                rhs,
                correction,
                self.solver_options.linear_solver,
                raise_on_failure=False,
            )
            tangent.destroy()
            rhs.destroy()
            if not linear.converged:
                correction.destroy()
                break
            base = self.solution.x.array.copy()
            direction = correction.array_r.copy()
            correction.destroy()
            alpha, accepted_trial = self._line_search(
                base, direction, norm, target_factor, target_factor - start_factor
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
                ),
            )
            if alpha == 0.0:
                break
        summary = result.summary() if result is not None else {}
        return SmallStrainMaterialIncrementInfo(
            increment=increment,
            attempt=attempt,
            start_load_factor=start_factor,
            load_factor=target_factor,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=float(initial_norm or 0.0),
            residual_norm=float(norm),
            minimum_suggested_time_scale=float(
                summary.get("minimum_suggested_time_scale", 1.0)
            ),
            applicability_counts=dict(summary.get("applicability_counts", {})),
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

    @staticmethod
    def _emit(reporter, event):
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
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        from ..checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
        )
        from ..results import CheckpointRecord

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
            "schema": "agentfem.small-strain-material-step-checkpoint.v1",
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
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=True,
            metadata={
                "state_variables": ("U", *tuple(self.state.transaction.names)),
                "material": _material_summary(self.material),
            },
        )
        self.checkpoints.append(record)
        return manifest

    def load_checkpoint(self, path) -> None:
        from ..checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )

        manifest = Path(path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema") != "agentfem.small-strain-material-step-checkpoint.v1":
            raise ValueError("Unsupported small-strain material checkpoint schema.")
        if payload.get("step_identity") != json.loads(
            json.dumps(self._checkpoint_identity(), sort_keys=True)
        ):
            raise ValueError(
                "Small-strain material checkpoint scientific identity differs."
            )
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
        self.accepted_load_factor = float(payload["coordinate"])
        self.accepted_increments[:] = [
            SmallStrainMaterialIncrementInfo.from_dict(item)
            for item in payload["accepted_increments"]
        ]
        self.attempted_increments[:] = [
            SmallStrainMaterialIncrementInfo.from_dict(item)
            for item in payload["attempted_increments"]
        ]
        self.execution_events[:] = [
            SolveEvent.from_dict(item) for item in payload.get("execution_events", ())
        ]
        self.next_increment_size = payload.get("next_increment_size")
        self._apply_loading(self.accepted_load_factor)
        self._accepted_strain = self._evaluate_strain()
        self._update_material(
            time=self.accepted_load_factor,
            time_increment=np.finfo(float).eps,
        )
        self.last_solve_info = SmallStrainMaterialPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )

    def _checkpoint_identity(self):
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "material": _material_summary(self.material),
            "amplitude": self.amplitude.summary(),
            "incrementation": self.incrementation.summary(),
            "solution": function_portable_identity(self.solution),
            "quadrature": self.state.summary()["transaction"],
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
            "last_solve": None
            if self.last_solve_info is None
            else self.last_solve_info.as_dict(),
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
    name="small_strain_material",
) -> SmallStrainMaterialStep:
    if not isinstance(material, SmallStrainUserMaterial):
        raise TypeError(
            "small_strain_material_step requires a SmallStrainUserMaterial."
        )
    if not bool(getattr(material, "rate_independent", False)):
        raise NotImplementedError(
            "The first global small-strain material Step requires an explicit "
            "rate_independent=True declaration. Time/temperature-dependent "
            "providers need a physical-time procedure and restart contract."
        )
    domain = displacement.value.function_space.mesh
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
