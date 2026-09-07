"""Global small-strain generalized-Maxwell equilibrium.

The constitutive law owns the exact Prony-branch update.  This module owns the
finite-element procedure: quadrature state, equilibrium, accepted time,
rollback, progress, result fields, and constitutive energy evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from .. import amplitudes
from .. import procedures
from .. import steps as step_controls
from ..constitutive import elasticity
from ..constitutive.quadrature import (
    MaterialQuadratureState,
    QuadratureField,
    QuadratureMaterialMap,
)
from ..constitutive.viscoelasticity import IsotropicGeneralizedMaxwell
from ..diagnostics import (
    SolveEventRecorder,
    StandardRunReporter,
    comm_of,
    compose_reporters,
)
from ..solvers import NewtonSolverOptions, SolveEvent, newton, solve_matrix_system


@dataclass(frozen=True)
class ViscoelasticIncrementInfo:
    """Accepted equilibrium evidence for one physical-time increment."""

    increment: int
    start_time: float
    end_time: float
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    attempt: int = 1
    time_error_estimate: float | None = None
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "increment": self.increment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "time_increment": self.end_time - self.start_time,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": self.initial_residual_norm,
            "residual_norm": self.residual_norm,
            "attempt": self.attempt,
            "time_error_estimate": self.time_error_estimate,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, record) -> "ViscoelasticIncrementInfo":
        return cls(
            increment=int(record["increment"]),
            start_time=float(record["start_time"]),
            end_time=float(record["end_time"]),
            converged=bool(record["converged"]),
            iterations=int(record["iterations"]),
            initial_residual_norm=float(record["initial_residual_norm"]),
            residual_norm=float(record["residual_norm"]),
            attempt=int(record.get("attempt", 1)),
            time_error_estimate=(
                None
                if record.get("time_error_estimate") is None
                else float(record["time_error_estimate"])
            ),
            rejection_reason=record.get("rejection_reason"),
        )


@dataclass(frozen=True)
class ViscoelasticPathInfo:
    """Resolved fixed physical-time path."""

    increments: tuple[ViscoelasticIncrementInfo, ...]
    duration: float
    steps: int | None
    attempts: tuple[ViscoelasticIncrementInfo, ...] = ()
    incrementation: object | None = None

    @property
    def converged(self) -> bool:
        return bool(self.increments) and all(item.converged for item in self.increments)

    @property
    def completed_step(self) -> bool:
        return self.converged and np.isclose(
            self.increments[-1].end_time,
            self.duration,
            rtol=0.0,
            atol=1.0e-12 * max(1.0, self.duration),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "generalized_maxwell_time_path",
            "converged": self.converged,
            "completed_step": self.completed_step,
            "duration": self.duration,
            "steps": self.steps,
            "accepted_increments": len(self.increments),
            "attempted_increments": len(self.attempts),
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "increments": [item.as_dict() for item in self.increments],
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass(frozen=True)
class ViscoelasticEnergyFrame:
    """Exact constitutive work--storage--dissipation ledger."""

    time: float
    load_amplitude: float
    stored_energy: float
    viscous_dissipation: float
    material_work: float
    constitutive_energy_residual: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass
class ViscoelasticQuadratureState:
    """Typed committed/trial Maxwell state and current response fields."""

    state: MaterialQuadratureState
    stress: QuadratureField
    tangent: QuadratureField
    stored_energy: QuadratureField
    work_increment: QuadratureField

    @classmethod
    def create(
        cls,
        domain,
        material: IsotropicGeneralizedMaxwell | QuadratureMaterialMap,
        *,
        degree: int = 2,
        scheme: str = "default",
    ) -> "ViscoelasticQuadratureState":
        schema = (
            material.require_common_state_schema()
            if isinstance(material, QuadratureMaterialMap)
            else material.state_schema
        )
        common = {"degree": int(degree), "scheme": str(scheme)}
        return cls(
            state=MaterialQuadratureState.create(domain, schema, **common),
            stress=QuadratureField.create(
                domain, name="S", value_shape=(3, 3), **common
            ),
            tangent=QuadratureField.create(
                domain, name="DDSDDE", value_shape=(3, 3, 3, 3), **common
            ),
            stored_energy=QuadratureField.create(domain, name="SENER", **common),
            work_increment=QuadratureField.create(domain, name="DWORK", **common),
        )

    @property
    def domain(self):
        return self.state.domain

    @property
    def degree(self) -> int:
        return self.state.degree

    @property
    def scheme(self) -> str:
        return self.state.scheme

    @property
    def measure(self):
        return self.state.measure

    @property
    def accepted_strain(self) -> QuadratureField:
        return self.state.committed["strain"]

    @property
    def dissipated_energy(self) -> QuadratureField:
        return self.state.committed["dissipated_energy"]

    def compile_strain(self, expression):
        return self.state.compile_expression(expression, value_shape=(3, 3))

    def evaluate_strain(self, expression) -> np.ndarray:
        return self.state.evaluate_expression(expression, value_shape=(3, 3))

    def evaluate_scalar(self, expression) -> np.ndarray:
        return self.state.evaluate_expression(expression, value_shape=()).reshape(-1)

    def update(
        self,
        strains,
        material: IsotropicGeneralizedMaxwell | QuadratureMaterialMap,
        *,
        dt: float,
        temperature_values=None,
    ) -> dict[str, float | int | None]:
        """Refresh trial state and response from the committed boundary."""

        selected_strains = np.asarray(strains, dtype=float).reshape((-1, 3, 3))
        committed = self.state.committed_state_vectors()
        if len(selected_strains) != len(committed):
            raise ValueError("Strain and Maxwell quadrature layouts do not match.")
        temperatures = None
        if temperature_values is not None:
            temperatures = np.asarray(temperature_values, dtype=float).reshape(-1)
            if len(temperatures) != len(committed):
                raise ValueError("Temperature and Maxwell quadrature layouts do not match.")
            if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
                raise ValueError("Viscoelastic temperatures must be positive kelvin values.")

        stresses = np.empty_like(self.stress.values)
        tangents = np.empty_like(self.tangent.values)
        states = np.empty_like(committed)
        stored = np.empty(len(committed), dtype=float)
        work = np.empty(len(committed), dtype=float)
        points_per_cell = len(self.stress.points)
        self.state.begin()
        local_problem = None
        try:
            for index, strain in enumerate(selected_strains):
                selected_material = (
                    material.material_for_point(index, points_per_cell=points_per_cell)
                    if isinstance(material, QuadratureMaterialMap)
                    else material
                )
                try:
                    response = selected_material.update(
                        committed[index],
                        strain,
                        float(dt),
                        temperature=(
                            None if temperatures is None else float(temperatures[index])
                        ),
                    )
                except Exception as exc:
                    local_problem = (
                        "generalized-Maxwell update failed at local quadrature "
                        f"point {index}: {type(exc).__name__}: {exc}"
                    )
                    break
                stresses[index] = response.stress
                tangents[index] = response.consistent_tangent
                states[index] = response.state_new
                stored[index] = response.stored_energy_density
                work[index] = response.mechanical_work_increment
            problems = self.domain.comm.allgather(local_problem)
            if any(problem is not None for problem in problems):
                rank = next(
                    rank for rank, problem in enumerate(problems) if problem is not None
                )
                raise RuntimeError(f"Rank {rank}: {problems[rank]}")
            self.state.assign_trial_state_vectors(states)
            self.stress.assign(stresses)
            self.tangent.assign(tangents)
            self.stored_energy.assign(stored)
            self.work_increment.assign(work)
        except Exception:
            self.state.rollback()
            raise

        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        owned_points = int(cell_map.size_local) * points_per_cell
        minimum_temperature = None
        maximum_temperature = None
        if temperatures is not None:
            local = temperatures[:owned_points]
            minimum_temperature = float(
                self.domain.comm.allreduce(np.min(local, initial=np.inf), op=MPI.MIN)
            )
            maximum_temperature = float(
                self.domain.comm.allreduce(np.max(local, initial=-np.inf), op=MPI.MAX)
            )
        return {
            "points": int(self.domain.comm.allreduce(owned_points, op=MPI.SUM)),
            "minimum_temperature": minimum_temperature,
            "maximum_temperature": maximum_temperature,
        }

    def commit(self) -> None:
        self.state.commit()

    def rollback(self) -> None:
        self.state.rollback()

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.snapshot(),
            "stress": self.stress.values.copy(),
            "tangent": self.tangent.values.copy(),
            "stored_energy": self.stored_energy.values.copy(),
            "work_increment": self.work_increment.values.copy(),
        }

    def restore(self, snapshot) -> None:
        self.state.restore(snapshot["state"])
        self.stress.assign(snapshot["stress"])
        self.tangent.assign(snapshot["tangent"])
        self.stored_energy.assign(snapshot["stored_energy"])
        self.work_increment.assign(snapshot["work_increment"])

    def refresh_response(
        self,
        material: IsotropicGeneralizedMaxwell | QuadratureMaterialMap,
        *,
        tangent_dt: float | None = None,
        temperature_values=None,
    ) -> None:
        """Rebuild accepted response fields without advancing material time."""

        committed = self.state.committed_state_vectors()
        temperatures = None
        if temperature_values is not None:
            temperatures = np.asarray(temperature_values, dtype=float).reshape(-1)
            if len(temperatures) != len(committed):
                raise ValueError(
                    "Temperature and Maxwell quadrature layouts do not match."
                )
        stresses = np.empty_like(self.stress.values)
        tangents = np.empty_like(self.tangent.values)
        stored = np.empty(len(committed), dtype=float)
        points_per_cell = len(self.stress.points)
        for index, state in enumerate(committed):
            selected_material = (
                material.material_for_point(index, points_per_cell=points_per_cell)
                if isinstance(material, QuadratureMaterialMap)
                else material
            )
            stress, tangent, energy = selected_material.accepted_response(
                state,
                tangent_dt=tangent_dt,
                temperature=(
                    None if temperatures is None else float(temperatures[index])
                ),
            )
            stresses[index] = stress
            tangents[index] = tangent
            stored[index] = energy
        self.stress.assign(stresses)
        self.tangent.assign(tangents)
        self.stored_energy.assign(stored)
        self.work_increment.assign(np.zeros(len(committed), dtype=float))
        self.rollback()

    def save(self, path, *, material=None) -> Path:
        """Collectively save committed Maxwell state by physical cell identity."""

        return self.state.save(path, material=material)

    def load(self, path, *, material=None) -> None:
        """Collectively restore committed Maxwell state by physical cell identity."""

        self.state.load(path, material=material)

    def equivalent_stress(self) -> QuadratureField:
        stress = self.stress.values
        trace = np.trace(stress, axis1=-2, axis2=-1)
        deviator = stress - trace[:, None, None] * np.eye(3) / 3.0
        output = QuadratureField.create(
            self.domain, name="MISES", degree=self.degree, scheme=self.scheme
        )
        output.assign(np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1))))
        return output

    def summary(self) -> dict[str, object]:
        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        points_per_cell = len(self.stress.points)
        return {
            "kind": "generalized_maxwell_quadrature_state",
            "degree": self.degree,
            "scheme": self.scheme,
            "points_owned": int(cell_map.size_local) * points_per_cell,
            "points_global": int(cell_map.size_global) * points_per_cell,
            "portable_cell_identity": "dolfinx_original_cell_index",
            "transaction": self.state.transaction.summary(),
        }


@dataclass
class QuasistaticViscoelasticStep:
    """Incremental equilibrium for a small-strain generalized-Maxwell solid."""

    name: str
    solution: object
    material: IsotropicGeneralizedMaxwell | QuadratureMaterialMap
    state: ViscoelasticQuadratureState
    residual_form: object
    tangent_form: object
    external_force: object | None
    load_factor: object
    amplitude: amplitudes.Amplitude
    bcs: tuple[object, ...]
    prescribed_values: tuple[tuple[object, np.ndarray, object], ...]
    time_dependent_constraints: tuple[object, ...]
    duration: float
    steps: int | None
    solver_options: NewtonSolverOptions
    time_points: tuple[float, ...] | None = None
    incrementation: object | None = None
    time_error_tolerance: float | None = None
    temperature: object | None = None
    time_unit: str | None = None
    study: object | None = None
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    step_number: int = 1
    procedure: object = field(default_factory=procedures.quasistatic_viscoelasticity)
    accepted_time: float = field(default=0.0, init=False)
    accepted_increments: list[ViscoelasticIncrementInfo] = field(
        default_factory=list, init=False
    )
    attempted_increments: list[ViscoelasticIncrementInfo] = field(
        default_factory=list, init=False
    )
    execution_events: list[object] = field(default_factory=list, init=False)
    energy_history: list[ViscoelasticEnergyFrame] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    last_solve_info: ViscoelasticPathInfo | None = field(default=None, init=False)
    next_increment_size: float | None = field(default=None, init=False)
    _strain_evaluator: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.duration = float(self.duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("Viscoelastic duration must be finite and positive.")
        automatic = isinstance(
            self.incrementation, step_controls.AutomaticIncrementation
        )
        if self.incrementation is not None and not automatic:
            raise TypeError(
                "Viscoelastic incrementation currently accepts steps.automatic(...); "
                "use steps= or time_points= for a prescribed path."
            )
        if automatic and (self.steps is not None or self.time_points is not None):
            raise ValueError(
                "Specify adaptive incrementation or a prescribed steps/time_points "
                "path, not both."
            )
        if automatic:
            grid = None
        elif self.time_points is None:
            if self.steps is None or int(self.steps) <= 0:
                raise ValueError(
                    "Viscoelastic steps must be positive when time_points is omitted."
                )
            self.steps = int(self.steps)
            grid = np.linspace(0.0, self.duration, self.steps + 1)
        else:
            grid = np.asarray(self.time_points, dtype=float)
            if (
                grid.ndim != 1
                or len(grid) < 2
                or not np.all(np.isfinite(grid))
                or not np.isclose(grid[0], 0.0)
                or not np.isclose(grid[-1], self.duration)
                or np.any(np.diff(grid) <= 0.0)
            ):
                raise ValueError(
                    "Viscoelastic time_points must be a finite, strictly increasing "
                    "grid from zero through duration."
                )
            declared_steps = len(grid) - 1
            if self.steps is not None and int(self.steps) != declared_steps:
                raise ValueError(
                    "steps must equal len(time_points) - 1 when both are supplied."
                )
            self.steps = declared_steps
        self.time_points = (
            None if grid is None else tuple(float(value) for value in grid)
        )
        if self.time_error_tolerance is not None:
            self.time_error_tolerance = float(self.time_error_tolerance)
            if (
                not np.isfinite(self.time_error_tolerance)
                or self.time_error_tolerance <= 0.0
            ):
                raise ValueError("time_error_tolerance must be finite and positive.")
            if not automatic:
                raise ValueError(
                    "time_error_tolerance requires adaptive incrementation."
                )
        if not np.isclose(self.amplitude(0.0), 0.0):
            raise ValueError(
                "The first global viscoelastic provider requires a load amplitude "
                "that starts at zero. Represent an initial step by a short explicit ramp."
            )
        materials = (
            tuple(self.material.materials.values())
            if isinstance(self.material, QuadratureMaterialMap)
            else (self.material,)
        )
        shifted = tuple(item.shift is not None for item in materials)
        if any(shifted) and self.temperature is None:
            raise ValueError("A shifted viscoelastic material requires temperature=.")
        if not any(shifted) and self.temperature is not None:
            raise ValueError("temperature= requires a declared viscoelastic shift law.")
        if len(set(shifted)) > 1:
            raise ValueError(
                "One regional viscoelastic Step cannot mix shifted and unshifted materials."
            )
        self._strain_evaluator = self.state.compile_strain(
            elasticity.strain(self.solution, study=self.study)
        )

    def _snapshot_accepted_boundary(self) -> dict[str, object]:
        """Capture every mutable channel before accepted-state finalization."""

        return {
            "solution": self.solution.x.array.copy(),
            "state": self.state.snapshot(),
            "accepted_time": float(self.accepted_time),
            "accepted_increments": list(self.accepted_increments),
            "attempted_increments": list(self.attempted_increments),
            "execution_events": list(self.execution_events),
            "energy_history": list(self.energy_history),
            "checkpoints": list(self.checkpoints),
            "last_solve_info": self.last_solve_info,
            "next_increment_size": self.next_increment_size,
        }

    def _restore_accepted_boundary(self, boundary: dict[str, object]) -> None:
        """Undo a failed post-solve finalization as one scientific transaction."""

        self.solution.x.array[:] = boundary["solution"]
        self.solution.x.scatter_forward()
        self.state.restore(boundary["state"])
        self.accepted_time = float(boundary["accepted_time"])
        self.accepted_increments[:] = boundary["accepted_increments"]
        self.attempted_increments[:] = boundary["attempted_increments"]
        self.execution_events[:] = boundary["execution_events"]
        self.energy_history[:] = boundary["energy_history"]
        self.checkpoints[:] = boundary["checkpoints"]
        self.last_solve_info = boundary["last_solve_info"]
        self.next_increment_size = boundary["next_increment_size"]
        self._apply_loading(self.accepted_time)

    @property
    def time_increment(self) -> float | None:
        if self.time_points is None:
            return None
        increments = np.diff(self.time_points)
        if np.allclose(increments, increments[0], rtol=1.0e-12, atol=0.0):
            return float(increments[0])
        return None

    def solve(self, *, until: float | None = None):
        """Advance on the declared or automatically resolved physical-time path."""

        if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
            return self._solve_adaptive(until=until)

        selected_until = self.duration if until is None else float(until)
        grid = np.asarray(self.time_points, dtype=float)
        candidates = np.flatnonzero(grid > self.accepted_time + 1.0e-12)
        target_indices = [
            index for index in candidates if grid[index] <= selected_until + 1e-12
        ]
        if not target_indices or not np.isclose(grid[target_indices[-1]], selected_until):
            raise ValueError("until must be a future point on the fixed time grid.")

        reporter = self._reporter()
        self._emit(
            reporter,
            SolveEvent(
                "step_started",
                self.name,
                step_number=self.step_number,
                incrementation="exact generalized-Maxwell / declared physical time",
                time=self.accepted_time,
            ),
        )
        for grid_index in target_indices:
            boundary = self._snapshot_accepted_boundary()
            start_time = float(grid[grid_index - 1])
            end_time = float(grid[grid_index])
            increment = len(self.accepted_increments) + 1
            self._emit(
                reporter,
                SolveEvent(
                    "increment_started",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=1,
                    start_factor=start_time / self.duration,
                    target_factor=end_time / self.duration,
                    time=end_time,
                ),
            )
            displacement_snapshot = self.solution.x.array.copy()
            state_snapshot = self.state.snapshot()
            self._apply_loading(end_time)
            info = self._solve_increment(
                increment=increment,
                attempt=1,
                start_time=start_time,
                end_time=end_time,
                reporter=reporter,
            )
            self.attempted_increments.append(info)
            if not info.converged:
                self.solution.x.array[:] = displacement_snapshot
                self.solution.x.scatter_forward()
                self.state.restore(state_snapshot)
                self._apply_loading(start_time)
                self.last_solve_info = self._path_info()
                message = (
                    f"{self.name}: equilibrium failed at t={end_time:g}; "
                    f"residual={info.residual_norm:.6g}."
                )
                self._emit(
                    reporter,
                    SolveEvent(
                        "step_failed",
                        self.name,
                        step_number=self.step_number,
                        increment=increment,
                        start_factor=start_time / self.duration,
                        target_factor=end_time / self.duration,
                        residual_norm=info.residual_norm,
                        message=message,
                        time=end_time,
                    ),
                )
                if self.solver_options.error_if_not_converged:
                    raise RuntimeError(message)
                return self.solution
            self.state.commit()
            self.accepted_time = end_time
            self.accepted_increments.append(info)
            self._record_energy(end_time)
            self._emit(
                reporter,
                SolveEvent(
                    "increment_converged",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=1,
                    start_factor=start_time / self.duration,
                    target_factor=end_time / self.duration,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    time=end_time,
                ),
            )
            self._finalize_scheduled_checkpoint(boundary)
        self.last_solve_info = self._path_info()
        self._emit(
            reporter,
            SolveEvent(
                "step_completed" if self.last_solve_info.completed_step else "step_paused",
                self.name,
                step_number=self.step_number,
                increment=len(self.accepted_increments),
                time=self.accepted_time,
            ),
        )
        return self.solution

    def _solve_adaptive(self, *, until: float | None = None):
        """Advance with atomic cutback and optional step-doubling control."""

        selected_until = self.duration if until is None else float(until)
        if not self.accepted_time < selected_until <= self.duration:
            raise ValueError(
                "until must be greater than accepted_time and no larger than duration."
            )
        control = self.incrementation
        reporter = self._reporter()
        proposed_size = (
            control.initial
            if self.next_increment_size is None
            else self.next_increment_size
        )
        consecutive_cutbacks = 0
        self._emit(
            reporter,
            SolveEvent(
                "step_started",
                self.name,
                step_number=self.step_number,
                incrementation=(
                    "exact generalized-Maxwell / automatic physical time"
                ),
                time=self.accepted_time,
            ),
        )
        while self.accepted_time < selected_until - self._time_tolerance():
            if len(self.accepted_increments) >= control.max_increments:
                raise RuntimeError(
                    "Viscoelastic analysis reached max_increments before the step end."
                )
            start_time = self.accepted_time
            start_factor = start_time / self.duration
            target_factor = min(
                selected_until / self.duration,
                start_factor + proposed_size,
            )
            end_time = target_factor * self.duration
            increment = len(self.accepted_increments) + 1
            attempt = consecutive_cutbacks + 1
            boundary = self._snapshot_accepted_boundary()
            self._emit(
                reporter,
                SolveEvent(
                    "increment_started",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=start_factor,
                    target_factor=target_factor,
                    time=end_time,
                ),
            )
            info, work_increment = self._attempt_adaptive_increment(
                increment=increment,
                attempt=attempt,
                start_time=start_time,
                end_time=end_time,
            )
            self.attempted_increments.append(info)
            if info.converged:
                self.state.commit()
                self.accepted_time = end_time
                self.accepted_increments.append(info)
                self._record_energy(end_time, work_increment=work_increment)
                accepted_size = target_factor - start_factor
                proposed_size = control.after_convergence(
                    accepted_size,
                    info.iterations,
                )
                if (
                    self.time_error_tolerance is not None
                    and info.time_error_estimate is not None
                    and info.time_error_estimate
                    < 0.125 * self.time_error_tolerance
                    and info.iterations < control.slow_iterations
                ):
                    proposed_size = min(
                        control.maximum,
                        max(proposed_size, accepted_size * control.growth_factor),
                    )
                self.next_increment_size = proposed_size
                consecutive_cutbacks = 0
                self._emit(
                    reporter,
                    SolveEvent(
                        "increment_converged",
                        self.name,
                        step_number=self.step_number,
                        increment=increment,
                        attempt=attempt,
                        start_factor=start_factor,
                        target_factor=target_factor,
                        iteration=info.iterations,
                        residual_norm=info.residual_norm,
                        time=end_time,
                        message=(
                            None
                            if info.time_error_estimate is None
                            else f"time_error={info.time_error_estimate:.6g}"
                        ),
                    ),
                )
                self._finalize_scheduled_checkpoint(boundary)
                continue

            consecutive_cutbacks += 1
            proposed_size = control.after_failure(target_factor - start_factor)
            self.next_increment_size = proposed_size
            if (
                consecutive_cutbacks > control.max_cutbacks
                or proposed_size < control.minimum
            ):
                self.last_solve_info = self._path_info()
                reason = info.rejection_reason or "global equilibrium did not converge"
                message = (
                    f"{self.name}: automatic time incrementation failed near "
                    f"t={end_time:g}: {reason}."
                )
                self._emit(
                    reporter,
                    SolveEvent(
                        "step_failed",
                        self.name,
                        step_number=self.step_number,
                        increment=increment,
                        attempt=attempt,
                        start_factor=start_factor,
                        target_factor=target_factor,
                        residual_norm=info.residual_norm,
                        message=message,
                        time=end_time,
                    ),
                )
                if self.solver_options.error_if_not_converged:
                    raise RuntimeError(message)
                return self.solution
            self._emit(
                reporter,
                SolveEvent(
                    "increment_cutback",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=start_factor,
                    target_factor=target_factor,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    next_increment=proposed_size,
                    message=info.rejection_reason,
                    time=end_time,
                ),
            )

        self.last_solve_info = self._path_info()
        self._emit(
            reporter,
            SolveEvent(
                "step_completed"
                if self.accepted_time >= self.duration - self._time_tolerance()
                else "step_paused",
                self.name,
                step_number=self.step_number,
                increment=len(self.accepted_increments),
                attempt=len(self.attempted_increments),
                time=self.accepted_time,
            ),
        )
        return self.solution

    def _attempt_adaptive_increment(
        self, *, increment: int, attempt: int, start_time: float, end_time: float
    ) -> tuple[ViscoelasticIncrementInfo, float]:
        """Try one interval and leave only an accepted fine trial in memory."""

        displacement_start = self.solution.x.array.copy()
        state_start = self.state.snapshot()
        self._apply_loading(end_time)
        coarse = self._solve_increment(
            increment=increment,
            attempt=attempt,
            start_time=start_time,
            end_time=end_time,
            reporter=None,
        )
        if not coarse.converged or self.time_error_tolerance is None:
            if not coarse.converged:
                self._restore_attempt(displacement_start, state_start, start_time)
                coarse = replace(
                    coarse,
                    rejection_reason="global equilibrium did not converge",
                )
                return coarse, 0.0
            return coarse, self._integral(self.state.work_increment)

        displacement_coarse = self.solution.x.array.copy()
        stress_coarse = self.state.stress.values.copy()
        self._restore_attempt(displacement_start, state_start, start_time)
        midpoint = 0.5 * (start_time + end_time)
        self._apply_loading(midpoint)
        first = self._solve_increment(
            increment=increment,
            attempt=attempt,
            start_time=start_time,
            end_time=midpoint,
            reporter=None,
        )
        if not first.converged:
            self._restore_attempt(displacement_start, state_start, start_time)
            return replace(
                first,
                start_time=start_time,
                end_time=end_time,
                rejection_reason="first error-control half-step did not converge",
            ), 0.0
        first_work = self._integral(self.state.work_increment)
        self.state.commit()
        self._apply_loading(end_time)
        second = self._solve_increment(
            increment=increment,
            attempt=attempt,
            start_time=midpoint,
            end_time=end_time,
            reporter=None,
        )
        if not second.converged:
            self._restore_attempt(displacement_start, state_start, start_time)
            return replace(
                second,
                start_time=start_time,
                rejection_reason="second error-control half-step did not converge",
            ), 0.0
        second_work = self._integral(self.state.work_increment)
        error = max(
            self._relative_endpoint_error(
                displacement_coarse,
                self.solution.x.array,
                owned_count=self._owned_solution_values(),
            ),
            self._relative_endpoint_error(
                stress_coarse,
                self.state.stress.values,
                owned_count=self._owned_quadrature_points(),
            ),
        )
        info = replace(
            second,
            start_time=start_time,
            iterations=max(first.iterations, second.iterations),
            initial_residual_norm=max(
                first.initial_residual_norm,
                second.initial_residual_norm,
            ),
            time_error_estimate=error,
        )
        if error > self.time_error_tolerance:
            self._restore_attempt(displacement_start, state_start, start_time)
            return replace(
                info,
                converged=False,
                rejection_reason=(
                    f"step-doubling estimate {error:.6g} exceeds "
                    f"{self.time_error_tolerance:.6g}"
                ),
            ), 0.0
        return info, first_work + second_work

    def _restore_attempt(self, displacement, state, time: float) -> None:
        self.solution.x.array[:] = displacement
        self.solution.x.scatter_forward()
        self.state.restore(state)
        self._apply_loading(time)

    def _owned_solution_values(self) -> int:
        dofmap = self.solution.function_space.dofmap
        return int(dofmap.index_map.size_local) * int(dofmap.index_map_bs)

    def _owned_quadrature_points(self) -> int:
        cell_map = self.state.domain.topology.index_map(self.state.domain.topology.dim)
        return int(cell_map.size_local) * len(self.state.stress.points)

    def _relative_endpoint_error(self, coarse, fine, *, owned_count: int) -> float:
        coarse_values = np.asarray(coarse)[:owned_count].reshape(-1)
        fine_values = np.asarray(fine)[:owned_count].reshape(-1)
        delta = fine_values - coarse_values
        local_difference = float(np.dot(delta, delta))
        local_coarse = float(np.dot(coarse_values, coarse_values))
        local_fine = float(np.dot(fine_values, fine_values))
        difference = self.state.domain.comm.allreduce(local_difference, op=MPI.SUM)
        scale = max(
            self.state.domain.comm.allreduce(local_coarse, op=MPI.SUM),
            self.state.domain.comm.allreduce(local_fine, op=MPI.SUM),
            np.finfo(float).tiny,
        )
        return float(np.sqrt(difference / scale))

    def _time_tolerance(self) -> float:
        return 1.0e-12 * max(1.0, self.duration)

    def _path_info(self) -> ViscoelasticPathInfo:
        return ViscoelasticPathInfo(
            tuple(self.accepted_increments),
            self.duration,
            self.steps,
            attempts=tuple(self.attempted_increments),
            incrementation=self.incrementation,
        )

    def _solve_increment(
        self, *, increment, attempt, start_time, end_time, reporter
    ):
        initial_norm = None
        norm = float("inf")
        converged = False
        iteration = 0
        for iteration in range(self.solver_options.maximum_iterations + 1):
            self.state.update(
                self.state.evaluate_strain(self._strain_evaluator),
                self.material,
                dt=end_time - start_time,
                temperature_values=self._temperature_values(),
            )
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
                break
            direction = correction.array_r.copy()
            correction.destroy()
            self.solution.x.array[: len(direction)] += direction
            self.solution.x.scatter_forward()
            self._emit(
                reporter,
                SolveEvent(
                    "iteration",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=start_time / self.duration,
                    target_factor=end_time / self.duration,
                    iteration=iteration + 1,
                    residual_norm=norm,
                    step_length=1.0,
                    time=end_time,
                ),
            )
        return ViscoelasticIncrementInfo(
            increment=increment,
            start_time=start_time,
            end_time=end_time,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=float(initial_norm or 0.0),
            residual_norm=float(norm),
            attempt=attempt,
        )

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

    def _apply_loading(self, time: float) -> None:
        if self.temperature is not None and hasattr(self.temperature, "apply"):
            self.temperature.apply(time)
        factor = self.amplitude(time)
        self.load_factor.value = PETSc.ScalarType(factor)
        for constant, target, _bc in self.prescribed_values:
            selected = factor * target
            constant.value = (
                PETSc.ScalarType(selected.item())
                if selected.ndim == 0 or selected.size == 1
                else np.asarray(selected, dtype=PETSc.ScalarType)
            )
        for constraint in self.time_dependent_constraints:
            constraint.update(time)

    def _temperature_values(self):
        if self.temperature is None:
            return None
        selected = getattr(self.temperature, "value", self.temperature)
        if hasattr(selected, "function_space"):
            return self.state.evaluate_scalar(selected)
        value = np.asarray(selected, dtype=float)
        if value.size != 1:
            raise ValueError("Viscoelastic temperature must be scalar or a scalar field.")
        return np.full(len(self.state.stress.values), float(value.reshape(-1)[0]))

    def _temperature_summary(self) -> dict[str, object] | None:
        values = self._temperature_values()
        if values is None:
            return None
        canonical = np.ascontiguousarray(values, dtype=np.float64)
        selected = getattr(self.temperature, "value", self.temperature)
        summary = {
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
            summary["history"] = self.temperature.scientific_identity()
            summary["active_time"] = getattr(self.temperature, "active_time", None)
        return summary

    def _integral(self, field_value: QuadratureField) -> float:
        local = fem.assemble_scalar(fem.form(field_value.function * self.state.measure))
        return float(self.state.domain.comm.allreduce(local, op=MPI.SUM))

    def _record_energy(
        self, time: float, *, work_increment: float | None = None
    ) -> None:
        stored = self._integral(self.state.stored_energy)
        dissipation = self._integral(self.state.dissipated_energy)
        increment = (
            self._integral(self.state.work_increment)
            if work_increment is None
            else float(work_increment)
        )
        material_work = increment + (
            0.0 if not self.energy_history else self.energy_history[-1].material_work
        )
        self.energy_history.append(
            ViscoelasticEnergyFrame(
                time=time,
                load_amplitude=float(self.amplitude(time)),
                stored_energy=stored,
                viscous_dissipation=dissipation,
                material_work=material_work,
                constitutive_energy_residual=material_work - stored - dissipation,
            )
        )

    def reaction_field(self, *, name: str = "RF"):
        residual = fem_petsc.assemble_vector(self.residual_form)
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        reaction = fem.Function(self.solution.function_space, name=name)
        reaction.x.array[: len(residual.array_r)] = residual.array_r
        reaction.x.scatter_forward()
        residual.destroy()
        return reaction

    def _finalize_scheduled_checkpoint(self, boundary) -> None:
        """Publish a due checkpoint or restore the previous accepted boundary."""

        policy = self.checkpoint_policy
        if policy is None:
            return
        increment = len(self.accepted_increments)
        due = increment % int(policy.every) == 0
        due = due or (
            bool(policy.final)
            and self.accepted_time >= self.duration - self._time_tolerance()
        )
        if not due:
            return
        problem = None
        try:
            self.save_checkpoint(
                policy.path(step_name=self.name, increment=increment),
                portable=(
                    True
                    if self.state.domain.comm.size > 1
                    else bool(policy.portable)
                ),
            )
            record = self.checkpoints[-1]
            self.checkpoints[-1] = replace(
                record,
                metadata={**record.metadata, "role": "scheduled_checkpoint"},
            )
            if not self.checkpoints[-1].portable:
                self.checkpoints[-1].write_manifest()
            self._prune_scheduled_checkpoints()
        except BaseException as exc:
            problem = f"{type(exc).__name__}: {exc}"
        problems = self.state.domain.comm.allgather(problem)
        if any(item is not None for item in problems):
            self._restore_accepted_boundary(boundary)
            rank = next(
                index for index, item in enumerate(problems) if item is not None
            )
            raise RuntimeError(
                "Viscoelastic accepted-state checkpoint failed collectively; "
                f"rank {rank}: {problems[rank]}"
            )

    def _prune_scheduled_checkpoints(self) -> None:
        policy = self.checkpoint_policy
        keep_last = None if policy is None else policy.keep_last
        scheduled = [
            record
            for record in self.checkpoints
            if record.metadata.get("role") == "scheduled_checkpoint"
        ]
        if keep_last is None or len(scheduled) <= int(keep_last):
            return
        obsolete = scheduled[: -int(keep_last)]
        for record in obsolete:
            self._remove_checkpoint_record(record)
        removed = {id(record) for record in obsolete}
        self.checkpoints[:] = [
            record for record in self.checkpoints if id(record) not in removed
        ]

    def _remove_checkpoint_record(self, record) -> None:
        """Collectively remove only files declared by one viscoelastic checkpoint."""

        from ..checkpointing import (
            remove_serial_checkpoint,
            remove_stateful_checkpoint,
        )

        comm = self.state.domain.comm
        if record.portable:
            remove_stateful_checkpoint(
                record.path,
                comm=comm,
                expected_schema="agentfem.generalized-maxwell-step-checkpoint.v2",
            )
        else:
            remove_serial_checkpoint(
                record.path,
                comm=comm,
                expected_schema="agentfem.generalized-maxwell-step-checkpoint.v1",
            )

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        """Save an accepted boundary, portable across MPI rank counts on request."""

        comm = self.state.domain.comm
        selected_portable = comm.size != 1 if portable is None else bool(portable)
        if selected_portable:
            return self._save_portable_checkpoint(path)
        if comm.size != 1:
            raise ValueError("Distributed viscoelastic checkpoints must use portable=True.")
        selected = Path(path)
        if selected.suffix != ".npz":
            selected = selected.with_suffix(".npz")
        selected.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.state.state.snapshot()
        from ..checkpointing import atomic_savez

        atomic_savez(
            selected,
            schema="agentfem.generalized-maxwell-step-checkpoint.v1",
            step_identity=json.dumps(self._checkpoint_identity(), sort_keys=True),
            accepted_time=self.accepted_time,
            displacement=self.solution.x.array,
            state_names=json.dumps(tuple(snapshot)),
            **snapshot,
            stress=self.state.stress.values,
            tangent=self.state.tangent.values,
            stored_energy=self.state.stored_energy.values,
            work_increment=self.state.work_increment.values,
            increments=json.dumps([item.as_dict() for item in self.accepted_increments]),
            attempts=json.dumps([item.as_dict() for item in self.attempted_increments]),
            energy=json.dumps([item.as_dict() for item in self.energy_history]),
            next_increment_size=(
                np.nan
                if self.next_increment_size is None
                else self.next_increment_size
            ),
        )
        from ..results import CheckpointRecord

        record = CheckpointRecord(
            name=f"{self.name}_{self.accepted_time:g}",
            path=selected,
            schema="agentfem.generalized-maxwell-step-checkpoint.v1",
            step_name=self.name,
            coordinate_name="time",
            coordinate_value=self.accepted_time,
            portable=False,
            metadata={
                "reason": "serial nodal and generalized-Maxwell quadrature state",
                "state_variables": ("U", *self.state.state.transaction.names),
                "identity": self._checkpoint_identity(),
            },
        )
        record.write_manifest()
        self.checkpoints.append(record)
        return selected

    def load_checkpoint(self, path) -> None:
        selected = Path(path)
        comm = self.state.domain.comm
        manifest = (
            selected
            if selected.name.endswith(".checkpoint.json")
            else selected.with_suffix("").with_name(
                selected.with_suffix("").name + ".checkpoint.json"
            )
        )
        manifest_exists = comm.bcast(
            manifest.is_file() if comm.rank == 0 else None,
            root=0,
        )
        if manifest_exists:
            packet = None
            if comm.rank == 0:
                try:
                    packet = {
                        "payload": json.loads(manifest.read_text(encoding="utf-8")),
                        "error": None,
                    }
                except Exception as exc:
                    packet = {
                        "payload": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            packet = comm.bcast(packet, root=0)
            if packet["error"] is not None:
                raise RuntimeError(
                    "Viscoelastic checkpoint manifest read failed: "
                    f"{packet['error']}"
                )
            payload = packet["payload"]
            if payload.get("schema") == "agentfem.generalized-maxwell-step-checkpoint.v2":
                self._load_portable_checkpoint(manifest, payload)
                return
        if self.state.domain.comm.size != 1:
            raise ValueError(
                "This legacy viscoelastic checkpoint is partition-bound; use a v2 "
                "portable checkpoint for distributed restart."
            )
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != "agentfem.generalized-maxwell-step-checkpoint.v1":
                raise ValueError("Unsupported generalized-Maxwell checkpoint schema.")
            stored_identity = json.loads(str(data["step_identity"]))
            if stored_identity != json.loads(
                json.dumps(self._checkpoint_identity(), sort_keys=True)
            ):
                raise ValueError(
                    "Viscoelastic checkpoint material, time grid, amplitude, "
                    "temperature, or quadrature identity differs."
                )
            self.solution.x.array[:] = np.asarray(data["displacement"])
            self.solution.x.scatter_forward()
            names = tuple(json.loads(str(data["state_names"])))
            self.state.state.restore({name: data[name] for name in names})
            self.state.stress.assign(data["stress"])
            self.state.tangent.assign(data["tangent"])
            self.state.stored_energy.assign(data["stored_energy"])
            self.state.work_increment.assign(data["work_increment"])
            self.accepted_time = float(data["accepted_time"])
            self.accepted_increments[:] = [
                ViscoelasticIncrementInfo.from_dict(item)
                for item in json.loads(str(data["increments"]))
            ]
            self.attempted_increments[:] = [
                ViscoelasticIncrementInfo.from_dict(item)
                for item in json.loads(str(data["attempts"]))
            ] if "attempts" in data else list(self.accepted_increments)
            self.energy_history[:] = [
                ViscoelasticEnergyFrame(**item)
                for item in json.loads(str(data["energy"]))
            ]
            stored_next = (
                float(data["next_increment_size"])
                if "next_increment_size" in data
                else np.nan
            )
            self.next_increment_size = (
                None if np.isnan(stored_next) else stored_next
            )
        self._apply_loading(self.accepted_time)
        self.last_solve_info = self._path_info()

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
            "schema": "agentfem.generalized-maxwell-step-checkpoint.v2",
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
        comm = self.state.domain.comm
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
            raise RuntimeError(f"Viscoelastic checkpoint manifest write failed: {error}")
        comm.barrier()
        from ..results import CheckpointRecord

        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_{self.accepted_time:g}",
                path=manifest,
                schema="agentfem.generalized-maxwell-step-checkpoint.v2",
                step_name=self.name,
                coordinate_name="time",
                coordinate_value=self.accepted_time,
                portable=True,
                metadata={
                    "state_variables": (
                        "U",
                        *self.state.state.transaction.names,
                    ),
                    "portability": "physical nodal and quadrature identity",
                },
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
            raise ValueError("Portable viscoelastic checkpoint scientific identity differs.")

        displacement = self.solution.x.array.copy()
        state_snapshot = self.state.snapshot()
        coordinate = self.accepted_time
        accepted = list(self.accepted_increments)
        attempted = list(self.attempted_increments)
        events = list(self.execution_events)
        energy = list(self.energy_history)
        next_size = self.next_increment_size
        try:
            load_portable_state_bundle(
                manifest,
                state={"U": self.solution},
                record=payload["nodal_state"],
                identities=payload["nodal_identity"],
            )
            self.state.load(
                validate_checkpoint_record(
                    manifest.parent, payload["quadrature_state"]
                ),
                material=self.material,
            )
            self.accepted_time = float(payload["coordinate"])
            self.accepted_increments[:] = [
                ViscoelasticIncrementInfo.from_dict(item)
                for item in payload["accepted_increments"]
            ]
            self.attempted_increments[:] = [
                ViscoelasticIncrementInfo.from_dict(item)
                for item in payload["attempted_increments"]
            ]
            self.execution_events[:] = [
                SolveEvent.from_dict(item) for item in payload["execution_events"]
            ]
            self.energy_history[:] = [
                ViscoelasticEnergyFrame(**item) for item in payload["energy_history"]
            ]
            self.next_increment_size = payload.get("next_increment_size")
            self._apply_loading(self.accepted_time)
            self.state.refresh_response(
                self.material,
                tangent_dt=self._restart_tangent_dt(),
                temperature_values=self._temperature_values(),
            )
            self.last_solve_info = self._path_info()
        except Exception:
            self.solution.x.array[:] = displacement
            self.solution.x.scatter_forward()
            self.state.restore(state_snapshot)
            self.accepted_time = coordinate
            self.accepted_increments[:] = accepted
            self.attempted_increments[:] = attempted
            self.execution_events[:] = events
            self.energy_history[:] = energy
            self.next_increment_size = next_size
            self._apply_loading(self.accepted_time)
            self.last_solve_info = self._path_info()
            raise

    def _restart_tangent_dt(self) -> float | None:
        if self.accepted_time >= self.duration - self._time_tolerance():
            return None
        if self.next_increment_size is not None:
            return min(
                self.duration - self.accepted_time,
                self.duration * float(self.next_increment_size),
            )
        if self.time_points is not None:
            future = [
                time for time in self.time_points
                if time > self.accepted_time + self._time_tolerance()
            ]
            if future:
                return float(future[0] - self.accepted_time)
        return None

    def _portable_checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "duration": self.duration,
            "steps": self.steps,
            "time_points": self.time_points,
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "time_error_tolerance": self.time_error_tolerance,
            "amplitude": self.amplitude.summary(),
            "quadrature": self.state.summary()["transaction"],
            "temperature": self._portable_temperature_summary(),
            "solution": function_portable_identity(self.solution),
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
        return {
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "duration": self.duration,
            "steps": self.steps,
            "time_points": self.time_points,
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "time_error_tolerance": self.time_error_tolerance,
            "amplitude": self.amplitude.summary(),
            "quadrature": self.state.summary()["transaction"],
            "temperature": self._temperature_summary(),
            "solution_layout": {
                "global_size": int(self.solution.function_space.dofmap.index_map.size_global),
                "block_size": int(self.solution.function_space.dofmap.index_map_bs),
            },
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
        """Solve and publish Maxwell state through the common result lifecycle."""

        from ..results import (
            add_execution_trace,
            complete_result,
            from_solution,
            recover_integration_point_field,
        )

        if fields and output_fields:
            raise ValueError("Pass fields=... or output_fields=..., not both.")
        selected_fields = tuple(fields) or tuple(output_fields)
        if self.accepted_time < self.duration:
            self.solve()
        result = from_solution(
            self.solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "state": self.state.summary(),
            },
        )
        add_execution_trace(result, self.execution_events)
        quadrature_fields = (
            ("S", self.state.stress, "Generalized-Maxwell Cauchy stress."),
            ("E", self.state.accepted_strain, "Committed small strain."),
            ("SENER", self.state.stored_energy, "Recoverable viscoelastic energy density."),
            ("VDENER", self.state.dissipated_energy, "Cumulative viscous dissipation density."),
            ("MISES", self.state.equivalent_stress(), "Von Mises stress."),
        )
        for name, source, description in quadrature_fields:
            result.add_field(
                name,
                source.function,
                location="quadrature_points",
                description=description,
                processing={
                    "source_position": "quadrature_points",
                    "method": "exact_generalized_maxwell_update",
                    "representation": "quadrature_values",
                    "postprocessed": False,
                    "committed": name in {"E", "VDENER"},
                },
            )
            recovered = recover_integration_point_field(source, name=f"{name}_CELL")
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
            description="Full nodal residual for reaction extraction.",
            processing={"method": "assembled_equilibrium_residual"},
        )
        result.add_quantities(
            {
                "analysis_time": self.accepted_time,
                "maximum_viscous_dissipation_density": (
                    self.state.dissipated_energy.global_max()
                ),
            },
            kind="diagnostic",
        )
        temperature_values = self._temperature_values()
        if temperature_values is not None:
            selected_temperature = getattr(self.temperature, "value", self.temperature)
            if hasattr(selected_temperature, "function_space"):
                result.add_field(
                    "TEMP",
                    selected_temperature,
                    unit="K",
                    description="Temperature field consumed by the shift law.",
                    processing={
                        "method": "quadrature_interpolation_for_constitutive_update",
                        "postprocessed": False,
                    },
                )
            result.add_quantities(
                {
                    "minimum_viscoelastic_temperature": float(
                        np.min(temperature_values)
                    ),
                    "maximum_viscoelastic_temperature": float(
                        np.max(temperature_values)
                    ),
                },
                units={
                    "minimum_viscoelastic_temperature": "K",
                    "maximum_viscoelastic_temperature": "K",
                },
                kind="diagnostic",
            )
        if self.accepted_increments:
            times = np.asarray([item.end_time for item in self.accepted_increments])
            increment_histories = {
                "equilibrium_iterations": [
                    item.iterations for item in self.accepted_increments
                ],
                "time_increment": [
                    item.end_time - item.start_time
                    for item in self.accepted_increments
                ],
            }
            if all(
                item.time_error_estimate is not None
                for item in self.accepted_increments
            ):
                increment_histories["time_error_estimate"] = [
                    item.time_error_estimate for item in self.accepted_increments
                ]
            result.add_histories(
                times,
                increment_histories,
                abscissa_name="time",
                abscissa_unit=self.time_unit,
            )
        if self.energy_history:
            times = np.asarray([item.time for item in self.energy_history])
            result.add_histories(
                times,
                {
                    "load_amplitude": [item.load_amplitude for item in self.energy_history],
                    "stored_energy": [item.stored_energy for item in self.energy_history],
                    "viscous_dissipation": [
                        item.viscous_dissipation for item in self.energy_history
                    ],
                    "material_work": [item.material_work for item in self.energy_history],
                    "constitutive_energy_residual": [
                        item.constitutive_energy_residual for item in self.energy_history
                    ],
                },
                abscissa_name="time",
                abscissa_unit=self.time_unit,
            )
            result.metadata["viscoelastic_energy_ledger"] = {
                "identity": "material_work = stored_energy + viscous_dissipation",
                "scope": (
                    "exact constitutive ledger integrated over the domain; global "
                    "external work is a separate structural balance channel"
                ),
            }
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

    def summary(self) -> dict[str, object]:
        checkpoint_policy = None
        if self.checkpoint_policy is not None:
            checkpoint_policy = {
                **self.checkpoint_policy.summary(),
                "effective_portable": bool(self.checkpoint_policy.portable)
                or self.state.domain.comm.size > 1,
            }
        return {
            "kind": "quasistatic_viscoelastic_step",
            "name": self.name,
            "study": None if self.study is None else self.study.summary(),
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "state": self.state.summary(),
            "duration": self.duration,
            "steps": self.steps,
            "time_increment": self.time_increment,
            "time_grid": self._time_grid_summary(),
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "time_error_tolerance": self.time_error_tolerance,
            "accepted_time": self.accepted_time,
            "accepted_increments": len(self.accepted_increments),
            "attempted_increments": len(self.attempted_increments),
            "next_increment_size": self.next_increment_size,
            "time_unit": self.time_unit,
            "amplitude": self.amplitude.summary(),
            "temperature": self._temperature_summary(),
            "checkpoint_policy": checkpoint_policy,
            "checkpoint_count": len(self.checkpoints),
            "solver": self.solver_options.summary(),
            "last_solve": (
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
        }

    def _time_grid_summary(self) -> dict[str, object]:
        if self.time_points is None:
            return {
                "kind": "automatic",
                "points": None,
                "minimum_increment": self.duration * self.incrementation.minimum,
                "maximum_increment": self.duration * self.incrementation.maximum,
                "sha256": None,
            }
        increments = np.diff(self.time_points)
        return {
            "kind": "uniform" if self.time_increment is not None else "nonuniform",
            "points": len(self.time_points),
            "minimum_increment": float(np.min(increments)),
            "maximum_increment": float(np.max(increments)),
            "sha256": sha256(
                np.asarray(self.time_points, dtype=np.float64).tobytes()
            ).hexdigest(),
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


def quasistatic_viscoelastic_step(
    *,
    displacement,
    material,
    duration: float,
    steps: int | None = None,
    time_points=None,
    incrementation=None,
    time_error_tolerance: float | None = None,
    external_force=None,
    constraints=(),
    study=None,
    solver_options=None,
    quadrature_degree: int = 2,
    amplitude=None,
    temperature=None,
    time_unit: str | None = None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    name: str = "viscoelastic",
) -> QuasistaticViscoelasticStep:
    """Build a 3D quasi-static generalized-Maxwell Step."""

    if not isinstance(material, (IsotropicGeneralizedMaxwell, QuadratureMaterialMap)):
        raise TypeError(
            "quasistatic_viscoelastic_step requires IsotropicGeneralizedMaxwell "
            "or a compatible QuadratureMaterialMap."
        )
    domain = displacement.value.function_space.mesh
    if domain.geometry.dim != 3:
        raise NotImplementedError(
            "The first global generalized-Maxwell provider supports 3D solids."
        )
    if isinstance(material, QuadratureMaterialMap):
        if material.domain is not domain:
            raise ValueError("Viscoelastic material map belongs to another mesh.")
        if not all(
            isinstance(item, IsotropicGeneralizedMaxwell)
            for item in material.materials.values()
        ):
            raise TypeError("Every regional viscoelastic material must share this family.")
    selected_amplitude = (
        amplitudes.ramp(end_time=float(duration), name="viscoelastic_ramp")
        if amplitude is None
        else amplitudes.as_amplitude(
            amplitude, name="viscoelastic_load_amplitude"
        )
    )
    if incrementation is not None and (steps is not None or time_points is not None):
        raise ValueError(
            "Specify incrementation or a prescribed steps/time_points path, not both."
        )
    selected_incrementation = (
        None
        if incrementation is None
        else step_controls.normalize(incrementation)
    )
    load_factor = fem.Constant(domain, PETSc.ScalarType(selected_amplitude(0.0)))
    state = ViscoelasticQuadratureState.create(
        domain, material, degree=quadrature_degree
    )
    strain_test = elasticity.strain(displacement.test, study=study)
    strain_trial = elasticity.strain(displacement.trial, study=study)
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        state.tangent.function[i, j, k, l] * strain_trial[k, l], (i, j)
    )
    residual = ufl.inner(state.stress.function, strain_test) * state.measure
    if external_force is not None:
        residual -= load_factor * external_force.expression
    jacobian = ufl.inner(tangent_action, strain_test) * state.measure

    selected_bcs = []
    prescribed_values = []
    time_dependent = []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected_bcs.extend(item.bcs)
            candidates = tuple(getattr(item, "dirichlet", ()))
        elif hasattr(item, "bc"):
            selected_bcs.append(item.bc)
            candidates = (item,)
        else:
            selected_bcs.append(item)
            candidates = ()
        for candidate in candidates:
            if hasattr(candidate, "amplitude") and hasattr(candidate, "update"):
                time_dependent.append(candidate)
            else:
                value = getattr(candidate, "value", None)
                if value is not None and hasattr(value, "value"):
                    prescribed_values.append(
                        (value, np.asarray(value.value, dtype=float).copy(), candidate.bc)
                    )
    return QuasistaticViscoelasticStep(
        name=name,
        solution=displacement.value,
        material=material,
        state=state,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        external_force=external_force,
        load_factor=load_factor,
        amplitude=selected_amplitude,
        bcs=tuple(selected_bcs),
        prescribed_values=tuple(prescribed_values),
        time_dependent_constraints=tuple(time_dependent),
        duration=duration,
        steps=steps,
        time_points=(
            None if time_points is None else tuple(float(value) for value in time_points)
        ),
        incrementation=selected_incrementation,
        time_error_tolerance=time_error_tolerance,
        solver_options=(
            newton(maximum_iterations=4, line_search="basic")
            if solver_options is None
            else solver_options
        ),
        temperature=temperature,
        time_unit=time_unit,
        study=study,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
    )


__all__ = [
    "QuasistaticViscoelasticStep",
    "ViscoelasticEnergyFrame",
    "ViscoelasticIncrementInfo",
    "ViscoelasticPathInfo",
    "ViscoelasticQuadratureState",
    "quasistatic_viscoelastic_step",
]
