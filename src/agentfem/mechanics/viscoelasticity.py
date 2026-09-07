"""Global small-strain generalized-Maxwell equilibrium.

The constitutive law owns the exact Prony-branch update.  This module owns the
finite-element procedure: quadrature state, equilibrium, accepted time,
rollback, progress, result fields, and constitutive energy evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        )


@dataclass(frozen=True)
class ViscoelasticPathInfo:
    """Resolved fixed physical-time path."""

    increments: tuple[ViscoelasticIncrementInfo, ...]
    duration: float
    steps: int

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
            "increments": [item.as_dict() for item in self.increments],
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
    temperature: object | None = None
    time_unit: str | None = None
    study: object | None = None
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    procedure: object = field(default_factory=procedures.quasistatic_viscoelasticity)
    accepted_time: float = field(default=0.0, init=False)
    accepted_increments: list[ViscoelasticIncrementInfo] = field(
        default_factory=list, init=False
    )
    execution_events: list[object] = field(default_factory=list, init=False)
    energy_history: list[ViscoelasticEnergyFrame] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    last_solve_info: ViscoelasticPathInfo | None = field(default=None, init=False)
    _strain_evaluator: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.duration = float(self.duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("Viscoelastic duration must be finite and positive.")
        if self.time_points is None:
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
        self.time_points = tuple(float(value) for value in grid)
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

    @property
    def time_increment(self) -> float | None:
        increments = np.diff(self.time_points)
        if np.allclose(increments, increments[0], rtol=1.0e-12, atol=0.0):
            return float(increments[0])
        return None

    def solve(self, *, until: float | None = None):
        """Advance on the declared physical-time grid."""

        selected_until = self.duration if until is None else float(until)
        grid = np.asarray(self.time_points, dtype=float)
        candidates = np.flatnonzero(grid > self.accepted_time + 1.0e-12)
        target_indices = [index for index in candidates if grid[index] <= selected_until + 1e-12]
        if not target_indices or not np.isclose(grid[target_indices[-1]], selected_until):
            raise ValueError("until must be an undeclared future point on the fixed time grid.")

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
                start_time=start_time,
                end_time=end_time,
                reporter=reporter,
            )
            if not info.converged:
                self.solution.x.array[:] = displacement_snapshot
                self.solution.x.scatter_forward()
                self.state.restore(state_snapshot)
                self._apply_loading(start_time)
                self.last_solve_info = ViscoelasticPathInfo(
                    tuple((*self.accepted_increments, info)), self.duration, self.steps
                )
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
        self.last_solve_info = ViscoelasticPathInfo(
            tuple(self.accepted_increments), self.duration, self.steps
        )
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

    def _solve_increment(self, *, increment, start_time, end_time, reporter):
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
                    attempt=1,
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

    def _record_energy(self, time: float) -> None:
        stored = self._integral(self.state.stored_energy)
        dissipation = self._integral(self.state.dissipated_energy)
        increment = self._integral(self.state.work_increment)
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

    def save_checkpoint(self, path) -> Path:
        """Save a serial restart at an accepted time boundary."""

        if self.state.domain.comm.size != 1:
            raise NotImplementedError(
                "Distributed viscoelastic restart requires the portable state bundle."
            )
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
            energy=json.dumps([item.as_dict() for item in self.energy_history]),
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
        if self.state.domain.comm.size != 1:
            raise NotImplementedError(
                "Distributed viscoelastic restart requires the portable state bundle."
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
            self.energy_history[:] = [
                ViscoelasticEnergyFrame(**item)
                for item in json.loads(str(data["energy"]))
            ]
        self._apply_loading(self.accepted_time)
        self.last_solve_info = ViscoelasticPathInfo(
            tuple(self.accepted_increments), self.duration, self.steps
        )

    def _checkpoint_identity(self) -> dict[str, object]:
        return {
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "duration": self.duration,
            "steps": self.steps,
            "time_points": self.time_points,
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
            result.add_history(
                "equilibrium_iterations",
                times,
                [item.iterations for item in self.accepted_increments],
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
            "time_grid": {
                "kind": "uniform" if self.time_increment is not None else "nonuniform",
                "points": len(self.time_points),
                "minimum_increment": float(np.min(np.diff(self.time_points))),
                "maximum_increment": float(np.max(np.diff(self.time_points))),
                "sha256": sha256(
                    np.asarray(self.time_points, dtype=np.float64).tobytes()
                ).hexdigest(),
            },
            "accepted_time": self.accepted_time,
            "time_unit": self.time_unit,
            "amplitude": self.amplitude.summary(),
            "temperature": self._temperature_summary(),
            "solver": self.solver_options.summary(),
            "last_solve": (
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
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
    )


__all__ = [
    "QuasistaticViscoelasticStep",
    "ViscoelasticEnergyFrame",
    "ViscoelasticIncrementInfo",
    "ViscoelasticPathInfo",
    "ViscoelasticQuadratureState",
    "quasistatic_viscoelastic_step",
]
