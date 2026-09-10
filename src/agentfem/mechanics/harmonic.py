"""Direct harmonic finite-element procedure over inspectable K/M/C/F operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc

from .. import procedures
from ..backends._harmonic import PreparedHarmonicLinearProblem
from ..operators.harmonic import DirectHarmonicSystem
from ..provenance import content_fingerprint
from ..solvers import LinearSolveInfo, LinearSolverOptions


@dataclass
class DirectHarmonicStep:
    """One direct steady-state harmonic solve with separated ownership."""

    name: str
    solution_real: object
    solution_imaginary: object
    system: DirectHarmonicSystem
    angular_frequency: float
    bcs: tuple[object, ...]
    solver_options: LinearSolverOptions
    load_phase: float = 0.0
    study: object | None = None
    procedure: object = field(default_factory=procedures.direct_harmonic)
    last_solve_info: LinearSolveInfo | None = field(default=None, init=False)
    algebraic_equilibrium: dict[str, float] | None = field(default=None, init=False)
    cycle_input_energy: float | None = field(default=None, init=False)
    displacement_amplitude: object | None = field(default=None, init=False)
    displacement_phase: object | None = field(default=None, init=False)
    solved_angular_frequency: float | None = field(default=None, init=False)
    _prepared_problem: PreparedHarmonicLinearProblem | None = field(
        default=None, init=False, repr=False
    )
    _prepared_configuration_fingerprint: str | None = field(
        default=None, init=False, repr=False
    )
    _closed_backend_summary: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def frequency(self) -> float:
        return self.angular_frequency / (2.0 * np.pi)

    @property
    def complex_dofs(self) -> np.ndarray:
        return self.solution_real.x.array.copy() + 1j * self.solution_imaginary.x.array

    def solve(self):
        """Solve the current frequency and return the real displacement field."""

        self._require_open()
        current_configuration = self._require_prepared_configuration_current(
            collective=True
        )
        if self._prepared_problem is None:
            prefix_name = "".join(
                character if character.isalnum() else "_" for character in self.name
            )
            self._prepared_problem = PreparedHarmonicLinearProblem.from_system(
                self.system,
                solution_real=self.solution_real,
                solution_imaginary=self.solution_imaginary,
                bcs=self.bcs,
                solver_options=self.solver_options,
                angular_frequency=self.angular_frequency,
                load_phase=self.load_phase,
                petsc_options_prefix=f"agentfem_harmonic_{prefix_name}_",
            )
            self._prepared_configuration_fingerprint = current_configuration
        else:
            self._prepared_problem.set_angular_frequency(self.angular_frequency)
        self._clear_solve_evidence()
        evidence = self._prepared_problem.solve()
        self.last_solve_info = evidence.solve
        self.algebraic_equilibrium = evidence.equilibrium()
        self.cycle_input_energy = evidence.input_energy_per_cycle
        _require_solve_evidence(
            evidence,
            solver_options=self.solver_options,
            context=f"Direct harmonic solve at {self.frequency:.12g} Hz",
        )
        self._refresh_polar_fields()
        self.solved_angular_frequency = self.angular_frequency
        return self.solution_real

    def set_frequency(
        self,
        *,
        frequency: float | None = None,
        angular_frequency: float | None = None,
    ) -> None:
        """Select another frequency while retaining the prepared backend."""

        self._require_open()
        selected = _angular_frequency(
            frequency=frequency,
            angular_frequency=angular_frequency,
        )
        if selected != self.angular_frequency:
            self.angular_frequency = selected
            self._clear_solve_evidence()

    def _clear_solve_evidence(self) -> None:
        self.last_solve_info = None
        self.algebraic_equilibrium = None
        self.cycle_input_energy = None
        self.solved_angular_frequency = None
        self.solution_real.x.array[:] = 0.0
        self.solution_imaginary.x.array[:] = 0.0
        self.solution_real.x.scatter_forward()
        self.solution_imaginary.x.scatter_forward()
        for field_value in (self.displacement_amplitude, self.displacement_phase):
            if field_value is not None:
                field_value.x.array[:] = np.nan
                field_value.x.scatter_forward()

    def _refresh_polar_fields(self) -> None:
        real = np.asarray(self.solution_real.x.array, dtype=float)
        imaginary = np.asarray(self.solution_imaginary.x.array, dtype=float)
        amplitude = self.displacement_amplitude
        phase = self.displacement_phase
        if amplitude is None:
            amplitude = fem.Function(
                self.solution_real.function_space, name="U_AMPLITUDE"
            )
        if phase is None:
            phase = fem.Function(self.solution_real.function_space, name="U_PHASE")
        amplitude.x.array[:] = np.hypot(real, imaginary)
        phase.x.array[:] = np.arctan2(imaginary, real)
        amplitude.x.scatter_forward()
        phase.x.scatter_forward()
        self.displacement_amplitude = amplitude
        self.displacement_phase = phase

    def energy_evidence(self) -> dict[str, float]:
        """Return separated storage, inertia, and dissipation evidence."""

        self._require_prepared_configuration_current()
        if self.solved_angular_frequency != self.angular_frequency:
            raise RuntimeError(
                "Harmonic energy evidence requires a solution at the selected "
                "frequency. Call solve() after set_frequency()."
            )

        stored = 0.25 * self._quadratic_pair(self.system.storage)
        kinetic = (
            0.0
            if self.system.mass is None
            else 0.25
            * self.angular_frequency**2
            * self._quadratic_pair(self.system.mass)
        )
        material_loss = (
            0.0
            if self.system.loss is None
            else np.pi * self._quadratic_pair(self.system.loss)
        )
        viscous_loss = (
            0.0
            if self.system.damping is None
            else np.pi
            * self.angular_frequency
            * self._quadratic_pair(self.system.damping)
        )
        _require_finite_harmonic_values(
            {
                "mean stored energy": stored,
                "mean kinetic energy": kinetic,
                "material dissipated energy per cycle": material_loss,
                "viscous dissipated energy per cycle": viscous_loss,
            }
        )
        loss_scale = max(
            abs(stored),
            abs(kinetic),
            abs(material_loss),
            abs(viscous_loss),
            np.finfo(float).eps,
        )
        if material_loss < -1.0e-12 * loss_scale:
            raise RuntimeError(
                "The declared material-loss operator produced negative cycle work."
            )
        if viscous_loss < -1.0e-12 * loss_scale:
            raise RuntimeError(
                "The declared viscous-damping operator produced negative cycle work."
            )
        dissipated = material_loss + viscous_loss
        if self.cycle_input_energy is None:
            raise RuntimeError(
                "Harmonic cycle-energy evidence requires a completed solve."
            )
        balance_error = self.cycle_input_energy - dissipated
        balance_scale = max(
            abs(self.cycle_input_energy),
            abs(dissipated),
            np.sqrt(np.finfo(float).eps) * (abs(stored) + abs(kinetic)),
            np.finfo(float).eps,
        )
        evidence = {
            "mean_stored_energy": stored,
            "mean_kinetic_energy": kinetic,
            "material_dissipated_energy_per_cycle": material_loss,
            "viscous_dissipated_energy_per_cycle": viscous_loss,
            "dissipated_energy_per_cycle": dissipated,
            "input_energy_per_cycle": self.cycle_input_energy,
            "cycle_energy_balance_error": balance_error,
            "relative_cycle_energy_balance_error": abs(balance_error) / balance_scale,
            "mean_dissipated_power": (
                0.0
                if self.angular_frequency == 0.0
                else self.angular_frequency * dissipated / (2.0 * np.pi)
            ),
        }
        _require_finite_harmonic_values(evidence)
        return evidence

    def _quadratic_pair(self, operator) -> float:
        return _quadratic_integral(operator, self.solution_real) + _quadratic_integral(
            operator, self.solution_imaginary
        )

    def summary(self) -> dict[str, object]:
        self._require_prepared_configuration_current()
        return {
            "kind": "direct_harmonic_step",
            "name": self.name,
            "frequency": self.frequency,
            "angular_frequency": self.angular_frequency,
            "frequency_regime": (
                "static_limit" if self.angular_frequency == 0.0 else "harmonic"
            ),
            "phasor_convention": self.system.phasor_convention,
            "load_phase": self.load_phase,
            "system": self.system.summary(),
            "procedure": self.procedure.summary(),
            "solver": self.solver_options.summary(),
            "backend_execution": (
                self._backend_execution_summary()
            ),
            "solve": (
                None
                if self.last_solve_info is None
                else {
                    **self.last_solve_info.as_dict(),
                    "algebraic_equilibrium": self.algebraic_equilibrium,
                }
            ),
        }

    def _require_prepared_configuration_current(
        self, *, collective: bool = False
    ) -> str:
        current = _prepared_configuration_fingerprint(self)
        if self._prepared_configuration_fingerprint is None:
            return current
        changed = current != self._prepared_configuration_fingerprint
        if collective:
            comm = self.solution_real.function_space.mesh.comm
            changed = bool(comm.allreduce(changed, op=MPI.LOR))
        if changed:
            raise RuntimeError(
                "The direct harmonic system, coefficients, constraints, load "
                "phase, solution fields, or solver policy changed after the "
                "backend was prepared. Create a new Step so the executed "
                "operator and reported scientific contract cannot diverge."
            )
        return current

    @property
    def closed(self) -> bool:
        """Whether this Step's retained backend allocation has been released."""

        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"DirectHarmonicStep {self.name!r} is closed.")

    def _backend_execution_summary(self) -> dict[str, object] | None:
        if self._prepared_problem is not None:
            return self._prepared_problem.summary()
        if self._closed_backend_summary is None:
            return None
        return dict(self._closed_backend_summary)

    def close(self) -> None:
        """Deterministically release the retained harmonic PETSc allocation."""

        if self._closed:
            return
        prepared = self._prepared_problem
        if prepared is not None:
            backend_summary = prepared.summary()
            try:
                prepared.close()
            finally:
                self._closed_backend_summary = backend_summary
                self._prepared_problem = None
                self._closed = True
        else:
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def solve_result(self, *, output=None, strict_output: bool = False):
        """Solve and delegate result assembly to the Result owner."""

        from ..results._harmonic import from_harmonic_step

        self.solve()
        return from_harmonic_step(self, output=output, strict_output=strict_output)


@dataclass
class DirectHarmonicSweepStep:
    """A bounded-memory ordered frequency sweep over one prepared Step."""

    name: str
    point_step: DirectHarmonicStep
    frequencies: tuple[float, ...]
    responses: tuple[object, ...] = ()
    execution_order: str = "forward"
    scientific_assets: dict[str, object] | None = None
    status_file: object | None = None
    procedure: object = field(default_factory=procedures.direct_harmonic_sweep)
    records: dict[int, dict[str, object]] = field(default_factory=dict, init=False)
    failure: dict[str, object] | None = field(default=None, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    last_live_field_frequency: float | None = field(default=None, init=False)
    _event_recorder: object = field(default=None, init=False, repr=False)
    _checkpoint_field_identity_record: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )
    _frozen_executable_identity: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.frequencies = _frequency_axis(self.frequencies)
        order = str(self.execution_order).strip().lower().replace("-", "_")
        if order not in {"forward", "reverse"}:
            raise ValueError("execution_order must be 'forward' or 'reverse'.")
        self.execution_order = order
        self.responses = tuple(self.responses)
        if len({item.name for item in self.responses}) != len(self.responses):
            raise ValueError("Harmonic response names must be unique.")
        if any(
            not callable(getattr(item, "sample", None))
            or not callable(getattr(item, "summary", None))
            for item in self.responses
        ):
            raise TypeError("responses must be results.harmonic_response(...) objects.")
        from ..diagnostics import SolveEventRecorder

        self._event_recorder = SolveEventRecorder(self.execution_events)

    @property
    def completed(self) -> bool:
        return len(self.records) == len(self.frequencies) and self.failure is None

    @property
    def solution_real(self):
        return self.point_step.solution_real

    @property
    def solution_imaginary(self):
        return self.point_step.solution_imaginary

    @property
    def execution_event_capacity(self) -> int:
        return int(self._event_recorder.max_events)

    @property
    def dropped_execution_events(self) -> int:
        return int(self._event_recorder.dropped_events)

    def solve(self, *, max_points: int | None = None):
        """Advance pending frequency points, retaining only scalar records."""

        self._require_open()
        if max_points is not None:
            if isinstance(max_points, (bool, np.bool_)) or not isinstance(
                max_points, (int, np.integer)
            ):
                raise TypeError("max_points must be an integer when supplied.")
            if int(max_points) <= 0:
                raise ValueError("max_points must be positive when supplied.")
        self._checkpoint_policy()
        self._freeze_executable_identity()
        indices = list(range(len(self.frequencies)))
        if self.execution_order == "reverse":
            indices.reverse()
        pending = [index for index in indices if index not in self.records]
        if max_points is not None:
            pending = pending[: int(max_points)]
        if not pending and self.completed:
            return self
        reporter = self._reporter()
        from ..solvers import SolveEvent

        reporter.emit(
            SolveEvent(
                "sweep_resumed" if self.records else "sweep_started",
                self.name,
                increment=len(self.records),
                total_increments=len(self.frequencies),
                incrementation="independent_frequency_points",
            )
        )
        self.failure = None
        display_every = max(1, int(np.ceil(len(self.frequencies) / 20.0)))
        for index in pending:
            frequency = self.frequencies[index]
            try:
                self.point_step.set_frequency(frequency=frequency)
                self.point_step.solve()
                self.records[index] = self._record(index, frequency)
                self.last_live_field_frequency = frequency
                record = self.records[index]
                completed = len(self.records)
                reporter.emit(
                    SolveEvent(
                        "sweep_point",
                        self.name,
                        increment=completed,
                        total_increments=len(self.frequencies),
                        residual_norm=record["equilibrium"].get(
                            "relative_residual_norm"
                        ),
                        coordinate_name="frequency",
                        coordinate_value=frequency,
                        coordinate_unit="Hz",
                        metrics={
                            "maximum_displacement_vector_amplitude": record[
                                "maximum_displacement_vector_amplitude"
                            ],
                            "relative_cycle_energy_balance_error": record["energy"][
                                "relative_cycle_energy_balance_error"
                            ],
                        },
                        display=(
                            completed == 1
                            or completed == len(self.frequencies)
                            or completed % display_every == 0
                        ),
                    )
                )
                if index == pending[-1]:
                    reporter.emit(
                        SolveEvent(
                            ("sweep_completed" if self.completed else "sweep_paused"),
                            self.name,
                            increment=len(self.records),
                            total_increments=len(self.frequencies),
                        )
                    )
                self._write_scheduled_checkpoint()
            except Exception as exc:
                self.failure = {
                    "index": index,
                    "frequency": frequency,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                reporter.emit(
                    SolveEvent(
                        "sweep_failed",
                        self.name,
                        increment=len(self.records),
                        total_increments=len(self.frequencies),
                        coordinate_name="frequency",
                        coordinate_value=frequency,
                        coordinate_unit="Hz",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise RuntimeError(
                    f"Harmonic sweep failed at index={index}, "
                    f"frequency={frequency:.12g} Hz: {exc}"
                ) from exc
        return self

    def scientific_inputs(self, *, require_checkpoint_assets: bool = False):
        """Return the scientific identity shared by results and restart gates."""

        assets = self.scientific_assets
        executable_identity = None
        if require_checkpoint_assets:
            executable_identity = self._validated_executable_identity()
        return {
            "harmonic_system": self.point_step.system,
            "frequency_axis": {
                "values": self.frequencies,
                "unit": "Hz",
                "canonical_order": "ascending",
            },
            "responses": self.responses,
            "load_phase": self.point_step.load_phase,
            "solver": self.point_step.solver_options,
            "procedure": self.procedure,
            "model_assets": {} if assets is None else assets,
            "executable_identity": executable_identity,
        }

    def save_checkpoint(self, path, *, role: str = "manual_checkpoint"):
        """Persist accepted scalar evidence without retaining live FEM fields."""

        from .. import checkpointing
        from ..results import CheckpointRecord

        comm = self.solution_real.function_space.mesh.comm
        manifest = checkpointing.save_harmonic_sweep_checkpoint(
            path,
            step_name=self.name,
            frequencies=self.frequencies,
            records=self.records,
            scientific_inputs=self.scientific_inputs(require_checkpoint_assets=True),
            field_identity=self._checkpoint_field_identity(),
            execution_events=self.execution_events,
            comm=comm,
        )
        self.checkpoints[:] = [
            item for item in self.checkpoints if item.path != manifest
        ]
        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_frequency_checkpoint_{len(self.records)}",
                path=manifest,
                schema=checkpointing.HARMONIC_SWEEP_CHECKPOINT_SCHEMA,
                step_name=self.name,
                coordinate_name="completed_frequency_points",
                coordinate_value=len(self.records),
                portable=True,
                metadata={
                    "role": str(role),
                    "rank_count_at_write": int(comm.size),
                    "producer_software": {
                        "name": "AgentFEM",
                        "version": checkpointing._software_version(),
                    },
                    "requested_portable": self._requested_checkpoint_portability(),
                    "effective_portable": True,
                    "field_state": "not_stored_scalar_ledger_only",
                    "portability": (
                        "portable across MPI partitions, rank counts, and "
                        "frequency execution order"
                    ),
                },
            )
        )
        return manifest

    def load_checkpoint(self, path):
        """Restore accepted scalar evidence after a fail-closed identity check."""

        from .. import checkpointing
        from ..results import CheckpointRecord
        from ..solvers import SolveEvent

        comm = self.solution_real.function_space.mesh.comm
        frozen_before = self._frozen_executable_identity
        field_identity_before = self._checkpoint_field_identity_record
        try:
            payload = checkpointing.load_harmonic_sweep_checkpoint(
                path,
                step_name=self.name,
                frequencies=self.frequencies,
                scientific_inputs=self.scientific_inputs(
                    require_checkpoint_assets=True
                ),
                field_identity=self._checkpoint_field_identity(),
                comm=comm,
            )
            restored_events = tuple(
                SolveEvent.from_dict(record)
                for record in payload.get("execution_events", ())
            )
            manifest = payload["manifest_path"]
            restored_checkpoint = CheckpointRecord(
                name=f"{self.name}_restart_source_{len(payload['records'])}",
                path=manifest,
                schema=checkpointing.HARMONIC_SWEEP_CHECKPOINT_SCHEMA,
                step_name=self.name,
                coordinate_name="completed_frequency_points",
                coordinate_value=len(payload["records"]),
                portable=True,
                metadata={
                    "role": "restart_source",
                    "rank_count_at_write": payload["rank_count_at_write"],
                    "rank_count_at_read": int(comm.size),
                    "field_state": "not_restored_scalar_ledger_only",
                    "identity_fingerprint": payload["identity_fingerprint"],
                    "producer_software": dict(payload["software"]),
                    "requested_portable": self._requested_checkpoint_portability(),
                    "effective_portable": True,
                },
            )
        except Exception:
            self._frozen_executable_identity = frozen_before
            self._checkpoint_field_identity_record = field_identity_before
            raise

        # Every fallible parse and identity check is complete. Commit the
        # restored state only at this final transaction boundary.
        self.records.clear()
        self.records.update(payload["records"])
        self.failure = None
        self.point_step._clear_solve_evidence()
        self.last_live_field_frequency = None
        self._event_recorder.clear()
        for event in restored_events:
            self._event_recorder.emit(event)
        self.checkpoints.append(restored_checkpoint)
        return payload

    def _reporter(self):
        from ..diagnostics import StandardRunReporter, compose_reporters

        context = getattr(self, "execution_context", None)
        selected = None if context is None else context.policy.progress
        visible = None
        if selected is None or selected is True:
            visible = StandardRunReporter(
                self.solution_real.function_space.mesh.comm,
                status_file=self.status_file,
                show_iterations=False,
            )
        elif selected is not False:
            visible = selected
        return compose_reporters(self._event_recorder, visible)

    def _checkpoint_field_identity(self) -> dict[str, object]:
        if self._checkpoint_field_identity_record is None:
            from ..checkpointing import function_portable_identity

            self._checkpoint_field_identity_record = function_portable_identity(
                self.solution_real
            )
        return dict(self._checkpoint_field_identity_record)

    def _current_executable_identity(self) -> dict[str, object]:
        from ..operators.identity import harmonic_executable_identity

        identity = harmonic_executable_identity(
            self.point_step.system,
            solution=self.point_step.solution_real,
            bcs=self.point_step.bcs,
        )
        if not identity["complete"]:
            missing = ", ".join(str(item["path"]) for item in identity["missing"][:5])
            raise ValueError(
                "Harmonic sweep checkpointing cannot establish a portable "
                "executable identity for the actual operators or constraints: "
                f"{missing or 'unknown'}."
            )
        return identity

    def _freeze_executable_identity(self) -> dict[str, object]:
        current = self._current_executable_identity()
        if self._frozen_executable_identity is None:
            self._frozen_executable_identity = current
        elif current["fingerprint"] != self._frozen_executable_identity["fingerprint"]:
            raise RuntimeError(
                "The executable harmonic operators, coefficients, mesh tags, "
                "or constrained DOFs changed after the sweep began. Start a "
                "new sweep instead of mixing frequency evidence."
            )
        return dict(self._frozen_executable_identity)

    def _validated_executable_identity(self) -> dict[str, object]:
        return self._freeze_executable_identity()

    def _requested_checkpoint_portability(self) -> bool | None:
        policy = self._checkpoint_policy()
        return None if policy is None else bool(policy.portable)

    def _write_scheduled_checkpoint(self) -> None:
        policy = self._checkpoint_policy()
        if policy is None:
            return

        completed = len(self.records)
        if not policy.due(completed, len(self.frequencies)):
            return
        self.save_checkpoint(
            policy.path(step_name=self.name, increment=completed),
            role="scheduled_checkpoint",
        )
        self._prune_scheduled_checkpoints(policy.keep_last)

    def _checkpoint_policy(self):
        context = getattr(self, "execution_context", None)
        policy = None if context is None else context.policy.checkpoint
        if policy is None:
            return None
        from ..checkpointing import CheckpointPolicy

        if not isinstance(policy, CheckpointPolicy):
            raise TypeError(
                "Harmonic sweep checkpoint= must be checkpointing.every(...)."
            )
        return policy

    def _prune_scheduled_checkpoints(self, keep_last) -> None:
        if keep_last is None:
            return
        scheduled = [
            item
            for item in self.checkpoints
            if item.metadata.get("role") == "scheduled_checkpoint"
        ]
        obsolete = scheduled[: -int(keep_last)]
        if not obsolete:
            return
        from ..checkpointing import remove_harmonic_sweep_checkpoint

        comm = self.solution_real.function_space.mesh.comm
        for item in obsolete:
            remove_harmonic_sweep_checkpoint(item.path, comm=comm)
        removed = {id(item) for item in obsolete}
        self.checkpoints[:] = [
            item for item in self.checkpoints if id(item) not in removed
        ]

    def _record(self, index: int, frequency: float) -> dict[str, object]:
        real_field = self.point_step.solution_real
        imaginary_field = self.point_step.solution_imaginary
        owned = int(
            real_field.function_space.dofmap.index_map.size_local
            * real_field.function_space.dofmap.index_map_bs
        )
        block_size = int(real_field.function_space.dofmap.index_map_bs)
        local_amplitudes = _physical_cycle_vector_amplitudes(
            np.asarray(real_field.x.array[:owned], dtype=float),
            np.asarray(imaginary_field.x.array[:owned], dtype=float),
            block_size=block_size,
        )
        local_max = float(np.max(local_amplitudes)) if local_amplitudes.size else 0.0
        comm = real_field.function_space.mesh.comm
        maximum = float(comm.allreduce(local_max, op=MPI.MAX))
        response_values = {
            item.name: item.sample(self.point_step) for item in self.responses
        }
        return {
            "index": index,
            "frequency": frequency,
            "angular_frequency": 2.0 * np.pi * frequency,
            "maximum_displacement_vector_amplitude": maximum,
            "responses": response_values,
            "energy": self.point_step.energy_evidence(),
            "equilibrium": dict(self.point_step.algebraic_equilibrium or {}),
            "solve": (
                None
                if self.point_step.last_solve_info is None
                else self.point_step.last_solve_info.as_dict()
            ),
        }

    def canonical_records(self) -> tuple[dict[str, object], ...]:
        if not self.completed:
            raise RuntimeError(
                "A partial harmonic sweep cannot be published as completed."
            )
        return tuple(self.records[index] for index in range(len(self.frequencies)))

    def summary(self) -> dict[str, object]:
        checkpoint_policy = self._checkpoint_policy()
        checkpoint_summary = None
        if checkpoint_policy is not None:
            checkpoint_summary = {
                **checkpoint_policy.summary(),
                "requested_portable": bool(checkpoint_policy.portable),
                "effective_portable": True,
                "portable": True,
                "portability_mode": "scalar_evidence_ledger_no_field_state",
            }
        return {
            "kind": "direct_harmonic_frequency_sweep",
            "name": self.name,
            "frequency_axis": {
                "name": "frequency",
                "unit": "Hz",
                "values": self.frequencies,
                "count": len(self.frequencies),
                "canonical_order": "ascending",
            },
            "execution_order": self.execution_order,
            "completed_points": len(self.records),
            "completed": self.completed,
            "failure": self.failure,
            "execution_event_count": len(self.execution_events),
            "execution_event_capacity": self.execution_event_capacity,
            "execution_events_dropped": self.dropped_execution_events,
            "checkpoint_count": len(self.checkpoints),
            "checkpoint_policy": checkpoint_summary,
            "last_live_field_frequency": self.last_live_field_frequency,
            "field_retention": "last_executed_frequency_only",
            "operator_evolution": "frequency_invariant",
            "load_phasor": "real_spatial_force_times_one_global_phase",
            "maximum_displacement_vector_amplitude_semantics": (
                "maximum physical-cycle vector norm over owned nodal/DOF "
                "coefficient blocks and MPI ranks"
            ),
            "responses": tuple(item.summary() for item in self.responses),
            "procedure": self.procedure.summary(),
            "point_step": self.point_step.summary(),
        }

    def solve_result(self, *, output=None, strict_output: bool = False):
        """Complete the sweep and publish canonical frequency histories."""

        context = getattr(self, "execution_context", None)
        configured_output = (
            None if context is None else getattr(context, "configured_output", None)
        )
        if output is not None or configured_output is not None:
            raise NotImplementedError(
                "Frequency-sweep field output requires explicit snapshot frequencies; "
                "the sweep currently publishes bounded-memory scalar histories."
            )
        from ..results._harmonic import from_harmonic_sweep

        self.solve()
        return from_harmonic_sweep(self, strict_output=strict_output)

    @property
    def closed(self) -> bool:
        """Whether this sweep's retained point-solve backend is closed."""

        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"DirectHarmonicSweepStep {self.name!r} is closed.")

    def close(self) -> None:
        """Release the reusable point solve while preserving scalar records."""

        if self._closed:
            return
        try:
            self.point_step.close()
        finally:
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def harmonic_frequency_sweep_step(
    point_step: DirectHarmonicStep,
    *,
    frequencies,
    responses=(),
    execution_order: str = "forward",
    scientific_assets: dict[str, object] | None = None,
    status_file=None,
    name: str | None = None,
) -> DirectHarmonicSweepStep:
    """Create a reusable ordered sweep around one direct harmonic Step."""

    if not isinstance(point_step, DirectHarmonicStep):
        raise TypeError("harmonic_frequency_sweep_step requires DirectHarmonicStep.")
    return DirectHarmonicSweepStep(
        name=name or point_step.name,
        point_step=point_step,
        frequencies=tuple(frequencies),
        responses=tuple(responses or ()),
        execution_order=execution_order,
        scientific_assets=scientific_assets,
        status_file=status_file,
    )


def direct_harmonic_step(
    *,
    displacement,
    system: DirectHarmonicSystem,
    frequency: float | None = None,
    angular_frequency: float | None = None,
    constraints=(),
    load_phase: float = 0.0,
    study=None,
    solver_options=None,
    name: str = "direct_harmonic",
) -> DirectHarmonicStep:
    """Build one direct harmonic Step from explicit operator contributions."""

    if not isinstance(system, DirectHarmonicSystem):
        raise TypeError("direct_harmonic_step requires DirectHarmonicSystem.")
    system.check()
    if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
        raise NotImplementedError(
            "The real-block harmonic provider requires a real PETSc scalar build."
        )
    omega = _angular_frequency(frequency=frequency, angular_frequency=angular_frequency)
    if omega == 0.0 and system.loss is not None:
        raise ValueError(
            "A material loss operator is undefined at zero cyclic frequency. "
            "Use a positive frequency or omit K_loss for the static limit."
        )
    phase = float(load_phase)
    if not np.isfinite(phase):
        raise ValueError("load_phase must be finite radians.")
    if study is not None and hasattr(study, "require"):
        study.require(analysis="frequency_domain", physics="solid_mechanics")

    solution_real = displacement.value
    solution_real.name = "U_REAL"
    real_space = solution_real.function_space
    solution_imaginary = fem.Function(real_space.clone(), name="U_IMAG")
    selected_bcs = harmonic_strong_bcs(constraints)
    require_homogeneous_harmonic_bcs(solution_real, selected_bcs)
    imaginary_bcs = clone_zero_harmonic_bcs(
        real_space, solution_imaginary.function_space, selected_bcs
    )
    options = solver_options or LinearSolverOptions(
        ksp_type="gmres", pc_type="fieldsplit", rtol=1.0e-11, max_it=500
    )
    if not isinstance(options, LinearSolverOptions):
        raise TypeError("solver_options must be LinearSolverOptions.")
    if options.pc_type.lower() in {"cholesky", "icc"}:
        raise ValueError(
            "Direct harmonic real-block systems are generally indefinite and "
            "may be nonsymmetric; pc_type='cholesky' and pc_type='icc' are "
            "not valid harmonic solver policies. Use LU or a nonsymmetric "
            "iterative preconditioner."
        )
    if options.ksp_type.lower() in {
        "cg",
        "cr",
        "groppcg",
        "minres",
        "pipecg",
        "pipecgrr",
        "pipecr",
        "qcg",
        "symmlq",
    }:
        raise ValueError(
            "Direct harmonic real-block systems do not guarantee the symmetric "
            "operator properties required by "
            f"ksp_type={options.ksp_type!r}. The verified policies are "
            "GMRES/field-split and direct LU."
        )
    return DirectHarmonicStep(
        name=str(name),
        solution_real=solution_real,
        solution_imaginary=solution_imaginary,
        system=system,
        angular_frequency=omega,
        bcs=(*selected_bcs, *imaginary_bcs),
        solver_options=options,
        load_phase=phase,
        study=study,
    )


def harmonic_strong_bcs(constraints) -> tuple[object, ...]:
    selected = []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected.extend(item.bcs)
        elif hasattr(item, "bc"):
            selected.append(item.bc)
        elif callable(getattr(item, "dof_indices", None)):
            selected.append(item)
        else:
            raise NotImplementedError(
                "Direct harmonic response currently supports strong Dirichlet "
                "constraints; MPC and weak constraints require a complex dual contract."
            )
    return tuple(selected)


def require_homogeneous_harmonic_bcs(solution, bcs) -> None:
    if not bcs:
        return
    probe = fem.Function(solution.function_space)
    fem_petsc.set_bc(probe.x.petsc_vec, list(bcs))
    owned = int(
        probe.function_space.dofmap.index_map.size_local
        * probe.function_space.dofmap.index_map_bs
    )
    local = float(np.max(np.abs(probe.x.array[:owned]))) if owned else 0.0
    maximum = float(probe.function_space.mesh.comm.allreduce(local, op=MPI.MAX))
    if maximum > 64.0 * np.finfo(float).eps:
        raise NotImplementedError(
            "Direct harmonic response currently requires homogeneous strong "
            "constraints. Express harmonic excitation as a load phasor."
        )


def clone_zero_harmonic_bcs(source_space, target_space, bcs) -> tuple[object, ...]:
    if not bcs:
        return ()
    cloned = []
    value_size = int(np.prod(source_space.element.value_shape or (1,)))
    for bc in bcs:
        dofs, _owned = bc.dof_indices()
        selected_dofs = np.asarray(dofs, dtype=np.int32).reshape(-1)
        component = next(
            (
                index
                for index in range(value_size)
                if source_space.sub(index)._cpp_object.contains(bc.function_space)
            ),
            None,
        )
        if component is None:
            zero = fem.Function(target_space)
            cloned.append(fem.dirichletbc(zero, selected_dofs))
        else:
            zero = fem.Constant(target_space.mesh, PETSc.ScalarType(0.0))
            cloned.append(
                fem.dirichletbc(zero, selected_dofs, target_space.sub(component))
            )
    return tuple(cloned)


def _prepared_configuration_fingerprint(step: DirectHarmonicStep) -> str:
    """Fingerprint every mutable input captured by a prepared backend.

    This is deliberately a lightweight, process-lifetime guard rather than a
    portable checkpoint identity.  It hashes rank-local coefficient, mesh-tag,
    geometry and boundary-condition contents.  The solve path reduces the
    local drift decision collectively, so repeated frequency points avoid
    gathering a complete distributed mesh while every rank still fails
    together if any partition drifts.
    """

    operator_records = []
    for name, operator in (
        ("storage", step.system.storage),
        ("mass", step.system.mass),
        ("damping", step.system.damping),
        ("loss", step.system.loss),
        ("force", step.system.force),
    ):
        if operator is None:
            operator_records.append((name, None))
            continue
        expression = (
            operator.expression if hasattr(operator, "expression") else operator
        )
        signature = getattr(expression, "signature", None)
        operator_records.append(
            (
                name,
                {
                    "operator_id": id(operator),
                    "expression_id": id(expression),
                    "ufl_signature": (
                        str(signature()) if callable(signature) else None
                    ),
                    "coefficients": [
                        _runtime_value_identity(value)
                        for value in tuple(expression.coefficients())
                    ],
                    "constants": [
                        _runtime_value_identity(value)
                        for value in tuple(expression.constants())
                    ],
                    "subdomain_data": _runtime_subdomain_identity(expression),
                },
            )
        )

    boundary_records = []
    for bc in step.bcs:
        dofs, owned = bc.dof_indices()
        boundary_records.append(
            {
                "bc_id": id(bc),
                "owned_dofs": int(owned),
                "dofs": _runtime_array_identity(np.asarray(dofs)),
                "value": _runtime_value_identity(bc.g),
            }
        )

    domain = step.solution_real.function_space.mesh
    local_record = {
        "schema": "agentfem.prepared-harmonic-runtime-identity.v1",
        "system_id": id(step.system),
        "solution_real_id": id(step.solution_real),
        "solution_imaginary_id": id(step.solution_imaginary),
        "target_element": str(step.solution_real.ufl_element()),
        "geometry": _runtime_array_identity(domain.geometry.x),
        "operators": operator_records,
        "homogeneous_dirichlet": boundary_records,
        "load_phase": float(step.load_phase),
        "solver": step.solver_options.summary(),
        "phasor_convention": step.system.phasor_convention,
    }
    return content_fingerprint(local_record)


def _runtime_subdomain_identity(expression) -> list[dict[str, object]]:
    records = []
    for _domain, by_integral_type in expression.subdomain_data().items():
        for integral_type in sorted(by_integral_type):
            for value in by_integral_type[integral_type]:
                if value is None:
                    continue
                records.append(
                    {
                        "integral_type": str(integral_type),
                        "object_id": id(value),
                        "dimension": getattr(value, "dim", None),
                        "indices": _runtime_array_identity(
                            np.asarray(getattr(value, "indices", ()))
                        ),
                        "values": _runtime_array_identity(
                            np.asarray(getattr(value, "values", ()))
                        ),
                    }
                )
    return records


def _runtime_value_identity(value) -> dict[str, object]:
    record: dict[str, object] = {
        "object_id": id(value),
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    constant_value = getattr(value, "value", None)
    if constant_value is not None:
        record["value"] = _runtime_array_identity(constant_value)
        return record
    vector = getattr(value, "x", None)
    array = None if vector is None else getattr(vector, "array", None)
    if array is not None:
        record["value"] = _runtime_array_identity(array)
    return record


def _runtime_array_identity(value) -> dict[str, object]:
    selected = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": selected.dtype.str,
        "shape": list(selected.shape),
        "sha256": sha256(selected.tobytes(order="C")).hexdigest(),
    }


def _require_finite_harmonic_values(values) -> None:
    nonfinite = [name for name, value in values.items() if not np.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(
            "Direct harmonic evidence contains NaN or Inf in: "
            + ", ".join(nonfinite)
            + ". No energy evidence was accepted."
        )


def _physical_cycle_vector_amplitudes(
    real,
    imaginary,
    *,
    block_size: int,
) -> np.ndarray:
    r"""Return exact ``max_theta ||a cos(theta) - b sin(theta)||`` per block."""

    selected_block_size = int(block_size)
    if selected_block_size <= 0:
        raise ValueError("block_size must be positive.")
    real_values = np.asarray(real, dtype=float).reshape(-1)
    imaginary_values = np.asarray(imaginary, dtype=float).reshape(-1)
    if real_values.shape != imaginary_values.shape:
        raise ValueError("Real and imaginary harmonic arrays must have equal shape.")
    if real_values.size % selected_block_size:
        raise ValueError("Harmonic array size must be divisible by block_size.")
    if not np.all(np.isfinite(real_values)) or not np.all(
        np.isfinite(imaginary_values)
    ):
        raise RuntimeError("Harmonic displacement contains NaN or Inf values.")
    if not real_values.size:
        return np.empty(0, dtype=float)

    a = real_values.reshape((-1, selected_block_size))
    b = imaginary_values.reshape((-1, selected_block_size))
    scale = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1))
    nonzero = scale > 0.0
    normalized_a = np.zeros_like(a)
    normalized_b = np.zeros_like(b)
    normalized_a[nonzero] = a[nonzero] / scale[nonzero, None]
    normalized_b[nonzero] = b[nonzero] / scale[nonzero, None]
    a_squared = np.einsum("ij,ij->i", normalized_a, normalized_a)
    b_squared = np.einsum("ij,ij->i", normalized_b, normalized_b)
    coupling = np.einsum("ij,ij->i", normalized_a, normalized_b)
    largest_gram_eigenvalue = 0.5 * (
        a_squared + b_squared + np.hypot(a_squared - b_squared, 2.0 * coupling)
    )
    amplitudes = scale * np.sqrt(np.maximum(largest_gram_eigenvalue, 0.0))
    if not np.all(np.isfinite(amplitudes)):
        raise FloatingPointError("Harmonic physical-cycle vector amplitude overflowed.")
    return amplitudes


def _angular_frequency(*, frequency, angular_frequency) -> float:
    if (frequency is None) == (angular_frequency is None):
        raise ValueError("Specify exactly one of frequency or angular_frequency.")
    omega = (
        2.0 * np.pi * float(frequency)
        if frequency is not None
        else float(angular_frequency)
    )
    if not np.isfinite(omega) or omega < 0.0:
        raise ValueError("Harmonic frequency must be finite and nonnegative.")
    return omega


def _frequency_axis(values) -> tuple[float, ...]:
    selected = tuple(float(value) for value in values)
    if not selected:
        raise ValueError("A harmonic frequency sweep requires at least one point.")
    if any(not np.isfinite(value) or value < 0.0 for value in selected):
        raise ValueError("Sweep frequencies must be finite and nonnegative.")
    ordered = tuple(sorted(selected))
    if any(right == left for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Sweep frequencies must be unique.")
    return ordered


def _quadratic_integral(operator, value) -> float:
    expression = operator.expression if hasattr(operator, "expression") else operator
    arguments = tuple(expression.arguments())
    if len(arguments) != 2:
        raise ValueError("Harmonic energy evidence requires bilinear operators.")
    scalar = ufl.replace(expression, {argument: value for argument in arguments})
    local = fem.assemble_scalar(fem.form(scalar))
    return float(value.function_space.mesh.comm.allreduce(local, op=MPI.SUM))


def _require_solve_evidence(evidence, *, solver_options, context: str) -> None:
    info = evidence.solve
    if not info.converged and solver_options.error_if_not_converged:
        raise RuntimeError(
            f"{context} did not converge: reason={info.converged_reason}, "
            f"iterations={info.iterations}, residual={info.residual_norm:.6e}."
        )
    relative_tolerance = max(
        10.0 * (solver_options.rtol or 1.0e-8),
        100.0 * np.finfo(float).eps,
    )
    absolute_tolerance = 10.0 * (solver_options.atol or 0.0)
    if (
        evidence.relative_residual_norm > relative_tolerance
        and evidence.residual_norm > absolute_tolerance
        and solver_options.error_if_not_converged
    ):
        raise RuntimeError(
            f"{context} failed the assembled A*x-b check: relative_residual="
            f"{evidence.relative_residual_norm:.6e}, "
            f"tolerance={relative_tolerance:.6e}."
        )


__all__ = [
    "DirectHarmonicStep",
    "DirectHarmonicSweepStep",
    "direct_harmonic_step",
    "harmonic_frequency_sweep_step",
]
