"""Global small-strain J2 plasticity with integration-point state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc

from .. import _axisymmetric
from .. import procedures
from .. import amplitudes
from .. import steps as step_controls
from ..constitutive import elasticity
from ..constitutive.plasticity import J2LinearIsotropicHardening
from ..constitutive.quadrature import (
    J2QuadratureState,
    QuadratureField,
    QuadratureMaterialMap,
)
from ..diagnostics import (
    SolveEventRecorder,
    StandardRunReporter,
    comm_of,
    compose_reporters,
)
from ..solvers import (
    NewtonSolverOptions,
    SolveEvent,
    newton,
    solve_matrix_system,
)


@dataclass(frozen=True)
class J2IncrementInfo:
    increment: int
    attempt: int
    start_load_factor: float
    load_factor: float
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    plastic_points: int
    maximum_plastic_increment: float
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        def finite_or_none(value):
            selected = float(value)
            return selected if np.isfinite(selected) else None

        return {
            "increment": self.increment,
            "attempt": self.attempt,
            "start_load_factor": self.start_load_factor,
            "load_factor": self.load_factor,
            "increment_size": self.load_factor - self.start_load_factor,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": finite_or_none(self.initial_residual_norm),
            "residual_norm": finite_or_none(self.residual_norm),
            "plastic_points": self.plastic_points,
            "maximum_plastic_increment": finite_or_none(
                self.maximum_plastic_increment
            ),
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "J2IncrementInfo":
        """Restore one increment record from a portable checkpoint."""

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
            plastic_points=int(record["plastic_points"]),
            maximum_plastic_increment=(
                float("inf")
                if record["maximum_plastic_increment"] is None
                else float(record["maximum_plastic_increment"])
            ),
            rejection_reason=record.get("rejection_reason"),
        )


@dataclass(frozen=True)
class J2LoadPathInfo:
    increments: tuple[J2IncrementInfo, ...]
    attempts: tuple[J2IncrementInfo, ...]
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
        return self.converged and abs(
            self.increments[-1].load_factor - 1.0
        ) <= 1.0e-12

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "j2_nonlinear_load_path",
            "converged": self.converged,
            "completed_step": self.completed_step,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": [item.as_dict() for item in self.increments],
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass(frozen=True)
class J2EnergyFrame:
    """Accepted energy/work evidence for one path coordinate."""

    step_coordinate: float
    load_amplitude: float
    elastic_strain_energy: float
    isotropic_hardening_energy: float
    plastic_dissipation: float
    internal_energy: float
    generalized_reaction: float | None
    external_work: float | None
    energy_balance_error: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "J2EnergyFrame":
        return cls(
            **{
                field: (
                    None
                    if record[field] is None
                    else float(record[field])
                )
                for field in cls.__dataclass_fields__
            }
        )


@dataclass
class J2PlasticityStep:
    """Incremental global equilibrium for 3D small-strain J2 plasticity."""

    name: str
    solution: object
    material: J2LinearIsotropicHardening | QuadratureMaterialMap
    state: J2QuadratureState
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
    procedure: object = field(default_factory=lambda: procedures.nonlinear_static(stateful=True))
    accepted_load_factor: float = field(default=0.0, init=False)
    last_solve_info: J2LoadPathInfo | None = field(default=None, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    accepted_increments: list[J2IncrementInfo] = field(default_factory=list, init=False)
    attempted_increments: list[J2IncrementInfo] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    energy_history: list[J2EnergyFrame] = field(default_factory=list, init=False)
    next_increment_size: float | None = field(default=None, init=False)
    _strain_evaluator: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._strain_evaluator = self.state.compile_strain(
            elasticity.strain(self.solution, study=self.study)
        )

    def solve(self, *, until: float = 1.0):
        """Advance the load path, optionally stopping at a checkpoint factor."""

        selected_until = float(until)
        if not self.accepted_load_factor < selected_until <= 1.0:
            raise ValueError(
                "until must be greater than the accepted factor and at most 1."
            )
        reporter = self._reporter()
        accepted: list[J2IncrementInfo] = []
        attempts: list[J2IncrementInfo] = []
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
            increment = len(self.accepted_increments) + len(accepted) + 1
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                if (
                    len(self.accepted_increments) + len(accepted)
                    >= self.incrementation.max_increments
                ):
                    raise RuntimeError(
                        "J2 load path reached max_increments before load factor 1."
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
                        "Fixed incrementation has no factor beyond the restored state."
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
            displacement_snapshot = self.solution.x.array.copy()
            state_snapshot = self.state.snapshot()
            self._apply_loading(target)
            info = self._solve_increment(
                increment=increment,
                attempt=attempt,
                start_factor=accepted_factor,
                target_factor=target,
                reporter=reporter,
            )
            if (
                info.converged
                and isinstance(
                    self.incrementation,
                    step_controls.AutomaticIncrementation,
                )
                and self.incrementation.maximum_inelastic_increment is not None
                and info.maximum_plastic_increment
                > self.incrementation.maximum_inelastic_increment
            ):
                info = replace(
                    info,
                    converged=False,
                    rejection_reason=(
                        "maximum equivalent plastic-strain increment "
                        f"{info.maximum_plastic_increment:.6g} exceeds "
                        f"{self.incrementation.maximum_inelastic_increment:.6g}"
                    ),
                )
            attempts.append(info)
            if info.converged:
                self.state.commit()
                accepted.append(info)
                accepted_size = target - accepted_factor
                accepted_factor = target
                self.accepted_load_factor = target
                self._record_energy(target)
                consecutive_cutbacks = 0
                if isinstance(
                    self.incrementation,
                    step_controls.AutomaticIncrementation,
                ):
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
                        start_factor=info.start_load_factor,
                        target_factor=target,
                        iteration=info.iterations,
                        residual_norm=info.residual_norm,
                    ),
                )
                continue

            self.solution.x.array[:] = displacement_snapshot
            self.solution.x.scatter_forward()
            self.state.restore(state_snapshot)
            self._apply_loading(accepted_factor)
            self.state.update(
                self.state.evaluate_strain(self._strain_evaluator),
                self.material,
            )
            if not isinstance(
                self.incrementation,
                step_controls.AutomaticIncrementation,
            ):
                self.accepted_increments.extend(accepted)
                self.attempted_increments.extend(attempts)
                self._fail(reporter, info, "fixed increment did not converge")
                return self.solution
            consecutive_cutbacks += 1
            proposed_size = self.incrementation.after_failure(
                target - accepted_factor
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

        self.accepted_increments.extend(accepted)
        self.attempted_increments.extend(attempts)
        self.last_solve_info = J2LoadPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        self._emit(
            reporter,
            SolveEvent(
                (
                    "step_completed"
                    if accepted_factor >= 1.0 - 1.0e-12
                    else "step_paused"
                ),
                self.name,
                step_number=self.step_number,
                increment=len(self.accepted_increments),
                attempt=len(self.attempted_increments),
                target_factor=accepted_factor,
            ),
        )
        return self.solution

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        """Save displacement, accepted load factor, and committed state."""

        comm = self.solution.function_space.mesh.comm
        selected_portable = comm.size != 1 if portable is None else bool(portable)
        if selected_portable:
            return self._save_portable_checkpoint(path)
        if comm.size != 1:
            raise ValueError("Distributed J2 checkpoints must use portable=True.")
        selected = Path(path)
        if selected.suffix != ".npz":
            selected = selected.with_suffix(".npz")
        selected.parent.mkdir(parents=True, exist_ok=True)
        state = self.state.snapshot()
        identity = self._checkpoint_identity()
        from ..checkpointing import atomic_savez

        atomic_savez(
            selected,
            schema="agentfem.j2-step-checkpoint.v4",
            step_identity=json.dumps(identity, sort_keys=True),
            displacement=self.solution.x.array,
            accepted_load_factor=self.accepted_load_factor,
            amplitude_summary=json.dumps(self.amplitude.summary()),
            plastic_strain=state["plastic_strain"],
            equivalent_plastic_strain=state["equivalent_plastic_strain"],
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
                np.nan
                if self.next_increment_size is None
                else self.next_increment_size
            ),
        )
        from ..results import CheckpointRecord

        record = CheckpointRecord(
            name=f"{self.name}_{self.accepted_load_factor:g}",
            path=selected,
            schema="agentfem.j2-step-checkpoint.v4",
            step_name=self.name,
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=False,
            metadata={
                "reason": "serial dof and quadrature layout checkpoint",
                "state_variables": ("U", "PE", "PEEQ"),
                "amplitude": self.amplitude.summary(),
                "identity": identity,
            },
        )
        record.write_manifest()
        self.checkpoints.append(record)
        return selected

    def load_checkpoint(self, path) -> None:
        """Restore a serial checkpoint into the same mesh/function layout."""

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
            if payload.get("schema") == "agentfem.j2-step-checkpoint.v5":
                self._load_portable_checkpoint(manifest, payload)
                return
        if self.solution.function_space.mesh.comm.size != 1:
            raise ValueError(
                "This legacy J2 checkpoint is partition-bound; use a v5 "
                "portable checkpoint for distributed restart."
            )
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"])
            if schema not in {
                "agentfem.j2-step-checkpoint.v1",
                "agentfem.j2-step-checkpoint.v2",
                "agentfem.j2-step-checkpoint.v3",
                "agentfem.j2-step-checkpoint.v4",
            }:
                raise ValueError("Unsupported J2 step checkpoint schema.")
            displacement = np.asarray(data["displacement"])
            if schema == "agentfem.j2-step-checkpoint.v4":
                stored_identity = json.loads(str(data["step_identity"]))
                current_identity = json.loads(
                    json.dumps(self._checkpoint_identity(), sort_keys=True)
                )
                if stored_identity != current_identity:
                    raise ValueError(
                        "J2 checkpoint material, procedure, increment control, "
                        "quadrature state, or mesh/function layout differs from "
                        "the current step."
                    )
            if displacement.size != self.solution.x.array.size:
                raise ValueError("Checkpoint displacement layout does not match.")
            self.solution.x.array[:] = displacement
            self.solution.x.scatter_forward()
            self.state.restore(
                {
                    "plastic_strain": data["plastic_strain"],
                    "equivalent_plastic_strain": data[
                        "equivalent_plastic_strain"
                    ],
                }
            )
            self.accepted_load_factor = float(data["accepted_load_factor"])
            if "amplitude_summary" in data:
                restored_amplitude = json.loads(str(data["amplitude_summary"]))
                if restored_amplitude != self.amplitude.summary():
                    raise ValueError(
                        "Checkpoint load amplitude differs from the current step."
                    )
            self._apply_loading(self.accepted_load_factor)
            self.accepted_increments.clear()
            self.attempted_increments.clear()
            self.execution_events.clear()
            self.energy_history.clear()
            self.next_increment_size = None
            if schema in {
                "agentfem.j2-step-checkpoint.v2",
                "agentfem.j2-step-checkpoint.v3",
                "agentfem.j2-step-checkpoint.v4",
            }:
                self.accepted_increments.extend(
                    J2IncrementInfo.from_dict(item)
                    for item in json.loads(str(data["accepted_increments"]))
                )
                self.attempted_increments.extend(
                    J2IncrementInfo.from_dict(item)
                    for item in json.loads(str(data["attempted_increments"]))
                )
                self.execution_events.extend(
                    SolveEvent.from_dict(item)
                    for item in json.loads(str(data["execution_events"]))
                )
                if "energy_history" in data:
                    self.energy_history.extend(
                        J2EnergyFrame.from_dict(item)
                        for item in json.loads(str(data["energy_history"]))
                    )
                if "next_increment_size" in data:
                    selected_size = float(data["next_increment_size"])
                    self.next_increment_size = (
                        selected_size if np.isfinite(selected_size) else None
                    )
            self.last_solve_info = J2LoadPathInfo(
                tuple(self.accepted_increments),
                tuple(self.attempted_increments),
                self.incrementation,
            )
            self.state.update(
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
            "schema": "agentfem.j2-step-checkpoint.v5",
            "step_identity": self._portable_checkpoint_identity(),
            "coordinate": float(self.accepted_load_factor),
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
            raise RuntimeError(f"J2 checkpoint manifest write failed: {error}")
        comm.barrier()
        from ..results import CheckpointRecord

        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_{self.accepted_load_factor:g}",
                path=manifest,
                schema="agentfem.j2-step-checkpoint.v5",
                step_name=self.name,
                coordinate_name="load_factor",
                coordinate_value=self.accepted_load_factor,
                portable=True,
                metadata={"state_variables": ("U", "PE", "PEEQ")},
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
            raise ValueError("Portable J2 checkpoint scientific identity differs.")
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
            J2IncrementInfo.from_dict(item) for item in payload["accepted_increments"]
        ]
        self.attempted_increments[:] = [
            J2IncrementInfo.from_dict(item) for item in payload["attempted_increments"]
        ]
        self.execution_events[:] = [
            SolveEvent.from_dict(item) for item in payload["execution_events"]
        ]
        self.energy_history[:] = [
            J2EnergyFrame.from_dict(item) for item in payload["energy_history"]
        ]
        self.next_increment_size = payload.get("next_increment_size")
        self._apply_loading(self.accepted_load_factor)
        self.last_solve_info = J2LoadPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        self.state.update(
            self.state.evaluate_strain(self._strain_evaluator), self.material
        )

    def _portable_checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "amplitude": self.amplitude.summary(),
            "incrementation": self.incrementation.summary(),
            "solution": function_portable_identity(self.solution),
            "quadrature": self.state.summary()["transaction"],
        }

    def _checkpoint_identity(self) -> dict[str, object]:
        """Return the stateful procedure identity required for safe restart."""

        from ..checkpointing import function_partition_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "incrementation": self.incrementation.summary(),
            "solution": function_partition_identity(self.solution),
            "plastic_strain": function_partition_identity(
                self.state.plastic_strain.function
            ),
            "equivalent_plastic_strain": function_partition_identity(
                self.state.equivalent_plastic_strain.function
            ),
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
        """Solve and publish constitutive state through the common result path.

        Raw integration-point state remains attached to the result.  When
        ``output`` is requested, the primary solution, recovered ``*_CELL``
        fields, and nodal reaction field share the standard completed-result
        XDMF/HDF5 writer.
        """
        from ..results import (
            add_execution_trace,
            complete_result,
            from_solution,
            recover_integration_point_field,
        )

        if fields and output_fields:
            raise ValueError("Pass fields=... or output_fields=..., not both.")
        selected_output_fields = tuple(fields) or tuple(output_fields)

        solution = (
            self.solve()
            if self.accepted_load_factor < 1.0 - 1.0e-12
            else self.solution
        )
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
        result.add_field(
            "S",
            self.state.stress.function,
            location="quadrature_points",
            description="Cauchy stress at constitutive integration points.",
            processing={
                "source_position": "quadrature_points",
                "method": "constitutive_update",
                "representation": "quadrature_values",
                "postprocessed": False,
                "accepted": True,
            },
        )
        result.add_field(
            "PE",
            self.state.plastic_strain.function,
            location="quadrature_points",
            description="Plastic strain at constitutive integration points.",
            processing={
                "source_position": "quadrature_points",
                "method": "constitutive_state",
                "representation": "quadrature_values",
                "postprocessed": False,
                "committed": True,
            },
        )
        result.add_field(
            "PEEQ",
            self.state.equivalent_plastic_strain.function,
            location="quadrature_points",
            description="Equivalent plastic strain at integration points.",
            processing={
                "source_position": "quadrature_points",
                "method": "constitutive_state",
                "representation": "quadrature_values",
                "postprocessed": False,
                "committed": True,
            },
        )
        result.add_field(
            "MISES",
            self.state.equivalent_stress().function,
            location="quadrature_points",
            description="Pointwise von Mises invariant of quadrature stress.",
            processing={
                "source_position": "quadrature_points",
                "method": "pointwise_invariant",
                "representation": "quadrature_values",
                "derived_from": ("S",),
                "nodal_extrapolation": False,
                "interelement_smoothing": False,
            },
        )
        for source, recovered_name in (
            (self.state.stress, "S_CELL"),
            (self.state.plastic_strain, "PE_CELL"),
            (self.state.equivalent_plastic_strain, "PEEQ_CELL"),
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
            processing={
                "method": "assembled_equilibrium_residual",
                "representation": "finite_element_dofs",
                "postprocessed": False,
            },
        )
        reaction_map = reaction.function_space.dofmap.index_map
        reaction_owned = int(reaction_map.size_local) * int(
            reaction.function_space.dofmap.index_map_bs
        )
        reaction_local_squared = float(
            np.vdot(
                reaction.x.array[:reaction_owned],
                reaction.x.array[:reaction_owned],
            ).real
        )
        result.add_quantities(
            {
                "maximum_equivalent_plastic_strain": (
                    self.state.equivalent_plastic_strain.global_max()
                ),
                "plastic_integration_points": (
                    self.state.equivalent_plastic_strain.global_count_nonzero(
                        tolerance=1.0e-14
                    )
                ),
                "reaction_l2_norm": float(
                    np.sqrt(
                        self.state.domain.comm.allreduce(
                            reaction_local_squared
                        )
                    )
                ),
                **self.internal_energy(),
            },
            kind="diagnostic",
        )
        accepted = self.last_solve_info.increments
        if accepted:
            coordinates = np.asarray(
                [item.load_factor for item in accepted], dtype=float
            )
            result.add_history(
                "load_amplitude",
                coordinates,
                np.asarray([self.amplitude(value) for value in coordinates]),
                abscissa_name="step_coordinate",
                abscissa_unit=None,
                description=(
                    "Applied load/Dirichlet scale, which may be non-monotone "
                    "for cyclic paths."
                ),
            )
        if self.energy_history:
            coordinates = np.asarray(
                [item.step_coordinate for item in self.energy_history],
                dtype=float,
            )
            result.add_histories(
                coordinates,
                {
                    "elastic_strain_energy": [
                        item.elastic_strain_energy for item in self.energy_history
                    ],
                    "isotropic_hardening_energy": [
                        item.isotropic_hardening_energy
                        for item in self.energy_history
                    ],
                    "plastic_dissipation": [
                        item.plastic_dissipation for item in self.energy_history
                    ],
                    "internal_energy": [
                        item.internal_energy for item in self.energy_history
                    ],
                },
                abscissa_name="step_coordinate",
                abscissa_unit=None,
            )
            if all(item.external_work is not None for item in self.energy_history):
                result.add_histories(
                    coordinates,
                    {
                        "external_work": [
                            item.external_work for item in self.energy_history
                        ],
                        "energy_balance_error": [
                            item.energy_balance_error
                            for item in self.energy_history
                        ],
                        "generalized_reaction": [
                            item.generalized_reaction
                            for item in self.energy_history
                        ],
                    },
                    abscissa_name="step_coordinate",
                    abscissa_unit=None,
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

    def internal_energy(self) -> dict[str, float]:
        """Return elastic, hardening, dissipated, and total internal energy.

        For linear isotropic hardening the hardening contribution is treated
        as stored energy and ``yield_stress * PEEQ`` as rate-independent
        plastic dissipation.  This is a state diagnostic, not an external-work
        balance for a non-proportional loading history.
        """

        strain = elasticity.strain(self.solution, study=self.study)
        weight = _axisymmetric.integration_weight(self.solution, self.study)
        plastic_strain = self.state.plastic_strain.function
        peeq = self.state.equivalent_plastic_strain.function
        stress = self.state.stress.function
        elastic_density = 0.5 * ufl.inner(
            stress,
            strain - plastic_strain,
        )
        if isinstance(self.material, QuadratureMaterialMap):
            hardening = QuadratureField.create(
                self.state.domain,
                name="HARDENING_MODULUS",
                degree=self.state.degree,
                scheme=self.state.scheme,
            )
            yield_stress = QuadratureField.create(
                self.state.domain,
                name="YIELD_STRESS",
                degree=self.state.degree,
                scheme=self.state.scheme,
            )
            points_per_cell = len(self.state.stress.points)
            regions = np.repeat(self.material.cell_regions, points_per_cell)
            hardening.assign(
                [self.material.materials[int(region)].hardening_modulus for region in regions]
            )
            yield_stress.assign(
                [self.material.materials[int(region)].yield_stress for region in regions]
            )
            hardening_coefficient = hardening.function
            yield_coefficient = yield_stress.function
        else:
            hardening_coefficient = self.material.hardening_modulus
            yield_coefficient = self.material.yield_stress
        hardening_density = 0.5 * hardening_coefficient * peeq**2
        dissipation_density = yield_coefficient * peeq
        values = []
        for density in (
            elastic_density,
            hardening_density,
            dissipation_density,
        ):
            local = fem.assemble_scalar(
                fem.form(weight * density * self.state.measure)
            )
            values.append(float(self.state.domain.comm.allreduce(local)))
        elastic, hardening, dissipation = values
        return {
            "elastic_strain_energy": elastic,
            "isotropic_hardening_energy": hardening,
            "plastic_dissipation": dissipation,
            "internal_energy": elastic + hardening + dissipation,
        }

    def reaction_field(self, *, name: str = "RF"):
        """Return the converged full residual as a nodal reaction field."""

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

    def summary(self) -> dict[str, object]:
        return {
            "kind": "j2_plasticity_step",
            "name": self.name,
            "study": (
                None
                if self.study is None
                else self.study.summary()
            ),
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "state": self.state.summary(),
            "incrementation": self.incrementation.summary(),
            "solver": self.solver_options.summary(),
            "num_bcs": len(self.bcs),
            "loading": {
                "kind": self.amplitude.kind,
                "amplitude": self.amplitude.summary(),
                "natural_loads": True,
                "prescribed_values": len(self.prescribed_values),
            },
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "accepted_load_factor": self.accepted_load_factor,
            "energy_frame_count": len(self.energy_history),
            "next_increment_size": self.next_increment_size,
        }

    def _solve_increment(
        self,
        *,
        increment: int,
        attempt: int,
        start_factor: float,
        target_factor: float,
        reporter,
    ) -> J2IncrementInfo:
        initial_norm = None
        norm = float("inf")
        update_info = {
            "plastic_points": 0,
            "maximum_plastic_increment": 0.0,
        }
        converged = False
        iteration = 0
        for iteration in range(self.solver_options.maximum_iterations + 1):
            update_info = self.state.update(
                self.state.evaluate_strain(self._strain_evaluator),
                self.material,
            )
            rhs, norm = self._correction_rhs()
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
                        step_length=0.0,
                        message=(
                            "linear correction failed: "
                            f"KSP reason {linear_info.converged_reason}"
                        ),
                    ),
                )
                break
            base = self.solution.x.array.copy()
            direction = correction.array_r.copy()
            correction.destroy()
            alpha = self._line_search(base, direction, norm)
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
        return J2IncrementInfo(
            increment=increment,
            attempt=attempt,
            start_load_factor=start_factor,
            load_factor=target_factor,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=float(initial_norm or 0.0),
            residual_norm=float(norm),
            plastic_points=int(update_info["plastic_points"]),
            maximum_plastic_increment=float(
                update_info["maximum_plastic_increment"]
            ),
        )

    def _line_search(self, base, direction, base_norm: float) -> float:
        options = self.solver_options
        alpha = 1.0
        if options.line_search in {None, "basic"}:
            self._assign_trial(base, direction, alpha)
            return alpha
        while alpha + 1.0e-15 >= options.minimum_step_length:
            self._assign_trial(base, direction, alpha)
            self.state.update(
                self.state.evaluate_strain(self._strain_evaluator),
                self.material,
            )
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

    def _reporter(self):
        recorder = SolveEventRecorder(self.execution_events)
        if self.progress is True:
            return compose_reporters(
                recorder,
                StandardRunReporter(
                    comm_of(self.solution),
                    status_file=self.status_file,
                ),
            )
        if self.progress in (False, None):
            return recorder
        return compose_reporters(recorder, self.progress)

    def _set_prescribed_factor(self, factor: float) -> None:
        """Scale nonzero engineering Dirichlet targets with the step factor.

        DOLFINx boundary-condition objects retain references to their backing
        constants, so updating those constants changes the constrained value
        without rebuilding forms.  Homogeneous supports remain zero.  Raw
        backend BC objects are accepted but cannot be amplitude-controlled;
        model/constraint objects retain the required scientific intent.
        """

        for constant, target, _bc in self.prescribed_values:
            selected = float(factor) * target
            constant.value = (
                PETSc.ScalarType(selected.item())
                if selected.ndim == 0 or selected.size == 1
                else np.asarray(selected, dtype=PETSc.ScalarType)
            )

    def _apply_loading(self, step_coordinate: float) -> None:
        value = self.amplitude(step_coordinate)
        self.load_factor.value = PETSc.ScalarType(value)
        self._set_prescribed_factor(value)

    def _record_energy(self, step_coordinate: float) -> None:
        energies = self.internal_energy()
        load_amplitude = self.amplitude(step_coordinate)
        generalized = self._generalized_reaction()
        if generalized is None:
            external_work = None
            balance = None
        else:
            previous_amplitude = 0.0
            previous_generalized = 0.0
            previous_work = 0.0
            if self.energy_history:
                previous = self.energy_history[-1]
                previous_amplitude = previous.load_amplitude
                previous_generalized = float(previous.generalized_reaction)
                previous_work = float(previous.external_work)
            external_work = previous_work + 0.5 * (
                previous_generalized + generalized
            ) * (load_amplitude - previous_amplitude)
            balance = external_work - energies["internal_energy"]
        self.energy_history.append(
            J2EnergyFrame(
                step_coordinate=float(step_coordinate),
                load_amplitude=float(load_amplitude),
                generalized_reaction=generalized,
                external_work=external_work,
                energy_balance_error=balance,
                **energies,
            )
        )

    def _generalized_reaction(self) -> float | None:
        active = [
            (target, bc)
            for _constant, target, bc in self.prescribed_values
            if np.any(np.abs(target) > 0.0)
        ]
        if not active:
            return None
        reaction = self.reaction_field().x.array
        generalized = 0.0
        for target, bc in active:
            dofs, owned = bc.dof_indices()
            selected = np.asarray(dofs[:owned], dtype=np.int32)
            scale = float(np.asarray(target).reshape(-1)[0])
            generalized += float(np.sum(reaction[selected])) * scale
        return float(self.state.domain.comm.allreduce(generalized))

    @staticmethod
    def _emit(reporter, event) -> None:
        if reporter is None:
            return
        reporter.emit(event) if hasattr(reporter, "emit") else reporter(event)

    def _fail(self, reporter, info: J2IncrementInfo, message: str) -> None:
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
        self.last_solve_info = J2LoadPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        if self.solver_options.error_if_not_converged:
            raise RuntimeError(f"{self.name}: {message}.")


def j2_plasticity_step(
    *,
    displacement,
    material,
    external_force,
    constraints=(),
    study=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    progress=True,
    status_file=None,
    amplitude=None,
    name: str = "j2_plasticity",
    _experimental_distributed: bool = False,
) -> J2PlasticityStep:
    """Build a global 3D or axisymmetric J2 step."""

    if not isinstance(material, (J2LinearIsotropicHardening, QuadratureMaterialMap)):
        raise TypeError(
            "j2_plasticity_step requires J2LinearIsotropicHardening or a "
            "QuadratureMaterialMap of that family."
        )
    domain = displacement.value.function_space.mesh
    axisymmetric = _axisymmetric.is_axisymmetric(study)
    if domain.geometry.dim != 3 and not (
        domain.geometry.dim == 2 and axisymmetric
    ):
        raise NotImplementedError(
            "Global J2 supports 3D or 2D axisymmetric small-strain solids. "
            "Plane stress needs a separate local return-map constraint."
        )
    if isinstance(material, QuadratureMaterialMap) and material.domain is not domain:
        raise ValueError("J2 quadrature material map belongs to a different mesh.")
    # ``_experimental_distributed`` is retained temporarily for source
    # compatibility with development cases. Distributed equilibrium is now
    # public after partition-interface, cross-rank restart, and external
    # thick-cylinder structural acceptance.
    state = J2QuadratureState.create(domain, degree=quadrature_degree)
    selected_amplitude = amplitudes.ramp() if amplitude is None else amplitudes.as_amplitude(
        amplitude,
        name="j2_load_amplitude",
    )
    if not np.isclose(selected_amplitude(0.0), 0.0):
        raise ValueError("A J2 load amplitude must start at zero.")
    load_factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    strain_test = elasticity.strain(displacement.test, study=study)
    strain_trial = elasticity.strain(displacement.trial, study=study)
    stress = state.stress.function
    tangent = state.tangent.function
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        tangent[i, j, k, l] * strain_trial[k, l],
        (i, j),
    )
    weight = _axisymmetric.integration_weight(displacement.value, study)
    residual = weight * ufl.inner(stress, strain_test) * state.measure
    if external_force is not None:
        residual -= load_factor * external_force.expression
    jacobian = weight * ufl.inner(tangent_action, strain_test) * state.measure
    selected_bcs = []
    prescribed_values = []
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
            value = getattr(item, "value", None)
            if value is not None and hasattr(value, "value"):
                prescribed_values.append(
                    (
                        value,
                        np.asarray(value.value, dtype=float).copy(),
                        item.bc,
                    )
                )
        else:
            selected_bcs.append(item)
    return J2PlasticityStep(
        name=name,
        solution=displacement.value,
        material=material,
        state=state,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        load_factor=load_factor,
        amplitude=selected_amplitude,
        bcs=tuple(selected_bcs),
        prescribed_values=tuple(prescribed_values),
        incrementation=step_controls.normalize(incrementation),
        solver_options=newton() if solver_options is None else solver_options,
        study=study,
        progress=progress,
        status_file=status_file,
    )
