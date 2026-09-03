"""Total-Lagrangian finite-strain plasticity equilibrium providers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc
from mpi4py import MPI

from .. import amplitudes
from .. import steps as step_controls
from ..constitutive import FiniteStrainJ2Logarithmic
from ..constitutive import MaterialQuadratureResponse
from ..constitutive.quadrature import QuadratureField, QuadratureMaterialMap
from ..solvers import NewtonSolverOptions, SolveEvent, newton, solve_matrix_system


def _raise_collective_transaction_problem(comm, local_problem, *, context: str) -> None:
    """Raise the same trial-state failure on every participating MPI rank."""

    problems = comm.allgather(local_problem)
    if not any(problem is not None for problem in problems):
        return
    rank = next(index for index, problem in enumerate(problems) if problem is not None)
    raise RuntimeError(f"Rank {rank}: {context} failed: {problems[rank]}")


@dataclass(frozen=True)
class FiniteStrainPlasticityIncrementInfo:
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
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, record) -> "FiniteStrainPlasticityIncrementInfo":
        return cls(
            increment=int(record["increment"]),
            attempt=int(record["attempt"]),
            start_load_factor=float(record["start_load_factor"]),
            load_factor=float(record["load_factor"]),
            converged=bool(record["converged"]),
            iterations=int(record["iterations"]),
            initial_residual_norm=float(record["initial_residual_norm"]),
            residual_norm=float(record["residual_norm"]),
            plastic_points=int(record["plastic_points"]),
            maximum_plastic_increment=float(record["maximum_plastic_increment"]),
            rejection_reason=record.get("rejection_reason"),
        )


@dataclass(frozen=True)
class FiniteStrainPlasticityPathInfo:
    """Accepted and attempted increments for a standard finite-strain J2 path."""

    increments: tuple[FiniteStrainPlasticityIncrementInfo, ...]
    attempts: tuple[FiniteStrainPlasticityIncrementInfo, ...]
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
            "kind": "finite_strain_j2_standard_load_path",
            "converged": self.converged,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": tuple(item.as_dict() for item in self.increments),
            "attempts": tuple(item.as_dict() for item in self.attempts),
        }


@dataclass
class FiniteStrainJ2StateTransaction:
    """Constraint-neutral trial/commit state for finite-strain J2 Newton.

    An equilibrium provider owns kinematic enforcement; this transaction owns
    only constitutive state and response fields. Keeping those responsibilities
    separate lets ordinary strong boundaries, serial affine elimination, and
    distributed ``dolfinx_mpc`` consume exactly the same material update.
    """

    solution: object
    accepted_solution: object
    material: FiniteStrainJ2Logarithmic | QuadratureMaterialMap
    response: MaterialQuadratureResponse
    deformation_gradient_old: object
    deformation_gradient_new: object
    deformation_gradient: QuadratureField
    equivalent_stress: QuadratureField
    accepted_factor: float = field(default=0.0, init=False)
    last_plastic_points: int = field(default=0, init=False)
    last_maximum_plastic_increment: float = field(default=0.0, init=False)

    def initialize(self) -> None:
        """Restore the declared initial state before the first snapshot."""

        if abs(self.accepted_factor) > 1.0e-15:
            raise RuntimeError(
                "This finite-strain J2 step has already accepted material "
                "history. Create a new Step for a fresh path or restore a "
                "checkpoint through the restart lifecycle."
            )
        self.accepted_solution.x.array[:] = self.solution.x.array
        self.accepted_solution.x.scatter_forward()
        self.response.rollback()
        self.refresh_trial(start_factor=0.0, target_factor=0.0)

    def prepare_resume(self) -> None:
        """Rebuild trial response fields from one restored accepted boundary."""

        self.solution.x.array[:] = self.accepted_solution.x.array
        self.solution.x.scatter_forward()
        self.response.rollback()
        self.refresh_trial(
            start_factor=self.accepted_factor,
            target_factor=self.accepted_factor,
        )

    def _evaluate_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.response.state.evaluate_expression(
                self.deformation_gradient_old,
                value_shape=(3, 3),
            ),
            self.response.state.evaluate_expression(
                self.deformation_gradient_new,
                value_shape=(3, 3),
            ),
        )

    def refresh_trial(
        self,
        *,
        start_factor: float,
        target_factor: float,
    ):
        comm = self.response.domain.comm
        old_gradient = None
        new_gradient = None
        committed_peeq = None
        setup_problem = None
        try:
            old_gradient, new_gradient = self._evaluate_gradients()
            committed_peeq = np.asarray(
                self.response.state.committed[
                    "equivalent_plastic_strain"
                ].values,
                dtype=float,
            ).reshape(-1)
        except Exception as exc:
            setup_problem = f"{type(exc).__name__}: {exc}"
        _raise_collective_transaction_problem(
            comm,
            setup_problem,
            context="finite-strain J2 trial setup",
        )
        result = self.response.update(
            self.material,
            deformation_gradient_old=old_gradient,
            deformation_gradient_new=new_gradient,
            time=float(target_factor),
            time_increment=max(
                float(target_factor) - float(start_factor),
                np.finfo(float).eps,
            ),
            commit=False,
        )
        equivalent_stress = None
        local_count = 0
        local_maximum = 0.0
        diagnostics_problem = None
        try:
            stress = np.asarray(result.cauchy_stress, dtype=float)
            mean = np.trace(stress, axis1=1, axis2=2) / 3.0
            deviator = stress - mean[:, None, None] * np.eye(3)
            equivalent_stress = np.sqrt(
                np.maximum(
                    0.0,
                    1.5 * np.sum(deviator * deviator, axis=(1, 2)),
                )
            )
            trial_peeq = np.asarray(
                self.response.state.trial[
                    "equivalent_plastic_strain"
                ].values,
                dtype=float,
            ).reshape(-1)
            increments = trial_peeq - committed_peeq
            point_count = len(self.response.state.reference_field.points)
            cell_map = self.response.domain.topology.index_map(
                self.response.domain.topology.dim
            )
            owned_points = int(cell_map.size_local) * point_count
            local_count = int(
                np.count_nonzero(increments[:owned_points] > 1.0e-14)
            )
            local_maximum = float(
                np.max(increments[:owned_points], initial=0.0)
            )
        except Exception as exc:
            diagnostics_problem = f"{type(exc).__name__}: {exc}"
        _raise_collective_transaction_problem(
            comm,
            diagnostics_problem,
            context="finite-strain J2 trial diagnostics",
        )
        self.deformation_gradient.assign(new_gradient)
        self.equivalent_stress.assign(equivalent_stress)
        self.last_plastic_points = int(
            comm.allreduce(local_count, op=MPI.SUM)
        )
        self.last_maximum_plastic_increment = float(
            comm.allreduce(local_maximum, op=MPI.MAX)
        )
        return result

    def commit_increment(
        self,
        *,
        start_factor: float,
        target_factor: float,
    ) -> None:
        del start_factor
        self.response.commit()
        self.accepted_solution.x.array[:] = self.solution.x.array
        self.accepted_solution.x.scatter_forward()
        self.accepted_factor = float(target_factor)

    def rollback_increment(self, *, accepted_factor: float) -> None:
        self.response.rollback()
        self.accepted_factor = float(accepted_factor)
        self.refresh_trial(
            start_factor=self.accepted_factor,
            target_factor=self.accepted_factor,
        )

    def snapshot_accepted_boundary(self) -> dict[str, object]:
        """Capture the compact state needed to undo a provisional commit.

        Copying the 81-component material tangent at every increment would be
        prohibitive for a large RVE.  The committed internal variables and the
        accepted nodal field are the irreducible history; all response fields
        are deterministically reconstructed from them during restoration.
        """

        return {
            "accepted_factor": float(self.accepted_factor),
            "accepted_solution": self.accepted_solution.x.array.copy(),
            "committed_state": self.response.snapshot(),
            "last_plastic_points": int(self.last_plastic_points),
            "last_maximum_plastic_increment": float(
                self.last_maximum_plastic_increment
            ),
        }

    def restore_accepted_boundary(self, snapshot: dict[str, object]) -> None:
        """Restore one committed boundary after output/checkpoint failure."""

        self.response.restore(snapshot["committed_state"])
        self.accepted_solution.x.array[:] = snapshot["accepted_solution"]
        self.accepted_solution.x.scatter_forward()
        self.accepted_factor = float(snapshot["accepted_factor"])
        self.prepare_resume()
        self.last_plastic_points = int(snapshot["last_plastic_points"])
        self.last_maximum_plastic_increment = float(
            snapshot["last_maximum_plastic_increment"]
        )

    def snapshot_runtime_state(self) -> dict[str, object]:
        """Capture the complete mutable transaction boundary for recovery."""

        return {
            "accepted_factor": float(self.accepted_factor),
            "last_plastic_points": int(self.last_plastic_points),
            "last_maximum_plastic_increment": float(
                self.last_maximum_plastic_increment
            ),
            "committed_state": self.response.snapshot(),
            "trial_state": {
                name: np.asarray(selected.values).copy()
                for name, selected in self.response.state.trial.items()
            },
            "first_piola_stress": self.response.first_piola_stress.values.copy(),
            "cauchy_stress": self.response.cauchy_stress.values.copy(),
            "tangent": self.response.tangent.values.copy(),
            "strain_energy_density": (
                self.response.strain_energy_density.values.copy()
            ),
            "stored_energy_density_components": {
                name: field.values.copy()
                for name, field in (
                    self.response.stored_energy_density_components.items()
                )
            },
            "deformation_gradient": self.deformation_gradient.values.copy(),
            "equivalent_stress": self.equivalent_stress.values.copy(),
        }

    def restore_runtime_state(self, snapshot: dict[str, object]) -> None:
        """Restore state and derived response fields after a failed operation."""

        self.response.restore(snapshot["committed_state"])
        for name, selected in self.response.state.trial.items():
            selected.assign(snapshot["trial_state"][name])
        self.response.first_piola_stress.assign(
            snapshot["first_piola_stress"]
        )
        self.response.cauchy_stress.assign(snapshot["cauchy_stress"])
        self.response.tangent.assign(snapshot["tangent"])
        self.response.strain_energy_density.assign(
            snapshot["strain_energy_density"]
        )
        for name, field in self.response.stored_energy_density_components.items():
            field.assign(snapshot["stored_energy_density_components"][name])
        self.deformation_gradient.assign(snapshot["deformation_gradient"])
        self.equivalent_stress.assign(snapshot["equivalent_stress"])
        self.accepted_factor = float(snapshot["accepted_factor"])
        self.last_plastic_points = int(snapshot["last_plastic_points"])
        self.last_maximum_plastic_increment = float(
            snapshot["last_maximum_plastic_increment"]
        )

    def snapshot(self) -> dict[str, object]:
        """Return the complete mutable boundary for generic state ownership."""

        return self.snapshot_runtime_state()

    def restore(self, snapshot: dict[str, object]) -> None:
        """Restore a generic state snapshot without changing increment policy."""

        self.restore_runtime_state(snapshot)

    def snapshot_fields(self) -> dict[str, object]:
        fields = {
            "F": self.deformation_gradient,
            "P": self.response.first_piola_stress,
            "S": self.response.cauchy_stress,
            "MISES": self.equivalent_stress,
            "SENER": self.response.strain_energy_density,
            "FP": self.response.state.committed["plastic_deformation_gradient"],
            "PEEQ": self.response.state.committed["equivalent_plastic_strain"],
            "PDENER": self.response.state.committed["plastic_dissipation"],
        }
        fields.update(self.response.stored_energy_density_components)
        return fields

    def populate_result(self, result) -> tuple[object, ...]:
        from ..results import recover_integration_point_field

        descriptions = {
            "F": "Deformation gradient at accepted integration points.",
            "P": "First Piola stress at accepted integration points.",
            "S": "Cauchy stress at accepted integration points.",
            "MISES": "Von Mises stress at accepted integration points.",
            "SENER": "Stored elastic and hardening energy density.",
            "ELENER": "Recoverable Hencky elastic free-energy density.",
            "HARDENER": "Stored linear-isotropic-hardening free-energy density.",
            "FP": "Committed plastic deformation gradient.",
            "PEEQ": "Committed equivalent plastic strain.",
            "PDENER": (
                "Committed cumulative irrecoverable plastic dissipation "
                "density per reference volume."
            ),
        }
        recovered_fields = []
        for name, source in self.snapshot_fields().items():
            result.add_field(
                name,
                source.function,
                location="quadrature_points",
                description=descriptions[name],
                processing={
                    "source_position": "quadrature_points",
                    "method": (
                        "constitutive_state" if name in {"FP", "PEEQ", "PDENER"}
                        else "accepted_constitutive_response"
                    ),
                    "representation": "quadrature_values",
                    "postprocessed": False,
                    "committed": name in {"FP", "PEEQ", "PDENER"},
                },
            )
            recovered = recover_integration_point_field(
                source,
                name=f"{name}_CELL",
            )
            result.add_field(
                recovered.name,
                recovered.field,
                unit=recovered.unit,
                location=recovered.location,
                description=recovered.description,
                processing=recovered.processing,
            )
            recovered_fields.append(recovered.field)
        peeq = self.response.state.committed["equivalent_plastic_strain"]
        pdener = self.response.state.committed["plastic_dissipation"]
        weights = pdener.owned_physical_weights()
        local_dissipation = float(
            np.dot(weights, np.asarray(pdener.owned_values).reshape(-1))
        )
        plastic_dissipation = float(
            pdener.function.function_space.mesh.comm.allreduce(
                local_dissipation,
                op=MPI.SUM,
            )
        )
        result.add_quantities(
            {
                "maximum_equivalent_plastic_strain": peeq.global_max(),
                "plastic_integration_points": peeq.global_count_nonzero(
                    tolerance=1.0e-14
                ),
                "accepted_load_factor": self.accepted_factor,
                "plastic_dissipation": plastic_dissipation,
            },
            kind="diagnostic",
        )
        return tuple(recovered_fields)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_j2_state_transaction",
            "accepted_factor": self.accepted_factor,
            "material": self.material.summary(),
            "response": self.response.summary(),
            "last_plastic_points": self.last_plastic_points,
            "last_maximum_plastic_increment": (
                self.last_maximum_plastic_increment
            ),
            "energy_scope": (
                "SENER = ELENER + HARDENER is recoverable stored energy; "
                "PDENER is cumulative irrecoverable plastic dissipation"
            ),
        }


@dataclass
class FiniteStrainJ2StandardProblem:
    """Stateful Total-Lagrangian J2 equilibrium with ordinary strong BCs.

    The standard-boundary and affine/MPC routes deliberately remain separate
    nonlinear lowerings: they eliminate constrained degrees of freedom in
    different ways.  They nevertheless consume the same quadrature material
    transaction, output fields, increment controls, and restart state.
    """

    name: str
    solution: object
    accepted_solution: object
    material: FiniteStrainJ2Logarithmic | QuadratureMaterialMap
    response: MaterialQuadratureResponse
    residual_form: object
    tangent_form: object
    deformation_gradient_old: object
    deformation_gradient_new: object
    load_factor: object
    amplitude: amplitudes.Amplitude
    bcs: tuple[object, ...]
    value_path: object
    incrementation: object
    solver_options: NewtonSolverOptions
    state_transaction: FiniteStrainJ2StateTransaction
    output_every: int | None = 1
    output_factors: tuple[float, ...] = ()
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    procedure: object | None = None
    step_number: int = 1
    quadrature_degree: int = 2
    constraint_identity: tuple[object, ...] = ()
    load_identity: object | None = None
    accepted_load_factor: float = field(default=0.0, init=False)
    accepted_increments: list[FiniteStrainPlasticityIncrementInfo] = field(
        default_factory=list, init=False
    )
    attempted_increments: list[FiniteStrainPlasticityIncrementInfo] = field(
        default_factory=list, init=False
    )
    next_increment_size: float | None = field(default=None, init=False)
    snapshots: list[object] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    last_solve_info: FiniteStrainPlasticityPathInfo | None = field(
        default=None,
        init=False,
    )

    def _apply_loading(self, coordinate: float) -> None:
        factor = self.amplitude(coordinate)
        self.load_factor.value = PETSc.ScalarType(factor)
        self.value_path.update(factor)

    def _snapshot_loading_state(self) -> dict[str, object]:
        return {
            "load_factor": np.asarray(self.load_factor.value).copy(),
            "prescribed_values": self.value_path.snapshot_runtime_state(),
        }

    def _restore_loading_state(self, state: dict[str, object]) -> None:
        self.load_factor.value = state["load_factor"]
        self.value_path.restore_runtime_state(state["prescribed_values"])

    def _evaluate_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.response.state.evaluate_expression(
                self.deformation_gradient_old,
                value_shape=(3, 3),
            ),
            self.response.state.evaluate_expression(
                self.deformation_gradient_new,
                value_shape=(3, 3),
            ),
        )

    def _update_response(
        self,
        *,
        start_factor: float,
        target_factor: float,
    ):
        result = self.state_transaction.refresh_trial(
            start_factor=start_factor,
            target_factor=target_factor,
        )
        return result, self.state_transaction.last_maximum_plastic_increment

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

    def _assign_trial(self, base, direction, alpha: float) -> None:
        self.solution.x.array[:] = base
        self.solution.x.array[: len(direction)] += alpha * direction
        self.solution.x.scatter_forward()

    def _line_search(
        self,
        base,
        direction,
        base_norm: float,
        *,
        start_factor: float,
        target_factor: float,
    ) -> float:
        options = self.solver_options
        alpha = 1.0
        if options.line_search in {None, "basic"}:
            self._assign_trial(base, direction, alpha)
            return alpha
        while alpha + 1.0e-15 >= options.minimum_step_length:
            self._assign_trial(base, direction, alpha)
            self._update_response(
                start_factor=start_factor,
                target_factor=target_factor,
            )
            rhs, trial_norm = self._correction_rhs()
            rhs.destroy()
            if np.isfinite(trial_norm) and trial_norm < base_norm:
                return alpha
            alpha *= options.line_search_reduction
        self.solution.x.array[:] = base
        self.solution.x.scatter_forward()
        self.response.rollback()
        return 0.0

    def _solve_increment(
        self,
        *,
        increment: int,
        attempt: int,
        start_factor: float,
        target_factor: float,
    ) -> FiniteStrainPlasticityIncrementInfo:
        initial_norm = None
        norm = float("inf")
        maximum_increment = 0.0
        plastic_points = 0
        converged = False
        iteration = 0
        for iteration in range(self.solver_options.maximum_iterations + 1):
            result, maximum_increment = self._update_response(
                start_factor=start_factor,
                target_factor=target_factor,
            )
            del result
            plastic_points = self.state_transaction.last_plastic_points
            maximum_increment = (
                self.state_transaction.last_maximum_plastic_increment
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
            alpha = self._line_search(
                base,
                direction,
                norm,
                start_factor=start_factor,
                target_factor=target_factor,
            )
            if alpha == 0.0:
                break
        return FiniteStrainPlasticityIncrementInfo(
            increment=increment,
            attempt=attempt,
            start_load_factor=start_factor,
            load_factor=target_factor,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=float(initial_norm or 0.0),
            residual_norm=float(norm),
            plastic_points=plastic_points,
            maximum_plastic_increment=maximum_increment,
        )

    def _restore_accepted(self) -> None:
        self.solution.x.array[:] = self.accepted_solution.x.array
        self.solution.x.scatter_forward()
        self._apply_loading(self.accepted_load_factor)
        self.state_transaction.rollback_increment(
            accepted_factor=self.accepted_load_factor,
        )

    def _snapshot_increment_boundary(self) -> dict[str, object]:
        """Capture the accepted boundary before one provisional attempt."""

        return {
            "solution": self.solution.x.array.copy(),
            "state": self.state_transaction.snapshot_accepted_boundary(),
            "accepted_load_factor": float(self.accepted_load_factor),
            "accepted_increments": list(self.accepted_increments),
            "attempted_increments": list(self.attempted_increments),
            "snapshots": list(self.snapshots),
            "checkpoints": list(self.checkpoints),
            "execution_events": list(self.execution_events),
            "next_increment_size": self.next_increment_size,
            "loading": self._snapshot_loading_state(),
        }

    def _restore_increment_boundary(self, boundary: dict[str, object]) -> None:
        """Undo every mutable effect of one failed increment attempt."""

        self.solution.x.array[:] = boundary["solution"]
        self.solution.x.scatter_forward()
        self.state_transaction.restore_accepted_boundary(boundary["state"])
        self.accepted_load_factor = float(boundary["accepted_load_factor"])
        self.accepted_increments[:] = boundary["accepted_increments"]
        self.attempted_increments[:] = boundary["attempted_increments"]
        self.snapshots[:] = boundary["snapshots"]
        self.checkpoints[:] = boundary["checkpoints"]
        self.execution_events[:] = boundary["execution_events"]
        self.next_increment_size = boundary["next_increment_size"]
        self._restore_loading_state(boundary["loading"])

    def _record_failure(
        self,
        emit,
        message: str,
        *,
        info: FiniteStrainPlasticityIncrementInfo | None = None,
        target_factor: float | None = None,
    ) -> None:
        """Emit one terminal event without weakening rollback semantics."""

        selected_target = (
            self.accepted_load_factor
            if target_factor is None
            else float(target_factor)
        )
        emit(
            SolveEvent(
                "step_failed",
                self.name,
                step_number=self.step_number,
                increment=(
                    len(self.accepted_increments) + 1
                    if info is None
                    else info.increment
                ),
                attempt=(
                    len(self.attempted_increments)
                    if info is None
                    else info.attempt
                ),
                start_factor=(
                    self.accepted_load_factor
                    if info is None
                    else info.start_load_factor
                ),
                target_factor=selected_target,
                iteration=0 if info is None else info.iterations,
                residual_norm=None if info is None else info.residual_norm,
                message=str(message),
            )
        )
        self.last_solve_info = FiniteStrainPlasticityPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )

    def solve(self, *, until: float = 1.0):
        """Advance the normalized load path with commit/cutback discipline."""

        selected_until = float(until)
        if not self.accepted_load_factor < selected_until <= 1.0:
            raise ValueError(
                "until must exceed the accepted load factor and be at most one."
            )
        if self.output_every is not None and int(self.output_every) <= 0:
            raise ValueError("Finite-strain J2 output_every must be positive.")

        from ..diagnostics import (
            SolveEventRecorder,
            StandardRunReporter,
            compose_reporters,
        )
        from ..problems import _load_snapshot

        recorder = SolveEventRecorder(self.execution_events)
        if self.progress is True:
            visible = StandardRunReporter(
                self.solution.function_space.mesh.comm,
                status_file=self.status_file,
                show_iterations=False,
            )
        elif self.progress in (False, None):
            visible = None
        else:
            visible = self.progress
        reporter = compose_reporters(recorder, visible)

        def emit(event) -> None:
            if reporter is not None:
                reporter.emit(event)

        accepted = self.accepted_load_factor
        entry_solution = self.solution.x.array.copy()
        entry_accepted_solution = self.accepted_solution.x.array.copy()
        entry_runtime = self.state_transaction.snapshot_runtime_state()
        entry_events = list(self.execution_events)
        entry_snapshots = list(self.snapshots)
        entry_loading = self._snapshot_loading_state()
        try:
            self._apply_loading(accepted)
            if accepted <= 1.0e-12 and not self.accepted_increments:
                self.execution_events.clear()
                self.state_transaction.initialize()
            else:
                if (
                    abs(self.state_transaction.accepted_factor - accepted)
                    > 1.0e-12
                ):
                    raise RuntimeError(
                        "Finite-strain J2 problem and material transaction "
                        "accepted factors differ; restore both through the "
                        "checkpoint lifecycle before resuming."
                    )
                self.state_transaction.prepare_resume()
            self.snapshots.clear()
            self.snapshots.append(
                _load_snapshot(
                    len(self.accepted_increments),
                    accepted,
                    self.solution,
                    field_factory=lambda: (
                        self.solution,
                        self.state_transaction.snapshot_fields(),
                    ),
                )
            )
        except Exception as exc:
            self.solution.x.array[:] = entry_solution
            self.solution.x.scatter_forward()
            self.accepted_solution.x.array[:] = entry_accepted_solution
            self.accepted_solution.x.scatter_forward()
            self.state_transaction.restore_runtime_state(entry_runtime)
            self.execution_events[:] = entry_events
            self.snapshots[:] = entry_snapshots
            self._restore_loading_state(entry_loading)
            self._record_failure(
                emit,
                f"initialization failed: {type(exc).__name__}: {exc}",
                target_factor=accepted,
            )
            raise
        del (
            entry_solution,
            entry_accepted_solution,
            entry_runtime,
            entry_events,
            entry_snapshots,
            entry_loading,
        )
        emit(
            SolveEvent(
                "step_resumed" if accepted > 1.0e-12 else "step_started",
                self.name,
                step_number=self.step_number,
                incrementation=self.incrementation.summary()["kind"],
            )
        )
        proposed = (
            (
                self.incrementation.initial
                if self.next_increment_size is None
                else float(self.next_increment_size)
            )
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation)
            else None
        )
        cutbacks = 0
        while accepted < selected_until - 1.0e-12:
            increment = len(self.accepted_increments) + 1
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                if len(self.accepted_increments) >= self.incrementation.max_increments:
                    message = "Finite-strain J2 reached max_increments"
                    self._record_failure(emit, message, target_factor=accepted)
                    raise RuntimeError(message + ".")
                target = min(selected_until, accepted + proposed)
                later_output = tuple(
                    value
                    for value in self.output_factors
                    if value > accepted + 1.0e-12
                )
                if later_output:
                    target = min(target, min(later_output))
            else:
                remaining = [
                    value
                    for value in self.incrementation.load_factors
                    if value > accepted + 1.0e-12
                ]
                if not remaining:
                    message = "Fixed finite-strain J2 path is incomplete"
                    self._record_failure(emit, message, target_factor=accepted)
                    raise RuntimeError(message + ".")
                target = min(selected_until, remaining[0])
            attempt = cutbacks + 1
            emit(
                SolveEvent(
                    "increment_started",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=accepted,
                    target_factor=target,
                )
            )
            boundary = self._snapshot_increment_boundary()
            rollback_proposed = proposed
            rollback_cutbacks = cutbacks
            loading_problem = None
            try:
                self._apply_loading(target)
            except BaseException as exc:
                loading_problem = f"{type(exc).__name__}: {exc}"
            try:
                _raise_collective_transaction_problem(
                    self.solution.function_space.mesh.comm,
                    loading_problem,
                    context="finite-strain J2 loading update",
                )
            except BaseException as exc:
                self._restore_increment_boundary(boundary)
                proposed = rollback_proposed
                cutbacks = rollback_cutbacks
                self._record_failure(
                    emit,
                    f"loading update failed: {type(exc).__name__}: {exc}",
                    target_factor=target,
                )
                raise
            try:
                info = self._solve_increment(
                    increment=increment,
                    attempt=attempt,
                    start_factor=accepted,
                    target_factor=target,
                )
            except Exception as exc:
                self._restore_increment_boundary(boundary)
                proposed = rollback_proposed
                cutbacks = rollback_cutbacks
                self._record_failure(
                    emit,
                    f"increment evaluation failed: {type(exc).__name__}: {exc}",
                    target_factor=target,
                )
                raise
            limit = getattr(self.incrementation, "maximum_inelastic_increment", None)
            if (
                info.converged
                and limit is not None
                and info.maximum_plastic_increment > limit
            ):
                info = FiniteStrainPlasticityIncrementInfo(
                    **{
                        **info.as_dict(),
                        "converged": False,
                        "rejection_reason": (
                            "maximum equivalent plastic-strain increment "
                            f"{info.maximum_plastic_increment:.6g} exceeds {limit:.6g}"
                        ),
                    }
                )
            self.attempted_increments.append(info)
            if info.converged:
                finalization_problem = None
                try:
                    self.state_transaction.commit_increment(
                        start_factor=accepted,
                        target_factor=target,
                    )
                    self.accepted_increments.append(info)
                    size = target - accepted
                    accepted = target
                    self.accepted_load_factor = target
                    cutbacks = 0
                    if isinstance(
                        self.incrementation, step_controls.AutomaticIncrementation
                    ):
                        proposed = self.incrementation.after_convergence(
                            size, info.iterations
                        )
                        self.next_increment_size = proposed
                    save_by_increment = (
                        self.output_every is not None
                        and len(self.accepted_increments) % int(self.output_every) == 0
                    )
                    save_by_factor = any(
                        abs(target - value) <= 1.0e-12
                        for value in self.output_factors
                    )
                    if (
                        save_by_increment
                        or save_by_factor
                        or abs(target - selected_until) <= 1.0e-12
                    ):
                        self.snapshots.append(
                            _load_snapshot(
                                len(self.accepted_increments),
                                target,
                                self.solution,
                                solve_info=info,
                                field_factory=lambda: (
                                    self.solution,
                                    self.state_transaction.snapshot_fields(),
                                ),
                            )
                        )
                    emit(
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
                        )
                    )
                except BaseException as exc:
                    finalization_problem = f"{type(exc).__name__}: {exc}"
                try:
                    _raise_collective_transaction_problem(
                        self.solution.function_space.mesh.comm,
                        finalization_problem,
                        context="finite-strain J2 accepted-state finalization",
                    )
                except BaseException as exc:
                    self._restore_increment_boundary(boundary)
                    accepted = self.accepted_load_factor
                    proposed = rollback_proposed
                    cutbacks = rollback_cutbacks
                    self._record_failure(
                        emit,
                        f"accepted-state finalization failed: {type(exc).__name__}: {exc}",
                        info=info,
                        target_factor=target,
                    )
                    raise
                checkpoint_problem = None
                try:
                    self._write_scheduled_checkpoint()
                except BaseException as exc:
                    checkpoint_problem = f"{type(exc).__name__}: {exc}"
                try:
                    _raise_collective_transaction_problem(
                        self.solution.function_space.mesh.comm,
                        checkpoint_problem,
                        context="finite-strain J2 checkpoint finalization",
                    )
                except BaseException as exc:
                    self._restore_increment_boundary(boundary)
                    accepted = self.accepted_load_factor
                    proposed = rollback_proposed
                    cutbacks = rollback_cutbacks
                    self._record_failure(
                        emit,
                        f"accepted-state finalization failed: "
                        f"{type(exc).__name__}: {exc}",
                        info=info,
                        target_factor=target,
                    )
                    raise
                continue

            self._restore_increment_boundary(boundary)
            self.attempted_increments.append(info)
            proposed = rollback_proposed
            cutbacks = rollback_cutbacks
            if not isinstance(
                self.incrementation, step_controls.AutomaticIncrementation
            ):
                message = f"fixed increment failed at load factor {target}"
                self._record_failure(
                    emit, message, info=info, target_factor=target
                )
                raise RuntimeError(f"{self.name}: {message}.")
            cutbacks += 1
            proposed = self.incrementation.after_failure(target - accepted)
            self.next_increment_size = proposed
            if (
                cutbacks > self.incrementation.max_cutbacks
                or proposed < self.incrementation.minimum
            ):
                message = "automatic incrementation exhausted cutbacks"
                self._record_failure(
                    emit, message, info=info, target_factor=target
                )
                raise RuntimeError(f"{self.name}: {message}.")
            emit(
                SolveEvent(
                    "increment_cutback",
                    self.name,
                    step_number=self.step_number,
                    increment=increment,
                    attempt=attempt,
                    start_factor=accepted,
                    target_factor=target,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    next_increment=proposed,
                    message=info.rejection_reason or "",
                )
            )
        emit(
            SolveEvent(
                (
                    "step_completed"
                    if self.accepted_load_factor >= 1.0 - 1.0e-12
                    else "step_paused"
                ),
                self.name,
                step_number=self.step_number,
                increment=len(self.accepted_increments),
                attempt=len(self.attempted_increments),
                target_factor=self.accepted_load_factor,
            )
        )
        self.last_solve_info = FiniteStrainPlasticityPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        return self.solution

    def _write_scheduled_checkpoint(self) -> None:
        policy = self.checkpoint_policy
        if policy is None:
            return
        increment = len(self.accepted_increments)
        due = increment % int(policy.every) == 0
        due = due or (
            bool(policy.final) and self.accepted_load_factor >= 1.0 - 1.0e-12
        )
        if not due:
            return
        path = self.save_checkpoint(
            policy.path(step_name=self.name, increment=increment),
            portable=bool(policy.portable),
        )
        from ..results import CheckpointRecord

        portable = bool(policy.portable) or self.solution.function_space.mesh.comm.size > 1
        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_checkpoint_{increment}",
                path=path,
                schema=(
                    "agentfem.finite-strain-j2-standard-checkpoint.v2"
                    if portable
                    else "agentfem.finite-strain-j2-standard-checkpoint.v1"
                ),
                step_name=self.name,
                coordinate_name="load_factor",
                coordinate_value=self.accepted_load_factor,
                portable=portable,
                metadata={"role": "scheduled_checkpoint"},
            )
        )
        from ..problems import _prune_affine_checkpoints

        _prune_affine_checkpoints(self)

    def _checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_partition_identity

        return {
            "step_name": self.name,
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "material": self.material.summary(),
            "state_schema": self.response.state.state_schema.summary(),
            "incrementation": self.incrementation.summary(),
            "amplitude": self.amplitude.summary(),
            "quadrature_degree": self.quadrature_degree,
            "constraints": self.constraint_identity,
            "prescribed_value_path": self.value_path.summary(),
            "external_load": self.load_identity,
            "solver": self.solver_options.summary(),
            "solution": function_partition_identity(self.solution),
        }

    def _snapshot_checkpoint_restore_boundary(self) -> dict[str, object]:
        """Capture all mutable state before reading an external checkpoint."""

        return {
            "solution": self.solution.x.array.copy(),
            "accepted_solution": self.accepted_solution.x.array.copy(),
            "transaction": self.state_transaction.snapshot_runtime_state(),
            "accepted_load_factor": float(self.accepted_load_factor),
            "accepted_increments": list(self.accepted_increments),
            "attempted_increments": list(self.attempted_increments),
            "next_increment_size": self.next_increment_size,
            "execution_events": list(self.execution_events),
            "last_solve_info": self.last_solve_info,
            "snapshots": list(self.snapshots),
            "checkpoints": list(self.checkpoints),
            "loading": self._snapshot_loading_state(),
        }

    def _restore_checkpoint_restore_boundary(
        self,
        boundary: dict[str, object],
    ) -> None:
        """Undo a failed serial or portable checkpoint restore."""

        self.solution.x.array[:] = boundary["solution"]
        self.solution.x.scatter_forward()
        self.accepted_solution.x.array[:] = boundary["accepted_solution"]
        self.accepted_solution.x.scatter_forward()
        self.state_transaction.restore_runtime_state(boundary["transaction"])
        self.accepted_load_factor = float(boundary["accepted_load_factor"])
        self.accepted_increments[:] = boundary["accepted_increments"]
        self.attempted_increments[:] = boundary["attempted_increments"]
        self.next_increment_size = boundary["next_increment_size"]
        self.execution_events[:] = boundary["execution_events"]
        self.last_solve_info = boundary["last_solve_info"]
        self.snapshots[:] = boundary["snapshots"]
        self.checkpoints[:] = boundary["checkpoints"]
        self._restore_loading_state(boundary["loading"])

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        """Save the accepted global and material state."""

        comm = self.solution.function_space.mesh.comm
        local_problem = None
        if abs(
            float(self.state_transaction.accepted_factor)
            - self.accepted_load_factor
        ) > 1.0e-12:
            local_problem = (
                "Checkpointing is permitted only at a fully accepted "
                "material state."
            )
        elif not np.allclose(
            self.solution.x.array,
            self.accepted_solution.x.array,
            rtol=0.0,
            atol=1.0e-12,
        ):
            local_problem = (
                "Checkpointing is permitted only when U equals U_ACCEPTED."
            )
        _raise_collective_transaction_problem(
            comm,
            local_problem,
            context="finite-strain J2 checkpoint precondition",
        )
        selected_portable = comm.size != 1 if portable is None else bool(portable)
        if selected_portable:
            return self._save_portable_checkpoint(path)
        if comm.size != 1:
            raise ValueError("Distributed checkpoints require portable=True.")
        from ..checkpointing import atomic_savez

        selected = Path(path)
        if selected.suffix != ".npz":
            selected = selected.with_suffix(".npz")
        snapshot = self.response.snapshot()
        atomic_savez(
            selected,
            schema="agentfem.finite-strain-j2-standard-checkpoint.v1",
            identity=json.dumps(self._checkpoint_identity(), sort_keys=True),
            solution=self.solution.x.array,
            accepted_solution=self.accepted_solution.x.array,
            accepted_load_factor=self.accepted_load_factor,
            state_names=json.dumps(tuple(snapshot)),
            **snapshot,
            accepted_increments=json.dumps(
                [item.as_dict() for item in self.accepted_increments]
            ),
            attempted_increments=json.dumps(
                [item.as_dict() for item in self.attempted_increments]
            ),
            execution_events=json.dumps(
                [item.as_dict() for item in self.execution_events]
            ),
            next_increment_size=(
                np.nan
                if self.next_increment_size is None
                else self.next_increment_size
            ),
        )
        return selected

    def load_checkpoint(self, path) -> None:
        """Restore a serial or cross-partition portable checkpoint."""

        selected = Path(path)
        comm = self.solution.function_space.mesh.comm
        if selected.suffix == ".json" or selected.name.endswith(".checkpoint.json"):
            envelope = None
            if comm.rank == 0:
                try:
                    envelope = {
                        "payload": json.loads(selected.read_text(encoding="utf-8")),
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
                    "Finite-strain J2 checkpoint manifest read failed: "
                    f"{envelope['error']}"
                )
            payload = envelope["payload"]
            schema = payload.get("schema")
            if schema == "agentfem.finite-strain-j2-standard-checkpoint.v2":
                self._load_portable_checkpoint(selected, payload)
                return
            if schema == "agentfem.finite-strain-j2-experimental-checkpoint.v2":
                raise ValueError(
                    "Legacy experimental finite-strain J2 checkpoints do not "
                    "contain the complete constraint, load, procedure, and "
                    "solver identity required by the public workflow. Re-run "
                    "from a declared model and write the standard v2 schema."
                )
            raise ValueError("Unsupported finite-strain J2 checkpoint schema.")
        if comm.size != 1:
            raise ValueError(
                "This finite-strain J2 checkpoint is partition-bound; use the "
                "portable v2 manifest for distributed restore."
            )
        with np.load(selected, allow_pickle=False) as data:
            schema = str(data["schema"])
            if schema == "agentfem.finite-strain-j2-experimental-checkpoint.v1":
                raise ValueError(
                    "Legacy experimental finite-strain J2 checkpoints do not "
                    "contain the complete scientific identity required by the "
                    "public workflow. Re-run from a declared model and write "
                    "the standard v1 or portable v2 schema."
                )
            if schema != "agentfem.finite-strain-j2-standard-checkpoint.v1":
                raise ValueError("Unsupported finite-strain J2 checkpoint schema.")
            stored_identity = json.loads(str(data["identity"]))
            current_identity = json.loads(
                json.dumps(self._checkpoint_identity(), sort_keys=True)
            )
            if stored_identity != current_identity:
                raise ValueError(
                    "Finite-strain J2 checkpoint material, state, loading, "
                    "increment control, or function layout differs."
                )
            solution = np.asarray(data["solution"]).copy()
            accepted_solution = np.asarray(data["accepted_solution"]).copy()
            if (
                solution.size != self.solution.x.array.size
                or accepted_solution.size != self.accepted_solution.x.array.size
            ):
                raise ValueError("Finite-strain J2 checkpoint dof layout differs.")
            state_names = tuple(json.loads(str(data["state_names"])))
            if state_names != tuple(self.response.state.transaction.names):
                raise ValueError("Finite-strain J2 checkpoint state names differ.")
            state = {
                name: np.asarray(data[name]).copy() for name in state_names
            }
            coordinate = float(data["accepted_load_factor"])
            accepted = [
                FiniteStrainPlasticityIncrementInfo.from_dict(item)
                for item in json.loads(str(data["accepted_increments"]))
            ]
            attempted = [
                FiniteStrainPlasticityIncrementInfo.from_dict(item)
                for item in json.loads(str(data["attempted_increments"]))
            ]
            events = [
                SolveEvent.from_dict(item)
                for item in json.loads(str(data["execution_events"]))
            ]
            size = float(data["next_increment_size"])
        if not np.allclose(solution, accepted_solution, rtol=0.0, atol=1.0e-12):
            raise ValueError("Finite-strain J2 checkpoint U and U_ACCEPTED differ.")
        if accepted:
            if abs(accepted[-1].load_factor - coordinate) > 1.0e-12:
                raise ValueError(
                    "Finite-strain J2 checkpoint coordinate and accepted "
                    "history disagree."
                )
        elif abs(coordinate) > 1.0e-12:
            raise ValueError(
                "A nonzero finite-strain J2 checkpoint requires accepted history."
            )
        boundary = self._snapshot_checkpoint_restore_boundary()
        try:
            self.solution.x.array[:] = solution
            self.solution.x.scatter_forward()
            self.accepted_solution.x.array[:] = accepted_solution
            self.accepted_solution.x.scatter_forward()
            self.response.restore(state)
            self.accepted_load_factor = coordinate
            self.state_transaction.accepted_factor = coordinate
            self.accepted_increments[:] = accepted
            self.attempted_increments[:] = attempted
            self.next_increment_size = size if np.isfinite(size) else None
            self.execution_events[:] = events
            self.last_solve_info = FiniteStrainPlasticityPathInfo(
                tuple(accepted),
                tuple(attempted),
                self.incrementation,
            )
            self._apply_loading(coordinate)
            self.state_transaction.prepare_resume()
        except Exception:
            self._restore_checkpoint_restore_boundary(boundary)
            raise
        from ..results import CheckpointRecord

        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_restart_{len(accepted)}",
                path=selected,
                schema="agentfem.finite-strain-j2-standard-checkpoint.v1",
                step_name=self.name,
                coordinate_name="load_factor",
                coordinate_value=coordinate,
                portable=False,
                metadata={"role": "restart_source"},
            )
        )

    def _portable_checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "material": self.material.summary(),
            "state_schema": self.response.state.state_schema.summary(),
            "incrementation": self.incrementation.summary(),
            "amplitude": self.amplitude.summary(),
            "quadrature_degree": self.quadrature_degree,
            "constraints": self.constraint_identity,
            "prescribed_value_path": self.value_path.summary(),
            "external_load": self.load_identity,
            "solver": self.solver_options.summary(),
            "solution": function_portable_identity(self.solution),
        }

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
        bundle = save_portable_state_bundle(
            manifest,
            state={"U": self.solution, "U_ACCEPTED": self.accepted_solution},
        )
        quadrature = self.response.state.save(
            manifest.with_name(
                f"{selected.name}.{bundle['generation']}.quadrature"
            ),
            material=self.material,
        )
        comm = self.solution.function_space.mesh.comm
        payload = {
            "schema": "agentfem.finite-strain-j2-standard-checkpoint.v2",
            "identity": self._portable_checkpoint_identity(),
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
            "execution_events": [
                item.as_dict() for item in self.execution_events
            ],
            "next_increment_size": self.next_increment_size,
            "writer_rank_count": int(comm.size),
        }
        error = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest,
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(
                f"Finite-strain J2 checkpoint manifest write failed: {error}"
            )
        comm.barrier()
        return manifest

    def _load_portable_checkpoint(self, manifest: Path, payload: dict) -> None:
        from ..checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )

        comm = self.solution.function_space.mesh.comm
        current = json.loads(
            json.dumps(self._portable_checkpoint_identity(), sort_keys=True)
        )
        if payload.get("identity") != current:
            raise ValueError(
                "Portable finite-strain J2 checkpoint scientific identity differs."
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
                "Finite-strain J2 checkpoint payload validation failed: "
                f"{validation['error']}"
            )

        parsed_problem = None
        accepted = attempted = events = None
        coordinate = None
        try:
            coordinate = float(payload["coordinate"])
            accepted = [
                FiniteStrainPlasticityIncrementInfo.from_dict(item)
                for item in payload["accepted_increments"]
            ]
            attempted = [
                FiniteStrainPlasticityIncrementInfo.from_dict(item)
                for item in payload["attempted_increments"]
            ]
            events = [
                SolveEvent.from_dict(item)
                for item in payload["execution_events"]
            ]
            if accepted:
                if abs(accepted[-1].load_factor - coordinate) > 1.0e-12:
                    raise ValueError(
                        "checkpoint coordinate and accepted history disagree"
                    )
            elif abs(coordinate) > 1.0e-12:
                raise ValueError(
                    "a nonzero checkpoint requires accepted history"
                )
        except Exception as exc:
            parsed_problem = f"{type(exc).__name__}: {exc}"
        _raise_collective_transaction_problem(
            comm,
            parsed_problem,
            context="finite-strain J2 checkpoint history validation",
        )

        boundary = self._snapshot_checkpoint_restore_boundary()
        try:
            load_problem = None
            try:
                load_portable_state_bundle(
                    manifest,
                    state={
                        "U": self.solution,
                        "U_ACCEPTED": self.accepted_solution,
                    },
                    record=payload["nodal_state"],
                    identities=payload["nodal_identity"],
                )
                self.response.state.load(
                    Path(validation["quadrature_path"]),
                    material=self.material,
                )
            except Exception as exc:
                load_problem = f"{type(exc).__name__}: {exc}"
            _raise_collective_transaction_problem(
                comm,
                load_problem,
                context="finite-strain J2 checkpoint state restore",
            )
            local_equal = np.allclose(
                self.solution.x.array,
                self.accepted_solution.x.array,
                rtol=0.0,
                atol=1.0e-12,
            )
            if not comm.allreduce(bool(local_equal), op=MPI.LAND):
                raise ValueError(
                    "Finite-strain J2 checkpoint U and U_ACCEPTED differ."
                )
            self.accepted_load_factor = coordinate
            self.state_transaction.accepted_factor = coordinate
            self.accepted_increments[:] = accepted
            self.attempted_increments[:] = attempted
            self.next_increment_size = payload.get("next_increment_size")
            self.execution_events[:] = events
            self.last_solve_info = FiniteStrainPlasticityPathInfo(
                tuple(accepted),
                tuple(attempted),
                self.incrementation,
            )
            loading_problem = None
            try:
                self._apply_loading(coordinate)
            except Exception as exc:
                loading_problem = f"{type(exc).__name__}: {exc}"
            _raise_collective_transaction_problem(
                comm,
                loading_problem,
                context="finite-strain J2 checkpoint loading restore",
            )
            self.state_transaction.prepare_resume()
        except Exception:
            self._restore_checkpoint_restore_boundary(boundary)
            raise

        from ..results import CheckpointRecord

        self.checkpoints.append(
            CheckpointRecord(
                name=f"{self.name}_restart_{len(accepted)}",
                path=manifest,
                schema=payload["schema"],
                step_name=self.name,
                coordinate_name="load_factor",
                coordinate_value=coordinate,
                portable=True,
                metadata={
                    "writer_rank_count": payload.get("writer_rank_count"),
                    "reader_rank_count": int(comm.size),
                    "restart_mode": "portable_coordinate_and_cell_keyed_state",
                    "role": "restart_source",
                },
            )
        )

    def reaction_field(self, *, name: str = "RF"):
        """Return the accepted full-space residual as a nodal reaction field."""

        from ..problems import _reaction_field

        self.state_transaction.refresh_trial(
            start_factor=self.accepted_load_factor,
            target_factor=self.accepted_load_factor,
        )
        return _reaction_field(self.residual_form, self.solution, name=name)

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve and complete the common stateful nonlinear result lifecycle."""

        from ..results import add_execution_trace, complete_result, from_solution

        solution = self.solve()
        result = from_solution(
            solution,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "state": self.state_transaction.summary(),
            },
        )
        transaction_fields = tuple(
            self.state_transaction.populate_result(result) or ()
        )
        reaction = self.reaction_field()
        result.add_field(
            "RF",
            reaction,
            location="nodes",
            description=(
                "Accepted full-space residual; constrained entries are nodal "
                "reactions under the declared strong boundary conditions."
            ),
            processing={
                "method": "assembled_total_lagrangian_residual",
                "representation": "nodal_vector",
                "postprocessed": True,
            },
        )
        for selected in fields:
            function = getattr(selected, "value", selected)
            result.add_field(
                getattr(function, "name", type(function).__name__),
                function,
            )
        factors = np.asarray(
            [0.0, *(item.load_factor for item in self.accepted_increments)],
            dtype=float,
        )
        result.add_history(
            "load_amplitude",
            factors,
            np.asarray([self.amplitude(value) for value in factors], dtype=float),
            abscissa_name="load_factor",
            description="Resolved dimensionless loading amplitude.",
        )
        result.add_history(
            "newton_iterations",
            factors,
            np.asarray(
                [0, *(item.iterations for item in self.accepted_increments)],
                dtype=float,
            ),
            abscissa_name="load_factor",
            description="Newton iterations for each accepted increment.",
        )
        result.add_history(
            "newton_residual",
            factors,
            np.asarray(
                [0.0, *(item.residual_norm for item in self.accepted_increments)],
                dtype=float,
            ),
            abscissa_name="load_factor",
            description="Accepted full-space nonlinear residual norm.",
        )
        result.add_history(
            "increment_size",
            factors,
            np.asarray(
                [
                    0.0,
                    *(
                        item.load_factor - item.start_load_factor
                        for item in self.accepted_increments
                    ),
                ],
                dtype=float,
            ),
            abscissa_name="load_factor",
            description="Accepted normalized load increment size.",
        )
        for checkpoint in self.checkpoints:
            result.add_checkpoint(checkpoint)
        add_execution_trace(result, self.execution_events)
        return complete_result(
            self,
            result,
            output=output,
            fields=(*transaction_fields, reaction, *tuple(fields)),
            strict_output=strict_output,
            metadata=metadata,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_j2_standard_problem",
            "name": self.name,
            "maturity": "experimental_global_mpi_restart",
            "evidence_level": "internal_serial_mpi_restart_verified",
            "accepted_load_factor": self.accepted_load_factor,
            "material": self.material.summary(),
            "response": self.response.summary(),
            "incrementation": self.incrementation.summary(),
            "solver": self.solver_options.summary(),
            "accepted_increments": tuple(
                item.as_dict() for item in self.accepted_increments
            ),
            "attempted_increments": tuple(
                item.as_dict() for item in self.attempted_increments
            ),
            "output_every": self.output_every,
            "output_factors": self.output_factors,
            "snapshot_count": len(self.snapshots),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "checkpoint_count": len(self.checkpoints),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "quadrature_degree": self.quadrature_degree,
            "constraints": self.constraint_identity,
            "external_load": self.load_identity,
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
        }


FiniteStrainJ2AffineTransaction = FiniteStrainJ2StateTransaction
ExperimentalFiniteStrainPlasticityStep = FiniteStrainJ2StandardProblem


def _finite_strain_j2_transaction(
    displacement,
    material: FiniteStrainJ2Logarithmic | QuadratureMaterialMap,
    *,
    quadrature_degree: int,
) -> FiniteStrainJ2StateTransaction:
    """Create the constraint-neutral material state used by both lowerings."""

    if not isinstance(
        material,
        (FiniteStrainJ2Logarithmic, QuadratureMaterialMap),
    ):
        raise TypeError(
            "Finite-strain J2 requires FiniteStrainJ2Logarithmic or a "
            "QuadratureMaterialMap of that family."
        )
    solution = displacement.value
    domain = solution.function_space.mesh
    if domain.geometry.dim != 3:
        raise NotImplementedError("Finite-strain J2 is currently three-dimensional.")
    if isinstance(material, QuadratureMaterialMap):
        if material.domain is not domain:
            raise ValueError(
                "Finite-strain J2 quadrature material map belongs to another mesh."
            )
        if not all(
            isinstance(item, FiniteStrainJ2Logarithmic)
            for item in material.materials.values()
        ):
            raise TypeError(
                "One finite-strain J2 Step cannot mix constitutive families."
            )
        state_schema = material.require_common_state_schema()
        material.require_common_tangent_convention()
        stored_energy_component_names = (
            material.require_common_stored_energy_component_names()
        )
    else:
        state_schema = material.state_schema
        stored_energy_component_names = material.stored_energy_component_names
    response = MaterialQuadratureResponse.create(
        domain,
        state_schema,
        degree=quadrature_degree,
        stored_energy_component_names=stored_energy_component_names,
    )
    accepted_solution = fem.Function(solution.function_space, name="U_ACCEPTED")
    identity = ufl.Identity(3)
    old_gradient = response.state.compile_expression(
        identity + ufl.grad(accepted_solution),
        value_shape=(3, 3),
    )
    new_gradient = response.state.compile_expression(
        identity + ufl.grad(solution),
        value_shape=(3, 3),
    )
    gradient_field = QuadratureField.create(
        domain,
        name="F",
        value_shape=(3, 3),
        degree=quadrature_degree,
    )
    gradient_field.assign(
        np.broadcast_to(np.eye(3), (len(gradient_field.values), 3, 3))
    )
    equivalent_stress = QuadratureField.create(
        domain,
        name="MISES",
        degree=quadrature_degree,
    )
    return FiniteStrainJ2StateTransaction(
        solution=solution,
        accepted_solution=accepted_solution,
        material=material,
        response=response,
        deformation_gradient_old=old_gradient,
        deformation_gradient_new=new_gradient,
        deformation_gradient=gradient_field,
        equivalent_stress=equivalent_stress,
    )


def finite_strain_j2_standard_problem(
    *,
    displacement,
    material: FiniteStrainJ2Logarithmic | QuadratureMaterialMap,
    external_force=None,
    load_identity=None,
    constraints=(),
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    amplitude=None,
    output_every: int | None = 1,
    output_factors=(),
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    name: str = "finite_strain_j2",
) -> FiniteStrainJ2StandardProblem:
    """Build stateful finite-strain J2 under ordinary strong boundaries."""

    transaction = _finite_strain_j2_transaction(
        displacement,
        material,
        quadrature_degree=quadrature_degree,
    )
    solution = transaction.solution
    domain = solution.function_space.mesh
    response = transaction.response
    gradient_test = ufl.grad(displacement.test)
    gradient_trial = ufl.grad(displacement.trial)
    first_piola = response.first_piola_stress.function
    tangent = response.tangent.function
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        tangent[i, j, k, l] * gradient_trial[k, l], (i, j)
    )
    load_factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    residual = ufl.inner(first_piola, gradient_test) * response.measure
    if external_force is not None:
        residual -= load_factor * external_force.expression
    jacobian = ufl.inner(tangent_action, gradient_test) * response.measure

    from .. import constraints as constraint_api

    concrete_constraints = constraint_api.constraint_assets(constraints)
    selected_bcs = []
    for item in concrete_constraints:
        if isinstance(item, constraint_api.TimeDependentDirichlet):
            raise NotImplementedError(
                "Finite-strain J2 uses one normalized Step amplitude. Declare "
                "the end-of-step strong value instead of an independent "
                "TimeDependentDirichlet history."
            )
        if hasattr(item, "bc"):
            selected_bcs.append(item.bc)
        elif callable(getattr(item, "dof_indices", None)):
            selected_bcs.append(item)
        else:
            raise TypeError(
                "finite_strain_j2_standard_problem accepts ordinary strong "
                "DirichletConstraint/RemoteDisplacementConstraint assets or "
                "raw DOLFINx DirichletBC objects; received "
                f"{type(item).__name__}."
            )
    if not selected_bcs:
        raise ValueError(
            "Finite-strain J2 standard equilibrium requires at least one "
            "strong boundary constraint."
        )
    value_path = constraint_api.prescribed_value_path(concrete_constraints)
    selected_amplitude = (
        amplitudes.ramp()
        if amplitude is None
        else amplitudes.as_amplitude(amplitude, name="finite_strain_j2_amplitude")
    )
    if not np.isclose(selected_amplitude(0.0), 0.0):
        raise ValueError("Finite-strain J2 amplitude must start at zero.")
    selected_options = newton() if solver_options is None else solver_options
    if not bool(getattr(selected_options, "error_if_not_converged", True)):
        raise ValueError(
            "Public stateful finite-strain J2 requires "
            "error_if_not_converged=True."
        )
    from .. import procedures

    selected_incrementation = step_controls.normalize(incrementation)
    selected_output_factors = tuple(float(value) for value in output_factors)
    if isinstance(selected_incrementation, step_controls.FixedIncrementation):
        missing = tuple(
            value
            for value in selected_output_factors
            if not any(
                abs(value - declared) <= 1.0e-12
                for declared in selected_incrementation.load_factors
            )
        )
        if missing:
            raise ValueError(
                "Fixed finite-strain J2 incrementation must include every "
                f"requested output factor; missing {missing}."
            )

    return FiniteStrainJ2StandardProblem(
        name=name,
        solution=solution,
        accepted_solution=transaction.accepted_solution,
        material=material,
        response=response,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        deformation_gradient_old=transaction.deformation_gradient_old,
        deformation_gradient_new=transaction.deformation_gradient_new,
        load_factor=load_factor,
        amplitude=selected_amplitude,
        bcs=tuple(selected_bcs),
        value_path=value_path,
        incrementation=selected_incrementation,
        solver_options=selected_options,
        state_transaction=transaction,
        output_every=None if output_every is None else int(output_every),
        output_factors=selected_output_factors,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        procedure=procedures.nonlinear_static(stateful=True),
        quadrature_degree=int(quadrature_degree),
        constraint_identity=tuple(
            item.summary()
            if hasattr(item, "summary")
            else {"kind": type(item).__name__}
            for item in concrete_constraints
        ),
        load_identity=(
            load_identity
            if load_identity is not None
            else (
                None
                if external_force is None
                else (
                    external_force.summary()
                    if hasattr(external_force, "summary")
                    else {"kind": type(external_force).__name__}
                )
            )
        ),
    )


def experimental_finite_strain_j2_step(
    *,
    displacement,
    material: FiniteStrainJ2Logarithmic,
    external_force=None,
    constraints=(),
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    amplitude=None,
    name: str = "finite_strain_j2_experimental",
) -> ExperimentalFiniteStrainPlasticityStep:
    """Compatibility alias for :func:`finite_strain_j2_standard_problem`."""

    return finite_strain_j2_standard_problem(
        displacement=displacement,
        material=material,
        external_force=external_force,
        constraints=constraints,
        incrementation=incrementation,
        solver_options=solver_options,
        quadrature_degree=quadrature_degree,
        amplitude=amplitude,
        progress=False,
        name=name,
    )


def finite_strain_j2_affine_problem(
    *,
    displacement,
    material: FiniteStrainJ2Logarithmic | QuadratureMaterialMap,
    constraint,
    external_force=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    output_every: int | None = 1,
    output_factors=(),
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    name: str = "finite_strain_j2",
):
    """Build stateful finite-strain J2 under exact affine/MPC kinematics.

    This is an expert lowering API retained for extension and compatibility.
    Application models should use the stable ``model.step(...)`` language,
    whose provider owns validation, result lifecycle and future evolution.
    """

    from .. import problems

    if external_force is not None:
        raise NotImplementedError(
            "The first stateful affine J2 route accepts prescribed macroscopic "
            "deformation only; affine cells with natural loads require a "
            "separate work-conjugate load contract."
        )
    if getattr(constraint, "target", None) is not displacement:
        raise ValueError("Affine finite-strain J2 constraint must target displacement.")
    transaction = _finite_strain_j2_transaction(
        displacement,
        material,
        quadrature_degree=quadrature_degree,
    )
    solution = transaction.solution
    response = transaction.response
    gradient_test = ufl.grad(displacement.test)
    gradient_trial = ufl.grad(displacement.trial)
    first_piola = response.first_piola_stress.function
    tangent = response.tangent.function
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        tangent[i, j, k, l] * gradient_trial[k, l],
        (i, j),
    )
    residual = ufl.inner(first_piola, gradient_test) * response.measure
    jacobian = ufl.inner(tangent_action, gradient_test) * response.measure

    def acceptance():
        limit = getattr(
            step_controls.normalize(incrementation),
            "maximum_inelastic_increment",
            None,
        )
        accepted = (
            limit is None
            or transaction.last_maximum_plastic_increment <= float(limit)
        )
        return {
            "accepted": accepted,
            "plastic_points": transaction.last_plastic_points,
            "maximum_plastic_increment": (
                transaction.last_maximum_plastic_increment
            ),
            "message": (
                ""
                if accepted
                else (
                    "maximum equivalent plastic-strain increment "
                    f"{transaction.last_maximum_plastic_increment:.6g} exceeds "
                    f"{float(limit):.6g}"
                )
            ),
        }

    selected_solver_options = newton() if solver_options is None else solver_options
    if not bool(getattr(selected_solver_options, "error_if_not_converged", True)):
        raise ValueError(
            "Public stateful finite-strain J2 requires "
            "error_if_not_converged=True so an incomplete material path cannot "
            "be returned as a completed SimulationResult."
        )
    problem = problems.affine_nonlinear(
        residual,
        solution,
        jacobian=jacobian,
        constraint=constraint,
        incrementation=step_controls.normalize(incrementation),
        solver_options=selected_solver_options,
        output_every=output_every,
        output_factors=output_factors,
        state_transaction=transaction,
        checkpoint_policy=checkpoint_policy,
        acceptance_check=acceptance,
        progress=progress,
        status_file=status_file,
        name=name,
    )
    def snapshot_fields():
        return solution, transaction.snapshot_fields()

    problem.snapshot_field_factory = snapshot_fields
    problem.material = material
    problem.response = response
    problem.accepted_solution = transaction.accepted_solution
    return problem
