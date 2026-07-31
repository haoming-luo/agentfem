"""Global small-strain J2 plasticity with integration-point state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc

from .. import procedures
from .. import steps as step_controls
from ..constitutive import elasticity
from ..constitutive.plasticity import J2LinearIsotropicHardening
from ..constitutive.quadrature import J2QuadratureState
from ..diagnostics import StandardRunReporter, comm_of
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

    def as_dict(self) -> dict[str, object]:
        return {
            "increment": self.increment,
            "attempt": self.attempt,
            "start_load_factor": self.start_load_factor,
            "load_factor": self.load_factor,
            "increment_size": self.load_factor - self.start_load_factor,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": self.initial_residual_norm,
            "residual_norm": self.residual_norm,
            "plastic_points": self.plastic_points,
            "maximum_plastic_increment": self.maximum_plastic_increment,
        }


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
            and abs(self.increments[-1].load_factor - 1.0) <= 1.0e-12
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "j2_nonlinear_load_path",
            "converged": self.converged,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": [item.as_dict() for item in self.increments],
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass
class J2PlasticityStep:
    """Incremental global equilibrium for 3D small-strain J2 plasticity."""

    name: str
    solution: object
    material: J2LinearIsotropicHardening
    state: J2QuadratureState
    residual_form: object
    tangent_form: object
    load_factor: object
    bcs: tuple[object, ...]
    incrementation: object
    solver_options: NewtonSolverOptions
    study: object | None = None
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    procedure: object = field(default_factory=lambda: procedures.nonlinear_static(stateful=True))
    accepted_load_factor: float = field(default=0.0, init=False)
    last_solve_info: J2LoadPathInfo | None = field(default=None, init=False)

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
            self.incrementation.initial
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
        self.load_factor.value = PETSc.ScalarType(accepted_factor)
        while accepted_factor < selected_until - 1.0e-12:
            increment = len(accepted) + 1
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                if len(accepted) >= self.incrementation.max_increments:
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
            self.load_factor.value = PETSc.ScalarType(target)
            info = self._solve_increment(
                increment=increment,
                attempt=attempt,
                start_factor=accepted_factor,
                target_factor=target,
                reporter=reporter,
            )
            attempts.append(info)
            if info.converged:
                self.state.commit()
                accepted.append(info)
                accepted_size = target - accepted_factor
                accepted_factor = target
                self.accepted_load_factor = target
                consecutive_cutbacks = 0
                if isinstance(
                    self.incrementation,
                    step_controls.AutomaticIncrementation,
                ):
                    proposed_size = self.incrementation.after_convergence(
                        accepted_size,
                        info.iterations,
                    )
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
            if not isinstance(
                self.incrementation,
                step_controls.AutomaticIncrementation,
            ):
                self._fail(reporter, info, "fixed increment did not converge")
                return self.solution
            consecutive_cutbacks += 1
            proposed_size = self.incrementation.after_failure(
                target - accepted_factor
            )
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
                ),
            )

        self.last_solve_info = J2LoadPathInfo(
            tuple(accepted),
            tuple(attempts),
            self.incrementation,
        )
        self._emit(
            reporter,
            SolveEvent(
                "step_completed",
                self.name,
                step_number=self.step_number,
                increment=len(accepted),
                attempt=len(attempts),
            ),
        )
        return self.solution

    def save_checkpoint(self, path) -> Path:
        """Save displacement, accepted load factor, and committed state."""

        if self.solution.function_space.mesh.comm.size != 1:
            raise NotImplementedError(
                "Portable distributed J2 checkpoints require global cell and "
                "dof identities and are not implemented yet."
            )
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        state = self.state.snapshot()
        np.savez(
            selected,
            schema="agentfem.j2-step-checkpoint.v1",
            displacement=self.solution.x.array,
            accepted_load_factor=self.accepted_load_factor,
            plastic_strain=state["plastic_strain"],
            equivalent_plastic_strain=state["equivalent_plastic_strain"],
        )
        return selected

    def load_checkpoint(self, path) -> None:
        """Restore a serial checkpoint into the same mesh/function layout."""

        if self.solution.function_space.mesh.comm.size != 1:
            raise NotImplementedError(
                "Portable distributed J2 checkpoints are not implemented yet."
            )
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != "agentfem.j2-step-checkpoint.v1":
                raise ValueError("Unsupported J2 step checkpoint schema.")
            displacement = np.asarray(data["displacement"])
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
            self.load_factor.value = PETSc.ScalarType(
                self.accepted_load_factor
            )

    def solve_result(self):
        from ..results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "state": self.state.summary(),
            },
        )

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
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "accepted_load_factor": self.accepted_load_factor,
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
                self.state.evaluate_strain(elasticity.strain(self.solution)),
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
            solve_matrix_system(
                tangent,
                rhs,
                correction,
                self.solver_options.linear_solver,
            )
            tangent.destroy()
            rhs.destroy()
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
                self.state.evaluate_strain(elasticity.strain(self.solution)),
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
        residual.scale(-1.0)
        fem_petsc.apply_lifting(
            residual,
            [self.tangent_form],
            [self.bcs],
            x0=[self.solution.x.petsc_vec],
            alpha=1.0,
        )
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        fem_petsc.set_bc(
            residual,
            self.bcs,
            x0=self.solution.x.petsc_vec,
            alpha=1.0,
        )
        return residual, float(residual.norm())

    def _reporter(self):
        if self.progress is True:
            return StandardRunReporter(
                comm_of(self.solution),
                status_file=self.status_file,
            )
        if self.progress in (False, None):
            return None
        return self.progress

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
            (),
            (info,),
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
    name: str = "j2_plasticity",
) -> J2PlasticityStep:
    """Build a global 3D J2 step from a displacement and load operator."""

    if not isinstance(material, J2LinearIsotropicHardening):
        raise TypeError("j2_plasticity_step requires J2LinearIsotropicHardening.")
    if displacement.value.function_space.mesh.geometry.dim != 3:
        raise NotImplementedError(
            "The first global J2 driver supports 3D small-strain solids. "
            "Plane stress needs a separate local return-map constraint."
        )
    domain = displacement.value.function_space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError(
            "The first global J2 provider is serial-only. Distributed "
            "quadrature ownership and portable cell identity must be verified "
            "before MPI execution is advertised."
        )
    state = J2QuadratureState.create(domain, degree=quadrature_degree)
    load_factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    strain_test = elasticity.strain(displacement.test)
    strain_trial = elasticity.strain(displacement.trial)
    stress = state.stress.function
    tangent = state.tangent.function
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        tangent[i, j, k, l] * strain_trial[k, l],
        (i, j),
    )
    residual = (
        ufl.inner(stress, strain_test) * state.measure
        - load_factor * external_force.expression
    )
    jacobian = ufl.inner(tangent_action, strain_test) * state.measure
    selected_bcs = []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected_bcs.extend(item.bcs)
        elif hasattr(item, "bc"):
            selected_bcs.append(item.bc)
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
        bcs=tuple(selected_bcs),
        incrementation=step_controls.normalize(incrementation),
        solver_options=newton() if solver_options is None else solver_options,
        study=study,
        progress=progress,
        status_file=status_file,
    )
