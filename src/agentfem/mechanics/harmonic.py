"""Direct harmonic finite-element procedure over inspectable K/M/C/F operators."""

from __future__ import annotations

from dataclasses import dataclass, field

from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc

from .. import procedures
from ..backends._harmonic import PreparedHarmonicLinearProblem
from ..operators.harmonic import DirectHarmonicSystem
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
    _prepared_problem: PreparedHarmonicLinearProblem | None = field(
        default=None, init=False, repr=False
    )

    @property
    def frequency(self) -> float:
        return self.angular_frequency / (2.0 * np.pi)

    @property
    def complex_dofs(self) -> np.ndarray:
        return self.solution_real.x.array.copy() + 1j * self.solution_imaginary.x.array

    def solve(self):
        """Solve the current frequency and return the real displacement field."""

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
        else:
            self._prepared_problem.set_angular_frequency(self.angular_frequency)
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
        return self.solution_real

    def set_frequency(
        self,
        *,
        frequency: float | None = None,
        angular_frequency: float | None = None,
    ) -> None:
        """Select another frequency while retaining the prepared backend."""

        self.angular_frequency = _angular_frequency(
            frequency=frequency,
            angular_frequency=angular_frequency,
        )

    def _refresh_polar_fields(self) -> None:
        real = np.asarray(self.solution_real.x.array, dtype=float)
        imaginary = np.asarray(self.solution_imaginary.x.array, dtype=float)
        amplitude = fem.Function(
            self.solution_real.function_space, name="U_AMPLITUDE"
        )
        phase = fem.Function(self.solution_real.function_space, name="U_PHASE")
        amplitude.x.array[:] = np.hypot(real, imaginary)
        phase.x.array[:] = np.arctan2(imaginary, real)
        amplitude.x.scatter_forward()
        phase.x.scatter_forward()
        self.displacement_amplitude = amplitude
        self.displacement_phase = phase

    def energy_evidence(self) -> dict[str, float]:
        """Return separated storage, inertia, and dissipation evidence."""

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
        return {
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

    def _quadratic_pair(self, operator) -> float:
        return _quadratic_integral(operator, self.solution_real) + _quadratic_integral(
            operator, self.solution_imaginary
        )

    def summary(self) -> dict[str, object]:
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
                None
                if self._prepared_problem is None
                else self._prepared_problem.summary()
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

    def solve_result(self, *, output=None, strict_output: bool = False):
        """Solve and delegate result assembly to the Result owner."""

        from ..results._harmonic import from_harmonic_step

        self.solve()
        return from_harmonic_step(
            self, output=output, strict_output=strict_output
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
    omega = _angular_frequency(
        frequency=frequency, angular_frequency=angular_frequency
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
    if options.pc_type.lower() != "fieldsplit":
        raise NotImplementedError(
            "The real-block harmonic provider currently requires "
            "pc_type='fieldsplit'."
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


__all__ = ["DirectHarmonicStep", "direct_harmonic_step"]
