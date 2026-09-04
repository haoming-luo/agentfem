"""Discrete problem and analysis-step containers for FEM workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from . import assembly
from . import fields
from . import time
from .diagnostics import PerformanceLedger
from .constraints.affine import AffineConstraintDualHistory
from .kernel import dofs
from .operators.core import LumpedMassOperator
from .solvers import (
    AffineNewtonOptions,
    LinearSolverOptions,
    NewtonSolverOptions,
    NonlinearSolverOptions,
    SolveEvent,
    prepare_linear_problem,
    prepare_mpc_linear_problem,
    solve_affine_nonlinear_path,
    solve_linear_problem,
    solve_nonlinear_problem,
)
from .state import (
    ExplicitDynamicsState,
    SecondOrderDynamicsState,
    TransientState,
    second_order_state,
)


@dataclass
class FEMProblem:
    """Lightweight finite-element problem description.

    This object is intentionally descriptive: it helps humans and agents inspect
    a model without hiding weak forms, assembly, or solver choices.
    """

    name: str
    domain: object
    spaces: dict[str, object] = field(default_factory=dict)
    fields: dict[str, object] = field(default_factory=dict)
    materials: list[object] = field(default_factory=list)
    constraints: list[object] = field(default_factory=list)
    loads: list[object] = field(default_factory=list)
    boundary_models: list[object] = field(default_factory=list)
    forms: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """Return a compact problem summary for logs or agent inspection."""

        return {
            "name": self.name,
            "topological_dim": self.domain.topology.dim,
            "geometric_dim": self.domain.geometry.dim,
            "spaces": tuple(self.spaces.keys()),
            "fields": tuple(self.fields.keys()),
            "materials": tuple(_describe_asset(material) for material in self.materials),
            "constraints": tuple(_describe_asset(item) for item in self.constraints),
            "loads": tuple(_describe_asset(item) for item in self.loads),
            "boundary_models": tuple(_describe_asset(item) for item in self.boundary_models),
            "forms": tuple(self.forms.keys()),
        }


@dataclass
class LinearVariationalProblem:
    """A standard linear variational problem, ``a(u, v) = L(v)``."""

    bilinear_form: object
    linear_form: object
    solution: object
    bcs: list = field(default_factory=list)
    solver_options: LinearSolverOptions | None = None
    last_solve_info: object | None = field(default=None, init=False)

    def solve(self):
        """Assemble and solve the problem into ``solution``."""

        solution, info = solve_linear_problem(
            self.bilinear_form,
            self.linear_form,
            self.solution,
            bcs=self.bcs,
            options=self.solver_options,
            return_info=True,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self, *, name: str = "linear_variational_result"):
        """Solve and wrap the solution in a scientific result object."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=name,
            metadata={"solve": self.last_solve_info.as_dict()},
        )


@dataclass
class LinearSystemProblem:
    """Engineering-level linear system problem, usually ``K x = F``."""

    system: object
    solution: object | None = None
    unknown: object | None = None
    bcs: list = field(default_factory=list)
    mpc_constraint: object | None = None
    solver_options: LinearSolverOptions | None = None
    last_solve_info: object | None = field(default=None, init=False)
    last_lifecycle_summary: dict[str, object] | None = field(default=None, init=False)

    @classmethod
    def from_operators(
        cls,
        K,
        F,
        *,
        unknown=None,
        solution=None,
        constraints=None,
        bcs=None,
        solver_options: LinearSolverOptions | None = None,
        name: str = "Kx_eq_F",
    ):
        """Create a linear-system problem from engineering notation."""

        from . import operators

        strong_bcs, mpc_constraint = _split_linear_constraints(
            constraints=constraints,
            bcs=bcs,
        )
        return cls(
            system=operators.linear_system(K, F, name=name),
            unknown=unknown,
            solution=solution,
            bcs=strong_bcs,
            mpc_constraint=mpc_constraint,
            solver_options=solver_options,
        )

    def solve(self):
        """Compile the system operators and solve into ``solution``."""

        prepared = self.prepare()
        try:
            return self.solve_prepared(prepared)
        finally:
            close = getattr(prepared, "close", None)
            if callable(close):
                close()

    def prepare(self):
        """Prepare the constant linear operator for one or more solves."""

        solution = self._solution()
        if self.mpc_constraint is None:
            return prepare_linear_problem(
                fem.form(self.system.lhs_form()),
                fem.form(self.system.rhs_form()),
                solution,
                bcs=self.bcs,
                options=self.solver_options,
            )
        return prepare_mpc_linear_problem(
            self.system.lhs_form(),
            self.system.rhs_form(),
            solution,
            self.mpc_constraint,
            bcs=self.bcs,
            options=self.solver_options,
            petsc_options_prefix="agentfem_linear_system_mpc_",
        )

    def solve_prepared(self, prepared):
        """Solve with a compatible prepared lifecycle and retain evidence."""

        solution = prepared.solve()
        self.last_solve_info = prepared.last_solve_info
        self.last_lifecycle_summary = dict(prepared.summary())
        return solution

    def solve_result(self, *, name: str | None = None):
        """Solve and return a :class:`SimulationResult`."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=name or getattr(self.system, "name", "linear_system_result"),
            metadata={"problem": self.summary()},
        )

    def summary(self) -> dict[str, object]:
        """Return an inspectable K/F problem summary."""

        return {
            "kind": "linear_system_problem",
            "system": self.system.summary() if hasattr(self.system, "summary") else repr(self.system),
            "solution": getattr(self._solution(), "name", repr(self._solution())),
            "num_bcs": len(self.bcs),
            "constraint_provider": (
                None
                if self.mpc_constraint is None
                else _describe_asset(self.mpc_constraint)
            ),
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else LinearSolverOptions().summary()
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "linear_lifecycle": self.last_lifecycle_summary,
        }

    def reaction_field(self, *, name: str = "RF"):
        """Return the unconstrained algebraic residual ``K u - F``.

        At converged free dofs this is zero up to solver tolerance; values at
        strongly constrained dofs are the nodal reactions. Weak, affine-MPC,
        and contact reactions require their own verified definitions.
        """

        import dolfinx.fem.petsc as fem_petsc
        from petsc4py import PETSc

        solution = self._solution()
        lhs = fem.form(self.system.lhs_form())
        rhs = fem.form(self.system.rhs_form())
        matrix = fem_petsc.assemble_matrix(lhs)
        matrix.assemble()
        external = fem_petsc.assemble_vector(rhs)
        external.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        residual = matrix.createVecLeft()
        matrix.mult(solution.x.petsc_vec, residual)
        residual.axpy(-1.0, external)
        reaction = fem.Function(solution.function_space, name=name)
        values = residual.array_r
        reaction.x.array[: len(values)] = values
        reaction.x.scatter_forward()
        residual.destroy()
        external.destroy()
        matrix.destroy()
        return reaction

    def _solution(self):
        if self.solution is not None:
            return self.solution
        if self.unknown is not None and hasattr(self.unknown, "value"):
            return self.unknown.value
        raise ValueError("LinearSystemProblem requires solution or unknown.")


@dataclass(frozen=True)
class ModalSolveInfo:
    """Convergence and filtering evidence for one modal solve."""

    converged_eigenpairs: int
    requested_modes: int
    accepted_modes: int
    constrained_dofs: int
    free_dofs: int
    residual_norms: tuple[float, ...]
    eigensolver: str
    target_frequency: float | None = None

    @property
    def converged(self) -> bool:
        return self.accepted_modes >= self.requested_modes

    def as_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "converged_eigenpairs": self.converged_eigenpairs,
            "requested_modes": self.requested_modes,
            "accepted_modes": self.accepted_modes,
            "constrained_dofs": self.constrained_dofs,
            "free_dofs": self.free_dofs,
            "residual_norms": self.residual_norms,
            "eigensolver": self.eigensolver,
            "target_frequency": self.target_frequency,
        }


@dataclass
class ModalAnalysisStep:
    """Constrained linear modes from ``K phi = lambda M phi``.

    Strong Dirichlet dofs are removed algebraically rather than assigned an
    artificial diagonal eigenvalue. This preserves the physical low spectrum
    and gives the same public Step/Result lifecycle as other analyses.
    """

    name: str
    target: object
    stiffness: object
    mass: object
    modes: int
    study: object | None = None
    constraints: tuple[object, ...] = ()
    bcs: tuple[object, ...] = ()
    target_frequency: float | None = None
    tolerance: float = 1.0e-9
    maximum_iterations: int = 1000
    rigid_mode_tolerance: float = 1.0e-10
    procedure: object | None = None
    eigenvalues: np.ndarray | None = field(default=None, init=False)
    mode_shapes: tuple[object, ...] = field(default=(), init=False)
    last_solve_info: ModalSolveInfo | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if int(self.modes) <= 0:
            raise ValueError("Modal analysis requires modes > 0.")
        if (
            not np.isfinite(self.tolerance)
            or self.tolerance <= 0.0
            or int(self.maximum_iterations) <= 0
        ):
            raise ValueError("Modal tolerance and maximum_iterations must be positive.")
        if (
            not np.isfinite(self.rigid_mode_tolerance)
            or self.rigid_mode_tolerance < 0.0
        ):
            raise ValueError("rigid_mode_tolerance must be nonnegative.")
        if self.target_frequency is not None and (
            not np.isfinite(self.target_frequency) or self.target_frequency < 0.0
        ):
            raise ValueError("target_frequency must be finite and nonnegative.")

    def solve(self):
        from .dependencies import require

        SLEPc = require(
            "slepc4py.SLEPc",
            extra="modal",
            capability="distributed structural modal analysis",
        )

        target = getattr(self.target, "value", self.target)
        V = target.function_space
        comm = V.mesh.comm
        selected_bcs = _collect_bcs(constraints=self.constraints, bcs=self.bcs)
        stiffness = self.stiffness.assemble_matrix(bcs=None)
        mass = self.mass.assemble_matrix(bcs=None)

        block_size = int(V.dofmap.index_map_bs)
        owned_blocks = int(V.dofmap.index_map.size_local)
        owned_scalar = owned_blocks * block_size
        constrained_local = []
        for bc in selected_bcs:
            indices, first_ghost = bc.dof_indices()
            constrained_local.extend(np.asarray(indices[:first_ghost], dtype=np.int64))
        constrained_local = np.unique(np.asarray(constrained_local, dtype=np.int64))
        constrained_local = constrained_local[constrained_local < owned_scalar]
        free_mask = np.ones(owned_scalar, dtype=bool)
        free_mask[constrained_local] = False
        free_local = np.flatnonzero(free_mask).astype(np.int32)

        local_blocks = free_local // block_size
        components = free_local % block_size
        global_blocks = V.dofmap.index_map.local_to_global(local_blocks)
        free_global = (
            np.asarray(global_blocks, dtype=PETSc.IntType) * block_size
            + components.astype(PETSc.IntType)
        )
        free_is = PETSc.IS().createGeneral(free_global, comm=comm)
        reduced_stiffness = stiffness.createSubMatrix(free_is, free_is)
        reduced_mass = mass.createSubMatrix(free_is, free_is)
        free_count = int(comm.allreduce(free_local.size, op=MPI.SUM))
        constrained_count = int(comm.allreduce(constrained_local.size, op=MPI.SUM))
        if free_count <= int(self.modes):
            raise ValueError(
                f"Modal analysis has {free_count} free dofs but requests {self.modes} modes."
            )

        eps = SLEPc.EPS().create(comm)
        eps.setOperators(reduced_stiffness, reduced_mass)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        requested = min(free_count - 1, int(self.modes) + min(8, free_count - int(self.modes) - 1))
        eps.setDimensions(requested)
        eps.setTolerances(tol=float(self.tolerance), max_it=int(self.maximum_iterations))
        target_eigenvalue = None
        if self.target_frequency is None:
            # Interior targeting at zero is substantially more reliable for
            # the smallest structural modes than an untransformed extremal
            # search, especially when stiffness and mass scales differ by
            # many orders of magnitude.
            eps.setTarget(0.0)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.getST().setType(SLEPc.ST.Type.SINVERT)
        else:
            target_eigenvalue = (2.0 * np.pi * float(self.target_frequency)) ** 2
            eps.setTarget(target_eigenvalue)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.getST().setType(SLEPc.ST.Type.SINVERT)
        eps.setFromOptions()
        eps.solve()

        converged = int(eps.getConverged())
        eigenvalues = []
        residual_norms = []
        mode_shapes = []
        reduced_vector = reduced_stiffness.createVecRight()
        candidates = []
        for index in range(converged):
            eigenvalue = eps.getEigenvalue(index)
            if abs(float(np.imag(eigenvalue))) > self.tolerance:
                continue
            candidates.append((float(np.real(eigenvalue)), index))
        scale = max(1.0, max((abs(item[0]) for item in candidates), default=1.0))
        candidates = [
            item
            for item in candidates
            if item[0] > self.rigid_mode_tolerance * scale
        ]
        if target_eigenvalue is None:
            selected_candidates = sorted(candidates, key=lambda item: item[0])[
                : int(self.modes)
            ]
        else:
            # SLEPc returns the eigenpairs closest to the declared shift. Keep
            # that physical selection when turning the solver output into an
            # ordered public result; sorting every candidate by eigenvalue
            # would silently turn a targeted solve back into a low-mode solve.
            selected_candidates = sorted(
                candidates,
                key=lambda item: abs(item[0] - target_eigenvalue),
            )[: int(self.modes)]
            selected_candidates.sort(key=lambda item: item[0])
        for eigenvalue, index in selected_candidates:
            eps.getEigenvector(index, reduced_vector)
            local_values = np.asarray(reduced_vector.array_r)
            if local_values.size != free_local.size:
                raise RuntimeError(
                    "Distributed modal subspace layout does not match the free-dof map."
                )
            mode = fem.Function(V, name=f"Mode_{len(mode_shapes) + 1}")
            mode.x.array[free_local] = np.real(local_values)
            mode.x.scatter_forward()
            eigenvalues.append(eigenvalue)
            residual_norms.append(
                float(eps.computeError(index, SLEPc.EPS.ErrorType.RELATIVE))
            )
            mode_shapes.append(mode)

        info = ModalSolveInfo(
            converged_eigenpairs=converged,
            requested_modes=int(self.modes),
            accepted_modes=len(mode_shapes),
            constrained_dofs=constrained_count,
            free_dofs=free_count,
            residual_norms=tuple(residual_norms),
            eigensolver=str(eps.getType()),
            target_frequency=self.target_frequency,
        )
        self.last_solve_info = info
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.mode_shapes = tuple(mode_shapes)

        reduced_vector.destroy()
        eps.destroy()
        reduced_stiffness.destroy()
        reduced_mass.destroy()
        free_is.destroy()
        stiffness.destroy()
        mass.destroy()
        if not info.converged:
            raise RuntimeError(
                f"Modal solve accepted {info.accepted_modes} of {info.requested_modes} requested modes."
            )
        return self.mode_shapes

    def solve_result(self, *, output=None, strict_output: bool = False):
        from .results import SimulationResult, complete_result

        modes = self.solve()
        assert self.eigenvalues is not None
        assert self.last_solve_info is not None
        result = SimulationResult(name=self.name)
        result.add_quantities(
            {
                "eigenvalues": self.eigenvalues,
                "angular_frequencies": np.sqrt(self.eigenvalues),
                "frequencies": np.sqrt(self.eigenvalues) / (2.0 * np.pi),
                "residual_norms": np.asarray(self.last_solve_info.residual_norms),
            },
            units={
                "eigenvalues": "rad^2/s^2",
                "angular_frequencies": "rad/s",
                "frequencies": "Hz",
            },
            kind="modal",
        )
        for index, mode in enumerate(modes, start=1):
            result.add_field(
                f"Mode_{index}",
                mode,
                unit="1",
                processing={
                    "method": "generalized_hermitian_eigenproblem",
                    "normalization": "mass",
                    "postprocessed": False,
                },
            )
        result.metadata["problem"] = self.summary()
        result.metadata["solve"] = self.last_solve_info.as_dict()
        return complete_result(
            self,
            result,
            output=output,
            strict_output=strict_output,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "modal_analysis_step",
            "name": self.name,
            "requested_modes": int(self.modes),
            "target_frequency": self.target_frequency,
            "constraints": len(_collect_bcs(constraints=self.constraints, bcs=self.bcs)),
            "stiffness": self.stiffness.summary(),
            "mass": self.mass.summary(),
            "procedure": None if self.procedure is None else self.procedure.summary(),
            "last_solve": None if self.last_solve_info is None else self.last_solve_info.as_dict(),
        }


@dataclass
class NonlinearVariationalProblem:
    """Nonlinear residual problem ``R(u; v) = 0`` solved by PETSc SNES."""

    residual_form: object
    solution: object
    bcs: list = field(default_factory=list)
    jacobian_form: object | None = None
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None
    name: str = "nonlinear_problem"
    petsc_options_prefix: str = "agentfem_nonlinear_"
    procedure: object | None = None
    last_solve_info: object | None = field(default=None, init=False)

    def solve(self):
        """Solve and return the live DOLFINx solution field."""

        solution, info = solve_nonlinear_problem(
            self.residual_form,
            self.solution,
            bcs=self.bcs,
            jacobian_form=self.jacobian_form,
            options=self.solver_options,
            petsc_options_prefix=self.petsc_options_prefix,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self):
        """Solve and return a result with SNES convergence evidence."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(
            solution,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "nonlinear_variational_problem",
            "name": self.name,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "num_bcs": len(self.bcs),
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else NonlinearSolverOptions().summary()
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }

    def reaction_field(self, *, name: str = "RF"):
        """Return the assembled nonlinear residual at the current solution."""

        return _reaction_field(self.residual_form, self.solution, name=name)


@dataclass(frozen=True)
class NonlinearLoadIncrementInfo:
    """Convergence evidence for one ordinary nonlinear load increment."""

    increment: int
    attempt: int
    start_load_factor: float
    load_factor: float
    converged: bool
    iterations: int
    residual_norm: float
    converged_reason: int
    message: str = ""
    checks: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "increment": self.increment,
            "attempt": self.attempt,
            "start_load_factor": self.start_load_factor,
            "load_factor": self.load_factor,
            "increment_size": self.load_factor - self.start_load_factor,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_norm": (
                float(self.residual_norm)
                if np.isfinite(self.residual_norm)
                else None
            ),
            "converged_reason": self.converged_reason,
            "message": self.message,
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class NonlinearLoadPathInfo:
    """Accepted and attempted increments for an ordinary nonlinear step."""

    increments: tuple[NonlinearLoadIncrementInfo, ...]
    attempts: tuple[NonlinearLoadIncrementInfo, ...]
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
            "kind": "nonlinear_load_path",
            "converged": self.converged,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": tuple(item.as_dict() for item in self.increments),
            "attempts": tuple(item.as_dict() for item in self.attempts),
        }


@dataclass
class IncrementalNonlinearVariationalProblem:
    """Ordinary nonlinear equilibrium with automatic load incrementation.

    Unlike :class:`AffineNonlinearVariationalProblem`, this path uses standard
    Dirichlet constraints. Natural loads and prescribed end-of-step values are
    driven by one normalized factor. Failed attempts restore the accepted
    field and boundary state before cutback.
    """

    residual_form: object
    solution: object
    factor: object
    value_path: object
    update_load: object | None = None
    acceptance_check: object | None = None
    bcs: list = field(default_factory=list)
    jacobian_form: object | None = None
    incrementation: object | None = None
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None
    output_every: int | None = 1
    progress: object = True
    status_file: object | None = None
    name: str = "incremental_nonlinear"
    petsc_options_prefix: str = "agentfem_incremental_nonlinear_"
    procedure: object | None = None
    result_field_factory: object | None = None
    snapshot_field_factory: object | None = None
    last_solve_info: NonlinearLoadPathInfo | None = field(default=None, init=False)
    snapshots: list = field(default_factory=list, init=False)
    execution_events: list = field(default_factory=list, init=False)

    def solve(self):
        from . import steps as step_controls
        from .diagnostics import (
            SolveEventRecorder,
            StandardRunReporter,
            comm_of,
            compose_reporters,
        )

        if self.output_every is not None and self.output_every <= 0:
            raise ValueError("Incremental nonlinear output_every must be positive.")
        control = step_controls.normalize(self.incrementation)
        selected_options = self.solver_options or NewtonSolverOptions()
        snes_options = (
            selected_options.for_snes()
            if isinstance(selected_options, NewtonSolverOptions)
            else selected_options
        )
        attempt_options = replace(snes_options, error_if_not_converged=False)
        recorder = SolveEventRecorder(self.execution_events)
        recorder.clear()
        if self.progress is True:
            reporter = compose_reporters(
                recorder,
                StandardRunReporter(
                    comm_of(self.solution),
                    status_file=self.status_file,
                    show_iterations=False,
                ),
            )
        elif self.progress in (False, None):
            reporter = recorder
        else:
            reporter = compose_reporters(recorder, self.progress)

        def emit(event) -> None:
            if reporter is not None:
                reporter.emit(event)

        self.snapshots.clear()
        self.snapshots.append(
            _load_snapshot(
                0,
                0.0,
                self.solution,
                field_factory=self.snapshot_field_factory,
            )
        )
        history: list[NonlinearLoadIncrementInfo] = []
        attempts: list[NonlinearLoadIncrementInfo] = []
        accepted_factor = 0.0
        total_attempts = 0
        cutbacks = 0
        proposed_size = (
            control.initial
            if isinstance(control, step_controls.AutomaticIncrementation)
            else control.load_factors[0]
        )
        self._update_factor(0.0)
        emit(
            SolveEvent(
                "step_started",
                self.name,
                incrementation=control.summary()["kind"],
            )
        )

        while accepted_factor < 1.0 - 1.0e-12:
            increment_number = len(history) + 1
            if isinstance(control, step_controls.AutomaticIncrementation):
                if len(history) >= control.max_increments:
                    self._fail(
                        history,
                        attempts,
                        control,
                        emit,
                        increment_number,
                        total_attempts,
                        accepted_factor,
                        "maximum accepted increments reached before load factor 1.0",
                    )
                factor = min(1.0, accepted_factor + proposed_size)
            else:
                factor = control.load_factors[len(history)]
            attempt_number = cutbacks + 1
            total_attempts += 1
            emit(
                SolveEvent(
                    "increment_started",
                    self.name,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=accepted_factor,
                    target_factor=factor,
                )
            )
            rollback = self.solution.x.array.copy()
            self._update_factor(factor)
            solve_info = None
            message = ""
            try:
                _, solve_info = solve_nonlinear_problem(
                    self.residual_form,
                    self.solution,
                    bcs=self.bcs,
                    jacobian_form=self.jacobian_form,
                    options=attempt_options,
                    petsc_options_prefix=(
                        f"{self.petsc_options_prefix}{increment_number}_{attempt_number}_"
                    ),
                )
                converged = solve_info.converged and bool(
                    np.all(np.isfinite(self.solution.x.array))
                )
            except (RuntimeError, ValueError) as exc:
                converged = False
                message = f"{type(exc).__name__}: {exc}"
            checks = {}
            if converged and self.acceptance_check is not None:
                checks = dict(self.acceptance_check())
                if not bool(checks.get("accepted", True)):
                    converged = False
                    message = str(
                        checks.get(
                            "message",
                            "increment failed its physical acceptance check",
                        )
                    )
            info = NonlinearLoadIncrementInfo(
                increment=increment_number,
                attempt=attempt_number,
                start_load_factor=accepted_factor,
                load_factor=factor,
                converged=converged,
                iterations=0 if solve_info is None else solve_info.iterations,
                residual_norm=(
                    float("nan") if solve_info is None else solve_info.function_norm
                ),
                converged_reason=(
                    0 if solve_info is None else solve_info.converged_reason
                ),
                message=message,
                checks=checks,
            )
            attempts.append(info)
            if converged:
                history.append(info)
                accepted_size = factor - accepted_factor
                accepted_factor = factor
                cutbacks = 0
                if (
                    self.output_every is not None
                    and (
                        len(history) % self.output_every == 0
                        or abs(factor - 1.0) <= 1.0e-12
                    )
                ):
                    self.snapshots.append(
                        _load_snapshot(
                            len(history),
                            factor,
                            self.solution,
                            solve_info=solve_info,
                            field_factory=self.snapshot_field_factory,
                        )
                    )
                emit(
                    SolveEvent(
                        "increment_converged",
                        self.name,
                        increment=increment_number,
                        attempt=attempt_number,
                        start_factor=info.start_load_factor,
                        target_factor=factor,
                        iteration=info.iterations,
                        residual_norm=info.residual_norm,
                    )
                )
                if isinstance(control, step_controls.AutomaticIncrementation):
                    proposed_size = control.after_convergence(
                        accepted_size,
                        info.iterations,
                    )
                continue

            self.solution.x.array[:] = rollback
            self.solution.x.scatter_forward()
            self._update_factor(accepted_factor)
            if isinstance(control, step_controls.FixedIncrementation):
                self._fail(
                    history,
                    attempts,
                    control,
                    emit,
                    increment_number,
                    total_attempts,
                    factor,
                    f"fixed increment failed at load factor {factor:.6g}",
                    info=info,
                )
            cutbacks += 1
            next_size = control.after_failure(factor - accepted_factor)
            if cutbacks > control.max_cutbacks or next_size < control.minimum - 1.0e-15:
                self._fail(
                    history,
                    attempts,
                    control,
                    emit,
                    increment_number,
                    total_attempts,
                    factor,
                    "automatic increment exhausted its cutback policy",
                    info=info,
                )
            proposed_size = max(control.minimum, next_size)
            emit(
                SolveEvent(
                    "increment_cutback",
                    self.name,
                    increment=increment_number,
                    attempt=attempt_number,
                    start_factor=accepted_factor,
                    target_factor=factor,
                    iteration=info.iterations,
                    residual_norm=info.residual_norm,
                    next_increment=proposed_size,
                )
            )

        self.last_solve_info = NonlinearLoadPathInfo(
            tuple(history), tuple(attempts), control
        )
        emit(
            SolveEvent(
                "step_completed",
                self.name,
                increment=len(history),
                attempt=total_attempts,
                target_factor=1.0,
            )
        )
        return self.solution

    def _update_factor(self, factor: float) -> None:
        self.factor.value = PETSc.ScalarType(factor)
        self.value_path.update(factor)
        if self.update_load is not None:
            self.update_load(float(factor))

    def _fail(
        self,
        history,
        attempts,
        control,
        emit,
        increment,
        total_attempts,
        factor,
        message,
        *,
        info=None,
    ) -> None:
        self.last_solve_info = NonlinearLoadPathInfo(
            tuple(history), tuple(attempts), control
        )
        emit(
            SolveEvent(
                "step_failed",
                self.name,
                increment=increment,
                attempt=total_attempts,
                target_factor=factor,
                iteration=0 if info is None else info.iterations,
                residual_norm=None if info is None else info.residual_norm,
                message=message,
            )
        )
        raise RuntimeError(message)

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve and complete one model-owned nonlinear result lifecycle."""

        from .results import add_execution_trace, complete_result, from_solution

        solution = self.solve()
        generated = (
            () if self.result_field_factory is None
            else tuple(self.result_field_factory())
        )
        primary = solution if not generated else generated[0]
        result = from_solution(
            primary,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
            },
        )
        for field in generated[1:]:
            result.add_field(
                getattr(field, "name", type(field).__name__),
                field,
                processing={
                    "method": "primary_mixed_finite_element_subfield",
                    "representation": "collapsed_finite_element_dofs",
                    "postprocessed": False,
                },
            )
        for selected in fields:
            function = _unwrap_result_field(selected)
            result.add_field(
                getattr(function, "name", type(function).__name__),
                function,
                location=_field_location(function),
            )
        add_execution_trace(result, self.execution_events)
        return complete_result(
            self,
            result,
            output=output,
            fields=fields,
            strict_output=strict_output,
            metadata=metadata,
        )

    def reaction_field(self, *, name: str = "RF"):
        return _reaction_field(self.residual_form, self.solution, name=name)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "incremental_nonlinear_variational_problem",
            "name": self.name,
            "num_bcs": len(self.bcs),
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "snapshot_count": len(self.snapshots),
            "primary_result_fields": (
                None if self.result_field_factory is None else tuple(self.primary_fields)
                if hasattr(self, "primary_fields") else "generated"
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class AffineNonlinearVariationalProblem:
    """Nonlinear equilibrium under an exact affine dof reduction."""

    residual_form: object
    jacobian_form: object
    solution: object
    constraint: object
    load_factors: tuple[float, ...] | None = None
    incrementation: object | None = None
    solver_options: AffineNewtonOptions | NewtonSolverOptions | None = None
    output_every: int | None = 1
    output_factors: tuple[float, ...] = ()
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    name: str = "affine_nonlinear_problem"
    procedure: object | None = None
    result_field_factory: object | None = None
    snapshot_field_factory: object | None = None
    state_transaction: object | None = None
    checkpoint_policy: object | None = None
    acceptance_check: object | None = None
    accepted_observers: tuple[object, ...] = ()
    last_solve_info: object | None = field(default=None, init=False)
    snapshots: list = field(default_factory=list, init=False)
    accepted_history_recorders: dict[str, object] = field(
        default_factory=dict,
        init=False,
    )
    execution_events: list[object] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    accepted_load_factor: float = field(default=0.0, init=False)
    accepted_increments: list[object] = field(default_factory=list, init=False)
    attempted_increments: list[object] = field(default_factory=list, init=False)
    next_increment_size: float | None = field(default=None, init=False)
    constraint_dual_history: AffineConstraintDualHistory = field(
        default_factory=AffineConstraintDualHistory,
        init=False,
    )

    def _capture_constraint_dual(self, load_factor: float) -> None:
        provider = getattr(self.constraint, "dual_evidence", None)
        if provider is None:
            return
        evidence = provider(self, load_factor=float(load_factor))
        if evidence is None:
            return
        outgoing_evidence = provider(
            self,
            load_factor=float(load_factor),
            path_side="right",
        )
        self.constraint_dual_history.append(
            load_factor,
            evidence,
            outgoing_evidence=outgoing_evidence,
        )

    def solve(self, *, until: float = 1.0):
        """Advance to an accepted load factor without discarding prior history."""

        selected_until = float(until)
        if not np.isfinite(selected_until) or not 0.0 < selected_until <= 1.0:
            raise ValueError("Affine solve until must be finite and in (0, 1].")
        if selected_until <= self.accepted_load_factor + 1.0e-12:
            raise ValueError(
                "Affine solve until must exceed the currently accepted load factor."
            )
        if self.output_every is not None and self.output_every <= 0:
            raise ValueError("Affine nonlinear output_every must be positive.")
        # Validate rollback support before initialization or any accepted
        # observer is allowed to mutate its history.
        initial_observer_state = _snapshot_accepted_observers(
            self.accepted_observers,
            comm=self.solution.function_space.mesh.comm,
        )
        initial_snapshots = list(self.snapshots)
        initial_events = list(self.execution_events)
        initial_solution = self.solution.x.array.copy()
        initial_transaction_state = (
            None
            if self.state_transaction is None
            else self.state_transaction.snapshot_accepted_boundary()
        )
        self.snapshots.clear()
        fresh = self.accepted_load_factor <= 1.0e-12
        try:
            if fresh:
                self.execution_events.clear()
                self.constraint_dual_history.clear()
                if self.state_transaction is not None and hasattr(
                    self.state_transaction, "initialize"
                ):
                    self.state_transaction.initialize()
            elif self.state_transaction is not None:
                transaction_factor = float(
                    getattr(self.state_transaction, "accepted_factor", -1.0)
                )
                if abs(transaction_factor - self.accepted_load_factor) > 1.0e-12:
                    raise RuntimeError(
                        "Affine problem and material transaction disagree on the "
                        "accepted load factor."
                    )
                if hasattr(self.state_transaction, "prepare_resume"):
                    self.state_transaction.prepare_resume()
            initial_snapshot = _load_snapshot(
                len(self.accepted_increments),
                self.accepted_load_factor,
                self.solution,
                zero=fresh,
                zero_fields=self.state_transaction is None,
                field_factory=self.snapshot_field_factory,
            )
            self.snapshots.append(initial_snapshot)
            for observer in self.accepted_observers:
                if not fresh and hasattr(observer, "prepare_resume"):
                    observer.prepare_resume(initial_snapshot)
                else:
                    observer.reset(initial_snapshot)
            if fresh:
                self._capture_constraint_dual(self.accepted_load_factor)
        except BaseException:
            self.solution.x.array[:] = initial_solution
            self.solution.x.scatter_forward()
            if self.state_transaction is not None:
                self.state_transaction.restore_accepted_boundary(
                    initial_transaction_state
                )
            self.snapshots[:] = initial_snapshots
            self.execution_events[:] = initial_events
            _restore_accepted_observers(initial_observer_state)
            raise

        pending_acceptance = None

        def acceptance_backup():
            return {
                "snapshots": list(self.snapshots),
                "observers": _snapshot_accepted_observers(
                    self.accepted_observers,
                    comm=self.solution.function_space.mesh.comm,
                ),
                "accepted_load_factor": self.accepted_load_factor,
                "accepted_increments": list(self.accepted_increments),
                "attempted_increments": list(self.attempted_increments),
                "next_increment_size": self.next_increment_size,
                "execution_events": list(self.execution_events),
                "checkpoints": list(self.checkpoints),
                "constraint_dual_history": (
                    self.constraint_dual_history.snapshot_runtime_state()
                ),
            }

        def restore_pending_acceptance():
            nonlocal pending_acceptance
            if pending_acceptance is None:
                return
            state = pending_acceptance
            self.snapshots[:] = state["snapshots"]
            _restore_accepted_observers(state["observers"])
            self.accepted_load_factor = state["accepted_load_factor"]
            self.accepted_increments[:] = state["accepted_increments"]
            self.attempted_increments[:] = state["attempted_increments"]
            self.next_increment_size = state["next_increment_size"]
            self.execution_events[:] = state["execution_events"]
            self.checkpoints[:] = state["checkpoints"]
            self.constraint_dual_history.restore_runtime_state(
                state["constraint_dual_history"]
            )
            pending_acceptance = None

        def capture(index, factor, solution, solve_info):
            nonlocal pending_acceptance
            if pending_acceptance is not None:
                raise RuntimeError(
                    "An accepted-boundary callback transaction is already active."
                )
            pending_acceptance = acceptance_backup()
            save_by_increment = (
                self.output_every is not None
                and index % self.output_every == 0
            )
            save_by_factor = any(
                abs(factor - value) <= 1.0e-12
                for value in self.output_factors
            )
            should_save = (
                save_by_increment
                or save_by_factor
                or abs(factor - 1.0) <= 1.0e-12
            )
            try:
                accepted_snapshot = None
                if self.accepted_observers or should_save:
                    accepted_snapshot = _load_snapshot(
                        index,
                        factor,
                        solution,
                        solve_info=solve_info,
                        field_factory=self.snapshot_field_factory,
                    )
                for observer in self.accepted_observers:
                    observer.accept(accepted_snapshot)
                self._capture_constraint_dual(factor)
                if should_save:
                    self.snapshots.append(accepted_snapshot)
            except BaseException:
                restore_pending_acceptance()
                raise

        def accept_boundary(
            _solution,
            accepted_history,
            attempted_history,
            next_increment_size,
        ):
            """Synchronize and checkpoint one fully committed boundary."""

            nonlocal pending_acceptance
            if pending_acceptance is None:
                pending_acceptance = acceptance_backup()
            try:
                self.accepted_increments[:] = list(accepted_history)
                self.attempted_increments[:] = list(attempted_history)
                self.accepted_load_factor = float(
                    accepted_history[-1].load_factor
                )
                self.next_increment_size = next_increment_size
                policy = self.checkpoint_policy
                if policy is not None:
                    increment = len(self.accepted_increments)
                    due = increment % int(policy.every) == 0
                    due = due or (
                        bool(policy.final)
                        and self.accepted_load_factor >= 1.0 - 1.0e-12
                    )
                    if due:
                        self.save_checkpoint(
                            policy.path(
                                step_name=self.name,
                                increment=increment,
                            ),
                            portable=bool(policy.portable),
                        )
                        _prune_affine_checkpoints(self)
            except BaseException:
                restore_pending_acceptance()
                raise
            pending_acceptance = None

        from .diagnostics import (
            SolveEventRecorder,
            StandardRunReporter,
            comm_of,
            compose_reporters,
        )

        recorder = SolveEventRecorder(self.execution_events)
        if self.progress is True:
            visible_reporter = StandardRunReporter(
                comm_of(self.solution),
                status_file=self.status_file,
            )
        elif self.progress in (False, None):
            visible_reporter = None
        else:
            visible_reporter = self.progress
        reporter = compose_reporters(recorder, visible_reporter)
        solution, info = solve_affine_nonlinear_path(
            self.residual_form,
            self.jacobian_form,
            self.solution,
            self.constraint,
            load_factors=self.load_factors,
            incrementation=self.incrementation,
            output_factors=self.output_factors,
            options=self.solver_options,
            on_increment=capture,
            on_accepted_boundary=accept_boundary,
            on_acceptance_failure=restore_pending_acceptance,
            acceptance_check=self.acceptance_check,
            state_transaction=self.state_transaction,
            stop_factor=selected_until,
            accepted_history=self.accepted_increments,
            attempted_history=self.attempted_increments,
            next_increment_size=self.next_increment_size,
            reporter=reporter,
            step_name=self.name,
            step_number=self.step_number,
        )
        self.last_solve_info = info
        self.accepted_increments[:] = list(info.increments)
        self.attempted_increments[:] = list(info.attempts)
        self.accepted_load_factor = (
            self.accepted_load_factor
            if not info.increments
            else float(info.increments[-1].load_factor)
        )
        self.next_increment_size = info.next_increment_size
        return solution

    def _checkpoint_identity(self) -> dict[str, object]:
        """Return the partition-independent scientific identity of this path."""

        from .checkpointing import function_portable_identity

        transaction = self.state_transaction
        material = getattr(transaction, "material", getattr(self, "material", None))
        response = getattr(transaction, "response", getattr(self, "response", None))
        constraint_identity = (
            self.constraint.scientific_identity()
            if hasattr(self.constraint, "scientific_identity")
            else self.constraint.summary()
        )
        return {
            "step_name": self.name,
            "procedure": (
                self.procedure.summary()
                if hasattr(self.procedure, "summary")
                else self.procedure
            ),
            "material": (
                material.summary() if hasattr(material, "summary") else material
            ),
            "state_schema": (
                response.state.state_schema.summary()
                if response is not None
                else None
            ),
            "quadrature_degree": (
                None if response is None else int(response.state.degree)
            ),
            "incrementation": (
                self.incrementation.summary()
                if hasattr(self.incrementation, "summary")
                else self.incrementation
            ),
            "solver_options": (
                self.solver_options.summary()
                if hasattr(self.solver_options, "summary")
                else self.solver_options
            ),
            "solution": function_portable_identity(self.solution),
            "constraint": constraint_identity,
            "accepted_history_recorders": {
                name: {"kind": type(recorder).__name__}
                for name, recorder in sorted(
                    self.accepted_history_recorders.items()
                )
            },
        }

    @staticmethod
    def _checkpoint_manifest_path(path) -> Path:
        selected = Path(path)
        if selected.name.endswith(".checkpoint.json"):
            return selected
        if selected.suffix:
            selected = selected.with_suffix("")
        return selected.with_name(selected.name + ".checkpoint.json")

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        """Save one accepted affine state and its complete path evidence.

        The stable stateful route always writes coordinate/cell-keyed state.
        This is deliberately stronger than a rank-local default: the same
        checkpoint can be resumed with a different compatible MPI partition.
        """

        if self.state_transaction is None:
            raise TypeError("Affine checkpointing requires a state transaction.")
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
            self.state_transaction.accepted_solution.x.array,
            rtol=0.0,
            atol=1.0e-12,
        ):
            local_problem = (
                "Checkpointing is permitted only when U equals U_ACCEPTED."
            )
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems)
                if problem is not None
            )
            raise RuntimeError(f"Rank {rank}: {problems[rank]}")
        # ``portable=False`` is retained for compatibility with the common
        # policy API; the public affine format is intentionally always portable.
        del portable
        from .checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
        )
        from .results import CheckpointRecord

        manifest = self._checkpoint_manifest_path(path)
        bundle = save_portable_state_bundle(
            manifest,
            state={
                "U": self.solution,
                "U_ACCEPTED": self.state_transaction.accepted_solution,
            },
        )
        quadrature = self.state_transaction.response.state.save(
            manifest.with_name(
                f"{manifest.name.removesuffix('.checkpoint.json')}."
                f"{bundle['generation']}.quadrature"
            ),
            material=self.state_transaction.material,
        )
        payload = {
            "schema": "agentfem.affine-stateful-checkpoint.v1",
            "identity": self._checkpoint_identity(),
            "coordinate_name": "load_factor",
            "coordinate": self.accepted_load_factor,
            "writer_rank_count": int(self.solution.function_space.mesh.comm.size),
            "portable": True,
            "nodal_state": bundle["record"],
            "nodal_identity": bundle["identities"],
            "quadrature_state": checkpoint_file_record(quadrature),
            "accepted_increments": [
                item.as_dict() for item in self.accepted_increments
            ],
            "attempted_increments": [
                item.as_dict() for item in self.attempted_increments
            ],
            "next_increment_size": self.next_increment_size,
            "execution_events": [
                item.as_dict() if hasattr(item, "as_dict") else dict(item)
                for item in self.execution_events
            ],
            "accepted_observer_state": {
                name: recorder.checkpoint_state()
                for name, recorder in sorted(
                    self.accepted_history_recorders.items()
                )
                if hasattr(recorder, "checkpoint_state")
            },
            "constraint_dual_history": (
                self.constraint_dual_history.checkpoint_state()
            ),
        }
        error = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest,
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"Affine checkpoint manifest write failed: {error}")
        comm.barrier()
        record = CheckpointRecord(
            name=f"{self.name}_increment_{len(self.accepted_increments)}",
            path=manifest,
            schema=payload["schema"],
            step_name=self.name,
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=True,
            metadata={
                "accepted_increment_count": len(self.accepted_increments),
                "writer_rank_count": int(comm.size),
                "role": "accepted_state",
            },
        )
        self.checkpoints[:] = [
            item for item in self.checkpoints if item.path != manifest
        ]
        self.checkpoints.append(record)
        return manifest

    def load_checkpoint(self, path) -> None:
        """Atomically restore an accepted affine state and resume metadata."""

        if self.state_transaction is None:
            raise TypeError("Affine checkpointing requires a state transaction.")
        from .checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )
        from .results import CheckpointRecord
        from .solvers import AffineLoadIncrementInfo

        manifest = self._checkpoint_manifest_path(path)
        comm = self.solution.function_space.mesh.comm
        envelope = None
        if comm.rank == 0:
            try:
                envelope = {
                    "payload": json.loads(manifest.read_text(encoding="utf-8")),
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
                f"Affine checkpoint manifest read failed: {envelope['error']}"
            )
        payload = envelope["payload"]
        if payload.get("schema") != "agentfem.affine-stateful-checkpoint.v1":
            raise ValueError("Unsupported affine stateful checkpoint schema.")
        current_identity = json.loads(
            json.dumps(self._checkpoint_identity(), sort_keys=True)
        )
        if payload.get("identity") != current_identity:
            raise ValueError(
                "Affine checkpoint material, state, mesh, increment/solver "
                "control, constraint equations, or deformation history differ "
                "from the current analysis."
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
                "Affine checkpoint payload validation failed: "
                f"{validation['error']}"
            )
        quadrature_path = Path(validation["quadrature_path"])
        solution_backup = self.solution.x.array.copy()
        accepted_backup = self.state_transaction.accepted_solution.x.array.copy()
        transaction_backup = self.state_transaction.snapshot_runtime_state()
        lifecycle_backup = {
            "accepted_load_factor": self.accepted_load_factor,
            "accepted_increments": list(self.accepted_increments),
            "attempted_increments": list(self.attempted_increments),
            "next_increment_size": self.next_increment_size,
            "execution_events": list(self.execution_events),
            "last_solve_info": self.last_solve_info,
            "snapshots": list(self.snapshots),
            "checkpoints": list(self.checkpoints),
            "constraint_dual_history": (
                self.constraint_dual_history.snapshot_runtime_state()
            ),
        }
        observer_backup = {
            name: recorder.snapshot_runtime_state()
            for name, recorder in self.accepted_history_recorders.items()
            if hasattr(recorder, "snapshot_runtime_state")
        }
        try:
            load_portable_state_bundle(
                manifest,
                state={
                    "U": self.solution,
                    "U_ACCEPTED": self.state_transaction.accepted_solution,
                },
                record=payload["nodal_state"],
                identities=payload["nodal_identity"],
            )
            self.state_transaction.response.state.load(
                quadrature_path,
                material=self.state_transaction.material,
            )
            local_equal = np.allclose(
                self.solution.x.array,
                self.state_transaction.accepted_solution.x.array,
                rtol=0.0,
                atol=1.0e-12,
            )
            if not comm.allreduce(bool(local_equal), op=MPI.LAND):
                raise ValueError("Affine checkpoint U and U_ACCEPTED differ.")
            accepted = [
                AffineLoadIncrementInfo.from_dict(item)
                for item in payload["accepted_increments"]
            ]
            attempted = [
                AffineLoadIncrementInfo.from_dict(item)
                for item in payload["attempted_increments"]
            ]
            coordinate = float(payload["coordinate"])
            if not accepted or abs(accepted[-1].load_factor - coordinate) > 1.0e-12:
                raise ValueError(
                    "Affine checkpoint coordinate and accepted history disagree."
                )
            self.accepted_increments[:] = accepted
            self.attempted_increments[:] = attempted
            self.accepted_load_factor = coordinate
            self.next_increment_size = payload.get("next_increment_size")
            self.execution_events[:] = [
                SolveEvent.from_dict(item)
                for item in payload.get("execution_events", ())
            ]
            self.constraint_dual_history.restore_checkpoint_state(
                payload.get("constraint_dual_history"),
                accepted_factor=coordinate,
            )
            self.state_transaction.accepted_factor = coordinate
            self.state_transaction.prepare_resume()
            observer_state = payload.get("accepted_observer_state", {})
            if set(observer_state) != set(self.accepted_history_recorders):
                raise ValueError(
                    "Affine checkpoint observer history differs from the "
                    "current output lifecycle."
                )
            current_snapshot = _load_snapshot(
                len(accepted),
                coordinate,
                self.solution,
                solve_info=accepted[-1],
                field_factory=self.snapshot_field_factory,
            )
            for name, record in observer_state.items():
                recorder = self.accepted_history_recorders[name]
                if not hasattr(recorder, "restore_checkpoint_state"):
                    raise TypeError(
                        f"Accepted observer {name!r} is not restartable."
                    )
                recorder.restore_checkpoint_state(
                    record,
                    current_snapshot=current_snapshot,
                )
        except Exception:
            self.solution.x.array[:] = solution_backup
            self.solution.x.scatter_forward()
            self.state_transaction.accepted_solution.x.array[:] = accepted_backup
            self.state_transaction.accepted_solution.x.scatter_forward()
            self.state_transaction.restore_runtime_state(transaction_backup)
            self.accepted_load_factor = lifecycle_backup[
                "accepted_load_factor"
            ]
            self.accepted_increments[:] = lifecycle_backup[
                "accepted_increments"
            ]
            self.attempted_increments[:] = lifecycle_backup[
                "attempted_increments"
            ]
            self.next_increment_size = lifecycle_backup[
                "next_increment_size"
            ]
            self.execution_events[:] = lifecycle_backup["execution_events"]
            self.last_solve_info = lifecycle_backup["last_solve_info"]
            self.snapshots[:] = lifecycle_backup["snapshots"]
            self.checkpoints[:] = lifecycle_backup["checkpoints"]
            self.constraint_dual_history.restore_runtime_state(
                lifecycle_backup["constraint_dual_history"]
            )
            for name, state in observer_backup.items():
                recorder = self.accepted_history_recorders.get(name)
                if recorder is not None and hasattr(
                    recorder,
                    "restore_runtime_state",
                ):
                    recorder.restore_runtime_state(state)
            raise
        record = CheckpointRecord(
            name=f"{self.name}_restart_{len(self.accepted_increments)}",
            path=manifest,
            schema=payload["schema"],
            step_name=self.name,
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=True,
            metadata={
                "writer_rank_count": payload.get("writer_rank_count"),
                "reader_rank_count": int(comm.size),
                "restart_mode": "portable_coordinate_and_cell_keyed_state",
                "role": "restart_source",
            },
        )
        self.checkpoints.append(record)

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve and complete one affine nonlinear result lifecycle."""

        from .results import add_execution_trace, complete_result, from_solution
        from . import constraints as constraint_api

        solution = self.solve()
        generated = (
            () if self.result_field_factory is None
            else tuple(self.result_field_factory())
        )
        primary = solution if not generated else generated[0]
        result = from_solution(
            primary,
            name=self.name,
            metadata={
                "problem": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "state": (
                    None
                    if self.state_transaction is None
                    else (
                        self.state_transaction.summary()
                        if hasattr(self.state_transaction, "summary")
                        else {"kind": type(self.state_transaction).__name__}
                    )
                ),
            },
        )
        for field in generated[1:]:
            result.add_field(
                getattr(field, "name", type(field).__name__),
                field,
                processing={
                    "method": "primary_mixed_finite_element_subfield",
                    "representation": "collapsed_finite_element_dofs",
                    "postprocessed": False,
                },
            )
        for selected in fields:
            function = _unwrap_result_field(selected)
            result.add_field(
                getattr(function, "name", type(function).__name__),
                function,
                location=_field_location(function),
            )
        transaction_fields = ()
        if self.state_transaction is not None and hasattr(
            self.state_transaction, "populate_result"
        ):
            transaction_fields = tuple(
                self.state_transaction.populate_result(result) or ()
            )
        for checkpoint in self.checkpoints:
            result.add_checkpoint(checkpoint)
        provider_duals = constraint_api.collect_provider_duals(
            (self.constraint,),
            self,
        )
        balance_contract = constraint_api.constraint_balance_contract(
            (self.constraint,),
            provider_duals=provider_duals,
        )
        result.metadata["constraint_balance_contract"] = balance_contract
        result.metadata["constraint_duals"] = tuple(
            item.summary() for item in provider_duals
        )
        if len(provider_duals) == 1:
            dual = provider_duals[0]
            result.add_quantities(
                {
                    "affine_path_generalized_reaction": float(dual.force[0]),
                    "affine_constraint_force_resultant": dual.resultant,
                },
                kind="diagnostic",
                descriptions={
                    "affine_path_generalized_reaction": (
                        "Virtual work of the converged full residual against a "
                        "unit increment of the prescribed affine path."
                    ),
                    "affine_constraint_force_resultant": (
                        "Physical-space resultant of the converged displacement "
                        "residual owned by the affine constraint provider."
                    ),
                },
            )
        dual_history = tuple(self.constraint_dual_history.records)
        complete_dual_path = bool(
            len(dual_history) >= 2
            and abs(float(dual_history[0]["load_factor"])) <= 1.0e-12
            and abs(
                float(dual_history[-1]["load_factor"])
                - self.accepted_load_factor
            )
            <= 1.0e-12
        )
        result.metadata["affine_constraint_path_work"] = {
            "status": "complete" if complete_dual_path else "unavailable",
            "sample_count": len(dual_history),
            "integration": "accepted_path_trapezoidal",
            "reason": (
                None
                if complete_dual_path
                else "Accepted generalized-force history does not start at zero."
            ),
        }
        if complete_dual_path:
            factors = self.constraint_dual_history.factors
            forces = self.constraint_dual_history.forces
            path_work = self.constraint_dual_history.work()
            result.add_history(
                "affine_path_generalized_reaction",
                factors,
                forces,
                abscissa_name="load_factor",
                description=(
                    "Full-residual generalized reaction conjugate to the "
                    "accepted affine path coordinate."
                ),
            )
            result.add_history(
                "affine_path_outgoing_generalized_reaction",
                factors,
                self.constraint_dual_history.outgoing_forces,
                abscissa_name="load_factor",
                description=(
                    "Right-sided generalized reaction at affine path knots; "
                    "equal to the incoming value on a smooth path."
                ),
            )
            result.add_quantity(
                "affine_constraint_path_work",
                path_work,
                kind="diagnostic",
                description=(
                    "Trapezoidal work of the affine generalized reaction over "
                    "all accepted load-path increments."
                ),
            )
            result.metadata["affine_constraint_path_work"]["value"] = path_work
        add_execution_trace(result, self.execution_events)
        selected_completion_fields = (*transaction_fields, *tuple(fields))
        return complete_result(
            self,
            result,
            output=output,
            fields=selected_completion_fields,
            strict_output=strict_output,
            metadata=metadata,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "affine_nonlinear_variational_problem",
            "name": self.name,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "constraint": self.constraint.summary(),
            "load_factors": self.load_factors,
            "incrementation": (
                None
                if self.incrementation is None
                else self.incrementation.summary()
            ),
            "output_every": self.output_every,
            "output_factors": self.output_factors,
            "snapshot_count": len(self.snapshots),
            "accepted_observers": tuple(
                observer.summary() for observer in self.accepted_observers
            ),
            "execution_event_count": len(self.execution_events),
            "accepted_load_factor": self.accepted_load_factor,
            "accepted_increment_count": len(self.accepted_increments),
            "attempted_increment_count": len(self.attempted_increments),
            "next_increment_size": self.next_increment_size,
            "constraint_dual_sample_count": len(
                self.constraint_dual_history.records
            ),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "checkpoint_count": len(self.checkpoints),
            "state_transaction": (
                None
                if self.state_transaction is None
                else (
                    self.state_transaction.summary()
                    if hasattr(self.state_transaction, "summary")
                    else {"kind": type(self.state_transaction).__name__}
                )
            ),
            "step_number": self.step_number,
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else AffineNewtonOptions().summary()
            ),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class AnalysisStep:
    """Inspectable analysis step that owns one algebraic solve.

    The step is the public workflow layer between a ``Study`` and an algebraic
    problem. It records the analysis intent and method, but keeps K/C/F-style
    operators visible through ``system`` and ``problem``.
    """

    name: str
    problem: LinearSystemProblem
    study: object | None = None
    method: str = "direct_linear_solve"
    dt: float | None = None
    procedure: object | None = None
    result_field_factory: object | None = None
    constraint_assets: tuple[object, ...] = ()
    constraint_dual_provider: object | None = None

    @property
    def system(self):
        """Return the engineering algebraic system used by this step."""

        return self.problem.system

    @property
    def bcs(self):
        """Return boundary conditions collected for this step."""

        return self.problem.bcs

    def solve(self, *, prepared=None):
        """Solve this analysis step."""

        if prepared is None:
            return self.problem.solve()
        return self.problem.solve_prepared(prepared)

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        field_variables=None,
        strict_output: bool = False,
        metadata=None,
    ):
        """Solve while retaining the existing ``solve()`` return contract.

        ``solve()`` continues to return the live DOLFINx solution field for
        backwards compatibility.  This method returns the higher-level result
        container used by post-processing, campaigns, and datasets.  Model-
        generated static-solid steps add their standard derived fields after
        convergence; ``output=...`` writes the final field set in one call.
        ``field_variables`` overrides the engineering default without exposing
        solver or projection plumbing in the top-level model.
        """

        from .results import (
            complete_result,
            from_solution,
            static_force_balance,
            static_work_balance,
        )
        from . import constraints as constraint_api

        solution = self.problem.solve()
        if fields and field_variables is not None:
            raise ValueError(
                "Choose explicit live fields or field_variables, not both."
            )
        if field_variables is not None and self.result_field_factory is None:
            raise ValueError(
                "This analysis step does not provide declarative derived fields."
            )
        generated = ()
        if not fields and self.result_field_factory is not None:
            generated = tuple(self.result_field_factory(field_variables))
        generated_ids = {id(item) for item in generated}
        selected_fields = tuple(fields) if fields else (solution, *generated)
        if not selected_fields:
            selected_fields = (solution,)
        result = from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "study": (
                    _describe_asset(self.study) if self.study is not None else None
                ),
            },
        )
        for item in selected_fields:
            function = _unwrap_result_field(item)
            if function is solution:
                continue
            result.add_field(
                getattr(function, "name", type(function).__name__),
                function,
                location=_field_location(function),
                description=(
                    "Constitutive result projected to a discontinuous finite-"
                    "element space; no nodal extrapolation or interelement "
                    "smoothing is applied."
                    if id(item) in generated_ids
                    else ""
                ),
                processing=(
                    _projected_field_processing(function)
                    if id(item) in generated_ids
                    else None
                ),
            )
        is_static_solid = bool(
            self.study is not None
            and getattr(self.study, "is_solid_mechanics", False)
            and len(tuple(getattr(solution, "ufl_shape", ()))) == 1
        )
        if is_static_solid:
            callback_duals = ()
            if self.constraint_dual_provider is not None:
                callback_duals = tuple(
                    self.constraint_dual_provider(self.problem) or ()
                )
            provider_duals = constraint_api.collect_provider_duals(
                self.constraint_assets,
                self.problem,
                extra=callback_duals,
            )
            balance_contract = constraint_api.constraint_balance_contract(
                self.constraint_assets,
                provider_duals=provider_duals,
            )
            result.metadata["constraint_balance_contract"] = balance_contract
            try:
                equilibrium = static_force_balance(
                    self.problem,
                    constraints=self.constraint_assets,
                    provider_duals=provider_duals,
                )
            except NotImplementedError as exc:
                equilibrium = None
                result.metadata["static_equilibrium"] = {
                    "status": "unavailable",
                    "reason": str(exc),
                    "reaction_scope": balance_contract["reaction_scope"],
                }
            try:
                work = static_work_balance(
                    self.problem,
                    constraints=self.constraint_assets,
                    provider_duals=provider_duals,
                )
            except NotImplementedError as exc:
                work = None
                result.metadata["static_work"] = {
                    "status": "unavailable",
                    "reason": str(exc),
                }
            if equilibrium is not None:
                result.add_quantities(
                    {
                        "external_force_resultant": equilibrium.external,
                        "reaction_force_resultant": equilibrium.reaction,
                        "provider_reaction_force_resultant": (
                            equilibrium.provider_reaction
                        ),
                        "force_balance_residual": equilibrium.residual,
                        "relative_force_balance_error": equilibrium.relative_error,
                    },
                    kind="diagnostic",
                    descriptions={
                        "external_force_resultant": (
                            "Resultant of the assembled linear-system right-hand side."
                        ),
                        "reaction_force_resultant": (
                            "Resultant of all declared constraint reactions included "
                            "by the balance contract."
                        ),
                        "provider_reaction_force_resultant": (
                            "Physical-space resultant supplied by MPC, weak, or "
                            "contact providers."
                        ),
                        "force_balance_residual": "Reaction plus external-force resultant.",
                        "relative_force_balance_error": (
                            "Norm of the force-balance residual divided by the larger "
                            "external or reaction resultant norm."
                        ),
                    },
                )
                result.metadata["static_equilibrium"] = equilibrium.as_dict()
            if work is not None:
                result.add_quantities(
                    {
                        "strain_energy": work.strain_energy,
                        "natural_load_work": work.natural_load_work,
                        "prescribed_motion_work": work.prescribed_motion_work,
                        "provider_constraint_work": work.provider_constraint_work,
                        "external_work": work.external_work,
                        "energy_balance_error": work.balance_error,
                    },
                    kind="diagnostic",
                )
                result.metadata["static_work"] = work.as_dict()
        return complete_result(
            self,
            result,
            output=output,
            strict_output=strict_output,
            metadata=metadata,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact, agent-readable step summary."""

        return {
            "kind": "analysis_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "method": self.method,
            "dt": self.dt,
            "problem": self.problem.summary(),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "constraint_dual_provider": (
                None
                if self.constraint_dual_provider is None
                else getattr(
                    self.constraint_dual_provider,
                    "__name__",
                    type(self.constraint_dual_provider).__name__,
                )
            ),
        }


@dataclass
class ExplicitDynamicsStep:
    """Inspectable second-order explicit dynamics step.

    This is the workflow layer for the standard loop
    ``predict -> apply constraints -> residual -> acceleration -> advance``.
    The integrator and residual operator remain explicit so advanced users can
    inspect or replace them.
    """

    name: str
    state: object
    integrator: object
    residual: object
    dt: float
    steps: int
    study: object | None = None
    prescribed: tuple[object, ...] = ()
    constraints: tuple[object, ...] = ()
    update_load: object | None = None
    save_every: int | None = None
    print_every: int | None = None
    history_every: int = 1
    procedure: object | None = None
    history_monitor: object | None = None
    stability: object | None = None
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    last_output_backend: str | None = field(default=None, init=False)
    last_output_layout: str | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    performance: PerformanceLedger = field(
        default_factory=PerformanceLedger,
        init=False,
    )

    def initialize_from_preload(
        self,
        displacement,
        *,
        source_step=None,
        initial_velocity=None,
        mode: str = "equilibrium",
        force_tolerance: float = 1.0e-8,
        source_energy: float | None = None,
    ):
        """Transfer a quasi-static configuration into this Explicit step.

        Reactions at held strong constraints are excluded from the free-force
        equilibrium norm.  The transfer uses the step's own mass, residual,
        constraint projection, and energy monitor so the public workflow does
        not need to reconstruct solver plumbing.
        """

        from . import fracture
        from .time import explicit as explicit_time

        monitor = getattr(self.history_monitor, "energy", None)

        def project(field) -> None:
            explicit_time.project_homogeneous_kinematics(
                field,
                prescribed=self.prescribed,
                constraints=self.constraints,
            )

        return fracture.transfer_preload_to_explicit(
            displacement,
            state=self.state,
            mass=self.integrator.mass,
            residual=self.residual,
            initial_velocity=initial_velocity,
            mode=mode,
            force_tolerance=force_tolerance,
            acceleration_projection=project,
            energy_monitor=monitor,
            source_energy=source_energy,
            source_step=getattr(source_step, "name", source_step),
            destination_step=self.name,
        )

    def run(
        self,
        *,
        output=None,
        domain=None,
        fields=(),
        progress=None,
        comm=None,
        until_step: int | None = None,
        history=(),
    ):
        """Run the explicit dynamics step with optional output and progress text."""

        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.state.u)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        run_started = perf_counter()
        self.integrator.performance = self.performance
        _bind_performance_ledger(self.residual, self.performance)
        if self.completed_steps == 0:
            self.performance.reset()
            self.execution_events.clear()
            self.accepted_times.clear()
            self.history_records.clear()
        reporter = _transient_reporter(
            selected_progress,
            selected_comm,
            self.execution_events,
            self.status_file,
        )
        stop_step = _transient_stop_step(self, until_step)
        save_every = _save_interval(self.save_every, output=output, steps=self.steps)
        print_every = _print_interval(self.print_every, self.steps)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=save_every,
            print_every=print_every,
            start_step=self.completed_steps + 1,
            stop_step=stop_step,
        )
        selected_fields = fields or (
            self.state.u.value,
            self.state.v.value,
            self.state.a.value,
        )
        output_fields, live_field_sets = _transient_output_fields(selected_fields)
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = output_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )

        _emit_transient_started(reporter, self)
        _record_transient_history(self, self.completed_steps * self.dt)

        if output is None:
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self,
                    info,
                    reporter,
                    selected_progress,
                    self.state,
                    selected_comm,
                )
            _emit_transient_completed(reporter, self)
            self.performance.add("run_wall", perf_counter() - run_started)
            return self

        if domain is None:
            domain = self.state.u.function_space.mesh
        series, actual_output, backend, layout = _transient_result_series(
            self.last_output,
            domain,
        )
        self.last_output = actual_output
        self.last_output_backend = backend
        self.last_output_layout = layout
        with series as xdmf:
            _refresh_transient_output_fields(live_field_sets)
            xdmf.write_fields(self.completed_steps * self.dt, *output_fields)
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self,
                    info,
                    reporter,
                    selected_progress,
                    self.state,
                    selected_comm,
                )
                if info.should_save:
                    _refresh_transient_output_fields(live_field_sets)
                    xdmf.write_fields(info.time, *output_fields)
        _emit_transient_completed(reporter, self)
        self.performance.add("run_wall", perf_counter() - run_started)
        return self

    def solve(self):
        self.run()
        return self.state.u.value

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        history=(),
        progress=None,
        comm=None,
        metadata=None,
    ):
        return _solve_transient_result(
            self,
            solution=self.state.u.value,
            default_fields=(
                self.state.u.value,
                self.state.v.value,
                self.state.a.value,
            ),
            output=output,
            fields=fields,
            history=history,
            progress=progress,
            comm=comm,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool = False) -> Path:
        """Save explicit state, optionally portable across MPI partitions."""

        return _save_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
            portable=portable,
        )

    def load_checkpoint(self, path) -> None:
        """Restore explicit state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
        )
        self.integrator.last_residual_owned = None

    def _advance_one(self, t: float) -> None:
        if self.update_load is not None:
            self.update_load(t)
        try:
            self.integrator.step(
                self.dt,
                time=t,
                residual_operator=self.residual,
                prescribed=self.prescribed,
                constraints=self.constraints,
            )
        except Exception:
            if hasattr(self.residual, "rollback"):
                self.residual.rollback()
            raise
        else:
            if hasattr(self.residual, "commit"):
                self.residual.commit()

    def summary(self) -> dict[str, object]:
        """Return a compact, agent-readable explicit step summary."""

        return {
            "kind": "explicit_dynamics_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "dt": self.dt,
            "steps": self.steps,
            "completed_steps": self.completed_steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
            "history_every": self.history_every,
            "history_evaluation_every": 1,
            "performance": self.performance.summary(),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "stability": (
                None
                if self.stability is None
                else (
                    self.stability.summary()
                    if hasattr(self.stability, "summary")
                    else self.stability
                )
            ),
            "integrator": (
                self.integrator.summary()
                if hasattr(self.integrator, "summary")
                else repr(self.integrator)
            ),
            "residual": (
                self.residual.summary() if hasattr(self.residual, "summary") else repr(self.residual)
            ),
            "num_prescribed": len(self.prescribed),
            "num_constraints": len(self.constraints),
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
        }


@dataclass
class ImplicitDynamicsStep:
    """Linear Newmark/generalized-alpha structural-dynamics step."""

    name: str
    state: object
    problem: LinearVariationalProblem
    parameters: object
    dt: float
    steps: int
    displacement_predictor: object
    velocity_predictor: object
    displacement_alpha_predictor: object
    velocity_alpha_predictor: object
    study: object | None = None
    update_load: object | None = None
    save_every: int | None = None
    print_every: int | None = None
    procedure: object | None = None
    history_monitor: object | None = None
    progress: object = True
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    last_output_backend: str | None = field(default=None, init=False)
    last_output_layout: str | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)

    def run(
        self,
        *,
        output=None,
        fields=(),
        progress=None,
        comm=None,
        until_step: int | None = None,
        history=(),
    ):
        """Advance the implicit dynamics step with standard progress output."""

        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.state.u)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        if self.completed_steps == 0:
            self.execution_events.clear()
            self.accepted_times.clear()
            self.history_records.clear()
        reporter = _transient_reporter(
            selected_progress,
            selected_comm,
            self.execution_events,
            self.status_file,
        )
        stop_step = _transient_stop_step(self, until_step)
        save_every = _save_interval(self.save_every, output=output, steps=self.steps)
        print_every = _print_interval(self.print_every, self.steps)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=save_every,
            print_every=print_every,
            start_step=self.completed_steps + 1,
            stop_step=stop_step,
        )
        output_fields = tuple(fields) or (
            self.state.u.value,
            self.state.v.value,
            self.state.a.value,
        )
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = output_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )
        _emit_transient_started(reporter, self)
        _record_transient_history(self, self.completed_steps * self.dt)
        if output is None:
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self, info, reporter, selected_progress, self.state,
                    selected_comm,
                )
            _emit_transient_completed(reporter, self)
            return self
        domain = self.state.u.function_space.mesh
        series, actual_output, backend, layout = _transient_result_series(
            self.last_output,
            domain,
        )
        self.last_output = actual_output
        self.last_output_backend = backend
        self.last_output_layout = layout
        with series as xdmf:
            xdmf.write_fields(self.completed_steps * self.dt, *output_fields)
            for info in stepper:
                self._advance_one(info.time)
                _accept_transient_increment(
                    self, info, reporter, selected_progress, self.state,
                    selected_comm,
                )
                if info.should_save:
                    xdmf.write_fields(info.time, *output_fields)
        _emit_transient_completed(reporter, self)
        return self

    def solve(self):
        self.run()
        return self.state.u.value

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        history=(),
        progress=None,
        comm=None,
        metadata=None,
    ):
        return _solve_transient_result(
            self,
            solution=self.state.u.value,
            default_fields=(
                self.state.u.value,
                self.state.v.value,
                self.state.a.value,
            ),
            output=output,
            fields=fields,
            history=history,
            progress=progress,
            comm=comm,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool = False) -> Path:
        """Save implicit state, optionally portable across MPI partitions."""

        return _save_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
            portable=portable,
        )

    def load_checkpoint(self, path) -> None:
        """Restore implicit state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {"displacement": self.state.u, "velocity": self.state.v, "acceleration": self.state.a},
        )

    def _advance_one(self, time_value: float) -> None:
        p = self.parameters
        dt = self.dt
        u = self.state.u.value
        v = self.state.v.value
        a = self.state.a.value
        u_predictor = self.displacement_predictor
        v_predictor = self.velocity_predictor
        u_predictor.x.array[:] = (
            u.x.array
            + dt * v.x.array
            + dt**2 * (0.5 - p.beta) * a.x.array
        )
        v_predictor.x.array[:] = (
            v.x.array + dt * (1.0 - p.gamma) * a.x.array
        )
        self.displacement_alpha_predictor.x.array[:] = (
            (1.0 - p.alpha_f) * u_predictor.x.array
            + p.alpha_f * u.x.array
        )
        self.velocity_alpha_predictor.x.array[:] = (
            (1.0 - p.alpha_f) * v_predictor.x.array
            + p.alpha_f * v.x.array
        )
        for function in (
            u_predictor,
            v_predictor,
            self.displacement_alpha_predictor,
            self.velocity_alpha_predictor,
        ):
            function.x.scatter_forward()
        if self.update_load is not None:
            evaluation_time = (
                (1.0 - p.alpha_f) * time_value
                + p.alpha_f * (time_value - dt)
            )
            self.update_load(evaluation_time)
        self.problem.solve()
        self.state.u_next.value.x.array[:] = (
            u_predictor.x.array
            + p.beta * dt**2 * self.state.a_next.value.x.array
        )
        self.state.v_next.value.x.array[:] = (
            v_predictor.x.array
            + p.gamma * dt * self.state.a_next.value.x.array
        )
        self.state.u_next.value.x.scatter_forward()
        self.state.v_next.value.x.scatter_forward()
        self.state.advance_state()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "implicit_dynamics_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "integration": self.parameters.summary(),
            "dt": self.dt,
            "steps": self.steps,
            "completed_steps": self.completed_steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "problem": {
                "num_bcs": len(self.problem.bcs),
                "solver": (
                    self.problem.solver_options.summary()
                    if self.problem.solver_options is not None
                    else LinearSolverOptions().summary()
                ),
            },
        }


@dataclass
class FirstOrderTransientStep:
    """Reusable implicit-Euler step loop for heat/diffusion problems."""

    name: str
    problem: AnalysisStep
    current: object
    previous: object
    dt: float
    steps: int
    study: object | None = None
    update_load: object | None = None
    save_every: int | None = None
    print_every: int | None = None
    progress: object = True
    procedure: object | None = None
    history_monitor: object | None = None
    status_file: object | None = None
    checkpoint_policy: object | None = None
    history_requests: tuple[object, ...] = field(default_factory=tuple, init=False)
    accepted_times: list[float] = field(default_factory=list, init=False)
    execution_events: list[object] = field(default_factory=list, init=False)
    last_output: Path | None = field(default=None, init=False)
    last_output_fields: tuple[object, ...] = field(default=(), init=False)
    last_output_start_time: float | None = field(default=None, init=False)
    last_output_backend: str | None = field(default=None, init=False)
    last_output_layout: str | None = field(default=None, init=False)
    completed_steps: int = field(default=0, init=False)
    history_records: list[dict[str, float]] = field(default_factory=list, init=False)
    checkpoints: list[object] = field(default_factory=list, init=False)
    captured_histories: list[object] = field(default_factory=list, init=False)

    def capture_history(
        self,
        source=None,
        *,
        name: str | None = None,
        unit: str | None = None,
        every: int = 1,
        interpolation: str = "linear",
        outside: str = "error",
    ):
        """Capture a scalar field at accepted physical times.

        The returned :class:`agentfem.histories.FieldHistory` can be passed
        directly to a later thermo-mechanical or creep step.
        """

        from . import histories

        selected = self.current if source is None else source
        recorder = histories.FieldHistory(
            selected,
            name=name or getattr(getattr(selected, "value", selected), "name", "field"),
            unit=unit,
            every=every,
            interpolation=interpolation,
            outside=outside,
            metadata={
                "source_step": getattr(self, "name", type(self).__name__),
                "source_procedure": (
                    self.procedure.summary()
                    if hasattr(self.procedure, "summary")
                    else None
                ),
                "source_study": (
                    self.study.summary()
                    if hasattr(self.study, "summary")
                    else None
                ),
                "accepted_time_only": True,
                "transfer_role": "sequential_field_input",
            },
        )
        self.captured_histories.append(recorder)
        return recorder

    def run(
        self,
        *,
        output=None,
        fields=(),
        progress=None,
        comm=None,
        until_step: int | None = None,
        history=(),
    ):
        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.current)
        _configure_transient_history(self, history)
        selected_progress = self.progress if progress is None else progress
        if self.completed_steps >= self.steps:
            return self
        if self.completed_steps == 0:
            self.execution_events.clear()
            self.accepted_times.clear()
            self.history_records.clear()
        reporter = _transient_reporter(
            selected_progress,
            selected_comm,
            self.execution_events,
            self.status_file,
        )
        stop_step = _transient_stop_step(self, until_step)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=_save_interval(
                self.save_every,
                output=output,
                steps=self.steps,
            ),
            print_every=_print_interval(self.print_every, self.steps),
            start_step=self.completed_steps + 1,
            stop_step=stop_step,
        )
        selected_fields = tuple(fields) or (self.current,)
        self.last_output = None if output is None else Path(output)
        self.last_output_fields = selected_fields if output is not None else ()
        self.last_output_start_time = (
            None if output is None else float(self.completed_steps) * float(self.dt)
        )

        _emit_transient_started(reporter, self)
        _record_transient_history(self, self.completed_steps * self.dt)
        self._record_captured_histories(force=True)
        linear_problem = getattr(self.problem, "problem", None)
        prepare = getattr(linear_problem, "prepare", None)
        prepared = prepare() if callable(prepare) else None

        def advance(info):
            if self.update_load is not None:
                self.update_load(info.time)
            current_rollback = self.current.x.array.copy()
            previous_rollback = self.previous.x.array.copy()
            try:
                if prepared is None:
                    self.problem.solve()
                else:
                    self.problem.solve(prepared=prepared)
            except Exception:
                self.current.x.array[:] = current_rollback
                self.current.x.scatter_forward()
                self.previous.x.array[:] = previous_rollback
                self.previous.x.scatter_forward()
                raise
            self.previous.x.array[:] = self.current.x.array
            self.previous.x.scatter_forward()
            _accept_transient_increment(
                self, info, reporter, selected_progress, self.current,
                selected_comm,
            )
            self._record_captured_histories(
                force=self.completed_steps == self.steps
            )

        try:
            if output is None:
                for info in stepper:
                    advance(info)
                _emit_transient_completed(reporter, self)
                return self
            domain = self.current.function_space.mesh
            series, actual_output, backend, layout = _transient_result_series(
                self.last_output,
                domain,
            )
            self.last_output = actual_output
            self.last_output_backend = backend
            self.last_output_layout = layout
            with series as xdmf:
                xdmf.write_fields(self.completed_steps * self.dt, *selected_fields)
                for info in stepper:
                    advance(info)
                    if info.should_save:
                        xdmf.write_fields(info.time, *selected_fields)
            _emit_transient_completed(reporter, self)
            return self
        finally:
            close = getattr(prepared, "close", None)
            if callable(close):
                close()

    def _record_captured_histories(self, *, force: bool = False) -> None:
        selected_time = self.completed_steps * self.dt
        for recorder in self.captured_histories:
            if force or self.completed_steps % recorder.every == 0:
                recorder.record(selected_time)

    def solve(self):
        self.run()
        return self.current

    def solve_result(
        self,
        *,
        output=None,
        fields=(),
        history=(),
        progress=None,
        comm=None,
        metadata=None,
    ):
        return _solve_transient_result(
            self,
            solution=self.current,
            default_fields=(self.current,),
            output=output,
            fields=fields,
            history=history,
            progress=progress,
            comm=comm,
            metadata=metadata,
        )

    def save_checkpoint(self, path, *, portable: bool = False) -> Path:
        """Save first-order state, optionally portable across MPI partitions."""

        return _save_transient_checkpoint(
            self,
            path,
            {"current": self.current, "previous": self.previous},
            portable=portable,
        )

    def load_checkpoint(self, path) -> None:
        """Restore first-order state and the accepted time/history position."""

        _load_transient_checkpoint(
            self,
            path,
            {"current": self.current, "previous": self.previous},
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "first_order_transient_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "procedure": (
                None if self.procedure is None else self.procedure.summary()
            ),
            "dt": self.dt,
            "steps": self.steps,
            "completed_steps": self.completed_steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
            "checkpoint_policy": (
                None
                if self.checkpoint_policy is None
                else self.checkpoint_policy.summary()
            ),
            "history_requests": [
                request.summary() for request in self.history_requests
            ],
            "captured_histories": [
                recorder.summary() for recorder in self.captured_histories
            ],
            "problem": self.problem.summary(),
        }


def _solve_transient_result(
    step,
    *,
    solution,
    default_fields,
    output,
    fields,
    history,
    progress,
    comm,
    metadata,
):
    """Shared transient solve/output/result lifecycle for all equation orders."""

    from .results import add_execution_trace, from_solution

    context = getattr(step, "execution_context", None)
    selected_output = output
    if selected_output is None and context is not None:
        selected_output = context.configured_output
    selected_history = tuple(history)
    if not selected_history and context is not None:
        selected_history = context.configured_history
    if selected_output is not None:
        from .results import OutputPlan

        if isinstance(selected_output, OutputPlan):
            raise ValueError(
                "Declarative OutputPlan currently describes finite-strain "
                "static output. Pass an XDMF path to a transient Step."
            )
    _configure_transient_history(step, selected_history)
    if selected_output is not None and step.completed_steps >= step.steps:
        raise RuntimeError(
            "Transient field output must be requested before completion; "
            "completed steps cannot be reconstructed from final state alone."
        )
    if selected_output is not None or step.completed_steps != step.steps:
        step.run(
            output=selected_output,
            fields=fields,
            history=selected_history,
            progress=progress,
            comm=comm,
        )
    result = from_solution(
        solution,
        name=step.name,
        metadata={"step": step.summary()},
    )
    if metadata:
        result.metadata.update(dict(metadata))
    if context is not None:
        result.metadata.setdefault("execution_context", context.summary())
    add_execution_trace(result, step.execution_events)
    _attach_transient_output(
        result,
        step,
        tuple(fields) or step.last_output_fields or tuple(default_fields),
    )
    return result


def _attach_transient_output(result, step, output_fields) -> None:
    """Attach the accepted time axis and one single-geometry field dataset."""

    result.metadata["accepted_times"] = tuple(float(item) for item in step.accepted_times)
    output_start = step.last_output_start_time
    result.metadata["transient"] = {
        "completed_steps": int(step.completed_steps),
        "total_steps": int(step.steps),
        "output_start_time": output_start,
        "output_scope": (
            None
            if step.last_output is None
            else "complete"
            if output_start == 0.0
            else "continuation_segment"
        ),
    }
    if step.history_requests:
        result.metadata["transient"]["history_requests"] = [
            request.summary() for request in step.history_requests
        ]
    if step.history_records:
        coordinates = [item["time"] for item in step.history_records]
        names = tuple(
            name
            for name in step.history_records[0]
            if name != "time"
        )
        requests = {
            request.name: request
            for request in getattr(step, "history_requests", ())
        }
        result.add_histories(
            coordinates,
            {
                name: [item[name] for item in step.history_records]
                for name in names
            },
            abscissa_name="time",
            abscissa_unit="s",
            units={
                name: getattr(requests.get(name), "unit", None)
                for name in names
            },
            descriptions={
                name: (
                    getattr(requests.get(name), "description", "")
                    or _TRANSIENT_HISTORY_DESCRIPTIONS.get(name, "")
                )
                for name in names
            },
        )
    for checkpoint in step.checkpoints:
        result.add_checkpoint(checkpoint)
    path = step.last_output
    if path is None:
        return
    primary = (
        None if not output_fields else _unwrap_result_field(output_fields[0])
    )
    primary_shape = () if primary is None else tuple(getattr(primary, "ufl_shape", ()))
    vector_primary = len(primary_shape) == 1
    domain = None if primary is None else primary.function_space.mesh
    backend = getattr(step, "last_output_backend", None)
    storage_name = (
        None
        if not vector_primary
        else (
            "U"
            if backend == "agentfem_unified_xdmf"
            else str(getattr(primary, "name", "Displacement"))
        )
    )
    semantic_name = "Displacement" if vector_primary else None
    is_paraview = path.suffix.lower() == ".pvd"
    result.metadata["field_output"] = {
        "status": "completed",
        "backend": backend,
        "layout": getattr(step, "last_output_layout", None),
        "geometry": "reference",
        "scientific_artifact": None if is_paraview else str(path),
        "scientific_xdmf_layout": (
            "not_emitted" if is_paraview else "single_uniform_grid"
        ),
        "recommended_visualization_artifact": str(path),
        "visualization_geometry_datasets_per_time": 1,
        "visualization_requires_extract_block": False,
        "warp_field": storage_name,
        "warp_field_semantic": semantic_name,
        "physical_components": (
            int(primary_shape[0]) if vector_primary else None
        ),
        "stored_components": (
            None if not vector_primary or domain is None else int(domain.geometry.x.shape[1])
        ),
        "geometry_dimension": (
            None if domain is None else int(domain.geometry.x.shape[1])
        ),
        "physical_model_dimension": (
            None if domain is None else int(domain.geometry.dim)
        ),
        "warp_compatible": bool(vector_primary),
        "field_aliases": (
            {}
            if storage_name is None or semantic_name is None
            else {semantic_name: storage_name}
        ),
    }
    if path.suffix.lower() == ".pvd":
        result.add_artifact("fields_paraview", path)
    else:
        result.add_artifact("fields_xdmf", path)
        heavy_data = path.with_suffix(".h5")
        if heavy_data.is_file():
            result.add_artifact("fields_hdf5", heavy_data)
    for item in output_fields:
        function = getattr(item, "value", item)
        name = getattr(function, "name", type(function).__name__)
        result.add_field(
            name,
            function,
            artifact=path,
            description=(
                "Transient field in the shared single-geometry series; "
                f"this output segment starts at time {output_start:g}."
            ),
        )


_TRANSIENT_HISTORY_DESCRIPTIONS = {
    "kinetic_energy": "Discrete kinetic energy, one half v-transpose M v.",
    "strain_energy": "Recoverable linear strain energy, one half u-transpose K u.",
    "total_mechanical_energy": "Sum of discrete kinetic and recoverable strain energy.",
    "bulk_strain_energy": "Finite-strain constitutive energy integrated in the reference body.",
    "cohesive_stored_energy": "Recoverable energy currently stored by the cohesive interface.",
    "cohesive_fracture_dissipation": "Irreversible cohesive dissipation relative to the initial interface state.",
    "numerical_damping_dissipation": "Accepted nonnegative work dissipated by the declared viscous damping model.",
    "natural_load_work": "Accepted-path trapezoidal work of weak natural loads.",
    "prescribed_motion_work": "Accepted-path trapezoidal work of strong prescribed-motion reactions.",
    "external_work": "Sum of natural-load and prescribed-motion work.",
    "energy_balance_error": "Initial accounted energy plus external work minus current accounted energy.",
    "relative_energy_balance_error": "Absolute energy-balance error normalized by the largest energy scale.",
    "thermal_content": (
        "Discrete thermal content, one-transpose C T, relative to the model's "
        "temperature zero."
    ),
    "applied_heat_rate": "Applied volumetric, flux, and Robin-source heat rate, one-transpose Q.",
    "outward_heat_rate": "Net discrete conduction/Robin rate, one-transpose K T.",
    "heat_balance_residual": (
        "Implicit-Euler energy residual: delta thermal content plus dt times "
        "outward rate minus applied rate. Strong-temperature reactions appear "
        "in this residual until reported separately."
    ),
}


def _transient_reporter(progress, comm, events, status_file=None):
    from .diagnostics import SolveEventRecorder, compose_reporters

    recorder = SolveEventRecorder(events)
    if progress is True:
        from .diagnostics import StandardRunReporter

        return compose_reporters(
            recorder,
            StandardRunReporter(
                comm,
                status_file=status_file,
                show_iterations=False,
            ),
        )
    if hasattr(progress, "emit"):
        return compose_reporters(recorder, progress)
    return recorder


def _emit_transient_started(reporter, step) -> None:
    if reporter is None:
        return
    procedure = getattr(step, "procedure", None)
    algorithm = getattr(procedure, "algorithm", "time_integration")
    stability = getattr(step, "stability", None)
    stability_message = ""
    if stability is not None:
        selected = getattr(stability, "selected", None)
        controller = getattr(stability, "controller", None)
        if selected is not None:
            ratio = float(step.dt) / float(selected)
            stability_message = (
                f"dt={float(step.dt):.6g}, dt/dt_limit={ratio:.3g}"
                + ("" if controller is None else f", controller={controller}")
            )
    reporter.emit(
        SolveEvent(
            "transient_started" if step.completed_steps == 0 else "transient_resumed",
            step.name,
            incrementation=algorithm,
            total_increments=step.steps,
            message=stability_message,
        )
    )


def _report_transient_increment(
    reporter,
    selected_progress,
    info,
    step,
    state,
    comm,
) -> None:
    if reporter is not None:
        message = ""
        records = getattr(step, "history_records", ())
        if records:
            latest = records[-1]
            channels = []
            if "relative_energy_balance_error" in latest:
                channels.append(
                    "energy_err="
                    f"{float(latest['relative_energy_balance_error']):.3e}"
                )
            elif "energy_balance_error" in latest:
                channels.append(
                    f"energy_err={float(latest['energy_balance_error']):.3e}"
                )
            message = " | ".join(channels)
        reporter.emit(
            SolveEvent(
                "time_increment",
                step.name,
                increment=info.index,
                time=float(info.time),
                total_increments=step.steps,
                display=bool(info.should_print),
                message=message,
            )
        )
    if info.should_print and callable(selected_progress):
        from .diagnostics import print_on_root

        message = selected_progress(info, state)
        if message:
            print_on_root(comm, message)


def _accept_transient_increment(
    step,
    info,
    reporter,
    selected_progress,
    report_state,
    comm,
) -> None:
    """Commit one accepted time increment through the shared lifecycle."""

    step.completed_steps = int(info.index)
    step.accepted_times.append(float(info.time))
    history_every = int(getattr(step, "history_every", 1))
    store_history = info.index % history_every == 0 or info.index == step.steps
    _record_transient_history(step, info.time, store=store_history)
    _report_transient_increment(
        reporter,
        selected_progress,
        info,
        step,
        report_state,
        comm,
    )
    _write_scheduled_checkpoint(step)


def _write_scheduled_checkpoint(step) -> Path | None:
    policy = getattr(step, "checkpoint_policy", None)
    if policy is None or not policy.due(step.completed_steps, step.steps):
        return None
    written = step.save_checkpoint(
        policy.path(
            step_name=step.name,
            increment=step.completed_steps,
        ),
        portable=bool(getattr(policy, "portable", False)),
    )
    _apply_checkpoint_retention(step, policy)
    return written


def _emit_transient_completed(reporter, step) -> None:
    if reporter is None:
        return
    times = getattr(step, "accepted_times", ())
    reporter.emit(
        SolveEvent(
            (
                "transient_completed"
                if step.completed_steps >= step.steps
                else "transient_paused"
            ),
            step.name,
            increment=step.completed_steps,
            time=float(step.completed_steps) * float(step.dt),
            total_increments=step.steps,
        )
    )


def _transient_stop_step(step, until_step: int | None) -> int:
    stop = step.steps if until_step is None else int(until_step)
    if not step.completed_steps <= stop <= step.steps:
        raise ValueError(
            "until_step must lie between the completed step and total steps."
        )
    return stop


def _record_transient_history(
    step,
    time_value: float,
    *,
    store: bool = True,
) -> None:
    """Advance history monitors every increment and store at their cadence.

    Stateful scientific ledgers, notably external work, must consume every
    accepted increment. ``history_every`` controls retained records only; it
    must never change the computed balance merely by changing output cadence.
    """

    monitor = getattr(step, "history_monitor", None)
    requests = tuple(getattr(step, "history_requests", ()))
    if monitor is None and not requests:
        return
    selected_time = float(time_value)
    if store and step.history_records and np.isclose(
        step.history_records[-1]["time"],
        selected_time,
    ):
        if hasattr(monitor, "restore"):
            monitor.restore(step.history_records[-1])
        return
    values = {}
    monitor_started = perf_counter()
    if monitor is not None:
        if isinstance(step, FirstOrderTransientStep):
            values.update(monitor.evaluate(step.current))
        else:
            monitor_kwargs = {
                "displacement": step.state.u,
                "velocity": step.state.v,
            }
            if getattr(monitor, "accepts_accepted_residual", False):
                monitor_kwargs["residual_owned"] = (
                    None
                    if step.completed_steps == 0
                    else getattr(
                        step.integrator,
                        "last_residual_owned",
                        None,
                    )
                )
            if not store and hasattr(monitor, "advance"):
                monitor.advance(**monitor_kwargs)
            elif hasattr(monitor, "evaluate"):
                values.update(monitor.evaluate(**monitor_kwargs))
            else:
                values.update(monitor(step, selected_time))
    performance = getattr(step, "performance", None)
    if monitor is not None and performance is not None:
        performance.add(
            "history_snapshot" if store else "history_advance",
            perf_counter() - monitor_started,
        )
    if not store:
        return
    for request in requests:
        if not hasattr(request, "evaluate_transient"):
            raise TypeError(
                f"History request {type(request).__name__} cannot be evaluated "
                "during a transient Step."
            )
        name = str(request.name)
        if name == "time" or name in values:
            raise ValueError(
                f"Transient history name {name!r} conflicts with an existing "
                "history channel."
            )
        values[name] = request.evaluate_transient(step, selected_time)
    frame = {"time": selected_time}
    for name, value in values.items():
        selected = float(value)
        if not np.isfinite(selected):
            raise ValueError(f"Transient history {name!r} is not finite.")
        frame[str(name)] = selected
    if step.history_records:
        expected = set(step.history_records[0])
        actual = set(frame)
        if actual != expected:
            raise RuntimeError(
                "Transient history channels changed after the analysis began: "
                f"expected {tuple(sorted(expected))}, received "
                f"{tuple(sorted(actual))}. When restarting, pass "
                "the same history requests used by the original Step."
            )
    if len(frame) > 1:
        step.history_records.append(frame)


def _bind_performance_ledger(consumer, ledger, visited=None) -> None:
    """Attach one timing ledger to nested solver consumers without ownership."""

    if consumer is None:
        return
    selected_visited = set() if visited is None else visited
    identity = id(consumer)
    if identity in selected_visited:
        return
    selected_visited.add(identity)
    try:
        consumer.performance = ledger
    except (AttributeError, TypeError):
        pass
    for name in ("base", "bulk", "cohesive", "energy"):
        nested = getattr(consumer, name, None)
        if nested is not None:
            _bind_performance_ledger(nested, ledger, selected_visited)


def _configure_transient_history(step, requests) -> None:
    """Bind immutable accepted-increment requests to one transient Step."""

    selected = tuple(requests)
    if not selected:
        return
    names = [str(getattr(request, "name", "")) for request in selected]
    if any(not name for name in names):
        raise ValueError("Every transient history request requires a name.")
    if len(set(names)) != len(names):
        raise ValueError("Transient history request names must be unique.")
    current = tuple(getattr(step, "history_requests", ()))
    if current and current != selected:
        raise RuntimeError(
            "Transient history requests are fixed after execution begins; "
            "continue with the same request objects."
        )
    step.history_requests = selected


def _apply_checkpoint_retention(step, policy) -> None:
    """Prune published scheduled checkpoints beyond an explicit policy."""

    keep_last = getattr(policy, "keep_last", None)
    scheduled = [
        record
        for record in step.checkpoints
        if record.metadata.get("role") != "restart_source"
    ]
    if keep_last is None or len(scheduled) <= keep_last:
        return
    from . import checkpointing

    obsolete = scheduled[:-keep_last]
    state = step.current if isinstance(step, FirstOrderTransientStep) else step.state.u
    function = getattr(state, "value", state)
    for record in obsolete:
        checkpointing._remove_transient_checkpoint(
            record.path,
            comm=function.function_space.mesh.comm,
        )
    removed = {id(record) for record in obsolete}
    step.checkpoints[:] = [
        record for record in step.checkpoints if id(record) not in removed
    ]


def _prune_affine_checkpoints(step) -> None:
    """Apply the common retention policy to portable affine checkpoints."""

    policy = getattr(step, "checkpoint_policy", None)
    keep_last = None if policy is None else getattr(policy, "keep_last", None)
    scheduled = [
        record
        for record in step.checkpoints
        if record.metadata.get("role") != "restart_source"
    ]
    if keep_last is None or len(scheduled) <= int(keep_last):
        return
    from .checkpointing import remove_stateful_checkpoint

    obsolete = scheduled[: -int(keep_last)]
    comm = step.solution.function_space.mesh.comm
    for record in obsolete:
        remove_stateful_checkpoint(record.path, comm=comm)
    removed = {id(record) for record in obsolete}
    step.checkpoints[:] = [
        record for record in step.checkpoints if id(record) not in removed
    ]


def _snapshot_accepted_observers(
    observers,
    *,
    comm=None,
) -> tuple[tuple[object, object], ...]:
    """Capture every accepted-history observer before a boundary mutation."""

    snapshots = []
    local_problem = None
    try:
        for observer in observers:
            capture = getattr(observer, "snapshot_runtime_state", None)
            restore = getattr(observer, "restore_runtime_state", None)
            if not callable(capture) or not callable(restore):
                raise TypeError(
                    "Accepted-increment observers must provide "
                    "snapshot_runtime_state() and restore_runtime_state() so "
                    "output/checkpoint failures cannot split the solve lifecycle; "
                    f"got {type(observer).__name__}."
                )
            snapshots.append((observer, capture()))
    except Exception as exc:
        if comm is None or comm.size == 1:
            raise
        local_problem = f"{type(exc).__name__}: {exc}"
    if comm is not None and comm.size > 1:
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(
                f"Rank {rank}: accepted-observer state snapshot failed: "
                f"{problems[rank]}"
            )
    return tuple(snapshots)


def _restore_accepted_observers(snapshots) -> None:
    """Restore observers captured by ``_snapshot_accepted_observers``."""

    for observer, state in snapshots:
        observer.restore_runtime_state(state)


def _save_transient_checkpoint(step, path, state, *, portable: bool = False) -> Path:
    from . import checkpointing
    from .results import CheckpointRecord

    manifest = checkpointing.save_transient_checkpoint(
        path,
        step_kind=step.summary()["kind"],
        step_name=step.name,
        procedure=step.procedure,
        dt=step.dt,
        total_steps=step.steps,
        completed_steps=step.completed_steps,
        state=state,
        accepted_times=step.accepted_times,
        execution_events=step.execution_events,
        history_records=step.history_records,
        auxiliary_state=(
            {"residual": step.residual.snapshot()}
            if hasattr(getattr(step, "residual", None), "snapshot")
            else None
        ),
        portable=portable,
    )
    record = CheckpointRecord(
        name=f"{step.name}_{step.completed_steps}",
        path=manifest,
        schema=checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,
        step_name=step.name,
        coordinate_name="time",
        coordinate_value=float(step.completed_steps) * float(step.dt),
        portable=bool(portable),
        metadata={
            "completed_steps": step.completed_steps,
            "total_steps": step.steps,
            "portability": (
                "nodal state portable across MPI partitions and rank counts"
                if portable
                else "same mesh partition and MPI size"
            ),
        },
    )
    step.checkpoints.append(record)
    return manifest


def _load_transient_checkpoint(step, path, state) -> None:
    from . import checkpointing
    from .results import CheckpointRecord

    metadata = checkpointing.load_transient_checkpoint(
        path,
        step_kind=step.summary()["kind"],
        step_name=step.name,
        procedure=step.procedure,
        dt=step.dt,
        total_steps=step.steps,
        state=state,
    )
    auxiliary = metadata.get("auxiliary_state")
    residual = getattr(step, "residual", None)
    if auxiliary is not None:
        if residual is None or not hasattr(residual, "restore"):
            raise ValueError(
                "Checkpoint contains auxiliary residual state, but the current "
                "step has no compatible residual-state consumer."
            )
        residual.restore(auxiliary["residual"])
    elif hasattr(residual, "restore"):
        raise ValueError(
            "The current step requires auxiliary residual state that is absent "
            "from this checkpoint."
        )
    step.completed_steps = int(metadata["completed_steps"])
    step.accepted_times[:] = [float(value) for value in metadata["accepted_times"]]
    step.execution_events[:] = [
        SolveEvent.from_dict(item) for item in metadata["execution_events"]
    ]
    step.history_records[:] = [
        {name: float(value) for name, value in item.items()}
        for item in metadata["history_records"]
    ]
    restart_time = float(step.completed_steps) * float(step.dt)
    if getattr(step, "update_load", None) is not None:
        step.update_load(restart_time)
    for item in tuple(getattr(step, "prescribed", ())):
        if hasattr(item, "update"):
            item.update(restart_time)
    step.checkpoints.append(
        CheckpointRecord(
            name=f"{step.name}_{step.completed_steps}_restart",
            path=Path(metadata["manifest_path"]),
            schema=checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,
            step_name=step.name,
            coordinate_name="time",
            coordinate_value=float(step.completed_steps) * float(step.dt),
            portable=bool(metadata.get("portable", False)),
            metadata={
                "role": "restart_source",
                "portability": metadata["portability"],
            },
        )
    )


def linear_system(
    K,
    F,
    *,
    unknown=None,
    solution=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    name: str = "Kx_eq_F",
) -> LinearSystemProblem:
    """Create a ``K x = F`` problem without exposing variational boilerplate."""

    return LinearSystemProblem.from_operators(
        K,
        F,
        unknown=unknown,
        solution=solution,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )


def linear_static(
    K,
    F,
    *,
    study=None,
    unknown=None,
    solution=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    result_field_factory=None,
    name: str = "linear_static",
) -> AnalysisStep:
    """Create a linear static analysis step in ``K x = F`` notation."""

    _require_study_analysis(study, "linear_static")
    problem = linear_system(
        K,
        F,
        unknown=unknown,
        solution=solution,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )
    from . import procedures

    return AnalysisStep(
        name=name,
        study=study,
        problem=problem,
        method="linear_static",
        procedure=procedures.linear_static(),
        result_field_factory=result_field_factory,
        constraint_assets=tuple(_as_list(constraints)),
    )


def nonlinear(
    residual,
    solution,
    *,
    jacobian=None,
    constraints=None,
    bcs=None,
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None,
    name: str = "nonlinear",
    petsc_options_prefix: str = "agentfem_nonlinear_",
) -> NonlinearVariationalProblem:
    """Create a general nonlinear residual problem."""

    from . import procedures

    return NonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        bcs=_collect_bcs(constraints=constraints, bcs=bcs),
        solver_options=solver_options,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
        procedure=procedures.nonlinear_static(),
    )


def incremental_nonlinear(
    residual,
    solution,
    *,
    factor,
    value_path,
    update_load=None,
    acceptance_check=None,
    jacobian=None,
    incrementation=None,
    constraints=None,
    bcs=None,
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None,
    output_every: int | None = 1,
    progress=True,
    status_file=None,
    name: str = "incremental_nonlinear",
    petsc_options_prefix: str = "agentfem_incremental_nonlinear_",
) -> IncrementalNonlinearVariationalProblem:
    """Create standard-BC nonlinear equilibrium over a normalized load path."""

    from . import procedures
    from . import steps as step_controls

    return IncrementalNonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        factor=factor,
        value_path=value_path,
        update_load=update_load,
        acceptance_check=acceptance_check,
        bcs=_collect_bcs(constraints=constraints, bcs=bcs),
        incrementation=step_controls.normalize(incrementation),
        solver_options=solver_options,
        output_every=output_every,
        progress=progress,
        status_file=status_file,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
        procedure=procedures.nonlinear_static(),
    )


def affine_nonlinear(
    residual,
    solution,
    *,
    jacobian,
    constraint,
    load_factors=None,
    incrementation=None,
    solver_options: AffineNewtonOptions | NewtonSolverOptions | None = None,
    output_every: int | None = 1,
    output_factors=(),
    state_transaction=None,
    checkpoint_policy=None,
    acceptance_check=None,
    progress=True,
    status_file=None,
    name: str = "affine_nonlinear",
    procedure=None,
) -> AffineNonlinearVariationalProblem:
    """Create a nonlinear problem reduced by an affine constraint map."""

    from . import procedures

    return AffineNonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        constraint=constraint,
        load_factors=(
            None
            if load_factors is None
            else tuple(float(value) for value in load_factors)
        ),
        incrementation=incrementation,
        solver_options=solver_options,
        output_every=(
            None if output_every is None else int(output_every)
        ),
        output_factors=tuple(float(value) for value in output_factors),
        state_transaction=state_transaction,
        checkpoint_policy=checkpoint_policy,
        acceptance_check=acceptance_check,
        progress=progress,
        status_file=status_file,
        name=name,
        procedure=(
            procedures.nonlinear_static(stateful=state_transaction is not None)
            if procedure is None
            else procedure
        ),
    )


@dataclass(frozen=True)
class LoadIncrementSnapshot:
    """A copied solution state at one nonlinear load factor."""

    index: int
    load_factor: float
    solution: object
    solve_info: object | None = None
    fields: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "index": self.index,
            "load_factor": self.load_factor,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
            "fields": tuple(self.fields),
            "solve": (
                None
                if self.solve_info is None
                else self.solve_info.as_dict()
            ),
        }


def _load_snapshot(
    index: int,
    load_factor: float,
    solution,
    *,
    solve_info=None,
    zero: bool = False,
    zero_fields: bool | None = None,
    field_factory=None,
) -> LoadIncrementSnapshot:
    auxiliary = {}
    if field_factory is None:
        selected = solution.value if hasattr(solution, "value") else solution
    else:
        selected, auxiliary = field_factory()
    copied = fem.Function(
        selected.function_space,
        name=getattr(selected, "name", "Solution"),
    )
    if not zero:
        copied.x.array[:] = selected.x.array
        copied.x.scatter_forward()
    selected_zero_fields = bool(zero) if zero_fields is None else bool(zero_fields)
    copied_fields = {}
    for name, value in dict(auxiliary).items():
        source = getattr(value, "function", value)
        field_copy = fem.Function(
            source.function_space,
            name=getattr(source, "name", str(name)),
        )
        if not selected_zero_fields:
            field_copy.x.array[:] = source.x.array
            field_copy.x.scatter_forward()
        if all(hasattr(value, item) for item in ("points", "weights", "value_shape")):
            from .constitutive.quadrature import QuadratureField

            copied_fields[str(name)] = QuadratureField(
                function=field_copy,
                points=np.asarray(value.points, dtype=float).copy(),
                weights=np.asarray(value.weights, dtype=float).copy(),
                value_shape=tuple(value.value_shape),
            )
        else:
            copied_fields[str(name)] = field_copy
    return LoadIncrementSnapshot(
        index=int(index),
        load_factor=float(load_factor),
        solution=copied,
        solve_info=solve_info,
        fields=copied_fields,
    )


def first_order_transient(
    *,
    capacity,
    stiffness,
    history,
    source=None,
    dt: float,
    study=None,
    unknown=None,
    solution=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    name: str = "first_order_transient_step",
    method: str = "implicit_euler",
) -> AnalysisStep:
    """Create a first-order transient step.

    This builds the common implicit Euler system
    ``(C / dt + K) x_next = C x_previous / dt + Q`` while keeping ``C``, ``K``,
    history, and source as explicit operator-level inputs.
    """

    from . import operators

    _require_study_analysis(study, "first_order_transient")
    if dt <= 0.0:
        raise ValueError("first_order_transient requires dt > 0.")

    target = solution if solution is not None else unknown
    try:
        target_function = fields.unwrap(target)
        inverse_dt = fem.Constant(
            target_function.function_space.mesh,
            PETSc.ScalarType(1.0 / dt),
        )
    except (AttributeError, TypeError):
        # Preserve compatibility for symbolic construction without a concrete
        # target field. Executable problems use the Constant route so changing
        # dt values do not create distinct compiled form signatures.
        inverse_dt = 1.0 / dt

    C_over_dt = operators.scale(
        capacity,
        inverse_dt,
        name="C_over_dt",
        kind=f"{method}_capacity_over_dt",
    )
    history_over_dt = operators.scale(
        history,
        inverse_dt,
        name="F_history_over_dt",
        kind=f"{method}_history_over_dt",
    )
    lhs = operators.combine(
        C_over_dt,
        stiffness,
        name="K_effective",
        kind=f"{method}_lhs",
    )
    rhs_terms = (history_over_dt,) if source is None else (history_over_dt, source)
    rhs = operators.combine(
        *rhs_terms,
        name="F_effective",
        kind=f"{method}_rhs",
    )
    problem = linear_system(
        lhs,
        rhs,
        unknown=unknown,
        solution=solution,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )
    from . import procedures

    return AnalysisStep(
        name=name,
        study=study,
        problem=problem,
        method=method,
        dt=dt,
        procedure=procedures.implicit_euler(),
    )


def first_order_transient_run(
    *,
    capacity,
    stiffness,
    history,
    current,
    previous,
    dt: float,
    steps: int,
    source=None,
    study=None,
    constraints=None,
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    name: str = "first_order_transient",
) -> FirstOrderTransientStep:
    """Create an executable implicit-Euler time step and loop."""

    from . import procedures
    from .diagnostics import ThermalBalanceMonitor

    if steps <= 0:
        raise ValueError("first_order_transient_run requires steps > 0.")
    problem = first_order_transient(
        capacity=capacity,
        stiffness=stiffness,
        history=history,
        source=source,
        dt=dt,
        study=study,
        solution=current,
        constraints=constraints,
        bcs=bcs,
        solver_options=solver_options,
        name=name,
    )
    return FirstOrderTransientStep(
        name=name,
        problem=problem,
        current=current,
        previous=previous,
        dt=float(dt),
        steps=int(steps),
        study=study,
        update_load=update_load,
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        procedure=procedures.implicit_euler(),
        history_monitor=ThermalBalanceMonitor(
            capacity=capacity,
            stiffness=stiffness,
            source=source,
            dt=float(dt),
        ),
    )


def nonlinear_first_order_transient_run(
    *,
    residual,
    jacobian,
    current,
    previous,
    dt: float,
    steps: int,
    study=None,
    constraints=None,
    bcs=None,
    solver_options: NonlinearSolverOptions | NewtonSolverOptions | None = None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    history_monitor=None,
    name: str = "nonlinear_first_order_transient",
    petsc_options_prefix: str = "agentfem_nonlinear_transient_",
) -> FirstOrderTransientStep:
    """Create a nonlinear implicit-Euler step with the shared lifecycle."""

    from . import procedures

    if dt <= 0.0:
        raise ValueError("nonlinear_first_order_transient_run requires dt > 0.")
    if steps <= 0:
        raise ValueError("nonlinear_first_order_transient_run requires steps > 0.")
    procedure = procedures.implicit_euler()
    problem = NonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=current,
        bcs=_collect_bcs(constraints=constraints, bcs=bcs),
        solver_options=solver_options,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
        procedure=procedure,
    )
    return FirstOrderTransientStep(
        name=name,
        problem=problem,
        current=current,
        previous=previous,
        dt=float(dt),
        steps=int(steps),
        study=study,
        update_load=update_load,
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        procedure=procedure,
        history_monitor=history_monitor,
    )


def explicit_dynamics(
    *,
    state,
    integrator,
    residual,
    stiffness=None,
    dt: float,
    steps: int,
    study=None,
    prescribed=(),
    constraints=(),
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    history_every: int = 1,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    history_monitor=None,
    stability=None,
    name: str = "explicit_dynamics",
) -> ExplicitDynamicsStep:
    """Create a second-order explicit dynamics step."""

    _require_study_analysis(study, "second_order_dynamics")
    if dt <= 0.0:
        raise ValueError("explicit_dynamics requires dt > 0.")
    if steps <= 0:
        raise ValueError("explicit_dynamics requires steps > 0.")
    if int(history_every) <= 0:
        raise ValueError("explicit_dynamics history_every must be positive.")
    from . import procedures
    from .diagnostics import MechanicalEnergyMonitor

    return ExplicitDynamicsStep(
        name=name,
        study=study,
        state=state,
        integrator=integrator,
        residual=residual,
        history_monitor=(
            MechanicalEnergyMonitor(
                mass=integrator.mass,
                stiffness=stiffness,
            )
            if history_monitor is None
            else history_monitor
        ),
        stability=stability,
        prescribed=tuple(_as_list(prescribed)),
        constraints=tuple(_as_list(constraints)),
        update_load=update_load,
        dt=dt,
        steps=int(steps),
        save_every=None if save_every is None else int(save_every),
        print_every=None if print_every is None else int(print_every),
        history_every=int(history_every),
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        procedure=procedures.central_difference(),
    )


def modal_analysis(
    *,
    target,
    mass,
    stiffness,
    modes: int,
    study=None,
    constraints=(),
    bcs=None,
    target_frequency: float | None = None,
    tolerance: float = 1.0e-9,
    maximum_iterations: int = 1000,
    rigid_mode_tolerance: float = 1.0e-10,
    name: str = "modal_analysis",
) -> ModalAnalysisStep:
    """Create an undamped linear structural modal analysis."""

    from . import procedures

    _require_study_analysis(study, "modal")
    return ModalAnalysisStep(
        name=name,
        target=target,
        stiffness=stiffness,
        mass=mass,
        modes=int(modes),
        study=study,
        constraints=tuple(_as_list(constraints)),
        bcs=tuple(_as_list(bcs)),
        target_frequency=target_frequency,
        tolerance=float(tolerance),
        maximum_iterations=int(maximum_iterations),
        rigid_mode_tolerance=float(rigid_mode_tolerance),
        procedure=procedures.modal(),
    )


def implicit_dynamics(
    *,
    state,
    mass,
    stiffness,
    force,
    damping=None,
    dt: float,
    steps: int,
    parameters=None,
    study=None,
    constraints=(),
    bcs=None,
    solver_options: LinearSolverOptions | None = None,
    update_load=None,
    progress=True,
    status_file=None,
    checkpoint_policy=None,
    save_every: int | None = None,
    print_every: int | None = None,
    name: str = "implicit_dynamics",
) -> ImplicitDynamicsStep:
    """Create a linear Newmark or generalized-alpha dynamics step.

    Strong constraints are imposed as zero acceleration. This is correct for
    time-invariant prescribed displacements; moving supports require a
    prescribed kinematic-history object and are rejected by the future
    validation layer rather than silently approximated here.
    """

    from . import procedures
    from .diagnostics import MechanicalEnergyMonitor
    from .time import implicit as implicit_time

    _require_study_analysis(study, "second_order_dynamics")
    if dt <= 0.0 or steps <= 0:
        raise ValueError("implicit_dynamics requires dt > 0 and steps > 0.")
    selected = implicit_time.newmark() if parameters is None else parameters
    am = selected.alpha_m
    af = selected.alpha_f
    beta = selected.beta
    gamma = selected.gamma
    effective_expression = (
        (1.0 - am) * mass.expression
        + (1.0 - af) * gamma * dt * (
            0 if damping is None else damping.expression
        )
        + (1.0 - af) * beta * dt**2 * stiffness.expression
    )
    V = state.a.value.function_space
    u_predictor = fem.Function(V, name="DisplacementPredictor")
    v_predictor = fem.Function(V, name="VelocityPredictor")
    u_alpha_predictor = fem.Function(V, name="DisplacementAlphaPredictor")
    v_alpha_predictor = fem.Function(V, name="VelocityAlphaPredictor")
    rhs = force.expression - am * ufl.action(
        mass.expression,
        state.a.value,
    )
    if damping is not None:
        rhs -= ufl.action(damping.expression, v_alpha_predictor)
    rhs -= ufl.action(stiffness.expression, u_alpha_predictor)
    source_bcs = _collect_bcs(constraints=constraints, bcs=bcs)
    acceleration_bcs = _zero_kinematic_bcs(source_bcs, V)
    problem = LinearVariationalProblem(
        bilinear_form=fem.form(effective_expression),
        linear_form=fem.form(rhs),
        solution=state.a_next.value,
        bcs=acceleration_bcs,
        solver_options=solver_options,
    )
    return ImplicitDynamicsStep(
        name=name,
        state=state,
        problem=problem,
        parameters=selected,
        dt=float(dt),
        steps=int(steps),
        displacement_predictor=u_predictor,
        velocity_predictor=v_predictor,
        displacement_alpha_predictor=u_alpha_predictor,
        velocity_alpha_predictor=v_alpha_predictor,
        study=study,
        update_load=update_load,
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint_policy,
        history_monitor=MechanicalEnergyMonitor(
            mass=mass,
            stiffness=stiffness,
        ),
        procedure=(
            procedures.newmark()
            if selected.method == "newmark"
            else procedures.generalized_alpha()
        ),
    )


def _zero_kinematic_bcs(source_bcs, V) -> list:
    result = []
    shape = V.element.value_shape
    value = (
        PETSc.ScalarType(0.0)
        if len(shape) == 0
        else np.zeros(shape, dtype=PETSc.ScalarType)
    )
    for bc in source_bcs:
        dof_indices = bc.dof_indices()
        dofs = dof_indices[0] if isinstance(dof_indices, tuple) else dof_indices
        result.append(fem.dirichletbc(value, dofs, V))
    return result


def _collect_bcs(*, constraints=None, bcs=None) -> list:
    result = []
    if bcs is not None:
        result.extend(_as_list(bcs))
    if constraints is not None:
        for item in _as_list(constraints):
            if hasattr(item, "bcs"):
                result.extend(item.bcs)
            elif hasattr(item, "bc"):
                result.append(item.bc)
            else:
                raise TypeError(
                    "AFM-CONSTRAINT-PROCEDURE-001: implicit dynamics received "
                    f"{type(item).__name__}, which is not a strong Dirichlet "
                    "constraint. Run model.check() and select an exact affine/MPC "
                    "backend for non-Dirichlet kinematic relations."
                )
    return result


def _split_linear_constraints(*, constraints=None, bcs=None) -> tuple[list, object | None]:
    """Separate strong data from one exact-MPC linear lowering provider.

    A linear system has two distinct assembly contracts: ordinary Dirichlet
    elimination and exact multi-point elimination.  Keeping the split here
    prevents a public MPC asset from being mistaken for a boundary condition,
    while unknown constraint providers continue to fail before assembly.
    """

    from . import constraints as constraint_api

    strong_bcs = [] if bcs is None else list(_as_list(bcs))
    exact_mpc = []
    for item in constraint_api.constraint_assets(constraints):
        capability = constraint_api.constraint_capabilities(item)
        if (
            capability is not None
            and capability.enforcement == "exact_multi_point_constraint"
        ):
            if not hasattr(item, "backend"):
                raise TypeError(
                    "AFM-CONSTRAINT-MPC-001: an exact-MPC provider must expose "
                    "its assembled backend through a `backend` attribute."
                )
            exact_mpc.append(item)
        elif hasattr(item, "bcs"):
            strong_bcs.extend(item.bcs)
        elif hasattr(item, "bc"):
            strong_bcs.append(item.bc)
        elif callable(getattr(item, "dof_indices", None)):
            strong_bcs.append(item)
        else:
            raise TypeError(
                "AFM-CONSTRAINT-PROCEDURE-001: linear analysis received "
                f"{type(item).__name__}, which has no supported strong or "
                "exact-MPC lowering contract. Run model.check() and select a "
                "compatible constraint provider."
            )
    if len(exact_mpc) > 1:
        names = tuple(
            str(getattr(item, "name", type(item).__name__)) for item in exact_mpc
        )
        raise ValueError(
            "AFM-CONSTRAINT-MPC-002: one linear system can consume exactly one "
            f"exact MPC provider; received {names!r}. Combine the relations in "
            "one provider before constructing the step."
        )
    return strong_bcs, (None if not exact_mpc else exact_mpc[0])


def _reaction_field(residual_form, solution, *, name: str):
    import dolfinx.fem.petsc as fem_petsc
    from petsc4py import PETSc

    residual = fem_petsc.assemble_vector(fem.form(residual_form))
    residual.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    reaction = fem.Function(solution.function_space, name=name)
    values = residual.array_r
    reaction.x.array[: len(values)] = values
    reaction.x.scatter_forward()
    residual.destroy()
    return reaction


def _print_interval(print_every: int | None, steps: int) -> int:
    if print_every is None:
        return max(1, int(steps) // 10)
    return int(print_every)


def _save_interval(save_every: int | None, *, output, steps: int) -> int:
    if save_every is not None:
        return int(save_every)
    if output is None:
        return 0
    return int(steps)


def _transient_output_fields(selected):
    """Flatten Functions and live derived-field groups for one writer."""

    output = []
    live = []

    def visit(item):
        if hasattr(item, "update") and hasattr(item, "fields"):
            live.append(item)
            for field in tuple(item.fields):
                visit(field)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)
        else:
            output.append(fields.unwrap(item))

    visit(selected)
    if not output:
        raise ValueError("Transient output requires at least one field.")
    return tuple(output), tuple(live)


def _transient_result_series(path, domain):
    """Choose one ParaView-readable dataset layout for transient output."""

    selected = Path(path)
    if selected.suffix.lower() == ".pvd" or domain.comm.size > 1:
        from . import io

        actual = (
            selected
            if selected.suffix.lower() == ".pvd"
            else selected.with_suffix(".pvd")
        )
        return (
            io.ParaViewTimeSeries(actual, domain),
            actual,
            "dolfinx_vtk_collective",
            "single_unstructured_grid_per_time",
        )

    from .results.output import UnifiedXDMFTimeSeries

    if selected.suffix.lower() != ".xdmf":
        raise ValueError("Transient field output must use an .xdmf or .pvd path.")
    return (
        UnifiedXDMFTimeSeries(selected, deformation_scale=0.0),
        selected,
        "agentfem_unified_xdmf",
        "single_uniform_grid",
    )


def _refresh_transient_output_fields(live_field_sets) -> None:
    for selected in live_field_sets:
        selected.update()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _unwrap_result_field(field):
    return fields.unwrap(field)


def _field_location(field) -> str:
    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    discontinuous = bool(
        getattr(element, "discontinuous", False)
        or getattr(basix_element, "discontinuous", False)
    )
    return "cells" if discontinuous else "nodes"


def _projected_field_processing(field) -> dict[str, object]:
    """Describe the post-processing contract of a projected result field."""

    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    degree = getattr(basix_element, "degree", None)
    family = getattr(basix_element, "family", None)
    selected_degree = None if degree is None else int(degree)
    return {
        "source_position": "constitutive_expression",
        "method": "global_l2_projection",
        "representation": (
            "cell_average" if selected_degree == 0 else "discontinuous_field"
        ),
        "space_family": (
            None if family is None else str(getattr(family, "name", family))
        ),
        "space_degree": selected_degree,
        "nodal_extrapolation": False,
        "interelement_smoothing": False,
        "material_boundary_averaging": False,
    }


def _describe_asset(asset) -> object:
    if hasattr(asset, "as_dict"):
        return asset.as_dict()
    if hasattr(asset, "summary"):
        return asset.summary()
    return getattr(asset, "name", repr(asset))


def _require_study_analysis(study, analysis: str) -> None:
    if study is not None and hasattr(study, "require"):
        study.require(analysis=analysis)
