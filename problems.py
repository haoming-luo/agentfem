"""Problem/state containers for standard finite-element workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dolfinx import fem

from . import assembly
from . import fields
from . import spaces
from . import time
from .kernel import dofs
from .solvers import (
    AffineNewtonOptions,
    LinearSolverOptions,
    NonlinearSolverOptions,
    solve_affine_nonlinear_path,
    solve_linear_problem,
    solve_nonlinear_problem,
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

    def solve(self):
        """Assemble and solve the problem into ``solution``."""

        return solve_linear_problem(
            self.bilinear_form,
            self.linear_form,
            self.solution,
            bcs=self.bcs,
            options=self.solver_options,
        )

    def solve_result(self, *, name: str = "linear_variational_result"):
        """Solve and wrap the solution in a scientific result object."""

        from .results import from_solution

        solution = self.solve()
        return from_solution(solution, name=name)


@dataclass
class LinearSystemProblem:
    """Engineering-level linear system problem, usually ``K x = F``."""

    system: object
    solution: object | None = None
    unknown: object | None = None
    bcs: list = field(default_factory=list)
    solver_options: LinearSolverOptions | None = None

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

        return cls(
            system=operators.linear_system(K, F, name=name),
            unknown=unknown,
            solution=solution,
            bcs=_collect_bcs(constraints=constraints, bcs=bcs),
            solver_options=solver_options,
        )

    def solve(self):
        """Compile the system operators and solve into ``solution``."""

        solution = self._solution()
        return solve_linear_problem(
            fem.form(self.system.lhs_form()),
            fem.form(self.system.rhs_form()),
            solution,
            bcs=self.bcs,
            options=self.solver_options,
        )

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
            "solver": (
                self.solver_options.summary()
                if self.solver_options is not None
                else LinearSolverOptions().summary()
            ),
        }

    def _solution(self):
        if self.solution is not None:
            return self.solution
        if self.unknown is not None and hasattr(self.unknown, "value"):
            return self.unknown.value
        raise ValueError("LinearSystemProblem requires solution or unknown.")


@dataclass
class NonlinearVariationalProblem:
    """Nonlinear residual problem ``R(u; v) = 0`` solved by PETSc SNES."""

    residual_form: object
    solution: object
    bcs: list = field(default_factory=list)
    jacobian_form: object | None = None
    solver_options: NonlinearSolverOptions | None = None
    name: str = "nonlinear_problem"
    petsc_options_prefix: str = "agentfem_nonlinear_"
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
    solver_options: AffineNewtonOptions | None = None
    output_every: int | None = 1
    output_factors: tuple[float, ...] = ()
    progress: object = True
    status_file: object | None = None
    step_number: int = 1
    name: str = "affine_nonlinear_problem"
    last_solve_info: object | None = field(default=None, init=False)
    snapshots: list = field(default_factory=list, init=False)

    def solve(self):
        if self.output_every is not None and self.output_every <= 0:
            raise ValueError("Affine nonlinear output_every must be positive.")
        self.snapshots.clear()
        self.snapshots.append(_load_snapshot(0, 0.0, self.solution, zero=True))

        def capture(index, factor, solution, solve_info):
            save_by_increment = (
                self.output_every is not None
                and index % self.output_every == 0
            )
            save_by_factor = any(
                abs(factor - value) <= 1.0e-12
                for value in self.output_factors
            )
            if save_by_increment or save_by_factor or abs(factor - 1.0) <= 1.0e-12:
                self.snapshots.append(
                    _load_snapshot(
                        index,
                        factor,
                        solution,
                        solve_info=solve_info,
                    )
                )

        from .diagnostics import StandardRunReporter, comm_of

        if self.progress is True:
            reporter = StandardRunReporter(
                comm_of(self.solution),
                status_file=self.status_file,
            )
        elif self.progress in (False, None):
            reporter = None
        else:
            reporter = self.progress
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
            reporter=reporter,
            step_name=self.name,
            step_number=self.step_number,
        )
        self.last_solve_info = info
        return solution

    def solve_result(self):
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

    @property
    def system(self):
        """Return the engineering algebraic system used by this step."""

        return self.problem.system

    @property
    def bcs(self):
        """Return boundary conditions collected for this step."""

        return self.problem.bcs

    def solve(self):
        """Solve this analysis step."""

        return self.problem.solve()

    def solve_result(self):
        """Solve while retaining the existing ``solve()`` return contract.

        ``solve()`` continues to return the live DOLFINx solution field for
        backwards compatibility.  This method returns the higher-level result
        container used by post-processing, campaigns, and datasets.
        """

        from .results import from_solution

        solution = self.problem.solve()
        return from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "study": (
                    _describe_asset(self.study) if self.study is not None else None
                ),
            },
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
    save_every: int | None = None
    print_every: int | None = None

    def run(
        self,
        *,
        output=None,
        domain=None,
        fields=(),
        progress=None,
        comm=None,
    ):
        """Run the explicit dynamics step with optional output and progress text."""

        from . import io
        from .diagnostics import comm_of, print_on_root

        selected_comm = comm if comm is not None else comm_of(self.state.u)
        save_every = _save_interval(self.save_every, output=output, steps=self.steps)
        print_every = _print_interval(self.print_every, self.steps)
        stepper = time.TimeStepper(
            total_steps=self.steps,
            dt=self.dt,
            save_every=save_every,
            print_every=print_every,
        )
        output_fields = tuple(fields)

        if output is None:
            for info in stepper:
                self._advance_one(info.time)
                if info.should_print and progress is not None:
                    message = progress(info, self.state)
                    if message:
                        print_on_root(selected_comm, message)
            return self

        if domain is None:
            domain = self.state.u.function_space.mesh
        with io.XDMFTimeSeries(output, domain) as xdmf:
            xdmf.write_fields(0.0, *output_fields)
            for info in stepper:
                self._advance_one(info.time)
                if info.should_save:
                    xdmf.write_fields(info.time, *output_fields)
                if info.should_print and progress is not None:
                    message = progress(info, self.state)
                    if message:
                        print_on_root(selected_comm, message)
        return self

    def _advance_one(self, t: float) -> None:
        self.integrator.step(
            self.dt,
            time=t,
            residual_operator=self.residual,
            prescribed=self.prescribed,
            constraints=self.constraints,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact, agent-readable explicit step summary."""

        return {
            "kind": "explicit_dynamics_step",
            "name": self.name,
            "study": _describe_asset(self.study) if self.study is not None else None,
            "dt": self.dt,
            "steps": self.steps,
            "save_every": self.save_every,
            "print_every": _print_interval(self.print_every, self.steps),
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
        }


@dataclass
class TransientState:
    """Current/next fields for a first-order transient unknown."""

    current: object
    next: object

    @classmethod
    def create(cls, V, *, name: str = "Field"):
        """Create a zero-initialized transient state on a function space."""

        return cls(
            current=spaces.named_function(V, name),
            next=spaces.named_function(V, name),
        )

    def accept_step(self) -> None:
        """Copy ``next`` into ``current``."""

        dofs.copy_function(self.current, self.next)


@dataclass
class SecondOrderDynamicsState:
    """Displacement/velocity/acceleration fields for second-order dynamics."""

    u: object
    v: object
    a: object
    v_mid: object
    u_next: object
    v_next: object
    a_next: object

    @classmethod
    def create(
        cls,
        V,
        *,
        displacement_name: str = "Displacement",
        velocity_name: str = "Velocity",
        acceleration_name: str = "Acceleration",
    ):
        """Create zero-initialized explicit dynamics fields for a space."""

        return cls(
            u=fields.wrap(spaces.named_function(V, displacement_name)),
            v=fields.wrap(spaces.named_function(V, velocity_name)),
            a=fields.wrap(spaces.named_function(V, acceleration_name)),
            v_mid=fields.wrap(spaces.named_function(V, f"{velocity_name}_Midstep")),
            u_next=fields.wrap(spaces.named_function(V, displacement_name)),
            v_next=fields.wrap(spaces.named_function(V, velocity_name)),
            a_next=fields.wrap(spaces.named_function(V, acceleration_name)),
        )

    def predict_displacement(self, dt: float) -> None:
        """Predict ``u_next`` from the current state."""

        time.central_difference_predict_displacement(self.u_next, self.u, self.v, self.a, dt)

    def set_acceleration_from_residual(self, residual, inv_mass: np.ndarray) -> None:
        """Set ``a_next`` from a residual vector and inverse lumped mass."""

        time.acceleration_from_residual(self.a_next, residual, inv_mass)

    def update_midstep_velocity(self, dt: float) -> None:
        """Update ``v_mid`` with the central-difference half-step formula."""

        time.central_difference_update_midstep_velocity(self.v_mid, self.v, self.a, dt)

    def correct_velocity(self, dt: float) -> None:
        """Correct ``v_next`` using ``a`` and ``a_next``."""

        time.central_difference_correct_velocity(self.v_next, self.v, self.a, self.a_next, dt)

    def update_velocity(self, dt: float) -> None:
        """Update ``v_next`` from ``v_mid`` and ``a_next``."""

        time.central_difference_update_velocity(self.v_next, self.v_mid, self.a_next, dt)

    def update_displacement(self) -> None:
        """Copy ``u_next`` into the current displacement state."""

        dofs.copy_function(self.u, self.u_next)

    def advance_state(self) -> None:
        """Copy next-step fields into current fields."""

        dofs.copy_function(self.u, self.u_next)
        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)

    def accept_step(self) -> None:
        """Compatibility alias for ``advance_state``."""

        self.advance_state()

    def accept_displacement(self) -> None:
        """Compatibility alias for ``update_displacement``."""

        self.update_displacement()

    def accept_velocity_acceleration(self) -> None:
        """Compatibility alias for advancing velocity and acceleration."""

        self.advance_velocity_acceleration()

    def advance_velocity_acceleration(self) -> None:
        """Advance velocity and acceleration to the next time level."""

        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)


@dataclass
class LumpedMassOperator:
    """Diagonal mass operator for explicit dynamics."""

    mass: np.ndarray
    inv_mass: np.ndarray

    @classmethod
    def assemble(cls, V, density=1.0, measure=None):
        """Assemble a lumped mass operator for a function space."""

        if measure is None:
            mass = assembly.assemble_lumped_mass(V, density)
        else:
            mass = assembly.assemble_lumped_mass(V, density, measure=measure)
        return cls(mass=mass, inv_mass=assembly.inverse_diagonal(mass))


ExplicitDynamicsState = SecondOrderDynamicsState


def second_order_state(field_or_space, **kwargs) -> SecondOrderDynamicsState:
    """Create a second-order dynamics state from a field or function space."""

    if hasattr(field_or_space, "space"):
        V = field_or_space.space
    elif hasattr(field_or_space, "function_space"):
        V = field_or_space.function_space
    else:
        V = field_or_space
    return SecondOrderDynamicsState.create(V, **kwargs)


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
    return AnalysisStep(
        name=name,
        study=study,
        problem=problem,
        method="linear_static",
    )


def nonlinear(
    residual,
    solution,
    *,
    jacobian=None,
    constraints=None,
    bcs=None,
    solver_options: NonlinearSolverOptions | None = None,
    name: str = "nonlinear",
    petsc_options_prefix: str = "agentfem_nonlinear_",
) -> NonlinearVariationalProblem:
    """Create a general nonlinear residual problem."""

    return NonlinearVariationalProblem(
        residual_form=residual,
        jacobian_form=jacobian,
        solution=solution,
        bcs=_collect_bcs(constraints=constraints, bcs=bcs),
        solver_options=solver_options,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
    )


def affine_nonlinear(
    residual,
    solution,
    *,
    jacobian,
    constraint,
    load_factors=None,
    incrementation=None,
    solver_options: AffineNewtonOptions | None = None,
    output_every: int | None = 1,
    output_factors=(),
    progress=True,
    status_file=None,
    name: str = "affine_nonlinear",
) -> AffineNonlinearVariationalProblem:
    """Create a nonlinear problem reduced by an affine constraint map."""

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
        progress=progress,
        status_file=status_file,
        name=name,
    )


@dataclass(frozen=True)
class LoadIncrementSnapshot:
    """A copied solution state at one nonlinear load factor."""

    index: int
    load_factor: float
    solution: object
    solve_info: object | None = None

    def summary(self) -> dict[str, object]:
        return {
            "index": self.index,
            "load_factor": self.load_factor,
            "solution": getattr(self.solution, "name", type(self.solution).__name__),
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
) -> LoadIncrementSnapshot:
    selected = solution.value if hasattr(solution, "value") else solution
    copied = fem.Function(
        selected.function_space,
        name=getattr(selected, "name", "Solution"),
    )
    if not zero:
        copied.x.array[:] = selected.x.array
        copied.x.scatter_forward()
    return LoadIncrementSnapshot(
        index=int(index),
        load_factor=float(load_factor),
        solution=copied,
        solve_info=solve_info,
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

    C_over_dt = operators.scale(
        capacity,
        1.0 / dt,
        name="C_over_dt",
        kind=f"{method}_capacity_over_dt",
    )
    history_over_dt = operators.scale(
        history,
        1.0 / dt,
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
    return AnalysisStep(
        name=name,
        study=study,
        problem=problem,
        method=method,
        dt=dt,
    )


def explicit_dynamics(
    *,
    state,
    integrator,
    residual,
    dt: float,
    steps: int,
    study=None,
    prescribed=(),
    constraints=(),
    save_every: int | None = None,
    print_every: int | None = None,
    name: str = "explicit_dynamics",
) -> ExplicitDynamicsStep:
    """Create a second-order explicit dynamics step."""

    _require_study_analysis(study, "second_order_dynamics")
    if dt <= 0.0:
        raise ValueError("explicit_dynamics requires dt > 0.")
    if steps <= 0:
        raise ValueError("explicit_dynamics requires steps > 0.")
    return ExplicitDynamicsStep(
        name=name,
        study=study,
        state=state,
        integrator=integrator,
        residual=residual,
        prescribed=tuple(_as_list(prescribed)),
        constraints=tuple(_as_list(constraints)),
        dt=dt,
        steps=int(steps),
        save_every=None if save_every is None else int(save_every),
        print_every=None if print_every is None else int(print_every),
    )


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
                result.append(item)
    return result


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


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _describe_asset(asset) -> object:
    if hasattr(asset, "as_dict"):
        return asset.as_dict()
    if hasattr(asset, "summary"):
        return asset.summary()
    return getattr(asset, "name", repr(asset))


def _require_study_analysis(study, analysis: str) -> None:
    if study is not None and hasattr(study, "require"):
        study.require(analysis=analysis)
