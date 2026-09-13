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
from ._solver_lifecycle import PreparedSolve
from .dynamics import ModalSolveInfo
from .diagnostics import PerformanceLedger
from .constraints.affine import AffineConstraintDualHistory
from .kernel import dofs
from .mechanics.modal import ModalAnalysisStep
from .operators.core import LumpedMassOperator
from ._problem_fields import reaction_field as _reaction_field
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


# Public compatibility types implemented by their current ownership modules.
# The documentation generator follows this explicit map; ordinary private
# imports remain private and are not inferred as API merely because imported.
_DOCUMENTED_REEXPORTS = {
    "PreparedSolve": "_solver_lifecycle",
    "AffineNonlinearVariationalProblem": "_nonlinear_problems",
    "IncrementalNonlinearVariationalProblem": "_nonlinear_problems",
    "LoadIncrementSnapshot": "_nonlinear_problems",
    "NonlinearLoadIncrementInfo": "_nonlinear_problems",
    "NonlinearLoadPathInfo": "_nonlinear_problems",
    "ExplicitDynamicsStep": "_transient_problems",
    "FirstOrderTransientStep": "_transient_problems",
    "ImplicitDynamicsStep": "_transient_problems",
}


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
            "materials": tuple(
                _describe_asset(material) for material in self.materials
            ),
            "constraints": tuple(_describe_asset(item) for item in self.constraints),
            "loads": tuple(_describe_asset(item) for item in self.loads),
            "boundary_models": tuple(
                _describe_asset(item) for item in self.boundary_models
            ),
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

        from .results._problem import from_linear_variational_problem

        solution = self.solve()
        return from_linear_variational_problem(
            self,
            solution,
            name=name,
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

    def prepare(self) -> PreparedSolve:
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

    def solve_prepared(self, prepared: PreparedSolve):
        """Solve with a compatible prepared lifecycle and retain evidence."""

        solution = prepared.solve()
        self.last_solve_info = prepared.last_solve_info
        self.last_lifecycle_summary = dict(prepared.summary())
        return solution

    def solve_result(self, *, name: str | None = None):
        """Solve and return a :class:`SimulationResult`."""

        from .results._problem import from_linear_system_problem

        solution = self.solve()
        return from_linear_system_problem(
            self,
            solution,
            name=name or getattr(self.system, "name", "linear_system_result"),
        )

    def summary(self) -> dict[str, object]:
        """Return an inspectable K/F problem summary."""

        return {
            "kind": "linear_system_problem",
            "system": self.system.summary()
            if hasattr(self.system, "summary")
            else repr(self.system),
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
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
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

        from .results._problem import from_nonlinear_variational_problem

        solution = self.solve()
        return from_nonlinear_variational_problem(
            self,
            solution,
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
                None if self.last_solve_info is None else self.last_solve_info.as_dict()
            ),
            "procedure": (None if self.procedure is None else self.procedure.summary()),
        }

    def reaction_field(self, *, name: str = "RF"):
        """Return the assembled nonlinear residual at the current solution."""

        return _reaction_field(self.residual_form, self.solution, name=name)


from ._nonlinear_problems import (
    AffineNonlinearVariationalProblem,
    IncrementalNonlinearVariationalProblem,
    LoadIncrementSnapshot,
    NonlinearLoadIncrementInfo,
    NonlinearLoadPathInfo,
)


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

        from .results._analysis_step import from_analysis_step

        solution = self.problem.solve()
        return from_analysis_step(
            self,
            solution,
            output=output,
            fields=fields,
            field_variables=field_variables,
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
            "procedure": (None if self.procedure is None else self.procedure.summary()),
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


from ._transient_problems import (
    ExplicitDynamicsStep,
    FirstOrderTransientStep,
    ImplicitDynamicsStep,
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
        output_every=(None if output_every is None else int(output_every)),
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
        modes=modes,
        study=study,
        constraints=tuple(_as_list(constraints)),
        bcs=tuple(_as_list(bcs)),
        target_frequency=target_frequency,
        tolerance=float(tolerance),
        maximum_iterations=maximum_iterations,
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
        + (1.0 - af) * gamma * dt * (0 if damping is None else damping.expression)
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


def _split_linear_constraints(
    *, constraints=None, bcs=None
) -> tuple[list, object | None]:
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
