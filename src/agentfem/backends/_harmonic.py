"""Private FEniCSx/PETSc execution for real-block harmonic systems.

The scientific operator and procedure layers decide what is assembled and
which frequency is solved. This module owns the reusable DOLFINx
``LinearProblem`` allocation, PETSc solve, and unpreconditioned algebraic
evidence. It is intentionally private while the FEniCSx-first backend seam is
still experimental.
"""

from __future__ import annotations

from dataclasses import dataclass

import dolfinx.fem.petsc as fem_petsc
from dolfinx import fem
import numpy as np
import ufl
from petsc4py import PETSc

from ..solvers import LinearSolveInfo, _destroy_dolfinx_linear_problem


def _finite_harmonic_scalar(value, *, quantity: str) -> float:
    """Return one finite evidence scalar or fail before it can be published."""

    selected = float(value)
    if not np.isfinite(selected):
        raise FloatingPointError(
            "Direct harmonic solve produced non-finite "
            f"{quantity}: {selected!r}. No solve evidence was accepted."
        )
    return selected


@dataclass(frozen=True)
class HarmonicLinearSolveEvidence:
    """One direct harmonic linear solve and its physical residual evidence."""

    solve: LinearSolveInfo
    residual_norm: float
    relative_residual_norm: float
    relative_real_block_residual_norm: float | None
    relative_imaginary_block_residual_norm: float | None
    input_energy_per_cycle: float

    def __post_init__(self) -> None:
        values = {
            "KSP residual norm": self.solve.residual_norm,
            "algebraic residual norm": self.residual_norm,
            "relative algebraic residual norm": self.relative_residual_norm,
            "input energy per cycle": self.input_energy_per_cycle,
        }
        if self.relative_real_block_residual_norm is not None:
            values["relative real-block residual norm"] = (
                self.relative_real_block_residual_norm
            )
        if self.relative_imaginary_block_residual_norm is not None:
            values["relative imaginary-block residual norm"] = (
                self.relative_imaginary_block_residual_norm
            )
        for quantity, value in values.items():
            _finite_harmonic_scalar(value, quantity=quantity)

    @property
    def converged(self) -> bool:
        return self.solve.converged

    def equilibrium(self) -> dict[str, float]:
        values = {
            "residual_norm": self.residual_norm,
            "relative_residual_norm": self.relative_residual_norm,
        }
        if self.relative_real_block_residual_norm is not None:
            values["relative_real_block_residual_norm"] = (
                self.relative_real_block_residual_norm
            )
        if self.relative_imaginary_block_residual_norm is not None:
            values["relative_imaginary_block_residual_norm"] = (
                self.relative_imaginary_block_residual_norm
            )
        return values


class PreparedHarmonicLinearProblem:
    """A reusable real-block DOLFINx linear problem.

    Repeated calls reassemble matrix values and the right-hand side into the
    same PETSc objects. The KSP object and allocations are reused; numerical
    factorization or preconditioner reuse is deliberately not claimed.
    """

    def __init__(
        self,
        bilinear_forms,
        linear_forms,
        *,
        solution_real,
        solution_imaginary,
        bcs,
        solver_options,
        matrix_kind: str = "nest",
        petsc_options_prefix: str,
    ) -> None:
        self._problem = fem_petsc.LinearProblem(
            bilinear_forms,
            linear_forms,
            u=[solution_real, solution_imaginary],
            bcs=list(bcs),
            kind=matrix_kind,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=solver_options.petsc_options(),
        )
        self._solve_count = 0
        self._matrix_kind = str(matrix_kind)
        self._petsc_matrix_type = self._problem.A.getType()
        self._closed = False

    @classmethod
    def from_system(
        cls,
        system,
        *,
        solution_real,
        solution_imaginary,
        bcs,
        solver_options,
        angular_frequency: float,
        load_phase: float,
        petsc_options_prefix: str,
    ) -> "PreparedHarmonicLinearProblem":
        """Lower one operator-level harmonic system to reusable real blocks."""

        system.check()
        domain = solution_real.function_space.mesh
        omega = fem.Constant(domain, PETSc.ScalarType(float(angular_frequency)))
        phase = float(load_phase)
        real_factor = fem.Constant(domain, PETSc.ScalarType(np.cos(phase)))
        imaginary_factor = fem.Constant(domain, PETSc.ScalarType(np.sin(phase)))
        real_space = solution_real.function_space
        imaginary_space = solution_imaginary.function_space
        real_trial = ufl.TrialFunction(real_space)
        real_test = ufl.TestFunction(real_space)
        imaginary_trial = ufl.TrialFunction(imaginary_space)
        imaginary_test = ufl.TestFunction(imaginary_space)

        storage_rr = _bind_bilinear(system.storage, real_trial, real_test)
        storage_ii = _bind_bilinear(
            system.storage,
            imaginary_trial,
            imaginary_test,
        )
        dynamic_rr = storage_rr
        dynamic_ii = storage_ii
        if system.mass is not None:
            dynamic_rr -= omega**2 * _bind_bilinear(
                system.mass,
                real_trial,
                real_test,
            )
            dynamic_ii -= omega**2 * _bind_bilinear(
                system.mass,
                imaginary_trial,
                imaginary_test,
            )

        coupling_ri = 0.0 * _bind_bilinear(
            system.storage,
            imaginary_trial,
            real_test,
        )
        coupling_ir = 0.0 * _bind_bilinear(
            system.storage,
            real_trial,
            imaginary_test,
        )
        if system.loss is not None:
            coupling_ri += _bind_bilinear(
                system.loss,
                imaginary_trial,
                real_test,
            )
            coupling_ir += _bind_bilinear(
                system.loss,
                real_trial,
                imaginary_test,
            )
        if system.damping is not None:
            coupling_ri += omega * _bind_bilinear(
                system.damping,
                imaginary_trial,
                real_test,
            )
            coupling_ir += omega * _bind_bilinear(
                system.damping,
                real_trial,
                imaginary_test,
            )

        real_load = real_factor * _bind_linear(system.force, real_test)
        imaginary_load = imaginary_factor * _bind_linear(
            system.force,
            imaginary_test,
        )
        matrix_kind = (
            "nest" if solver_options.pc_type.lower() == "fieldsplit" else "mpi"
        )
        prepared = cls(
            [[dynamic_rr, -coupling_ri], [coupling_ir, dynamic_ii]],
            [real_load, imaginary_load],
            solution_real=solution_real,
            solution_imaginary=solution_imaginary,
            bcs=bcs,
            solver_options=solver_options,
            matrix_kind=matrix_kind,
            petsc_options_prefix=petsc_options_prefix,
        )
        prepared._angular_frequency = omega
        prepared._system = system
        prepared._solution_real = solution_real
        prepared._solution_imaginary = solution_imaginary
        prepared._load_phase = phase
        if matrix_kind != "nest":
            prepared._input_energy_forms = _compile_input_energy_forms(
                system,
                solution_real=solution_real,
                solution_imaginary=solution_imaginary,
            )
            prepared._input_energy_comm = domain.comm
        return prepared

    @property
    def angular_frequency(self) -> float | None:
        coefficient = getattr(self, "_angular_frequency", None)
        return None if coefficient is None else float(coefficient.value)

    @property
    def closed(self) -> bool:
        """Whether the retained PETSc allocation has been released."""

        return self._closed

    def _require_open(self) -> None:
        # Treat legacy/test doubles created without ``__init__`` as open; the
        # closed state is an additive lifecycle guard, not a new prerequisite
        # for inspecting backend evidence in isolation.
        if getattr(self, "_closed", False):
            raise RuntimeError("PreparedHarmonicLinearProblem is closed.")

    def set_angular_frequency(self, value: float) -> None:
        """Update the frequency coefficient without reallocating the problem."""

        self._require_open()
        coefficient = getattr(self, "_angular_frequency", None)
        if coefficient is None:
            raise RuntimeError(
                "This prepared problem was built from fixed legacy block forms."
            )
        selected = float(value)
        if not np.isfinite(selected) or selected < 0.0:
            raise ValueError("Angular frequency must be finite and nonnegative.")
        coefficient.value = PETSc.ScalarType(selected)

    @property
    def solve_count(self) -> int:
        return self._solve_count

    def solve(self) -> HarmonicLinearSolveEvidence:
        """Reassemble, solve, and return KSP plus ``A*x-b`` evidence."""

        self._require_open()
        problem = self._problem
        problem.solve()
        self._solve_count += 1
        solver = problem.solver
        _finite_harmonic_scalar(problem.x.norm(), quantity="solution-vector norm")
        solve = LinearSolveInfo(
            converged_reason=int(solver.getConvergedReason()),
            iterations=int(solver.getIterationNumber()),
            residual_norm=_finite_harmonic_scalar(
                solver.getResidualNorm(), quantity="KSP residual norm"
            ),
        )

        action = problem.b.duplicate()
        residual = problem.b.duplicate()
        try:
            problem.A.mult(problem.x, action)
            action.copy(residual)
            residual.axpy(-1.0, problem.b)
            action_norm = _finite_harmonic_scalar(action.norm(), quantity="A*x norm")
            rhs_norm = _finite_harmonic_scalar(
                problem.b.norm(), quantity="right-hand-side norm"
            )
            residual_norm = _finite_harmonic_scalar(
                residual.norm(), quantity="A*x-b norm"
            )
            system_scale = max(action_norm, rhs_norm, np.finfo(float).tiny)
            relative_real = None
            relative_imaginary = None
            if self._matrix_kind == "nest":
                residual_blocks = residual.getNestSubVecs()
                rhs_blocks = problem.b.getNestSubVecs()
                solution_blocks = problem.x.getNestSubVecs()
                if not all(
                    len(blocks) == 2
                    for blocks in (residual_blocks, rhs_blocks, solution_blocks)
                ):
                    raise RuntimeError(
                        "Harmonic nested evidence requires exactly two PETSc blocks."
                    )
                for index, block in enumerate(solution_blocks):
                    _finite_harmonic_scalar(
                        block.norm(),
                        quantity=f"solution block {index} norm",
                    )
                relative_real = _finite_harmonic_scalar(
                    float(residual_blocks[0].norm()) / system_scale,
                    quantity="relative real-block residual norm",
                )
                relative_imaginary = _finite_harmonic_scalar(
                    float(residual_blocks[1].norm()) / system_scale,
                    quantity="relative imaginary-block residual norm",
                )
                input_energy = _finite_harmonic_scalar(
                    np.pi
                    * (
                        rhs_blocks[1].dot(solution_blocks[0])
                        - rhs_blocks[0].dot(solution_blocks[1])
                    ),
                    quantity="input energy per cycle",
                )
            else:
                input_energy = _input_energy_per_cycle(
                    self._input_energy_forms,
                    comm=self._input_energy_comm,
                    load_phase=self._load_phase,
                )
            return HarmonicLinearSolveEvidence(
                solve=solve,
                residual_norm=residual_norm,
                relative_residual_norm=_finite_harmonic_scalar(
                    residual_norm / system_scale,
                    quantity="relative algebraic residual norm",
                ),
                relative_real_block_residual_norm=relative_real,
                relative_imaginary_block_residual_norm=relative_imaginary,
                input_energy_per_cycle=input_energy,
            )
        finally:
            residual.destroy()
            action.destroy()

    def summary(self) -> dict[str, object]:
        summary = {
            "backend": "fenicsx_petsc_real_block",
            "problem_allocation_count": 1,
            "matrix_allocation_count": 1,
            "matrix_values_reassembled_each_solve": True,
            "ksp_object_reused": True,
            "factorization_reuse_claimed": False,
            "solve_count": self.solve_count,
        }
        if self.angular_frequency is not None:
            summary["angular_frequency"] = self.angular_frequency
            summary["matrix_layout"] = (
                "nested" if self._matrix_kind == "nest" else "monolithic"
            )
            summary["petsc_matrix_type"] = self._petsc_matrix_type
            summary["component_residuals_available"] = self._matrix_kind == "nest"
            if self._matrix_kind != "nest":
                summary["component_residuals_unavailable_reason"] = (
                    "The monolithic PETSc layout exposes the total real-block "
                    "residual; component residuals require an explicit index split."
                )
        return summary

    def close(self) -> None:
        """Release the retained DOLFINx/PETSc solve allocation once.

        DOLFINx 0.11 exposes destruction only through ``LinearProblem.__del__``.
        Harmonic steps can participate in the model/execution-context reference
        cycle, so waiting for cyclic garbage collection can retain distributed
        KSP, matrix, and vector objects until interpreter shutdown.  Clearing
        the owned PETSc attributes here preserves the upstream destructor while
        ensuring it cannot destroy the same handles for a second time.
        """

        if self._closed:
            return
        problem = self._problem
        self._problem = None
        self._closed = True
        _destroy_dolfinx_linear_problem(problem)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _expression(operator):
    return operator.expression if hasattr(operator, "expression") else operator


def _compile_input_energy_forms(
    system,
    *,
    solution_real,
    solution_imaginary,
) -> tuple[object, object]:
    """Compile load-action forms once for a reusable monolithic sweep."""

    expression = _expression(system.force)
    arguments = tuple(expression.arguments())
    if len(arguments) != 1:
        raise ValueError("Harmonic input work requires one linear force operator.")
    real_action = ufl.replace(expression, {arguments[0]: solution_real})
    imaginary_action = ufl.replace(expression, {arguments[0]: solution_imaginary})
    return fem.form(real_action), fem.form(imaginary_action)


def _input_energy_per_cycle(
    forms,
    *,
    comm,
    load_phase: float,
) -> float:
    """Integrate phasor load work without depending on PETSc block layout."""

    real_form, imaginary_form = forms
    real = _finite_harmonic_scalar(
        comm.allreduce(fem.assemble_scalar(real_form)),
        quantity="real load action",
    )
    imaginary = _finite_harmonic_scalar(
        comm.allreduce(fem.assemble_scalar(imaginary_form)),
        quantity="imaginary load action",
    )
    phase = _finite_harmonic_scalar(load_phase, quantity="load phase")
    return _finite_harmonic_scalar(
        np.pi * (np.sin(phase) * real - np.cos(phase) * imaginary),
        quantity="input energy per cycle",
    )


def _bind_bilinear(operator, trial, test):
    expression = _expression(operator)
    arguments = tuple(expression.arguments())
    if len(arguments) != 2:
        raise ValueError("A harmonic matrix operator must be bilinear.")
    replacements = {}
    for argument in arguments:
        replacements[argument] = trial if argument.number() == 1 else test
    return ufl.replace(expression, replacements)


def _bind_linear(operator, test):
    expression = _expression(operator)
    arguments = tuple(expression.arguments())
    if len(arguments) != 1:
        raise ValueError("A harmonic force operator must be linear.")
    return ufl.replace(expression, {arguments[0]: test})


__all__ = ()
