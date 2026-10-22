# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""One serial frozen-state affine Newton solve, without increment acceptance.

The load-path Procedure owns factors, cutbacks, commits, output and restart.
This helper owns only Newton trial vectors and current-trial full-space
Operator contributions. No geometric projection or force survives a changed
trial displacement, material refresh, or return to the owning Procedure.
"""

from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AffineEquilibrium:
    reduced_values: np.ndarray
    converged: bool
    iterations: int
    initial_norm: float
    residual_norm: float
    threshold: float
    step_lengths: tuple[float, ...]
    linear_iterations: tuple[int, ...]
    linear_reasons: tuple[int, ...]
    message: str
    residual_assemblies: int
    reused_assemblies: int
    procedure_checks: dict[str, object] = field(default_factory=dict)
    adaptation_iterations: int | None = None


class _TrialAssembly:
    """Sole owner of one full residual; contributions refer to the same trial."""

    def __init__(self, residual, contributions, transformation):
        self.residual = residual
        self.contributions = contributions
        reduced = None
        try:
            reduced = transformation.createVecRight()
            transformation.multTranspose(residual, reduced)
            self.norm = float(reduced.norm())
        except BaseException:
            self.close()
            raise
        finally:
            if reduced is not None:
                reduced.destroy()

    def close(self):
        if self.residual is not None:
            self.residual.destroy()
            self.residual = None
        self.contributions = ()


class _AffineLinearWorkspace:
    """PETSc allocation owner for one fixed-reduction augmentation attempt.

    The bounded plane Procedure guarantees an unchanged potential sparsity
    graph, as within its Newton loop. PtAP and setOperators still run for every
    correction. The optional managed Krylov owner may retain a preconditioner,
    never a tangent. Nothing survives acceptance, cutback or restart.
    """

    def __init__(self, transformation, options):
        self.transformation = transformation
        self.transformation_state = transformation.stateGet()
        self.options = options
        self.reduced_tangent = None
        self.linear_solver = None
        self.linear_correction = None
        self.closed = False

    def validate(self, transformation, options):
        if self.closed:
            raise RuntimeError("Affine linear workspace is already closed.")
        if (transformation is not self.transformation
                or transformation.stateGet() != self.transformation_state
                or options != self.options):
            raise ValueError("Affine linear workspace requires its unchanged reduction and options.")

    def __enter__(self):
        self.validate(self.transformation, self.options)
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True
        try:
            if self.linear_solver is not None:
                self.linear_solver.destroy()
        finally:
            self.linear_solver = None
            try:
                if self.linear_correction is not None:
                    self.linear_correction.close()
            finally:
                self.linear_correction = None
                if self.reduced_tangent is not None:
                    self.reduced_tangent.destroy()
                    self.reduced_tangent = None


def solve_serial_affine_equilibrium(
    residual, jacobian, function, reduction, transformation, reduced_values, *,
    options, operators, transaction, start_factor, target_factor,
    reporter, step_name, step_number, increment_number, attempt_number,
    reuse_accepted_assembly=True, reuse_linear_solver=True,
    _linear_workspace=None,
):
    """Solve fixed-factor equilibrium, retaining accepted line-search work.

    ``reuse_accepted_assembly=False`` is a private recomputation reference for
    tests/benchmarks, not a new user solver policy or checkpoint identity.
    Both modes use the same trial refresh, reduction, Newton and line search.
    ``reuse_linear_solver=False`` retains the private fresh-KSP comparison.
    Reuse owns an object only within this fixed reduction, not a stale tangent
    or frozen numerical preconditioner. Every solve sets the current operator.
    A bounded outer augmentation may own ``_linear_workspace`` across its
    frozen equilibria. Otherwise this routine releases all its PETSc objects.
    The caller always owns the transformation.
    """
    # Import at execution time: solvers owns the public options and existing
    # assembly/transaction primitives, and dispatches this private serial loop.
    from . import solvers as owner
    from .events import SolveEvent

    if function.function_space.mesh.comm.size != 1:
        raise NotImplementedError("Serial affine equilibrium does not admit MPI.")
    current = None
    if _linear_workspace is not None:
        if not reuse_linear_solver:
            raise ValueError("A shared workspace requires linear solver reuse.")
        _linear_workspace.validate(transformation, options.linear_options)
    reduced_tangent = None if _linear_workspace is None else _linear_workspace.reduced_tangent
    linear_solver = None if _linear_workspace is None else _linear_workspace.linear_solver
    linear_correction = None if _linear_workspace is None else _linear_workspace.linear_correction
    adaptive = options.linear_options.preconditioner_reuse == "adaptive"
    if adaptive and not reuse_linear_solver:
        raise ValueError("Adaptive preconditioning requires its linear-sequence owner.")
    correction_checks = ({
        "linear_preconditioner_reuses": 0,
        "linear_preconditioner_builds": 0,
        "linear_preconditioner_rebuild_retries": 0,
        "maximum_true_linear_residual_ratio": 0.,
    } if adaptive else {})
    assemblies = reused = 0
    steps, linear_iterations, linear_reasons = [], [], []

    def assemble():
        nonlocal assemblies
        full, contributions = owner._assemble_affine_full_residual(residual, operators)
        assemblies += 1
        return _TrialAssembly(full, contributions, transformation)

    def iteration_event(iteration, norm, alpha, message=""):
        owner._emit(reporter, SolveEvent(
            "iteration", step_name, step_number=step_number,
            increment=increment_number, attempt=attempt_number,
            start_factor=start_factor, target_factor=target_factor,
            iteration=iteration, residual_norm=norm, step_length=alpha,
            message=message,
        ))

    try:
        reuse_trial = getattr(transaction, "_reuse_trial_response", None)
        material_reused = bool(callable(reuse_trial) and reuse_trial(
            start_factor=start_factor, target_factor=target_factor,
        ))
        valid, message = ((True, "") if material_reused else owner._try_refresh_affine_trial_state(
            transaction, start_factor=start_factor, target_factor=target_factor,
        ))
        if valid:
            current = assemble()
        initial_norm = current.norm if current is not None else float("inf")
        current_norm = initial_norm
        threshold = (options.atol + options.rtol * initial_norm
                     if np.isfinite(initial_norm) else options.atol)
        converged = np.isfinite(current_norm) and current_norm <= threshold
        if current is not None and not reuse_accepted_assembly:
            current.close()
            current = None
        iteration = 0
        while not converged and iteration < options.max_it:
            if not np.isfinite(current_norm):
                break
            iteration += 1
            if current is None:
                current = assemble()
            else:
                reused += 1
            with ExitStack() as allocated:
                full_tangent = owner.fem_petsc.assemble_matrix(jacobian)
                allocated.callback(full_tangent.destroy)
                for operator, contribution in current.contributions:
                    operator.add_tangent(full_tangent, contribution)
                full_tangent.assemble()
                reduced_residual = transformation.createVecRight()
                allocated.callback(reduced_residual.destroy)
                transformation.multTranspose(current.residual, reduced_residual)
                reduced_tangent = full_tangent.PtAP(transformation, result=reduced_tangent)
                if _linear_workspace is not None:
                    _linear_workspace.reduced_tangent = reduced_tangent
                rhs = reduced_residual.copy()
                allocated.callback(rhs.destroy)
                rhs.scale(-1.0)
                correction = reduced_residual.duplicate()
                allocated.callback(correction.destroy)
                correction.set(0.0)
                if adaptive:
                    if linear_correction is None:
                        from ._adaptive_linear_correction import _AdaptiveLinearCorrection
                        linear_correction = _AdaptiveLinearCorrection(options.linear_options)
                        if _linear_workspace is not None:
                            _linear_workspace.linear_correction = linear_correction
                    info = linear_correction.solve(reduced_tangent, rhs, correction)
                    evidence = linear_correction.last_evidence
                    correction_checks["linear_preconditioner_reuses"] += int(evidence.reused)
                    correction_checks["linear_preconditioner_builds"] += int(
                        not evidence.reused or evidence.rebuilt)
                    correction_checks["linear_preconditioner_rebuild_retries"] += int(evidence.rebuilt)
                    ratio = evidence.true_residual / evidence.threshold
                    previous = correction_checks["maximum_true_linear_residual_ratio"]
                    correction_checks["maximum_true_linear_residual_ratio"] = (
                        max(previous, ratio)
                        if previous is not None and np.isfinite(ratio) else None
                    )
                else:
                    if reuse_linear_solver and linear_solver is None:
                        linear_solver = owner.create_ksp(
                            reduced_tangent.comm, options.linear_options,
                        )
                        if _linear_workspace is not None:
                            _linear_workspace.linear_solver = linear_solver
                    info = owner.solve_matrix_system(
                        reduced_tangent, rhs, correction, options.linear_options,
                        ksp=linear_solver, raise_on_failure=False,
                    )
                linear_iterations.append(int(info.iterations))
                linear_reasons.append(int(info.converged_reason))
                direction = correction.array_r.copy() if info.converged else None
            current.close()
            current = None
            if direction is None:
                iteration_event(iteration, current_norm, 0.,
                                f"linear correction failed: KSP reason {info.converged_reason}")
                break

            alpha, accepted, trial_failure = 1., False, ""
            while alpha + 1e-15 >= options.line_search_minimum:
                trial = reduced_values + alpha * direction
                owner._assign_reconstructed(function, reduction, trial)
                valid, trial_message = owner._try_refresh_affine_trial_state(
                    transaction, start_factor=start_factor, target_factor=target_factor,
                )
                if valid:
                    current = assemble()
                    trial_norm = current.norm
                else:
                    trial_norm = float("inf")
                    trial_failure = trial_message
                if np.isfinite(trial_norm) and (
                    options.line_search == "basic"
                    or trial_norm <= options.atol
                    or trial_norm <= current_norm * (1. - 1e-4 * alpha)
                ):
                    reduced_values = trial
                    current_norm = trial_norm
                    steps.append(alpha)
                    accepted = True
                    if not reuse_accepted_assembly:
                        current.close()
                        current = None
                    break
                if current is not None:
                    current.close()
                    current = None
                if options.line_search == "basic":
                    break
                alpha *= options.line_search_reduction
            if not accepted:
                owner._assign_reconstructed(function, reduction, reduced_values)
                restored, restore_message = owner._try_refresh_affine_trial_state(
                    transaction, start_factor=start_factor, target_factor=target_factor,
                )
                if not restored:
                    trial_failure = restore_message
                if trial_failure:
                    message = trial_failure
                iteration_event(iteration, current_norm, 0., trial_failure)
                break
            iteration_event(iteration, current_norm, steps[-1])
            converged = current_norm <= threshold
        return AffineEquilibrium(
            reduced_values, converged, iteration, initial_norm, current_norm,
            threshold, tuple(steps), tuple(linear_iterations),
            tuple(linear_reasons), message, assemblies, reused,
            procedure_checks={"material_response_reuses": int(material_reused), **correction_checks},
        )
    finally:
        if current is not None:
            current.close()
        if linear_solver is not None and _linear_workspace is None:
            linear_solver.destroy()
        if linear_correction is not None and _linear_workspace is None:
            linear_correction.close()
        if reduced_tangent is not None and _linear_workspace is None:
            reduced_tangent.destroy()
