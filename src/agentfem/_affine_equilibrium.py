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


def solve_serial_affine_equilibrium(
    residual, jacobian, function, reduction, transformation, reduced_values, *,
    options, operators, transaction, start_factor, target_factor,
    reporter, step_name, step_number, increment_number, attempt_number,
    reuse_accepted_assembly=True,
):
    """Solve fixed-factor equilibrium, retaining accepted line-search work.

    ``reuse_accepted_assembly=False`` is a private recomputation reference for
    tests/benchmarks, not a new user solver policy or checkpoint identity.
    Both modes use the same trial refresh, reduction, Newton and line search.
    The caller owns the transformation; this routine releases all its PETSc
    objects on success, a rejected trial, or an exception.
    """
    # Import at execution time: solvers owns the public options and existing
    # assembly/transaction primitives, and dispatches this private serial loop.
    from . import solvers as owner
    from .events import SolveEvent

    if function.function_space.mesh.comm.size != 1:
        raise NotImplementedError("Serial affine equilibrium does not admit MPI.")
    current = None
    reduced_tangent = None
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
        valid, message = owner._try_refresh_affine_trial_state(
            transaction, start_factor=start_factor, target_factor=target_factor,
        )
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
                rhs = reduced_residual.copy()
                allocated.callback(rhs.destroy)
                rhs.scale(-1.0)
                correction = reduced_residual.duplicate()
                allocated.callback(correction.destroy)
                correction.set(0.0)
                info = owner.solve_matrix_system(
                    reduced_tangent, rhs, correction, options.linear_options,
                    raise_on_failure=False,
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
                    trial_norm <= options.atol
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
        )
    finally:
        if current is not None:
            current.close()
        if reduced_tangent is not None:
            reduced_tangent.destroy()
