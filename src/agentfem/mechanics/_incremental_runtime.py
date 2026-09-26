# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light Newton lifecycle shared by incremental mechanics steps.

This module owns iteration policy and resource lifetime, not constitutive
equations, assembly, line search policy, or state commit/rollback.  Those stay
with the procedure that understands their physical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Generic, Protocol, TypeVar


PayloadT = TypeVar("PayloadT")
ResidualT = TypeVar("ResidualT")


class NewtonPolicy(Protocol):
    """Minimum option surface consumed by the incremental runtime."""

    absolute_tolerance: float
    relative_tolerance: float
    maximum_iterations: int


@dataclass(frozen=True)
class NewtonEvaluation(Generic[PayloadT, ResidualT]):
    """One fully assembled nonlinear residual and its procedure payload."""

    payload: PayloadT
    residual: ResidualT
    residual_norm: float


@dataclass(frozen=True)
class NewtonEvaluationRejection(Generic[PayloadT]):
    """A constitutive/local update rejected the increment before assembly."""

    payload: PayloadT | None
    reason: str
    residual_norm: float = float("inf")


@dataclass(frozen=True)
class NewtonCorrection(Generic[PayloadT, ResidualT]):
    """Outcome of one attempted Newton correction."""

    accepted: bool
    step_length: float = 1.0
    next_evaluation: NewtonEvaluation[PayloadT, ResidualT] | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class NewtonAttemptResult(Generic[PayloadT]):
    """Backend-neutral evidence from one nonlinear increment attempt."""

    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    payload: PayloadT | None
    rejection_reason: str | None = None


def run_newton_attempt(
    policy: NewtonPolicy,
    *,
    evaluate: Callable[
        [],
        NewtonEvaluation[PayloadT, ResidualT] | NewtonEvaluationRejection[PayloadT],
    ],
    correct: Callable[
        [NewtonEvaluation[PayloadT, ResidualT], int],
        NewtonCorrection[PayloadT, ResidualT],
    ],
    release: Callable[[ResidualT], None],
    on_iteration: Callable[[int, float, float], None] | None = None,
) -> NewtonAttemptResult[PayloadT]:
    """Run one Newton attempt while deterministically releasing residuals.

    ``correct`` may provide a line-search evaluation for the next iteration.
    The runtime then reuses it without repeating constitutive updates or
    assembly.  Every residual is released exactly once, including rejected and
    exceptional paths.
    """

    initial_norm: float | None = None
    norm = float("inf")
    payload: PayloadT | None = None
    pending: NewtonEvaluation[PayloadT, ResidualT] | None = None
    rejection_reason: str | None = None
    iteration = 0
    try:
        for iteration in range(int(policy.maximum_iterations) + 1):
            current = pending if pending is not None else evaluate()
            pending = None
            if isinstance(current, NewtonEvaluationRejection):
                payload = current.payload
                norm = float(current.residual_norm)
                rejection_reason = current.reason
                if initial_norm is None:
                    initial_norm = norm
                break
            payload = current.payload
            norm = float(current.residual_norm)
            if initial_norm is None:
                initial_norm = norm
            threshold = float(policy.absolute_tolerance) + (
                float(policy.relative_tolerance) * initial_norm
            )
            if isfinite(norm) and norm <= threshold:
                release(current.residual)
                return NewtonAttemptResult(
                    True,
                    iteration,
                    float(initial_norm),
                    norm,
                    payload,
                )
            if iteration == int(policy.maximum_iterations):
                release(current.residual)
                rejection_reason = "maximum nonlinear iterations reached"
                break
            try:
                correction = correct(current, iteration)
            finally:
                release(current.residual)
            try:
                if on_iteration is not None:
                    on_iteration(iteration + 1, norm, correction.step_length)
            except Exception:
                if correction.next_evaluation is not None:
                    release(correction.next_evaluation.residual)
                raise
            if not correction.accepted:
                rejection_reason = correction.rejection_reason
                if correction.next_evaluation is not None:
                    release(correction.next_evaluation.residual)
                break
            pending = correction.next_evaluation
    except Exception:
        if pending is not None:
            release(pending.residual)
        raise
    return NewtonAttemptResult(
        False,
        iteration,
        float(0.0 if initial_norm is None else initial_norm),
        norm,
        payload,
        rejection_reason,
    )
