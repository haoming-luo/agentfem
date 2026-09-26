from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentfem.mechanics._incremental_runtime import (
    NewtonCorrection,
    NewtonEvaluation,
    NewtonEvaluationRejection,
    run_newton_attempt,
)


class _Residual:
    def __init__(self, name):
        self.name = name
        self.releases = 0


def _policy(*, maximum_iterations=4):
    return SimpleNamespace(
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
        maximum_iterations=maximum_iterations,
    )


def test_incremental_runtime_converges_and_releases_every_residual_once():
    norms = iter((10.0, 1.0e-12))
    residuals = []
    iterations = []

    def evaluate():
        residual = _Residual(f"r{len(residuals)}")
        residuals.append(residual)
        return NewtonEvaluation("state", residual, next(norms))

    outcome = run_newton_attempt(
        _policy(),
        evaluate=evaluate,
        correct=lambda evaluation, iteration: NewtonCorrection(True),
        release=lambda residual: setattr(
            residual, "releases", residual.releases + 1
        ),
        on_iteration=lambda iteration, norm, alpha: iterations.append(
            (iteration, norm, alpha)
        ),
    )

    assert outcome.converged
    assert outcome.iterations == 1
    assert outcome.initial_residual_norm == pytest.approx(10.0)
    assert outcome.residual_norm == pytest.approx(1.0e-12)
    assert iterations == [(1, 10.0, 1.0)]
    assert [item.releases for item in residuals] == [1, 1]


def test_incremental_runtime_reuses_line_search_evaluation():
    initial = _Residual("initial")
    accepted = _Residual("accepted")
    evaluation_calls = 0

    def evaluate():
        nonlocal evaluation_calls
        evaluation_calls += 1
        return NewtonEvaluation("initial", initial, 4.0)

    def correct(_evaluation, _iteration):
        return NewtonCorrection(
            True,
            step_length=0.5,
            next_evaluation=NewtonEvaluation("accepted", accepted, 1.0e-12),
        )

    outcome = run_newton_attempt(
        _policy(),
        evaluate=evaluate,
        correct=correct,
        release=lambda residual: setattr(
            residual, "releases", residual.releases + 1
        ),
    )

    assert outcome.converged
    assert outcome.payload == "accepted"
    assert evaluation_calls == 1
    assert initial.releases == accepted.releases == 1


def test_incremental_runtime_preserves_rejection_reason():
    residual = _Residual("failed")
    outcome = run_newton_attempt(
        _policy(),
        evaluate=lambda: NewtonEvaluation(None, residual, 2.0),
        correct=lambda evaluation, iteration: NewtonCorrection(
            False,
            step_length=0.0,
            rejection_reason="linear correction solve did not converge",
        ),
        release=lambda selected: setattr(
            selected, "releases", selected.releases + 1
        ),
    )

    assert not outcome.converged
    assert outcome.iterations == 0
    assert outcome.rejection_reason == "linear correction solve did not converge"
    assert residual.releases == 1


def test_incremental_runtime_accepts_local_constitutive_rejection_before_assembly():
    outcome = run_newton_attempt(
        _policy(),
        evaluate=lambda: NewtonEvaluationRejection(
            {"local_iterations": 12},
            "local constitutive integration did not converge",
        ),
        correct=lambda evaluation, iteration: pytest.fail(
            "A rejected local update must not enter the global correction."
        ),
        release=lambda residual: pytest.fail(
            "No global residual exists for a rejected local update."
        ),
    )

    assert not outcome.converged
    assert outcome.payload == {"local_iterations": 12}
    assert outcome.rejection_reason == (
        "local constitutive integration did not converge"
    )


def test_incremental_runtime_releases_line_search_residual_if_reporter_fails():
    initial = _Residual("initial")
    accepted = _Residual("accepted")

    with pytest.raises(RuntimeError, match="reporter failed"):
        run_newton_attempt(
            _policy(),
            evaluate=lambda: NewtonEvaluation(None, initial, 2.0),
            correct=lambda evaluation, iteration: NewtonCorrection(
                True,
                next_evaluation=NewtonEvaluation(None, accepted, 1.0),
            ),
            release=lambda residual: setattr(
                residual, "releases", residual.releases + 1
            ),
            on_iteration=lambda iteration, norm, alpha: (_ for _ in ()).throw(
                RuntimeError("reporter failed")
            ),
        )

    assert initial.releases == accepted.releases == 1
