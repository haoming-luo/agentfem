from __future__ import annotations

import pytest

from agentfem import steps
from agentfem.diagnostics import StandardRunReporter
from agentfem.solvers import LinearSolverOptions, SolveEvent


def test_solver_options_are_inspectable():
    options = LinearSolverOptions(
        ksp_type="cg",
        pc_type="hypre",
        rtol=1.0e-8,
        max_it=500,
    )

    assert options.summary() == {
        "kind": "linear_solver_options",
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1.0e-8,
        "atol": None,
        "max_it": 500,
        "factor_solver_type": None,
    }


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rtol": 0.0},
        {"atol": -1.0},
        {"max_it": 0},
        {"pc_type": "gamg", "factor_solver_type": "mumps"},
    ),
)
def test_solver_options_reject_invalid_tolerances(kwargs):
    with pytest.raises(ValueError):
        LinearSolverOptions(**kwargs)


def test_automatic_incrementation_uses_a_limit_not_a_requested_count():
    control = steps.automatic(
        initial=0.1,
        minimum=1.0e-4,
        maximum=0.5,
        max_increments=10,
    )

    assert control.max_increments == 10
    assert control.after_convergence(0.1, iterations=3) == pytest.approx(0.15)
    assert control.after_convergence(0.4, iterations=3) == pytest.approx(0.5)
    assert control.after_failure(0.2) == pytest.approx(0.05)
    assert control.summary()["kind"] == "automatic_incrementation"


def test_fixed_increments_are_an_explicit_compatibility_mode():
    control = steps.fixed(4)

    assert control.load_factors == (0.25, 0.5, 0.75, 1.0)
    assert control.increments == 4


def test_standard_reporter_flushes_concise_status_and_iteration_text(
    tmp_path,
    capsys,
):
    comm = type("Comm", (), {"rank": 0})()
    status = tmp_path / "job.sta"
    reporter = StandardRunReporter(comm, status_file=status)
    base = {
        "step_name": "nonlinear_static",
        "step_number": 1,
        "increment": 1,
        "attempt": 1,
        "start_factor": 0.0,
        "target_factor": 0.25,
    }

    reporter.emit(
        SolveEvent(
            "step_started",
            "nonlinear_static",
            incrementation="automatic_incrementation",
        )
    )
    reporter.emit(SolveEvent("increment_started", **base))
    reporter.emit(
        SolveEvent(
            "iteration",
            **base,
            iteration=1,
            residual_norm=1.0e-5,
            step_length=1.0,
        )
    )
    reporter.emit(
        SolveEvent(
            "increment_converged",
            **base,
            iteration=1,
            residual_norm=1.0e-10,
        )
    )

    console = capsys.readouterr().out
    assert "[STEP 1]" in console
    assert "[INC 1 | ATT 1]" in console
    assert "ITER 01" in console
    assert "CONVERGED" in status.read_text(encoding="utf-8")
