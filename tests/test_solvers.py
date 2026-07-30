from __future__ import annotations

import pytest

from agentfem.solvers import LinearSolverOptions


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
    }


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rtol": 0.0},
        {"atol": -1.0},
        {"max_it": 0},
    ),
)
def test_solver_options_reject_invalid_tolerances(kwargs):
    with pytest.raises(ValueError):
        LinearSolverOptions(**kwargs)
