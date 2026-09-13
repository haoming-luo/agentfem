"""Private result assembly for directly solved discrete problems."""

from __future__ import annotations

from .core import from_solution


def from_linear_variational_problem(problem, solution, *, name: str):
    """Build the result of a direct linear variational solve."""

    return from_solution(
        solution,
        name=name,
        metadata={"solve": problem.last_solve_info.as_dict()},
    )


def from_linear_system_problem(problem, solution, *, name: str):
    """Build the result of a direct engineering-operator solve."""

    return from_solution(
        solution,
        name=name,
        metadata={"problem": problem.summary()},
    )


def from_nonlinear_variational_problem(problem, solution):
    """Build the result of a direct nonlinear variational solve."""

    return from_solution(
        solution,
        name=problem.name,
        metadata={
            "problem": problem.summary(),
            "solve": problem.last_solve_info.as_dict(),
        },
    )


__all__ = ()
