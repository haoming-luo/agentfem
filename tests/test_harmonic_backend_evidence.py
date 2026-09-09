from __future__ import annotations

import pytest

from agentfem.backends._harmonic import (
    HarmonicLinearSolveEvidence,
    PreparedHarmonicLinearProblem,
    _finite_harmonic_scalar,
)
from agentfem.solvers import LinearSolveInfo


def _evidence(**overrides) -> HarmonicLinearSolveEvidence:
    values = {
        "solve": LinearSolveInfo(
            converged_reason=2,
            iterations=1,
            residual_norm=1.0e-12,
        ),
        "residual_norm": 1.0e-12,
        "relative_residual_norm": 1.0e-13,
        "relative_real_block_residual_norm": 1.0e-13,
        "relative_imaginary_block_residual_norm": 1.0e-13,
        "input_energy_per_cycle": 2.0,
    }
    values.update(overrides)
    return HarmonicLinearSolveEvidence(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {
                "solve": LinearSolveInfo(
                    converged_reason=-9,
                    iterations=1,
                    residual_norm=float("nan"),
                )
            },
            "KSP residual norm",
        ),
        ({"residual_norm": float("inf")}, "algebraic residual norm"),
        (
            {"relative_residual_norm": float("nan")},
            "relative algebraic residual norm",
        ),
        (
            {"relative_real_block_residual_norm": float("inf")},
            "relative real-block residual norm",
        ),
        (
            {"relative_imaginary_block_residual_norm": float("nan")},
            "relative imaginary-block residual norm",
        ),
        ({"input_energy_per_cycle": float("inf")}, "input energy per cycle"),
    ),
)
def test_harmonic_solve_evidence_rejects_nonfinite_values(overrides, message):
    with pytest.raises(FloatingPointError, match=message):
        _evidence(**overrides)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_harmonic_solution_scalar_guard_is_fail_closed(value):
    with pytest.raises(FloatingPointError, match="solution-vector norm"):
        _finite_harmonic_scalar(value, quantity="solution-vector norm")


def test_prepared_harmonic_problem_rejects_nonfinite_solution_before_evidence():
    class _NonfiniteVector:
        @staticmethod
        def norm():
            return float("nan")

    class _Solver:
        @staticmethod
        def getConvergedReason():
            return 2

        @staticmethod
        def getIterationNumber():
            return 1

        @staticmethod
        def getResidualNorm():
            return 0.0

    class _Problem:
        x = _NonfiniteVector()
        solver = _Solver()

        @staticmethod
        def solve():
            return None

    prepared = object.__new__(PreparedHarmonicLinearProblem)
    prepared._problem = _Problem()
    prepared._solve_count = 0

    with pytest.raises(FloatingPointError, match="solution-vector norm"):
        prepared.solve()
    assert prepared.solve_count == 1


def test_harmonic_solve_evidence_allows_unavailable_component_residuals():
    evidence = _evidence(
        relative_real_block_residual_norm=None,
        relative_imaginary_block_residual_norm=None,
    )

    assert evidence.equilibrium() == {
        "residual_norm": pytest.approx(1.0e-12),
        "relative_residual_norm": pytest.approx(1.0e-13),
    }
