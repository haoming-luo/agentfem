"""PDEAgent-Bench adapter pinned to one audited public dataset revision."""

from .adapter import BenchmarkPolicy, BenchmarkSolveResult, solve, solve_case
from .schema import (
    BENCHMARK_COMMIT,
    BENCHMARK_NAME,
    BENCHMARK_SCHEMA,
    BenchmarkContractError,
    validate_case_spec,
)
from .report import BenchmarkReport, classify_official_result, read_official_summary

__all__ = [
    "BENCHMARK_COMMIT",
    "BENCHMARK_NAME",
    "BENCHMARK_SCHEMA",
    "BenchmarkContractError",
    "BenchmarkPolicy",
    "BenchmarkReport",
    "BenchmarkSolveResult",
    "solve",
    "solve_case",
    "classify_official_result",
    "read_official_summary",
    "validate_case_spec",
]
