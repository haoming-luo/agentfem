"""PDEAgent-Bench adapter pinned to one audited public dataset revision."""

from importlib import import_module as _import_module

from .schema import (
    BENCHMARK_COMMIT,
    BENCHMARK_NAME,
    BENCHMARK_SCHEMA,
    SUPPORTED_FAMILIES,
    BenchmarkContractError,
    validate_case_spec,
)
from .report import (
    BenchmarkReport,
    classify_official_result,
    combine_official_summaries,
    read_official_summary,
)


_ADAPTER_EXPORTS = {
    "BenchmarkPolicy",
    "BenchmarkSolveResult",
    "solve",
    "solve_case",
}


def __getattr__(name: str):
    if name in _ADAPTER_EXPORTS:
        from ...backends.runtime import require_capabilities

        require_capabilities(
            "petsc_linear_solve",
            operation=f"PDEAgent-Bench adapter export {name}",
        )
        value = getattr(_import_module(f"{__name__}.adapter"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "BENCHMARK_COMMIT",
    "BENCHMARK_NAME",
    "BENCHMARK_SCHEMA",
    "SUPPORTED_FAMILIES",
    "BenchmarkContractError",
    "BenchmarkPolicy",
    "BenchmarkReport",
    "BenchmarkSolveResult",
    "solve",
    "solve_case",
    "classify_official_result",
    "combine_official_summaries",
    "read_official_summary",
    "validate_case_spec",
]
