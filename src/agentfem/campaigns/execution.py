"""Safe execution providers for independent campaign cases."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import pickle
from typing import Callable, Iterable, TypeVar


_Case = TypeVar("_Case")
_Record = TypeVar("_Record")


@dataclass(frozen=True)
class ExecutionBatch:
    """Records and resource evidence returned by one execution provider."""

    records: tuple[object, ...]
    evidence: dict[str, object]


def run_local_processes(
    case_runner: Callable[[_Case, str], _Record],
    cases: Iterable[_Case],
    *,
    workers: int,
    fail_fast: bool,
) -> ExecutionBatch:
    """Run independent cases in spawned processes.

    ``spawn`` is deliberately fixed. Forking a process after MPI, PETSc, or a
    threaded numerical runtime has initialized is not a portable execution
    contract. Work is submitted in bounded batches so ``fail_fast`` never
    launches more than one worker batch beyond the first observed failure.
    """

    selected_cases = tuple(cases)
    if not selected_cases:
        return ExecutionBatch(
            records=(),
            evidence={
                "provider": "local_processes",
                "start_method": "spawn",
                "requested_workers": workers,
                "effective_workers": 0,
                "case_count": 0,
                "fail_fast_overshoot_limit": 0,
            },
        )
    selected_workers = min(int(workers), len(selected_cases))
    try:
        pickle.dumps((case_runner, selected_cases[0]))
    except Exception as exc:
        raise TypeError(
            "Local campaign processes require a serializable Campaign, build "
            "callable, evaluate callable, and case. Define reusable callables "
            "at module scope; use serial execution in notebooks or for closures."
        ) from exc

    context = multiprocessing.get_context("spawn")
    records: list[object] = []
    stopped_early = False
    with ProcessPoolExecutor(
        max_workers=selected_workers,
        mp_context=context,
    ) as executor:
        for start in range(0, len(selected_cases), selected_workers):
            batch = selected_cases[start : start + selected_workers]
            futures = [
                executor.submit(_execute_case, case_runner, case)
                for case in batch
            ]
            batch_records = tuple(future.result() for future in futures)
            records.extend(batch_records)
            if fail_fast and any(
                not bool(getattr(record, "successful", False))
                for record in batch_records
            ):
                stopped_early = start + selected_workers < len(selected_cases)
                break
    return ExecutionBatch(
        records=tuple(records),
        evidence={
            "provider": "local_processes",
            "start_method": "spawn",
            "requested_workers": workers,
            "effective_workers": selected_workers,
            "case_count": len(records),
            "stopped_early": stopped_early,
            "fail_fast_overshoot_limit": max(selected_workers - 1, 0),
        },
    )


def _execute_case(case_runner, case):
    return case_runner(case, "local_process")


__all__ = ["ExecutionBatch", "run_local_processes"]
