# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable performance evidence attached to scientific results.

Timing is measured rank-locally and reduced before publication.  The maximum
rank time is the parallel critical path; min/mean values expose imbalance
without making a collective result manifest rank-dependent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class PerformanceEvidence:
    """Comparable execution-cost evidence for one result lifecycle."""

    stages: Mapping[str, object]
    workload: Mapping[str, object] = field(default_factory=dict)
    solver: Mapping[str, object] = field(default_factory=dict)
    parallel: Mapping[str, object] = field(default_factory=dict)
    scope: str = "solve_result_call"

    def __post_init__(self) -> None:
        selected_scope = str(self.scope).strip()
        if not selected_scope:
            raise ValueError("Performance evidence scope cannot be empty.")
        object.__setattr__(self, "scope", selected_scope)
        object.__setattr__(self, "stages", _validated_stages(self.stages))
        object.__setattr__(self, "workload", dict(self.workload))
        object.__setattr__(self, "solver", dict(self.solver))
        object.__setattr__(self, "parallel", dict(self.parallel))

    @property
    def wall_seconds(self) -> float | None:
        for name in ("total", "run_wall", "solve"):
            stage = self.stages.get(name)
            if isinstance(stage, Mapping):
                return float(stage["seconds"])
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.performance-evidence",
            "schema_version": "0.1.0",
            "scope": self.scope,
            "timing_basis": "wall_clock_perf_counter",
            "wall_seconds": self.wall_seconds,
            "stages": {name: dict(record) for name, record in self.stages.items()},
            "workload": dict(self.workload),
            "solver": dict(self.solver),
            "parallel": dict(self.parallel),
            "runtime_reference": "simulation-result.runtime",
            "measurement_boundary": _measurement_boundary(self.scope),
            "interpretation": (
                "stages may be nested and must not be summed; seconds is the "
                "maximum-rank critical-path time"
            ),
        }


def performance_evidence(
    *,
    stages: Mapping[str, object],
    solution=None,
    source=None,
    scope: str = "solve_result_call",
) -> PerformanceEvidence:
    """Build collective, rank-consistent evidence from local stage timings."""

    comm = _solution_comm(solution)
    normalized = _local_stage_records(stages)
    collective = _collective_stages(comm, normalized)
    return PerformanceEvidence(
        stages=collective,
        workload=_workload(solution),
        solver=_solver_evidence(source),
        parallel={
            "rank_count": int(getattr(comm, "size", 1)),
            "timing_aggregation": "min_mean_max_across_ranks",
            "critical_path": "maximum_rank_wall_time",
        },
        scope=scope,
    )


def attach_performance(
    result,
    *,
    stages: Mapping[str, object],
    solution=None,
    source=None,
    scope: str = "solve_result_call",
):
    """Attach normalized performance evidence and return ``result``."""

    result.add_performance(
        performance_evidence(
            stages=stages,
            solution=solution,
            source=source,
            scope=scope,
        )
    )
    return result


def _local_stage_records(stages: Mapping[str, object]) -> dict[str, dict[str, object]]:
    selected = stages.get("stages", stages)
    if not isinstance(selected, Mapping):
        raise TypeError("Performance stages must be a mapping.")
    records = {}
    for name, value in selected.items():
        stage_name = str(name).strip()
        if not stage_name:
            raise ValueError("Performance stage names cannot be empty.")
        if isinstance(value, Mapping):
            seconds = value.get("seconds")
            calls = value.get("calls", 1)
        else:
            seconds = value
            calls = 1
        elapsed = float(seconds)
        count = int(calls)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                f"Performance stage {stage_name!r} must be finite and nonnegative."
            )
        if count < 1:
            raise ValueError(
                f"Performance stage {stage_name!r} calls must be positive."
            )
        records[stage_name] = {"seconds": elapsed, "calls": count}
    if not records:
        raise ValueError("Performance evidence requires at least one stage.")
    return records


def _collective_stages(comm, local_stages) -> dict[str, dict[str, object]]:
    if int(getattr(comm, "size", 1)) == 1:
        gathered = (local_stages,)
    else:
        # Gather the complete local mapping once.  Calling one collective per
        # local stage can deadlock when a provider stage exists on only some
        # ranks.
        gathered = tuple(comm.allgather(local_stages))
    names = sorted({name for mapping in gathered for name in mapping})
    return {
        name: _collective_stage(name, gathered)
        for name in names
    }


def _collective_stage(name, gathered) -> dict[str, object]:
    records = tuple(mapping.get(name) for mapping in gathered)
    times = tuple(
        0.0 if record is None else float(record["seconds"])
        for record in records
    )
    counts = tuple(
        0 if record is None else int(record["calls"])
        for record in records
    )
    minimum = min(times)
    maximum = max(times)
    mean = sum(times) / len(times)
    calls_min = min(counts)
    calls_max = max(counts)
    participating_ranks = sum(record is not None for record in records)
    return {
        "seconds": maximum,
        "seconds_min": minimum,
        "seconds_mean": mean,
        "seconds_max": maximum,
        "calls": calls_min if calls_min == calls_max else None,
        "calls_min": calls_min,
        "calls_max": calls_max,
        "participating_ranks": participating_ranks,
        "seconds_per_call": (
            None if calls_max < 1 else maximum / calls_max
        ),
    }


def _workload(solution) -> dict[str, object]:
    if solution is None or not hasattr(solution, "function_space"):
        return {}
    space = solution.function_space
    domain = space.mesh
    dofmap = space.dofmap
    index_map = dofmap.index_map
    block_size = int(getattr(dofmap, "index_map_bs", 1))
    topology = domain.topology
    topological_dimension = int(topology.dim)
    cells = topology.index_map(topological_dimension)
    return {
        "global_dofs": int(index_map.size_global) * block_size,
        "global_cells": int(cells.size_global),
        "topological_dimension": topological_dimension,
        "geometric_dimension": int(domain.geometry.dim),
        "element": str(space.ufl_element()),
    }


def _solver_evidence(source) -> dict[str, object]:
    if source is None:
        return {}
    info = getattr(source, "last_solve_info", None)
    if info is None:
        problem = getattr(source, "problem", None)
        info = getattr(problem, "last_solve_info", None)
    options = getattr(source, "solver_options", None)
    if options is None:
        problem = getattr(source, "problem", None)
        options = getattr(problem, "solver_options", None)
    return {
        "convergence": _describe(info),
        "options": _describe(options),
    }


def _describe(value):
    if value is None:
        return None
    for name in ("as_dict", "summary"):
        method = getattr(value, name, None)
        if callable(method):
            return method()
    return {"kind": type(value).__name__}


def _solution_comm(solution):
    if solution is not None and hasattr(solution, "function_space"):
        return solution.function_space.mesh.comm
    from mpi4py import MPI

    return MPI.COMM_SELF


def _validated_stages(stages: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if not isinstance(stages, Mapping) or not stages:
        raise ValueError("PerformanceEvidence requires at least one stage.")
    validated = {}
    for name, record in stages.items():
        if not isinstance(record, Mapping):
            raise TypeError("Collective performance stage records must be mappings.")
        selected = dict(record)
        seconds = float(selected.get("seconds", -1.0))
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"Performance stage {name!r} has invalid seconds.")
        validated[str(name)] = selected
    return validated


def _measurement_boundary(scope: str) -> dict[str, str]:
    if scope == "transient_run_and_result_call":
        start = "entry to the first recorded transient run call"
        stop = "completed result assembly before evidence reduction"
        preexisting = "work before the first recorded run call is excluded"
    else:
        start = "entry to the measured public execution call"
        stop = "completed result assembly before evidence reduction"
        preexisting = "excluded"
    return {
        "start": start,
        "stop": stop,
        "preexisting_work": preexisting,
        "jit_policy": (
            "included only when compilation is triggered inside a measured stage"
        ),
    }


__all__ = (
    "PerformanceEvidence",
    "attach_performance",
    "performance_evidence",
)
