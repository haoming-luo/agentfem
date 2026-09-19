# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed diagnostics for finite-element fields."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from time import monotonic
from typing import Callable

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from . import fields as field_api
from .kernel import dofs


@dataclass
class PerformanceLedger:
    """Low-overhead, rank-local timing evidence for one solver lifecycle."""

    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float) -> None:
        selected = str(name).strip()
        value = float(elapsed)
        if not selected:
            raise ValueError("Performance timing name cannot be empty.")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Performance timing must be finite and nonnegative.")
        self.seconds[selected] = self.seconds.get(selected, 0.0) + value
        self.counts[selected] = self.counts.get(selected, 0) + 1

    def reset(self) -> None:
        self.seconds.clear()
        self.counts.clear()

    def summary(self) -> dict[str, object]:
        wall = float(self.seconds.get("run_wall", 0.0))
        stages = {}
        for name in sorted(self.seconds):
            elapsed = float(self.seconds[name])
            count = int(self.counts[name])
            stages[name] = {
                "seconds": elapsed,
                "calls": count,
                "seconds_per_call": elapsed / count,
                "fraction_of_run_wall": (
                    None if wall <= 0.0 else elapsed / wall
                ),
            }
        return {
            "kind": "rank_local_solver_performance",
            "run_wall_seconds": wall,
            "stages": stages,
            "interpretation": "nested stages may overlap and must not be summed",
        }


@dataclass(frozen=True)
class DiscreteInfSupEvidence:
    """Basis-normalized singular evidence for one mixed discretization."""

    beta: float
    singular_values: tuple[float, ...]
    rank: int
    primal_size: int
    multiplier_size: int
    full_row_rank: bool
    rank_tolerance: float
    condition_number: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "discrete_inf_sup_evidence",
            "beta": self.beta,
            "singular_values": list(self.singular_values),
            "rank": self.rank,
            "primal_size": self.primal_size,
            "multiplier_size": self.multiplier_size,
            "full_row_rank": self.full_row_rank,
            "rank_tolerance": self.rank_tolerance,
            "condition_number": self.condition_number,
            "normalization": "primal_and_multiplier_norm_matrices",
            "interpretation": (
                "one_mesh_rank_and_beta; mesh-independent stability requires "
                "a refinement sequence"
            ),
        }


@dataclass(frozen=True)
class DiscreteInfSupSample:
    """One normalized mixed-stability measurement on a declared mesh."""

    characteristic_size: float
    evidence: DiscreteInfSupEvidence
    label: str = ""

    def __post_init__(self) -> None:
        size = float(self.characteristic_size)
        if not np.isfinite(size) or size <= 0.0:
            raise ValueError(
                "Inf-sup sample characteristic_size must be finite and positive."
            )
        if not isinstance(self.evidence, DiscreteInfSupEvidence):
            raise TypeError("Inf-sup sample evidence must be DiscreteInfSupEvidence.")
        object.__setattr__(self, "characteristic_size", size)
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class DiscreteInfSupStudy:
    """Coarse-to-fine evidence for one family of mixed discretizations.

    Three levels are required deliberately: two full-rank matrices only show
    two isolated algebraic facts.  They do not reveal whether ``beta_h`` is
    tending to zero under refinement.  Acceptance thresholds remain explicit
    properties of the benchmark or element-family contract, not global
    AgentFEM defaults.
    """

    name: str
    samples: tuple[DiscreteInfSupSample, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        samples = tuple(self.samples)
        if not name:
            raise ValueError("DiscreteInfSupStudy.name must not be empty.")
        if len(samples) < 3:
            raise ValueError("DiscreteInfSupStudy requires at least three samples.")
        if not all(isinstance(item, DiscreteInfSupSample) for item in samples):
            raise TypeError("Inf-sup study samples must be DiscreteInfSupSample objects.")
        sizes = tuple(item.characteristic_size for item in samples)
        if any(fine >= coarse for coarse, fine in zip(sizes, sizes[1:])):
            raise ValueError(
                "Inf-sup samples must be ordered coarse-to-fine with strictly "
                "decreasing characteristic_size."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "samples", samples)

    @property
    def minimum_beta(self) -> float:
        return min(item.evidence.beta for item in self.samples)

    @property
    def finest_beta(self) -> float:
        return self.samples[-1].evidence.beta

    @property
    def beta_range_ratio(self) -> float:
        largest = max(item.evidence.beta for item in self.samples)
        return 0.0 if largest <= 0.0 else self.minimum_beta / largest

    @property
    def endpoint_decay_order(self) -> float | None:
        """Return ``p`` from endpoint scaling ``beta_h ~ h**p`` when defined."""

        coarse = self.samples[0]
        fine = self.samples[-1]
        if coarse.evidence.beta <= 0.0 or fine.evidence.beta <= 0.0:
            return None
        return float(
            np.log(fine.evidence.beta / coarse.evidence.beta)
            / np.log(fine.characteristic_size / coarse.characteristic_size)
        )

    def verify(
        self,
        *,
        minimum_beta: float,
        maximum_decay_order: float | None = None,
        reference: str = "declared mixed-discretization stability contract",
    ):
        """Return a scientific claim using caller-declared family thresholds."""

        lower_bound = float(minimum_beta)
        if not np.isfinite(lower_bound) or lower_bound < 0.0:
            raise ValueError("minimum_beta must be finite and nonnegative.")
        decay_limit = None
        if maximum_decay_order is not None:
            decay_limit = float(maximum_decay_order)
            if not np.isfinite(decay_limit) or decay_limit < 0.0:
                raise ValueError(
                    "maximum_decay_order must be finite and nonnegative."
                )
        order = self.endpoint_decay_order
        rank_passes = all(item.evidence.full_row_rank for item in self.samples)
        beta_passes = self.minimum_beta >= lower_bound
        order_available = decay_limit is None or order is not None
        order_passes = decay_limit is None or (order is not None and order <= decay_limit)
        if not order_available:
            status = "inconclusive"
        else:
            status = "passed" if rank_passes and beta_passes and order_passes else "failed"
        criterion = f"full row rank and minimum beta >= {lower_bound:g}"
        if decay_limit is not None:
            criterion += f" and endpoint beta decay order <= {decay_limit:g}"
        from .verification import VerificationClaim

        return VerificationClaim(
            name=self.name,
            observable="normalized discrete inf-sup sequence",
            reference=reference,
            status=status,
            criterion=criterion,
            actual={
                "minimum_beta": self.minimum_beta,
                "endpoint_decay_order": order,
                "all_full_row_rank": rank_passes,
            },
            expected={
                "minimum_beta": lower_bound,
                "maximum_decay_order": decay_limit,
                "all_full_row_rank": True,
            },
            evidence=self.as_dict(),
            message=(
                "The refinement sequence satisfies the declared mixed-stability contract."
                if status == "passed"
                else "The refinement sequence does not establish the declared mixed-stability contract."
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "discrete_inf_sup_study",
            "name": self.name,
            "sample_count": len(self.samples),
            "characteristic_sizes": [
                item.characteristic_size for item in self.samples
            ],
            "betas": [item.evidence.beta for item in self.samples],
            "minimum_beta": self.minimum_beta,
            "finest_beta": self.finest_beta,
            "beta_range_ratio": self.beta_range_ratio,
            "endpoint_decay_order": self.endpoint_decay_order,
            "all_full_row_rank": all(
                item.evidence.full_row_rank for item in self.samples
            ),
            "samples": [
                {
                    "label": item.label,
                    "evidence": item.evidence.as_dict(),
                }
                for item in self.samples
            ],
            "interpretation": (
                "finite refinement evidence; acceptance requires a declared "
                "element-family or benchmark contract"
            ),
        }


def _symmetric_positive_definite_matrix(value, *, name: str, size: int) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (size, size) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite {size}x{size} matrix.")
    if not np.allclose(selected, selected.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError(f"{name} must be symmetric.")
    try:
        np.linalg.cholesky(selected)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc
    return selected


def discrete_inf_sup(
    constraint_matrix,
    *,
    primal_norm,
    multiplier_norm,
    rank_tolerance: float | None = None,
) -> DiscreteInfSupEvidence:
    """Compute one norm-aware discrete inf-sup spectrum.

    For a coupling matrix ``B`` and symmetric positive-definite norm matrices
    ``M_u`` and ``M_lambda``, the reported singular values belong to
    ``L_lambda^-1 B L_u^-T``.  The value is invariant to nonsingular changes of
    basis when the norm matrices are transformed consistently.  It is evidence
    for one discretization only; a non-decaying refinement sequence is needed
    before claiming inf-sup stability.
    """

    coupling = np.asarray(constraint_matrix, dtype=float)
    if coupling.ndim != 2 or min(coupling.shape) < 1 or not np.all(np.isfinite(coupling)):
        raise ValueError("constraint_matrix must be one non-empty finite matrix.")
    multiplier_size, primal_size = coupling.shape
    primal = _symmetric_positive_definite_matrix(
        primal_norm,
        name="primal_norm",
        size=primal_size,
    )
    multiplier = _symmetric_positive_definite_matrix(
        multiplier_norm,
        name="multiplier_norm",
        size=multiplier_size,
    )
    primal_factor = np.linalg.cholesky(primal)
    multiplier_factor = np.linalg.cholesky(multiplier)
    right_normalized = np.linalg.solve(primal_factor, coupling.T).T
    normalized = np.linalg.solve(multiplier_factor, right_normalized)
    singular = np.linalg.svd(normalized, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    if rank_tolerance is None:
        tolerance = max(normalized.shape) * np.finfo(float).eps * largest
    else:
        tolerance = float(rank_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("rank_tolerance must be finite and nonnegative.")
    rank = int(np.count_nonzero(singular > tolerance))
    full_row_rank = rank == multiplier_size
    beta = float(singular[-1]) if full_row_rank else 0.0
    condition = float(largest / beta) if beta > 0.0 else float("inf")
    return DiscreteInfSupEvidence(
        beta=beta,
        singular_values=tuple(float(value) for value in singular),
        rank=rank,
        primal_size=primal_size,
        multiplier_size=multiplier_size,
        full_row_rank=full_row_rank,
        rank_tolerance=tolerance,
        condition_number=condition,
    )


def comm_of(obj=None, default=MPI.COMM_WORLD):
    """Return the MPI communicator associated with an object when possible."""

    if obj is None:
        return default
    if hasattr(obj, "rank") and hasattr(obj, "size"):
        return obj
    if hasattr(obj, "comm"):
        return obj.comm
    if hasattr(obj, "mesh"):
        return obj.mesh.comm
    if hasattr(obj, "domain") and hasattr(obj.domain, "comm"):
        return obj.domain.comm
    if hasattr(obj, "function_space"):
        return obj.function_space.mesh.comm
    value = getattr(obj, "value", None)
    if value is not None and hasattr(value, "function_space"):
        return value.function_space.mesh.comm
    function = getattr(obj, "function", None)
    if function is not None and hasattr(function, "function_space"):
        return function.function_space.mesh.comm
    return default


def is_root(obj=None, *, root: int = 0) -> bool:
    """Return whether the current MPI rank is the selected reporting rank."""

    return comm_of(obj).rank == root


def print_on_root(obj, *args, root: int = 0, flush: bool = True, **kwargs) -> None:
    """Print a message only on the selected MPI root rank.

    ``flush`` defaults to ``True`` because long-running finite-element solves
    should show progress messages immediately.
    """

    if is_root(obj, root=root):
        print(*args, flush=flush, **kwargs)


@dataclass
class StandardRunReporter:
    """Immediate rank-zero progress for long-running analysis steps.

    The console is deliberately human-facing.  Transient and nonlinear output
    is throttled by the solver cadence and a wall-clock heartbeat, so a slow
    increment remains observable without retaining or printing one record per
    internal operation.  When ``status_file`` is given, every visible event is
    flushed immediately for terminals, schedulers, and agents.
    """

    comm: object
    status_file: str | Path | None = None
    show_iterations: bool | None = None
    verbosity: int | None = None
    heartbeat_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.heartbeat_seconds = float(self.heartbeat_seconds)
        if self.heartbeat_seconds < 0.0:
            raise ValueError("heartbeat_seconds must be nonnegative.")
        if self.verbosity is None:
            try:
                self.verbosity = int(os.environ.get("AGENTFEM_VERBOSITY", "0"))
            except ValueError:
                self.verbosity = 0
        self.verbosity = max(0, int(self.verbosity))
        if self.show_iterations is None:
            self.show_iterations = self.verbosity >= 2
        self.status_file = (
            None if self.status_file is None else Path(self.status_file)
        )
        self._started = monotonic()
        self._last_heartbeat = self._started
        self._status_initialized = False
        self._sweep_start_increment = 0

    def emit(self, event) -> None:
        """Report one solver event; non-root ranks remain silent."""

        if self.comm.rank != 0:
            return
        elapsed = monotonic() - self._started
        kind = event.kind
        visible = bool(getattr(event, "display", True))
        heartbeat = False
        if kind in {"time_increment", "sweep_point"} and not visible:
            now = monotonic()
            visible = now - self._last_heartbeat >= float(self.heartbeat_seconds)
            heartbeat = visible
        elif kind in {"iteration", "increment_converged"} and self.verbosity == 0:
            now = monotonic()
            heartbeat = (
                now - self._last_heartbeat >= float(self.heartbeat_seconds)
            )
        if not visible:
            return
        if kind in {"time_increment", "sweep_point"} or heartbeat:
            self._last_heartbeat = monotonic()
        if kind == "step_started":
            self._print(
                f"[STEP {event.step_number}] {event.step_name} "
                f"| {event.incrementation}"
            )
            self._write_status(
                "STEP INC ATT LOAD_FACTOR INCREMENT ITERATIONS RESIDUAL STATUS ELAPSED_S"
            )
        elif kind == "increment_started":
            if self.verbosity >= 1:
                self._print(
                    f"  [INC {event.increment} | ATT {event.attempt}] "
                    f"{event.start_factor:.6g} -> {event.target_factor:.6g} "
                    f"(d={event.target_factor - event.start_factor:.3g})"
                )
        elif kind == "iteration":
            if self.show_iterations:
                alpha = (
                    ""
                    if event.step_length is None
                    else f" | alpha={event.step_length:.3g}"
                )
                self._print(
                    f"    ITER {event.iteration:02d} "
                    f"| residual={event.residual_norm:.6e}{alpha}"
                )
            elif heartbeat:
                self._print(
                    f"  [INC {event.increment} | ITER {event.iteration}] RUNNING "
                    f"| progress={100.0 * event.target_factor:.1f}% "
                    f"| load={event.target_factor:.6g} "
                    f"| residual={_number(event.residual_norm)} "
                    f"| elapsed={elapsed:.1f}s"
                )
                self._write_status(
                    f"{event.step_number} {event.increment} {event.attempt} "
                    f"{event.target_factor:.16g} "
                    f"{event.target_factor - event.start_factor:.16g} "
                    f"{event.iteration} {_number(event.residual_norm)} "
                    f"RUNNING {elapsed:.6f}"
                )
        elif kind == "increment_converged":
            if self.verbosity >= 1 or heartbeat:
                self._print(
                    f"  [INC {event.increment}] CONVERGED "
                    f"| iterations={event.iteration} "
                    f"| residual={event.residual_norm:.6e} "
                    f"| elapsed={elapsed:.1f}s"
                )
            self._write_status(
                f"{event.step_number} {event.increment} {event.attempt} "
                f"{event.target_factor:.16g} "
                f"{event.target_factor - event.start_factor:.16g} "
                f"{event.iteration} {event.residual_norm:.16e} "
                f"CONVERGED {elapsed:.6f}"
            )
        elif kind == "increment_cutback":
            reason = (
                ""
                if not getattr(event, "message", None)
                else f" | reason={event.message}"
            )
            self._print(
                f"  [INC {event.increment} | ATT {event.attempt}] CUTBACK "
                f"| residual={_number(event.residual_norm)} "
                f"| next d={event.next_increment:.3g}{reason}"
            )
            self._write_status(
                f"{event.step_number} {event.increment} {event.attempt} "
                f"{event.target_factor:.16g} "
                f"{event.target_factor - event.start_factor:.16g} "
                f"{event.iteration} {_number(event.residual_norm)} "
                f"CUTBACK {elapsed:.6f}"
            )
        elif kind == "step_completed":
            self._print(
                f"[STEP {event.step_number}] COMPLETED "
                f"| increments={event.increment} "
                f"| attempts={event.attempt} "
                f"| elapsed={elapsed:.1f}s"
            )
            self._write_status(
                f"{event.step_number} {event.increment} {event.attempt} "
                f"1 0 0 0 COMPLETED {elapsed:.6f}"
            )
        elif kind == "step_paused":
            self._print(
                f"[STEP {event.step_number}] PAUSED "
                f"| increments={event.increment} "
                f"| load_factor={event.target_factor:.6g} "
                f"| elapsed={elapsed:.1f}s"
            )
            self._write_status(
                f"{event.step_number} {event.increment} {event.attempt} "
                f"{event.target_factor:.16g} 0 0 0 PAUSED {elapsed:.6f}"
            )
        elif kind == "step_failed":
            self._print(
                f"[STEP {event.step_number}] FAILED | {event.message}"
            )
            self._write_status(
                f"{event.step_number} {event.increment} {event.attempt} "
                f"{event.target_factor:.16g} 0 {event.iteration} "
                f"{_number(event.residual_norm)} FAILED {elapsed:.6f}"
            )
        elif kind in {"transient_started", "transient_resumed"}:
            detail = "" if not event.message else f" | {event.message}"
            state = "RESUMED" if kind == "transient_resumed" else "STARTED"
            self._print(
                f"[STEP {event.step_number}] {state} {event.step_name} "
                f"| {event.incrementation} | increments={event.total_increments}{detail}"
            )
            self._write_status(
                "STEP INC TIME STATUS ELAPSED_S"
            )
        elif kind == "time_increment":
            rate = event.increment / elapsed if elapsed > 0.0 else 0.0
            remaining = max(0, event.total_increments - event.increment)
            eta = remaining / rate if rate > 0.0 else 0.0
            percent = (
                100.0 * event.increment / event.total_increments
                if event.total_increments
                else 0.0
            )
            detail = "" if not event.message else f" | {event.message}"
            self._print(
                f"  [INC {event.increment}/{event.total_increments}] "
                f"t={event.time:.6g} | {percent:.1f}% | elapsed={elapsed:.1f}s "
                f"| rate={rate:.3g}/s | ETA~{eta:.0f}s{detail}"
            )
            self._write_status(
                f"{event.step_number} {event.increment} {event.time:.16g} "
                f"ACCEPTED {elapsed:.6f}"
            )
        elif kind == "transient_completed":
            self._print(
                f"[STEP {event.step_number}] COMPLETED "
                f"| increments={event.increment} | t={event.time:.6g} "
                f"| elapsed={elapsed:.1f}s"
            )
            self._write_status(
                f"{event.step_number} {event.increment} {event.time:.16g} "
                f"COMPLETED {elapsed:.6f}"
            )
        elif kind in {"sweep_started", "sweep_resumed"}:
            self._sweep_start_increment = int(event.increment)
            state = "RESUMED" if kind == "sweep_resumed" else "STARTED"
            self._print(
                f"[SWEEP {event.step_number}] {state} {event.step_name} "
                f"| points={event.increment}/{event.total_increments}"
            )
            self._write_status(
                "SWEEP POINT COORDINATE VALUE UNIT RESIDUAL STATUS ELAPSED_S"
            )
        elif kind == "sweep_point":
            advanced = max(0, event.increment - self._sweep_start_increment)
            rate = advanced / elapsed if elapsed > 0.0 else 0.0
            remaining = max(0, event.total_increments - event.increment)
            eta = remaining / rate if rate > 0.0 else 0.0
            percent = (
                100.0 * event.increment / event.total_increments
                if event.total_increments
                else 0.0
            )
            coordinate = _coordinate(event)
            metrics = getattr(event, "metrics", {})
            maximum = metrics.get("maximum_displacement_vector_amplitude")
            energy = metrics.get("relative_cycle_energy_balance_error")
            detail = (
                f" | max|U|={_number(maximum)}"
                f" | residual={_number(event.residual_norm)}"
                f" | energy_err={_number(energy)}"
            )
            self._print(
                f"  [POINT {event.increment}/{event.total_increments}] "
                f"{coordinate} | {percent:.1f}% | elapsed={elapsed:.1f}s "
                f"| ETA~{eta:.0f}s{detail}"
            )
            self._write_status(
                f"{event.step_number} {event.increment} "
                f"{event.coordinate_name or 'coordinate'} "
                f"{_number(event.coordinate_value)} "
                f"{event.coordinate_unit or '-'} {_number(event.residual_norm)} "
                f"ACCEPTED {elapsed:.6f}"
            )
        elif kind in {"sweep_paused", "sweep_completed"}:
            state = "PAUSED" if kind == "sweep_paused" else "COMPLETED"
            self._print(
                f"[SWEEP {event.step_number}] {state} "
                f"| points={event.increment}/{event.total_increments} "
                f"| elapsed={elapsed:.1f}s"
            )
            self._write_status(
                f"{event.step_number} {event.increment} frequency - Hz 0 "
                f"{state} {elapsed:.6f}"
            )
        elif kind == "sweep_failed":
            self._print(
                f"[SWEEP {event.step_number}] FAILED "
                f"| point={event.increment}/{event.total_increments} "
                f"| {_coordinate(event)} | {event.message}"
            )
            self._write_status(
                f"{event.step_number} {event.increment} "
                f"{event.coordinate_name or 'coordinate'} "
                f"{_number(event.coordinate_value)} "
                f"{event.coordinate_unit or '-'} {_number(event.residual_norm)} "
                f"FAILED {elapsed:.6f}"
            )

    def _print(self, message: str) -> None:
        print(message, flush=True)

    def _write_status(self, line: str) -> None:
        if self.status_file is None:
            return
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._status_initialized else "w"
        with self.status_file.open(mode, encoding="utf-8", buffering=1) as stream:
            stream.write(line + "\n")
            stream.flush()
        self._status_initialized = True


@dataclass
class SolveEventRecorder:
    """In-memory structured execution trace shared by every procedure.

    Unlike a progress printer, the recorder can retain hidden events, but its
    capacity is bounded. Repetitive hidden increments yield first to visible
    milestones, cutbacks, failures, and completion evidence.
    """

    events: list[object] = field(default_factory=list)
    max_events: int = 4096
    dropped_events: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.max_events = int(self.max_events)
        if self.max_events < 16:
            raise ValueError("SolveEventRecorder.max_events must be at least 16.")
        if len(self.events) > self.max_events:
            original = len(self.events)
            retained = [self.events[0], *self.events[-(self.max_events - 1) :]]
            self.events[:] = retained
            self.dropped_events = original - len(retained)

    def emit(self, event) -> None:
        if len(self.events) < self.max_events:
            self.events.append(event)
            return
        repetitive = getattr(event, "kind", "") in {
            "time_increment",
            "sweep_point",
        }
        important = bool(getattr(event, "display", True)) or not repetitive
        if important:
            for index, existing in enumerate(self.events):
                if (
                    getattr(existing, "kind", "")
                    in {"time_increment", "sweep_point"}
                    and not bool(getattr(existing, "display", True))
                ):
                    del self.events[index]
                    break
            else:
                del self.events[0]
            self.events.append(event)
        self.dropped_events += 1

    def clear(self) -> None:
        self.events.clear()
        self.dropped_events = 0

    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            event.as_dict() if hasattr(event, "as_dict") else dict(event)
            for event in self.events
        )


@dataclass(frozen=True)
class ReporterGroup:
    """Fan one solver event out to several independent consumers."""

    reporters: tuple[object, ...]

    def emit(self, event) -> None:
        for reporter in self.reporters:
            if hasattr(reporter, "emit"):
                reporter.emit(event)
            else:
                reporter(event)


def compose_reporters(*reporters) -> object | None:
    """Compose progress, persistence, and agent observers without coupling."""

    selected = tuple(reporter for reporter in reporters if reporter is not None)
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return ReporterGroup(selected)


def _number(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6e}"


def _coordinate(event) -> str:
    name = getattr(event, "coordinate_name", None) or "coordinate"
    value = _number(getattr(event, "coordinate_value", None))
    unit = getattr(event, "coordinate_unit", None)
    return f"{name}={value}{'' if not unit else ' ' + str(unit)}"


def kinetic_energy(mass_lumped: np.ndarray, velocity: fem.Function) -> float:
    """Global kinetic energy from a lumped mass vector and velocity field."""

    velocity = field_api.unwrap(velocity)
    local = 0.5 * float(np.sum(mass_lumped * dofs.owned_array(velocity) ** 2))
    return velocity.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


@dataclass(frozen=True)
class MechanicalEnergy:
    """Kinetic, recoverable strain, and total mechanical energy."""

    kinetic: float
    strain: float

    @property
    def total(self) -> float:
        return self.kinetic + self.strain

    def summary(self) -> dict[str, float]:
        return {
            "kinetic": self.kinetic,
            "strain": self.strain,
            "total": self.total,
        }


def mechanical_energy(*, mass, stiffness, displacement, velocity) -> MechanicalEnergy:
    """Evaluate ``1/2 v^T M v`` and ``1/2 u^T K u`` from visible operators."""

    from . import operators

    return MechanicalEnergy(
        kinetic=0.5 * operators.quadratic_form(mass, velocity),
        strain=0.5 * operators.quadratic_form(stiffness, displacement),
    )


@dataclass(frozen=True)
class LinearStaticEnergy:
    """Energy closure for a proportional linear-static load path."""

    strain_energy: float
    external_work: float
    balance_error: float

    def summary(self) -> dict[str, float]:
        return {
            "strain_energy": self.strain_energy,
            "external_work": self.external_work,
            "energy_balance_error": self.balance_error,
        }


def linear_static_energy(*, stiffness, force, displacement) -> LinearStaticEnergy:
    """Evaluate energy for loads ramped proportionally from zero to ``force``.

    The external work is ``1/2 u^T F``. Non-zero prescribed displacements add
    reaction work and therefore require a separate displacement-control path.
    """

    from . import operators

    strain = 0.5 * operators.quadratic_form(stiffness, displacement)
    external = 0.5 * operators.dual_product(force, displacement)
    return LinearStaticEnergy(
        strain_energy=float(strain),
        external_work=float(external),
        balance_error=float(external - strain),
    )


@dataclass
class MechanicalEnergyMonitor:
    """Cache visible M/K operators and sample mechanical energy in time.

    Matrix-valued engineering operators are assembled once on first use.  A
    lumped explicit mass remains a diagonal array.  This keeps energy output a
    diagnostic consumer of the same operators used by the procedure instead of
    rebuilding a separate physical model at every frame.
    """

    mass: object
    stiffness: object | None = None
    _compiled_mass: object | None = field(default=None, init=False, repr=False)
    _compiled_stiffness: object | None = field(default=None, init=False, repr=False)

    def evaluate(self, *, displacement, velocity) -> dict[str, float]:
        """Return the energy components currently supported by the model."""

        from . import operators

        if self._compiled_mass is None:
            self._compiled_mass = _energy_operator(self.mass)
        kinetic = 0.5 * operators.quadratic_form(
            self._compiled_mass,
            velocity,
        )
        values = {"kinetic_energy": float(kinetic)}
        if self.stiffness is not None:
            if self._compiled_stiffness is None:
                self._compiled_stiffness = _energy_operator(self.stiffness)
            strain = 0.5 * operators.quadratic_form(
                self._compiled_stiffness,
                displacement,
            )
            values["strain_energy"] = float(strain)
            values["total_mechanical_energy"] = float(kinetic + strain)
        return values


@dataclass
class ThermalBalanceMonitor:
    """Sample discrete heat content, applied rate, outflow, and closure."""

    capacity: object
    stiffness: object
    source: object | None
    dt: float
    _compiled_capacity: object | None = field(default=None, init=False, repr=False)
    _compiled_stiffness: object | None = field(default=None, init=False, repr=False)
    _unit_field: object | None = field(default=None, init=False, repr=False)
    _previous_content: float | None = field(default=None, init=False, repr=False)

    def evaluate(self, temperature) -> dict[str, float]:
        """Return sensible thermal content relative to the model's zero."""

        from . import operators

        selected = field_api.unwrap(temperature)
        if self._compiled_capacity is None:
            self._compiled_capacity = _energy_operator(self.capacity)
        if self._unit_field is None:
            self._unit_field = fem.Function(
                selected.function_space,
                name="UnitTemperatureWeight",
            )
            self._unit_field.interpolate(
                lambda x: np.ones((1, x.shape[1]), dtype=float)
            )
        content = float(operators.xtmy(
            self._unit_field,
            self._compiled_capacity,
            selected,
        ))
        if self._compiled_stiffness is None:
            self._compiled_stiffness = _energy_operator(self.stiffness)
        outward_rate = float(
            operators.xtmy(self._unit_field, self._compiled_stiffness, selected)
        )
        input_rate = (
            0.0
            if self.source is None
            else float(operators.dual_product(self.source, self._unit_field))
        )
        residual = (
            0.0
            if self._previous_content is None
            else content
            - self._previous_content
            + float(self.dt) * (outward_rate - input_rate)
        )
        self._previous_content = content
        return {
            "thermal_content": content,
            "applied_heat_rate": input_rate,
            "outward_heat_rate": outward_rate,
            "heat_balance_residual": float(residual),
        }

    def restore(self, record) -> None:
        """Restore monitor memory from a checkpointed history frame."""

        if "thermal_content" in record:
            self._previous_content = float(record["thermal_content"])


@dataclass
class StateDependentThermalBalanceMonitor:
    """Heat ledger for nonlinear conductivity and heat-capacity models.

    ``content_form`` is a scalar sensible-enthalpy functional evaluated at the
    live temperature. ``outward_forms`` contains natural-boundary heat rates;
    strong prescribed-temperature reactions are intentionally reported through
    the closure residual until a reaction consumer accounts for them.
    """

    content_form: object
    source: object | None
    dt: float
    outward_forms: tuple[object, ...] = ()
    _unit_field: object | None = field(default=None, init=False, repr=False)
    _previous_content: float | None = field(default=None, init=False, repr=False)

    def _assemble_scalar(self, expression, temperature) -> float:
        selected = field_api.unwrap(temperature)
        local = fem.assemble_scalar(fem.form(expression))
        return float(selected.function_space.mesh.comm.allreduce(local, op=MPI.SUM))

    def evaluate(self, temperature) -> dict[str, float]:
        """Return enthalpy content and discrete heat-balance evidence."""

        from . import operators

        selected = field_api.unwrap(temperature)
        if self._unit_field is None:
            self._unit_field = fem.Function(
                selected.function_space,
                name="UnitTemperatureWeight",
            )
            self._unit_field.interpolate(
                lambda x: np.ones((1, x.shape[1]), dtype=float)
            )
        content = self._assemble_scalar(self.content_form, selected)
        outward_rate = sum(
            self._assemble_scalar(item, selected) for item in self.outward_forms
        )
        input_rate = (
            0.0
            if self.source is None
            else float(operators.dual_product(self.source, self._unit_field))
        )
        residual = (
            0.0
            if self._previous_content is None
            else content
            - self._previous_content
            + float(self.dt) * (outward_rate - input_rate)
        )
        self._previous_content = content
        return {
            "thermal_content": content,
            "applied_heat_rate": input_rate,
            "outward_heat_rate": float(outward_rate),
            "heat_balance_residual": float(residual),
        }

    def restore(self, record) -> None:
        """Restore monitor memory from a checkpointed history frame."""

        if "thermal_content" in record:
            self._previous_content = float(record["thermal_content"])


@dataclass
class ThermalContentMonitor:
    """Backwards-compatible sensible-heat monitor without balance terms."""

    capacity: object
    _compiled_capacity: object | None = field(default=None, init=False, repr=False)
    _unit_field: object | None = field(default=None, init=False, repr=False)

    def evaluate(self, temperature) -> dict[str, float]:
        from . import operators

        selected = field_api.unwrap(temperature)
        if self._compiled_capacity is None:
            self._compiled_capacity = _energy_operator(self.capacity)
        if self._unit_field is None:
            self._unit_field = fem.Function(
                selected.function_space,
                name="UnitTemperatureWeight",
            )
            self._unit_field.interpolate(
                lambda x: np.ones((1, x.shape[1]), dtype=float)
            )
        return {
            "thermal_content": float(
                operators.xtmy(
                    self._unit_field,
                    self._compiled_capacity,
                    selected,
                )
            )
        }


def _energy_operator(operator):
    if hasattr(operator, "assemble_matrix"):
        return operator.assemble_matrix()
    if hasattr(operator, "mass") and isinstance(operator.mass, np.ndarray):
        return operator.mass
    return operator


def max_abs(function: fem.Function) -> float:
    """Global max absolute value of a finite-element field."""

    function = field_api.unwrap(function)
    local = float(np.max(np.abs(function.x.array)))
    return function.function_space.mesh.comm.allreduce(local, op=MPI.MAX)


def max_magnitude(function) -> float:
    """Global maximum magnitude of a scalar or vector finite-element field."""

    return magnitude_stats(function).max


@dataclass(frozen=True)
class FieldStats:
    """Distributed scalar statistics for a finite-element field."""

    name: str
    maximum: float
    mean: float
    minimum: float = 0.0
    count: int = 0

    @property
    def max(self) -> float:
        """Compatibility alias for the maximum value."""

        return self.maximum

    def summary(self) -> dict[str, object]:
        """Return a compact agent-readable summary."""

        return {
            "name": self.name,
            "kind": "field_stats",
            "max": self.maximum,
            "mean": self.mean,
            "min": self.minimum,
            "count": self.count,
        }


@dataclass(frozen=True)
class ScalarDiagnostic:
    """Named scalar diagnostic evaluated on demand."""

    name: str
    value: Callable[[], float]

    def evaluate(self) -> float:
        return float(self.value())


@dataclass(frozen=True)
class DiagnosticSet:
    """Ordered collection of scalar diagnostics."""

    diagnostics: tuple[ScalarDiagnostic, ...]

    @classmethod
    def create(cls, *diagnostics: ScalarDiagnostic):
        return cls(tuple(diagnostics))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(diagnostic.name for diagnostic in self.diagnostics)

    def evaluate(self) -> dict[str, float]:
        return {
            diagnostic.name: diagnostic.evaluate()
            for diagnostic in self.diagnostics
        }


def magnitude_stats(function, *, on=None, name: str | None = None) -> FieldStats:
    """Return distributed magnitude statistics for a scalar or vector field.

    ``on`` may be a geometric marker callable or a named mesh region with a
    marker, such as ``mesh.face(...)``. Cell regions are not supported yet
    because nodal field statistics need a dof-selection marker.
    """

    function = field_api.unwrap(function)
    marker = getattr(on, "marker", on)
    if marker is None:
        values = _field_magnitudes(function)
    else:
        values = _field_magnitudes_on_marker(function, marker)
    comm = function.function_space.mesh.comm
    local_count = int(len(values))
    if local_count:
        local_max = float(np.max(values))
        local_min = float(np.min(values))
        local_sum = float(np.sum(values))
    else:
        local_max = 0.0
        local_min = np.inf
        local_sum = 0.0
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    if global_count == 0:
        global_min = 0.0
    return FieldStats(
        name=name or f"{function.name}_magnitude",
        maximum=global_max,
        mean=0.0 if global_count == 0 else global_sum / global_count,
        minimum=float(global_min),
        count=int(global_count),
    )


def field_stats(function, *, on=None, name: str | None = None) -> FieldStats:
    """Alias for ``magnitude_stats`` for application-level diagnostics."""

    return magnitude_stats(function, on=on, name=name)


def _field_magnitudes(function) -> np.ndarray:
    values = dofs.owned_array(function)
    shape = getattr(function, "ufl_shape", ())
    if len(shape) == 1:
        dim = int(shape[0])
        if dim > 0 and len(values) % dim == 0:
            return np.linalg.norm(values.reshape((-1, dim)), axis=1)
    return np.abs(values)


def _field_magnitudes_on_marker(function, marker) -> np.ndarray:
    V = function.function_space
    values = function.x.array
    shape = getattr(function, "ufl_shape", ())
    if len(shape) != 1:
        dofs_selected = fem.locate_dofs_geometrical(V, marker)
        return np.abs(values[np.asarray(dofs_selected, dtype=np.int32)])

    components = []
    for component in range(V.num_sub_spaces):
        Vc, _ = V.sub(component).collapse()
        parent, _ = fem.locate_dofs_geometrical((V.sub(component), Vc), marker)
        components.append(values[np.asarray(parent, dtype=np.int32)])
    if not components or any(len(component) == 0 for component in components):
        return np.zeros(0, dtype=float)
    return np.sqrt(sum(component**2 for component in components))
