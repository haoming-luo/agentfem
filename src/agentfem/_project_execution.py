# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Operational handoff around an already constructed serial transient Step.

The Step remains the sole integration, state and checkpoint owner. This module
does not choose inertia, constitutive updates, contact or a time increment.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class RunPolicy:
    """Operational controls only; physical time grids remain in model.step."""
    until_step: int | None = None
    restart: str | Path | None = None
    wall_limit: float | None = None
    progress_every: int = 100
    journal_every: int = 10
    observation_every: int = 100
    export_partial: bool = False
    prepare_only: bool = False

    def __post_init__(self):
        for key in ("progress_every", "journal_every", "observation_every"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"AFM-RUN-POLICY: {key} must be a positive integer")
        if self.until_step is not None and (type(self.until_step) is not int or self.until_step < 0):
            raise ValueError("AFM-RUN-POLICY: until_step must be a nonnegative integer")
        if self.wall_limit is not None and (isinstance(self.wall_limit, bool) or not math.isfinite(self.wall_limit) or self.wall_limit <= 0):
            raise ValueError("AFM-RUN-POLICY: wall_limit must be positive and finite")
        for key in ("export_partial", "prepare_only"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"AFM-RUN-POLICY: {key} must be boolean")
        if self.prepare_only and self.restart is not None:
            raise ValueError("AFM-RUN-POLICY: preparation-only cannot also restore state")


def execute_step(step, *, context, policy=None, metadata=None, observations=None,
                 export=None, progress_quantities=None):
    """Run a serial transient Step and publish complete or paused evidence.

    ``observations`` maps names to pure accepted-state callbacks ``fn(step)``.
    ``export(step, directory)`` returns named file paths; it runs only at the
    final accepted station (including partial stations when requested).
    ``progress_quantities(step)`` returns explicit ``name: (value, unit)`` pairs.
    The wrapper never promotes trust, changes numerical controls or executes a
    second solver. On failure it attempts an accepted checkpoint and re-raises
    the original error. MPI is rejected before file creation; collective
    observation/export requires its own reviewed contract.
    """
    from . import checkpointing, results
    from .diagnostics import comm_of
    from .time import ProgressPrinter

    policy = RunPolicy() if policy is None else policy
    if not isinstance(policy, RunPolicy):
        raise TypeError("Expected RunPolicy")
    solution = getattr(step, "current", None)
    if solution is None:
        solution = step.state.u
    comm = comm_of(solution)
    if comm.size != 1:
        raise NotImplementedError("AFM-RUN-MPI: execute_step currently requires one MPI process")
    for name in ("run", "save_checkpoint", "load_checkpoint", "solve_result"):
        if not callable(getattr(step, name, None)):
            raise TypeError(f"AFM-RUN-STEP: missing transient method {name}")
    until = step.steps if policy.until_step is None else policy.until_step
    if until > step.steps or until < step.completed_steps:
        raise ValueError("AFM-RUN-POLICY: until_step outside accepted/terminal interval")
    observers = {} if observations is None else dict(observations)
    if any(not isinstance(n, str) or not n or not callable(f) for n, f in observers.items()):
        raise ValueError("AFM-RUN-OBSERVATIONS: require named callbacks")
    if progress_quantities is not None and not callable(progress_quantities):
        raise TypeError("progress_quantities must be callable")
    if export is not None and not callable(export):
        raise TypeError("export must be callable")
    # Copy and validate metadata before advancing any state.
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    output = context.output_directory
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError("AFM-RUN-OUTPUT: use a fresh run directory")
    started = perf_counter()
    report = {"schema": "agentfem.step-execution", "schema_version": "0.1.0",
              "complete": False, "metadata": metadata, "timing_scope": "execution_and_final_export"}
    artifacts = {}
    report["artifacts"] = artifacts
    printer = ProgressPrinter(step.steps, policy.progress_every, comm)

    def write(path, value):
        checkpointing.atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")

    try:
        if policy.prepare_only:
            report.update(status="prepared", accepted_steps=step.completed_steps)
            context.write_execution("prepared", structured_result=False)
            return report
        if policy.restart is not None:
            restart = Path(policy.restart).resolve()
            step.load_checkpoint(restart)
            report["restart_source"] = {"path": str(restart), "sha256": hashlib.sha256(restart.read_bytes()).hexdigest()}
        if until < step.completed_steps:
            raise ValueError("AFM-RUN-POLICY: requested endpoint precedes restored state")
        report["initial_step"] = step.completed_steps
        journal = results.AcceptedHistoryJournal(output / "accepted-history.jsonl")
        accepted = journal.reporter(step, every=policy.journal_every)
        observation_journal = results.AcceptedHistoryJournal(output / "observations.jsonl") if observers else None

        def observe():
            if observation_journal is None or observation_journal.last_increment == step.completed_steps:
                return
            observation_journal.append_records([{"increment": step.completed_steps,
                "time": step.completed_steps * step.dt,
                "observations": {name: fn(step) for name, fn in observers.items()}}])

        class Observer:
            def emit(self, event):
                accepted.emit(event)
                if event.kind != "time_increment":
                    return
                if event.increment % policy.observation_every == 0:
                    observe()
                if printer.should_print(event.increment):
                    printer.update(event.increment, time=event.time, elapsed_seconds=perf_counter()-started,
                                   quantities={} if progress_quantities is None else progress_quantities(step))
                if policy.wall_limit is not None and perf_counter()-started > policy.wall_limit:
                    raise TimeoutError("AFM-RUN-WALL-LIMIT: reached at an accepted station")

        step.run(until_step=until, progress=Observer())
        if step.completed_steps != until:
            raise RuntimeError("AFM-RUN-INCOMPLETE: Step returned before requested endpoint")
        observe()
        report["journal"] = journal.summary()
        artifacts["accepted_history"] = str(journal.path)
        if observation_journal is not None:
            report["observations"] = observation_journal.summary()
            artifacts["observations"] = str(observation_journal.path)
        checkpoint = step.save_checkpoint(output / "accepted-final")
        artifacts["checkpoint"] = str(checkpoint)
        complete = step.completed_steps == step.steps
        if export is not None and (complete or policy.export_partial):
            exported = dict(export(step, output))
            if set(exported) & set(artifacts):
                raise ValueError("AFM-RUN-ARTIFACT: duplicate artifact name")
            for name, value in exported.items():
                path = Path(value)
                if not path.is_absolute():
                    path = output / path
                if not path.is_file():
                    raise ValueError(f"AFM-RUN-ARTIFACT: missing export {name}")
                artifacts[name] = str(path)
        report.update(complete=complete, status="completed" if complete else "paused",
                      accepted_steps=step.completed_steps, summary=step.summary())
        if complete:
            result = step.solve_result(metadata=metadata)
            for name, path in artifacts.items():
                result.add_artifact(name, path)
            context.publish(result)
        else:
            context.write_execution("paused", structured_result=False)
        printer.status("Run completed" if complete else "Partial run completed")
        return report
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        try:
            report["failure_checkpoint"] = str(step.save_checkpoint(output / "accepted-failure"))
        except BaseException as checkpoint_error:
            report["checkpoint_failure"] = str(checkpoint_error)
        try:
            context.write_execution("failed", error=report["failure"], structured_result=False)
        except Exception as reporting_error:
            report["reporting_failure"] = str(reporting_error)
        raise
    finally:
        report["accepted_steps"] = step.completed_steps
        report["elapsed_seconds"] = perf_counter()-started
        if "initial_step" in report:
            report["advanced_steps_this_run"] = step.completed_steps-report["initial_step"]
        try:
            write(report_path, report)
        except Exception:
            if "failure" not in report:
                raise
