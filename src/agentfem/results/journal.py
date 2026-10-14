# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-writer accepted-report journal, independent of solver checkpoints.

Only newly accepted reporting stations are appended. They need not include
every increment. A digest chain detects corruption/reordering, not authorship
or removal of a complete suffix; retain the returned tip in final evidence.
Restart creates a new segment rather than editing an earlier journal.
"""
import hashlib
import json
import os
from pathlib import Path

_SCHEMA = "agentfem.accepted-history.v1"


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=True)


def _increment(record, previous):
    increment = record.get("increment")
    if type(increment) is not int or increment < 0:
        raise ValueError("history.increment must be a nonnegative integer")
    if previous is not None and increment <= previous:
        raise ValueError("history.increment must strictly increase")
    return increment


class AcceptedHistoryJournal:
    """Durable, exclusive-create evidence stream. Not a state/restart owner.

    The caller submits accepted records only, then publishes the returned tip.
    A failed write poisons this writer: do not retry into a partial journal.
    No MPI synchronization is implicit; one designated writer owns each file.
    """

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(_encode({"schema": _SCHEMA}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.count = 0
        self.last_increment = None
        self.tip = ""
        self._failed = False

    def append_records(self, records):
        if self._failed:
            raise RuntimeError("Journal write previously failed; start a new segment")
        pending = []
        last, tip = self.last_increment, self.tip
        for record in records:
            last = _increment(record, last)
            payload = _encode(record)
            digest = hashlib.sha256((tip + "\n" + payload).encode()).hexdigest()
            pending.append(_encode({"previous": tip, "sha256": digest,
                                    "payload": payload}) + "\n")
            tip = digest
        if pending:
            try:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write("".join(pending))
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self._failed = True
                raise
            self.count += len(pending)
            self.last_increment, self.tip = last, tip
        return self.summary()

    def summary(self):
        """Return the current digest tip without touching the file."""
        return {"path": self.path.name, "records": self.count,
                "sha256_tip": self.tip, "checkpoint": False,
                "scope": "accepted reporting stations"}

    def reporter(self, step, *, every=1):
        """Observe serial transient history through the existing event stream.

        Pass the returned observer as ``progress`` to ``run`` or ``solve_result``.
        Sampling uses absolute accepted increments, independently of display
        cadence. It does not advance the solver, evaluate physics or save state.
        Create a new journal segment after restarting from a checkpoint.
        """
        from .._transient_problems import (
            ExplicitDynamicsStep, ImplicitDynamicsStep, FirstOrderTransientStep,
        )
        from ..diagnostics import comm_of

        if type(every) is not int or every <= 0:
            raise ValueError("Journal every must be a positive integer")
        if not isinstance(step, (ExplicitDynamicsStep, ImplicitDynamicsStep,
                                 FirstOrderTransientStep)):
            raise TypeError("Journal reporter requires a supported transient Step")
        solution = step.current if isinstance(step, FirstOrderTransientStep) else step.state.u
        if comm_of(solution).size != 1:
            raise NotImplementedError("Journal reporter is serial; MPI collection is not implicit")
        history_every = int(getattr(step, "history_every", 1))
        if every % history_every:
            raise ValueError("Journal every must be a multiple of history_every")
        if self.count or self._failed:
            raise ValueError("Bind the reporter to a fresh journal segment")
        return _TransientHistoryReporter(self, step, every)


class _TransientHistoryReporter:
    """Read accepted scalar histories; never own or mutate numerical state."""

    def __init__(self, journal, step, every):
        self.journal = journal
        self.step = step
        self.every = every

    def emit(self, event):
        if event.kind != "time_increment":
            return
        step = self.step
        increment = event.increment
        if event.step_name != step.name or increment != step.completed_steps:
            raise ValueError("Journal event does not identify the bound accepted Step")
        if increment % self.every and increment != step.steps:
            return
        records = step.history_records
        if not records or records[-1].get("time") != event.time:
            raise ValueError("Journal requires a current accepted history frame; refusing stale data")
        frame = dict(records[-1])
        if "increment" in frame or "step_name" in frame:
            raise ValueError("History channel conflicts with journal station identity")
        frame.update(increment=int(increment), step_name=str(step.name))
        self.journal.append_records([frame])


def read_accepted_history(path, *, expected_tip=None):
    """Read a complete segment; optionally require the externally saved tip."""
    records, last, tip = [], None, ""
    with Path(path).open(encoding="utf-8") as stream:
        header = stream.readline()
        if not header.endswith("\n") or json.loads(header) != {"schema": _SCHEMA}:
            raise ValueError("Unknown or truncated accepted-history header")
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Truncated accepted-history record")
            entry = json.loads(line)
            payload = entry["payload"]
            digest = hashlib.sha256((tip + "\n" + payload).encode()).hexdigest()
            if entry["previous"] != tip or entry["sha256"] != digest:
                raise ValueError("Corrupt accepted-history chain")
            record = json.loads(payload)
            # Re-encoding rejects NaN/Inf even with an otherwise valid digest.
            _encode(record)
            last = _increment(record, last)
            records.append(record)
            tip = digest
    if expected_tip is not None and tip != expected_tip:
        raise ValueError("Accepted-history final tip mismatch")
    return records
