# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-call, accepted-only reporting through ordinary transient events."""
from types import SimpleNamespace

import numpy as np
import pytest

from agentfem import results
from agentfem.events import SolveEvent
from test_response_material_step import make_step
from test_transient_restart import _dynamic_step, _heat_step


@pytest.mark.parametrize("kind", ["response", "explicit", "implicit", "heat"])
def test_one_solve_records_absolute_stations_and_final(tmp_path, kind):
    def build():
        if kind == "response":
            return make_step(material_storage="owned_arrays")
        if kind == "heat":
            return _heat_step()
        return _dynamic_step(implicit=kind == "implicit")

    step = build()
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    observer = journal.reporter(step, every=3)
    result = step.solve_result(progress=observer)
    records = results.read_accepted_history(journal.path,
        expected_tip=journal.summary()["sha256_tip"])
    assert [row["increment"] for row in records] == [3, 4]
    for row in records:
        frame = next(item for item in step.history_records if item["time"] == row["time"])
        assert {key: row[key] for key in frame} == frame
    assert result is not None
    # Observing again after completion neither reruns nor duplicates the tail.
    step.run(progress=observer)
    assert journal.count == 2


def test_restart_and_partial_calls_do_not_shift_schedule(tmp_path):
    baseline = make_step(steps=7)
    baseline.run()
    partial = make_step(steps=7)
    partial.run(until_step=2)
    path = partial.save_checkpoint(tmp_path / "restart")
    step = make_step(steps=7)
    step.load_checkpoint(path)
    journal = results.AcceptedHistoryJournal(tmp_path / "continuation.jsonl")
    observer = journal.reporter(step, every=3)
    step.run(until_step=4, progress=observer)
    step.run(progress=observer)
    records = results.read_accepted_history(journal.path)
    assert [row["increment"] for row in records] == [3, 6, 7]
    np.testing.assert_array_equal(step.state.u.value.x.array, baseline.state.u.value.x.array)
    assert step.residual.snapshot() == baseline.residual.snapshot()


@pytest.mark.parametrize("every", [0, -1, 1.5, True])
def test_bad_cadence_is_rejected_before_execution(tmp_path, every):
    step = make_step()
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    with pytest.raises(ValueError, match="positive integer"):
        journal.reporter(step, every=every)
    assert step.completed_steps == 0


def test_sparse_history_must_cover_requested_stations(tmp_path):
    step = make_step(history_every=2, steps=7)
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    with pytest.raises(ValueError, match="multiple of history_every"):
        journal.reporter(step, every=3)
    step.run(progress=journal.reporter(step, every=4))
    assert [row["increment"] for row in results.read_accepted_history(journal.path)] == [4, 7]


def test_stale_or_wrong_step_event_is_not_recorded(tmp_path):
    step = make_step()
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    observer = journal.reporter(step)
    observer.emit(SolveEvent("failed", step.name))
    with pytest.raises(ValueError, match="bound accepted Step"):
        observer.emit(SolveEvent("time_increment", "another-step", increment=1))
    with pytest.raises(ValueError, match="refusing stale data"):
        observer.emit(SolveEvent("time_increment", step.name, time=0.0))
    assert journal.count == 0


def test_writer_failure_keeps_accepted_state_recoverable(tmp_path, monkeypatch):
    step = make_step()
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    observer = journal.reporter(step)
    def fail(_):
        raise OSError("disk full")
    with monkeypatch.context() as patch:
        import agentfem.results.journal as module
        patch.setattr(module.os, "fsync", fail)
        with pytest.raises(OSError, match="disk full"):
            step.run(progress=observer)
    assert step.completed_steps == 1  # reporter runs after numerical acceptance
    path = step.save_checkpoint(tmp_path / "accepted-after-io-failure")
    resumed = make_step()
    resumed.load_checkpoint(path)
    resumed.run()
    reference = make_step()
    reference.run()
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    with pytest.raises(RuntimeError, match="previously failed"):
        journal.append_records([{"increment": 2}])


def test_failed_trial_is_never_reported_as_accepted(tmp_path, monkeypatch):
    step = make_step()
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    advance = step._advance_one
    def fail_third(time):
        if step.completed_steps == 2:
            raise RuntimeError("rejected trial")
        return advance(time)
    monkeypatch.setattr(step, "_advance_one", fail_third)
    with pytest.raises(RuntimeError, match="rejected trial"):
        step.run(progress=journal.reporter(step))
    assert step.completed_steps == 2
    assert [row["increment"] for row in results.read_accepted_history(journal.path)] == [1, 2]


def test_mpi_and_unsupported_procedures_fail_closed(tmp_path, monkeypatch):
    from agentfem import diagnostics
    journal = results.AcceptedHistoryJournal(tmp_path / "history.jsonl")
    with pytest.raises(TypeError, match="supported transient"):
        journal.reporter(object())
    step = make_step()
    monkeypatch.setattr(diagnostics, "comm_of", lambda value: SimpleNamespace(size=2))
    with pytest.raises(NotImplementedError, match="serial"):
        journal.reporter(step)
