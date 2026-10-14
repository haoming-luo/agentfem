# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
from agentfem.results.journal import AcceptedHistoryJournal, read_accepted_history


def test_roundtrip_and_tip(tmp_path):
    path = tmp_path / "history.jsonl"
    journal = AcceptedHistoryJournal(path)
    records = [{"increment": 0, "force": [1., 2.]}, {"increment": 20, "work": 3.}]
    journal.append_records(records[:1])
    tip = journal.append_records(records[1:])
    assert tip["records"] == 2 and not tip["checkpoint"]
    assert read_accepted_history(path, expected_tip=tip["sha256_tip"]) == records
    before = path.read_bytes()
    journal.append_records([])
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        AcceptedHistoryJournal(path)
    # A removed complete suffix needs the external tip to detect it.
    path.write_bytes(b"\n".join(before.splitlines()[:2]) + b"\n")
    with pytest.raises(ValueError, match="tip mismatch"):
        read_accepted_history(path, expected_tip=tip["sha256_tip"])


@pytest.mark.parametrize("records", [
    [{"increment": 2}, {"increment": 2}], [{"increment": True}],
    [{"increment": -1}], [{"increment": 3, "value": float("nan")}],
])
def test_invalid_batch_writes_nothing(tmp_path, records):
    path = tmp_path / "history.jsonl"
    journal = AcceptedHistoryJournal(path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        journal.append_records(records)
    assert path.read_bytes() == before


def test_corruption_and_truncation(tmp_path):
    path = tmp_path / "history.jsonl"
    journal = AcceptedHistoryJournal(path)
    journal.append_records([{"increment": 1}, {"increment": 2}])
    before = path.read_bytes()
    path.write_bytes(before[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        read_accepted_history(path)
    lines = before.splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[2] + lines[1])
    with pytest.raises(ValueError, match="Corrupt"):
        read_accepted_history(path)


def test_failed_write_cannot_be_retried(tmp_path, monkeypatch):
    import agentfem.results.journal as module
    journal = AcceptedHistoryJournal(tmp_path / "history.jsonl")
    def fail(_):
        raise OSError("disk failure")
    monkeypatch.setattr(module.os, "fsync", fail)
    with pytest.raises(OSError):
        journal.append_records([{"increment": 1}])
    assert journal.count == 0
    with pytest.raises(RuntimeError, match="previously failed"):
        journal.append_records([{"increment": 1}])
