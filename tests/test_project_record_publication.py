"""Operational records share safe publication without weakening their values."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from agentfem.project import RunContext, _write_json


def test_concurrent_latest_records_do_not_share_a_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "latest.json"
    rendezvous = Barrier(2)
    original_replace = Path.replace

    def simultaneous_replace(source, target):
        if Path(target) == destination:
            rendezvous.wait(timeout=10)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", simultaneous_replace)
    records = [{"run_id": "first", "status": "completed"},
               {"run_id": "second", "status": "paused"}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_write_json, destination, record) for record in records]
        for future in futures:
            future.result(timeout=15)
    assert json.loads(destination.read_text()) in records
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_record_preserves_prior_published_json(tmp_path, value):
    destination = tmp_path / "execution.json"
    original = {"run_id": "valid", "status": "paused"}
    _write_json(destination, original)
    with pytest.raises(ValueError, match="JSON"):
        _write_json(destination, {"error": {"invalid_metric": value}})
    assert json.loads(destination.read_text()) == original
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_record_replace_preserves_prior_record_and_cleans_temporary(tmp_path, monkeypatch):
    destination = tmp_path / "execution.json"
    _write_json(destination, {"status": "paused"})

    def fail_replace(*args):
        raise PermissionError("simulated read-only destination")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="read-only"):
        _write_json(destination, {"status": "completed"})
    assert json.loads(destination.read_text()) == {"status": "paused"}
    assert list(tmp_path.iterdir()) == [destination]


def test_invalid_case_record_becomes_a_persisted_failed_execution(tmp_path, capsys):
    from agentfem import cli

    (tmp_path / "agentfem.toml").write_text("[project]\nname='invalid-record'\n")
    (tmp_path / "case.py").write_text(
        "from agentfem import project\n"
        "context = project.current_run()\n"
        "context.artifact('accepted.txt').write_text('retained accepted output')\n"
        "context.write_execution('paused', structured_result=False)\n"
        "context.write_execution('failed', error={'metric': float('nan')})\n"
        "raise AssertionError('must not reach this line')\n"
    )
    assert cli.main(["run", "--project", str(tmp_path), "--name", "invalid", "--json"]) == 1

    def reject_constant(value):
        raise AssertionError(f"Invalid JSON constant: {value}")

    command = json.loads(capsys.readouterr().out, parse_constant=reject_constant)
    assert command["status"] == "failed"
    path, = (tmp_path / "outputs").rglob("execution.json")
    record = json.loads(path.read_text(), parse_constant=reject_constant)
    assert record["status"] == "failed"
    assert record["error"]["type"] == "builtins.ValueError"
    assert "JSON" in record["error"]["message"]
    assert (path.parent / "accepted.txt").read_text() == "retained accepted output"


def test_artifact_resolution_respects_physical_run_root(tmp_path):
    storage = tmp_path / "native-storage"
    storage.mkdir()
    linked_root = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        linked_root.symlink_to(storage, target_is_directory=True)
        (storage / "external").symlink_to(outside, target_is_directory=True)
        (storage / "internal").symlink_to(storage / "fields", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symbolic links unavailable on this test host: {exc}")
    run = RunContext(tmp_path, "linked", "test", linked_root,
                     linked_root / "result.json", linked_root / "execution.json")
    assert run.artifact("fields/result.xdmf") == storage / "fields/result.xdmf"
    assert run.artifact("internal/result.xdmf") == storage / "fields/result.xdmf"
    with pytest.raises(ValueError, match="symbolic links"):
        run.artifact("external/new-directory/result.json")
    assert list(outside.iterdir()) == []
