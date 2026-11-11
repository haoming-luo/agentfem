from __future__ import annotations

import json
from pathlib import Path

from mpi4py import MPI
import pytest

from agentfem import cli


@pytest.mark.parametrize("stage", ["create", "prepare"])
def test_run_directory_failure_reaches_every_rank(monkeypatch, tmp_path, stage):
    from agentfem.project import ProjectConfig, RunContext

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires two MPI ranks")
    root = Path(comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0))
    config = ProjectConfig(root=root, name="prepare-failure", entrypoint=root / "case.py",
                           output_directory=root / "outputs")

    def fail(*args, **kwargs):
        assert comm.rank == 0
        raise PermissionError(f"injected {stage} failure")

    monkeypatch.setattr(RunContext, stage, fail)
    with pytest.raises(RuntimeError, match=f"injected {stage} failure") as caught:
        cli._run_context(config, None, "test", None)
    assert len(set(comm.allgather(str(caught.value)))) == 1


def test_ir_write_failure_real_mpi(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agentfem import _model_inspection

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires two MPI ranks")
    model = SimpleNamespace(mesh=SimpleNamespace(comm=comm))
    monkeypatch.setattr(_model_inspection, "model_to_ir", lambda *a, **kw: {})
    # An existing directory is an invalid file destination on every platform.
    destination = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    with pytest.raises(RuntimeError, match="AFM-IR-WRITE") as caught:
        _model_inspection.write_model_ir(model, destination)
    errors = comm.allgather(str(caught.value))
    assert len(set(errors)) == 1


@pytest.mark.parametrize("contents", ["{broken json", "[]"])
def test_invalid_execution_record_reaches_every_rank(tmp_path, contents):
    from agentfem.project import RunContext

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires two MPI ranks")
    root = Path(comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0))
    context = RunContext(root, "record-failure", "test", root,
                         root / "result.json", root / "execution.json")
    if comm.rank == 0:
        context.execution_path.write_text(contents)
    comm.barrier()
    with pytest.raises(RuntimeError, match="AFM-RUN-RECORD") as caught:
        cli._completed_execution_record(context)
    assert len(set(comm.allgather(str(caught.value)))) == 1


def test_execution_record_write_failure_reaches_every_rank(tmp_path, monkeypatch):
    from agentfem.project import RunContext

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires two MPI ranks")
    root = Path(comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0))
    context = RunContext(root, "write-failure", "test", root,
                         root / "result.json", root / "execution.json")

    def fail(*args, **kwargs):
        assert comm.rank == 0
        raise PermissionError("injected record write failure")

    monkeypatch.setattr(RunContext, "write_execution", fail)
    with pytest.raises(RuntimeError, match="AFM-RUN-RECORD.*injected") as caught:
        cli._completed_execution_record(context)
    assert len(set(comm.allgather(str(caught.value)))) == 1


@pytest.mark.parametrize("failure", ["RuntimeError('injected rank-one failure')", "SystemExit(2)"])
def test_cli_collectively_reports_a_single_rank_failure(tmp_path, failure):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("collective CLI failure requires at least two MPI ranks")

    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    if MPI.COMM_WORLD.rank == 0:
        project = Path(root)
        (project / "agentfem.toml").write_text(
            "[project]\nname='rank-failure'\nentrypoint='case.py'\n",
            encoding="utf-8",
        )
        (project / "case.py").write_text(
            "from mpi4py import MPI\n"
            "if MPI.COMM_WORLD.rank == 1:\n"
            f"    raise {failure}\n",
            encoding="utf-8",
        )
    MPI.COMM_WORLD.barrier()

    return_code = cli.main(
        [
            "run",
            "--project",
            root,
            "--run-id",
            "mpi-failure",
            "--json",
            "--inside-mpi",
        ]
    )

    assert return_code == 1
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        execution = json.loads(
            (
                Path(root)
                / "outputs"
                / "rank-failure"
                / "mpi-failure"
                / "execution.json"
            ).read_text(encoding="utf-8")
        )
        assert execution["status"] == "failed"
        assert execution["error"]["rank"] == 1
        assert execution["error"]["rank_errors"][0]["message"] == (
            "2" if failure.startswith("SystemExit") else "injected rank-one failure"
        )
