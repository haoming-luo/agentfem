from __future__ import annotations

import json
from pathlib import Path

from mpi4py import MPI
import pytest

from agentfem import cli


def test_cli_collectively_reports_a_single_rank_failure(tmp_path):
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
            "    raise RuntimeError('injected rank-one failure')\n",
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
            "injected rank-one failure"
        )
