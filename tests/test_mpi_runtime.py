from __future__ import annotations

from pathlib import Path

import pytest

from agentfem import mpi_runtime


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("launcher", encoding="utf-8")
    return str(path)


def test_audit_recovers_environment_mpich_when_path_is_openmpi(tmp_path, monkeypatch):
    python = tmp_path / "env" / "bin" / "python"
    _touch(python)
    environment_launcher = _touch(python.parent / "mpiexec")
    path_launcher = _touch(tmp_path / "homebrew" / "bin" / "mpiexec")

    def inspect(path):
        if Path(path) == Path(environment_launcher):
            return "mpich", "HYDRA build details:"
        return "openmpi", "mpiexec (Open MPI) 5.0.9"

    monkeypatch.setattr(mpi_runtime, "_inspect_launcher", inspect)
    audit = mpi_runtime.audit_mpi_runtime(
        python_executable=python,
        vendor=("MPICH", (5, 0, 1)),
        rank_count=1,
        path_mpiexec=path_launcher,
        path_mpirun="",
    )

    assert audit.compatible is True
    assert audit.selected_launcher == str(Path(environment_launcher).resolve())
    assert audit.path_mismatch is True
    assert audit.code == "AFM-MPI-LAUNCHER-RECOVERED"
    assert audit.summary()["environment_launcher"] == str(
        Path(environment_launcher).resolve()
    )
    assert audit.summary()["launchers"][1]["family"] == "openmpi"


def test_audit_fails_closed_when_only_incompatible_launcher_exists(tmp_path, monkeypatch):
    python = tmp_path / "env" / "bin" / "python"
    _touch(python)
    path_launcher = _touch(tmp_path / "system" / "mpiexec")
    monkeypatch.setattr(
        mpi_runtime,
        "_inspect_launcher",
        lambda _path: ("openmpi", "mpiexec (Open MPI)"),
    )

    audit = mpi_runtime.audit_mpi_runtime(
        python_executable=python,
        vendor=("MPICH", (4, 3, 2)),
        rank_count=1,
        path_mpiexec=path_launcher,
        path_mpirun="",
    )

    assert audit.compatible is False
    assert audit.code == "AFM-MPI-LAUNCHER-MISMATCH"


def test_mpi_command_uses_verified_launcher(monkeypatch):
    monkeypatch.setattr(
        mpi_runtime,
        "compatible_mpi_launcher",
        lambda: "/trusted/bin/mpiexec",
    )

    assert mpi_runtime.mpi_command(2, ("python", "case.py")) == (
        "/trusted/bin/mpiexec",
        "-n",
        "2",
        "python",
        "case.py",
    )


@pytest.mark.parametrize("ranks", (0, -1))
def test_mpi_command_rejects_non_positive_rank_counts(ranks):
    with pytest.raises(ValueError, match="positive"):
        mpi_runtime.mpi_command(ranks, ("python", "case.py"))
