# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""The installed launcher's parent must not initialize a second MPI runtime."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agentfem import _cli_entry, mpi_runtime


def test_entry_import_and_help_do_not_load_mpi_or_full_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agentfem import _cli_entry; import sys; "
            "assert 'mpi4py.MPI' not in sys.modules; "
            "assert 'agentfem.cli' not in sys.modules; "
            "_cli_entry.main(['mpi-run', '--help'])",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert "--timeout" in result.stdout


def test_vendor_probe_disables_initialization_only_in_probe(monkeypatch):
    monkeypatch.setenv("MPI4PY_RC_INITIALIZE", "1")

    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert kwargs["env"]["MPI4PY_RC_INITIALIZE"] == "0"
        assert kwargs["env"]["MPI4PY_RC_FINALIZE"] == "0"
        return SimpleNamespace(stdout=json.dumps(["MPICH", [5, 0, 1]]))

    monkeypatch.setattr(subprocess, "run", run)
    assert _cli_entry._probe_vendor() == ("MPICH", (5, 0, 1))
    import os

    assert os.environ["MPI4PY_RC_INITIALIZE"] == "1"


def test_entry_passes_verified_audit_without_reinitialization(monkeypatch):
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    audit = SimpleNamespace(compatible=True, family="mpich", selected_launcher="/mpi")
    monkeypatch.setattr(_cli_entry, "_probe_vendor", lambda: ("MPICH", (5, 0, 1)))
    monkeypatch.setattr(mpi_runtime, "audit_mpi_runtime", lambda **kwargs: audit)
    calls = []
    monkeypatch.setattr(
        mpi_runtime,
        "run_mpi_command",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 7,
    )
    assert (
        _cli_entry.main(
            ["mpi-run", "-n", "2", "--timeout", "30", "--", "python", "case.py"]
        )
        == 7
    )
    assert calls == [((2, ["python", "case.py"]), {"timeout": 30.0, "audit": audit})]


def test_nested_launch_rejected_without_vendor_probe(monkeypatch):
    monkeypatch.setenv("PMI_SIZE", "2")
    monkeypatch.setattr(
        _cli_entry, "_probe_vendor", lambda: pytest.fail("unexpected probe")
    )
    assert _cli_entry.main(["mpi-run", "-n", "2", "--", "python"]) == 1


def test_interrupt_terminates_owned_process_group(monkeypatch):
    process = SimpleNamespace(
        wait=lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    monkeypatch.setattr(mpi_runtime, "mpi_command", lambda *args: ("/mpi",))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    cleaned = []
    monkeypatch.setattr(
        mpi_runtime, "_terminate_process_group", lambda p, **kwargs: cleaned.append(p)
    )
    with pytest.raises(KeyboardInterrupt):
        mpi_runtime.run_mpi_command(2, ["python"])
    assert cleaned == [process]
