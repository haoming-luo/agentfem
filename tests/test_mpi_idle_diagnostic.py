# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""The optional idle diagnostic changes only its bounded child environment."""

import json
import os
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import pytest


def test_provider_probe_is_explicit_isolated_and_bounded(monkeypatch):
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/diagnose_mpi_idle.py")
    )
    monkeypatch.setenv("FI_PROVIDER", "sockets")
    for key in ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    def execute(command, **options):
        assert options["env"]["FI_PROVIDER"] == "tcp"
        assert options["timeout"] == 12
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ranks": 1,
                    "wall_seconds": 2,
                    "user_cpu_seconds": 0.01,
                    "system_cpu_seconds": 0.01,
                    "idle_cpu_equivalent_cores": 0.01,
                }
            ),
        )

    monkeypatch.setattr(subprocess, "run", execute)
    record = module["probe"](provider="tcp")
    assert record["status"] == "completed"
    assert os.environ["FI_PROVIDER"] == "sockets"
    with pytest.raises(ValueError, match="0.1..10"):
        module["probe"](seconds=float("nan"))


def test_provider_probe_timeout_and_nested_launch_are_explicit(monkeypatch):
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/diagnose_mpi_idle.py")
    )
    for key in ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    def timeout(command, **options):
        raise subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert module["probe"]()["status"] == "timeout"
    monkeypatch.setenv("PMI_SIZE", "2")
    with pytest.raises(ValueError, match="outside"):
        module["probe"]()
