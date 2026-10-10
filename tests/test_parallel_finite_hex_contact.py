# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Distributed material/contact ownership without distributed bonded interfaces."""
import copy
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from test_finite_hex_contact_composition import prepare_contact

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="Two-rank contact acceptance")


def prepare(**options):
    return prepare_contact(public=True, comm=MPI.COMM_WORLD, duration=0.004, **options)


def test_parallel_contact_matches_serial_and_partition_restart(tmp_path):
    full = prepare()
    full.run()
    serial = prepare_contact(public=True, duration=0.004)
    serial.run()
    for key in ("bulk_stored_energy", "contact_motion_work", "kinetic_energy",
                "contact_potential_energy", "relative_energy_balance_error"):
        assert full.history_records[-1][key] == pytest.approx(serial.history_records[-1][key], rel=1e-9, abs=1e-13)
    partial = prepare()
    partial.run(until_step=17)
    directory = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    checkpoint = partial.save_checkpoint(directory / "contact")
    resumed = prepare()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    for key in ("u", "v", "a"):
        np.testing.assert_array_equal(getattr(resumed.state, key).value.x.array,
                                      getattr(full.state, key).value.x.array)
    assert resumed.residual.snapshot() == full.residual.snapshot()


def test_rank_local_contact_corruption_rejects_before_nested_collectives():
    step = prepare()
    step.run(until_step=17)
    before = step.residual.snapshot()
    corrupted = copy.deepcopy(before)
    if MPI.COMM_WORLD.rank == 1:
        corrupted["work_state"]["identity"] = "wrong"
    with pytest.raises((ValueError, RuntimeError), match="identity"):
        step.residual.restore(corrupted)
    assert step.residual.snapshot() == before


def test_rank_inconsistent_contact_presence_fails_before_trace_collectives():
    with pytest.raises((ValueError, RuntimeError), match="contact contract"):
        prepare(include_contact=MPI.COMM_WORLD.rank == 0)


def test_parallel_contact_with_empty_partition():
    step = prepare(single_cell=True)
    owned = step.material_residual.internal.displacement.function_space.mesh.topology.index_map(3).size_local
    assert min(MPI.COMM_WORLD.allgather(owned)) == 0
    step.run()
    assert step.history_records[-1]["contact_motion_work"] > 0
    assert np.isfinite(step.history_records[-1]["relative_energy_balance_error"])


def test_rank_local_failure_after_contact_commit_rolls_back(monkeypatch):
    step = prepare()
    step.run(until_step=17)
    before, nodal = step.residual.snapshot(), step.state.snapshot()
    original = step.residual.commit

    def failed():
        original()
        if MPI.COMM_WORLD.rank == 1:
            raise RuntimeError("injected distributed contact acceptance")

    monkeypatch.setattr(step.residual, "commit", failed)
    with pytest.raises((RuntimeError, ValueError), match="contact acceptance"):
        step.run()
    assert step.completed_steps == 17
    assert step.residual.snapshot() == before
    for name, value in nodal["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], value)
    monkeypatch.setattr(step.residual, "commit", original)
    step.run()
