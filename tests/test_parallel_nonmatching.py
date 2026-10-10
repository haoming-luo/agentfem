# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Ordinary finite-bulk/bond/tool distributed acceptance."""
from pathlib import Path
import numpy as np
import pytest
from mpi4py import MPI
from test_finite_hex_bond_contact import prepare

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="Two-rank nonmatching gate")


@pytest.mark.parametrize("plastic", [False, True])
def test_parallel_bond_contact_matches_serial_and_restarts(tmp_path, plastic):
    full = prepare(dt=4e-4, comm=MPI.COMM_WORLD, plastic=plastic)
    full.run()
    serial = prepare(dt=4e-4, plastic=plastic)
    serial.run()
    assert full.material_residual.stability.selected == pytest.approx(serial.material_residual.stability.selected, rel=1e-12)
    result = full.solve_result(field_variables=())
    assert result is not None
    for key in ("interface_stored_energy", "bulk_stored_energy", "contact_motion_work",
                "contact_potential_energy", "kinetic_energy"):
        assert full.history_records[-1][key] == pytest.approx(serial.history_records[-1][key], rel=1e-9, abs=1e-13)
    partial = prepare(dt=4e-4, comm=MPI.COMM_WORLD, plastic=plastic)
    partial.run(until_step=71)
    path = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    checkpoint = partial.save_checkpoint(path / "joint")
    resumed = prepare(dt=4e-4, comm=MPI.COMM_WORLD, plastic=plastic)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(getattr(full.state, name).value.x.array,
                                      getattr(resumed.state, name).value.x.array)
    assert full.residual.snapshot() == resumed.residual.snapshot()


def test_parallel_bond_post_commit_failure_rolls_back(monkeypatch):
    step = prepare(dt=4e-4, comm=MPI.COMM_WORLD)
    step.run(until_step=71)
    snapshot = step.residual.snapshot()
    nodal = step.state.snapshot()
    original = step.residual.commit

    def fail():
        original()
        if MPI.COMM_WORLD.rank == 1:
            raise RuntimeError("bond acceptance failure")

    monkeypatch.setattr(step.residual, "commit", fail)
    with pytest.raises(RuntimeError, match="bond acceptance failure"):
        step.run()
    assert step.completed_steps == 71
    assert step.residual.snapshot() == snapshot
    for key, value in nodal["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][key], value)
    monkeypatch.setattr(step.residual, "commit", original)
    step.run()


def test_parallel_interface_corruption_is_atomic():
    import copy
    step = prepare(dt=4e-4, comm=MPI.COMM_WORLD)
    step.run(until_step=71)
    before = step.residual.snapshot()
    record = copy.deepcopy(before)
    if MPI.COMM_WORLD.rank == 1:
        record["base_state"]["cohesive"]["pairing"] = "corrupt"
    with pytest.raises((RuntimeError, ValueError), match="identity"):
        step.residual.restore(record)
    assert step.residual.snapshot() == before
