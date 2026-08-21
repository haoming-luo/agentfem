from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    checkpointing,
    constitutive,
    fields,
    histories,
    materials,
    mesh,
    models,
    studies,
)


def _distributed_state_dependent_heat_step():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.25),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
    )
    temperature = model.field(fields.temperature(domain, value=300.0))
    model.material(
        constitutive.temperature_dependent_thermoelastic(
            young=1.0e9,
            poisson=0.3,
            density=1.0,
            thermal_expansion=1.0e-5,
            conductivity=materials.temperature_property(
                [300.0, 400.0], [1.0, 2.0], extrapolation="constant"
            ),
            specific_heat=materials.temperature_property(
                [300.0, 400.0], [1.0, 3.0], extrapolation="constant"
            ),
            reference_temperature=300.0,
        )
    )
    model.heat_source(150.0)
    return model.step(
        target=temperature,
        dt=1.0,
        steps=1,
        progress=False,
    )


def _distributed_heat_step(*, checkpoint=None):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.25),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
    )
    temperature = model.field(fields.temperature(domain, value=400.0))
    model.material(
        constitutive.thermoelastic(
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    model.convection(
        on=mesh.face(domain, axis="x", value=1.0, name="right", tag=1),
        coefficient=25.0,
        ambient_temperature=300.0,
    )
    return model.step(
        target=temperature,
        dt=0.5,
        steps=3,
        progress=False,
        checkpoint=checkpoint,
    )


def test_distributed_checkpoint_policy_writes_cadence_and_final(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed checkpoint cadence requires at least two MPI ranks")

    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    policy = checkpointing.every(2, directory=Path(root) / "automatic")
    step = _distributed_heat_step(checkpoint=policy)

    step.run()

    assert [item.coordinate_value for item in step.checkpoints] == [1.0, 1.5]
    if MPI.COMM_WORLD.rank == 0:
        for record in step.checkpoints:
            manifest = json.loads(record.path.read_text(encoding="utf-8"))
            assert len(manifest["shards"]) == 2


def test_distributed_heat_capture_uses_one_physical_time_contract_per_rank():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed field-history capture requires at least two MPI ranks")

    step = _distributed_heat_step()
    history = step.capture_history(name="temperature", unit="K")

    step.run()

    assert history.times == pytest.approx((0.0, 0.5, 1.0, 1.5))
    assert history.frame_count == 4
    assert np.all(np.isfinite(history.sample(0.75)))
    frame_counts = MPI.COMM_WORLD.allgather(history.frame_count)
    assert frame_counts == [4] * MPI.COMM_WORLD.size


def test_state_dependent_heat_conserves_enthalpy_across_two_ranks():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed nonlinear heat transfer requires two MPI ranks")

    step = _distributed_state_dependent_heat_step()
    step.run()

    expected = 300.0 + (-1.0 + np.sqrt(7.0)) / 0.02
    np.testing.assert_allclose(step.current.x.array, expected, rtol=2.0e-8)
    assert step.problem.last_solve_info.converged
    assert step.history_records[-1]["thermal_content"] == pytest.approx(37.5)
    assert step.history_records[-1]["heat_balance_residual"] == pytest.approx(
        0.0, abs=2.0e-8
    )


def test_distributed_temperature_history_has_portable_archive(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("portable field-history evidence requires at least two MPI ranks")

    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    step = _distributed_heat_step()
    history = step.capture_history(name="temperature", unit="K")
    step.run()
    archive = history.save(Path(root) / "temperature_history")

    receiving_step = _distributed_heat_step()
    restored = histories.FieldHistory.load(
        archive,
        source=receiving_step.current,
    )

    np.testing.assert_allclose(restored.sample(0.75), history.sample(0.75))
    assert restored.portable_identity() == history.portable_identity()
    if MPI.COMM_WORLD.rank == 0:
        with np.load(archive, allow_pickle=False) as payload:
            assert str(payload["schema"]) == "agentfem.field-history.v2"
            assert payload["snapshots"].shape[0] == 4


def test_distributed_checkpoint_retention_removes_complete_old_generations(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed checkpoint retention requires at least two MPI ranks")

    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    directory = Path(root) / "retained"
    policy = checkpointing.every(1, directory=directory, keep_last=1)
    step = _distributed_heat_step(checkpoint=policy)

    step.run()

    assert len(step.checkpoints) == 1
    assert step.checkpoints[0].coordinate_value == pytest.approx(1.5)
    if MPI.COMM_WORLD.rank == 0:
        assert len(tuple(directory.glob("*.checkpoint.json"))) == 1
        assert len(tuple(directory.glob("*.rank-*.npz"))) == 2


def test_partition_bound_transient_checkpoint_restarts_collectively(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed transient restart requires at least two MPI ranks")

    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    reference = _distributed_heat_step()
    reference.run()

    partial = _distributed_heat_step()
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(Path(root) / "distributed_heat")
    restarted = _distributed_heat_step()
    restarted.load_checkpoint(checkpoint)
    restarted.run()

    np.testing.assert_allclose(
        restarted.current.x.array,
        reference.current.x.array,
    )
    assert restarted.completed_steps == 3
    assert len(restarted.history_records) == 4
    if MPI.COMM_WORLD.rank == 0:
        assert checkpoint.is_file()
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert len(manifest["shards"]) == 2
        assert all(item["sha256"] for item in manifest["shards"])
        assert len(tuple(Path(root).glob("distributed_heat.*.rank-*.npz"))) == 2


def test_transient_checkpoint_reports_one_missing_rank_shard_collectively(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed transient restart requires at least two MPI ranks")

    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    partial = _distributed_heat_step()
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(Path(root) / "missing_shard")
    if MPI.COMM_WORLD.rank == 0:
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        (Path(root) / manifest["shards"][1]["path"]).unlink()
    MPI.COMM_WORLD.barrier()

    restarted = _distributed_heat_step()
    with pytest.raises(RuntimeError, match="rank 1"):
        restarted.load_checkpoint(checkpoint)
