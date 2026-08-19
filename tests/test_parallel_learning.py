from __future__ import annotations

import json
from pathlib import Path

from mpi4py import MPI
import pytest

from agentfem import learning, mesh, models, results, studies


def _neural_field_spec():
    field = learning.FieldEncoding(
        name="temperature",
        role="output",
        unit="K",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
    )
    return learning.NeuralFieldSpec(
        fields=(field,),
        objectives=(
            learning.ObjectiveTerm(
                name="heat_balance",
                kind="residual",
                expression="-div(k*grad(T)) = 0",
                dependent_fields=(field.name,),
                form="strong",
                measure="domain",
            ),
        ),
        conditions=(
            learning.ConditionSpec(
                name="left_temperature",
                kind="boundary",
                target=field.name,
                on="left",
            ),
        ),
        representations=(
            learning.NeuralRepresentation(
                name="temperature_network",
                fields=(field.name,),
            ),
        ),
    )


def test_user_neural_field_executor_uses_model_comm_and_root_manifest(
    tmp_path,
    monkeypatch,
):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed neural-field lifecycle requires two ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=domain,
        name="distributed_neural_field",
    )
    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    observed = []

    def executor(request):
        observed.append((request.comm.rank, request.comm.size))
        result = results.SimulationResult(request.name)
        result.add_quantity("normalized_loss", 0.0, kind="optimization")
        return result

    if MPI.COMM_WORLD.rank != 0:
        def reject_non_root_manifest(*_args, **_kwargs):
            raise AssertionError("Non-root rank attempted to write result.json")

        monkeypatch.setattr(
            results.SimulationResult,
            "write_manifest",
            reject_non_root_manifest,
        )

    result = model.step(
        target=_neural_field_spec(),
        executor=executor,
        executor_name="test.distributed_executor",
        output=Path(root) / "learning",
    ).solve_result()

    assert observed == [(MPI.COMM_WORLD.rank, 2)]
    assert result.metadata["learning_execution"]["request"]["distributed"] is True
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        manifest = json.loads(
            (Path(root) / "learning" / "result.json").read_text(encoding="utf-8")
        )
        assert manifest["metadata"]["learning_execution"]["executor"]["name"] == (
            "test.distributed_executor"
        )
