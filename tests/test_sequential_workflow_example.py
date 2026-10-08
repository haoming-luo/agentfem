# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed-API example: durable upstream reuse, not a coupled checkpoint."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI


def test_public_example_resumes_without_recomputing_heat(tmp_path):
    comm = MPI.COMM_WORLD
    output = Path(comm.bcast(str(tmp_path / "sequential"), root=0))
    path = Path(__file__).resolve().parents[1] / "examples/thermal_stress_wall_2d.py"
    spec = importlib.util.spec_from_file_location("wall_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.main(output_dir=output, smoke=True)
    expected = original.fields["Displacement"].field.x.array.copy()
    checkpoint = output / "accepted_heat.checkpoint.json"
    before = checkpoint.read_bytes()
    resumed = module.main(output_dir=output, smoke=True, resume_heat=True)
    assert checkpoint.read_bytes() == before
    np.testing.assert_allclose(resumed.fields["Displacement"].field.x.array, expected)
    evidence = resumed.metadata["upstream_stage"]
    assert evidence["thermal_solves_this_execution"] == 0
    assert evidence["accepted_time"] == 180.0
    assert evidence["manifest_sha256"] == original.metadata["upstream_stage"]["manifest_sha256"]
    with pytest.raises(FileExistsError):
        module.main(output_dir=output, smoke=True)
    # Do not silently consume an accepted field from a different source recipe.
    if comm.rank == 0:
        result_path = output / "heat.result.json"
        record = json.loads(result_path.read_text())
        record["metadata"]["restart_recipe"]["example_sha256"] = "changed"
        result_path.write_text(json.dumps(record))
    comm.barrier()
    with pytest.raises((ValueError, RuntimeError), match="recipe changed"):
        module.main(output_dir=output, smoke=True, resume_heat=True)
