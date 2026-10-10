# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Retention refuses a tampered deletion set before removing any valid shard."""

import json

import pytest
from mpi4py import MPI

from agentfem.checkpointing import _remove_transient_checkpoint
from test_transient_restart import _heat_step


@pytest.mark.parametrize(
    "target",
    ["parent", "absolute", "neighbor", "other_generation", "symlink", "directory"],
)
@pytest.mark.parametrize("slot", ["shards", "portable_state"])
def test_retention_preflights_all_paths_before_deletion(tmp_path, target, slot):
    step = _heat_step()
    step.run(until_step=1)
    manifest = step.save_checkpoint(tmp_path / "safe" / "station")
    metadata = json.loads(manifest.read_text())
    valid = manifest.parent / metadata["shards"][0]["path"]
    valid_bytes = valid.read_bytes()
    unrelated = tmp_path / "unrelated.npz"
    unrelated.write_bytes(b"precious unrelated user data")
    prefix = "station." + metadata["generation"]
    names = {
        "parent": "../unrelated.npz",
        "absolute": str(unrelated),
        "neighbor": "neighbor.npz",
        "other_generation": "station.0123456789abcdef.rank-00000.npz",
        "symlink": prefix + ".link.npz",
        "directory": prefix + ".directory.npz",
    }
    name = names[target]
    if target == "symlink":
        (manifest.parent / name).symlink_to(unrelated)
    if target == "directory":
        (manifest.parent / name).mkdir()
    if slot == "shards":
        metadata["shards"].append({"path": name})
    else:
        metadata["portable_state"] = {"path": name}
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="checkpoint removal failed"):
        _remove_transient_checkpoint(manifest, comm=MPI.COMM_SELF)
    assert valid.read_bytes() == valid_bytes
    assert manifest.exists()
    assert unrelated.read_bytes() == b"precious unrelated user data"


def test_retention_removes_only_valid_generation(tmp_path):
    step = _heat_step()
    step.run(until_step=1)
    manifest = step.save_checkpoint(tmp_path / "station", portable=True)
    metadata = json.loads(manifest.read_text())
    payloads = [tmp_path / record["path"] for record in metadata["shards"]]
    payloads.append(tmp_path / metadata["portable_state"]["path"])
    neighbor = tmp_path / "station.0123456789abcdef.rank-00000.npz"
    neighbor.write_bytes(b"older generation not selected")
    _remove_transient_checkpoint(manifest, comm=MPI.COMM_SELF)
    assert not manifest.exists()
    assert all(not payload.exists() for payload in payloads)
    assert neighbor.exists()
