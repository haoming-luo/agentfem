"""Restart envelopes shared by transient finite-element procedures.

The first schema deliberately stores rank-local state shards.  It is safe for
restart with the same mesh partition and MPI size, and records that boundary
explicitly instead of presenting partition-bound arrays as portable data.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import fields


TRANSIENT_CHECKPOINT_SCHEMA = "agentfem.transient-checkpoint.v1"


def save_transient_checkpoint(
    path,
    *,
    step_kind: str,
    step_name: str,
    procedure,
    dt: float,
    total_steps: int,
    completed_steps: int,
    state: dict[str, object],
    accepted_times=(),
    execution_events=(),
    history_records=(),
):
    """Write one partition-bound transient restart and return its manifest."""

    functions = {name: fields.unwrap(value) for name, value in state.items()}
    if not functions:
        raise ValueError("A transient checkpoint requires at least one state field.")
    comm = next(iter(functions.values())).function_space.mesh.comm
    manifest = _manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shard = manifest.with_name(
        f"{manifest.name.removesuffix('.checkpoint.json')}.rank-{comm.rank:05d}.npz"
    )
    local_identity = None
    local_error = None
    try:
        np.savez(shard, **{name: value.x.array for name, value in functions.items()})
        local_identity = {
            name: function_partition_identity(value)
            for name, value in functions.items()
        }
    except Exception as exc:  # pragma: no cover - injected in MPI regression
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(comm, "write state shard", local_error)
    identities = comm.gather(local_identity, root=0)
    shard_names = comm.gather(shard.name, root=0)
    metadata = {
        "schema": TRANSIENT_CHECKPOINT_SCHEMA,
        "step_kind": str(step_kind),
        "step_name": str(step_name),
        "procedure": (
            procedure.summary() if hasattr(procedure, "summary") else procedure
        ),
        "dt": float(dt),
        "total_steps": int(total_steps),
        "completed_steps": int(completed_steps),
        "time": float(completed_steps) * float(dt),
        "rank_count": int(comm.size),
        "portable": False,
        "portability": "same mesh partition and MPI size",
        "state_names": list(functions),
        "accepted_times": [float(value) for value in accepted_times],
        "execution_events": [
            event.as_dict() if hasattr(event, "as_dict") else dict(event)
            for event in execution_events
        ],
        "history_records": [dict(item) for item in history_records],
    }
    root_error = None
    if comm.rank == 0:
        try:
            metadata["shards"] = list(shard_names)
            metadata["state_identity_by_rank"] = list(identities)
            manifest.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - filesystem failure
            root_error = f"{type(exc).__name__}: {exc}"
    root_error = comm.bcast(root_error, root=0)
    if root_error is not None:
        raise RuntimeError(f"Transient checkpoint manifest write failed: {root_error}")
    return manifest


def load_transient_checkpoint(
    path,
    *,
    step_kind: str,
    step_name: str,
    procedure,
    dt: float,
    total_steps: int,
    state: dict[str, object],
) -> dict[str, object]:
    """Restore a transient state after validating its scientific identity."""

    functions = {name: fields.unwrap(value) for name, value in state.items()}
    comm = next(iter(functions.values())).function_space.mesh.comm
    manifest = _manifest_path(path)
    payload = None
    if comm.rank == 0:
        try:
            payload = {
                "metadata": json.loads(manifest.read_text(encoding="utf-8")),
                "error": None,
            }
        except Exception as exc:
            payload = {
                "metadata": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    payload = comm.bcast(payload, root=0)
    if payload["error"] is not None:
        raise RuntimeError(
            f"Transient checkpoint manifest read failed: {payload['error']}"
        )
    metadata = payload["metadata"]
    if metadata.get("schema") != TRANSIENT_CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported transient checkpoint schema.")
    expected_procedure = (
        procedure.summary() if hasattr(procedure, "summary") else procedure
    )
    expected_procedure = json.loads(json.dumps(expected_procedure, sort_keys=True))
    checks = {
        "step kind": (metadata.get("step_kind"), str(step_kind)),
        "step name": (metadata.get("step_name"), str(step_name)),
        "procedure": (metadata.get("procedure"), expected_procedure),
        "time increment": (float(metadata.get("dt")), float(dt)),
        "total steps": (int(metadata.get("total_steps")), int(total_steps)),
        "MPI size": (int(metadata.get("rank_count")), int(comm.size)),
        "state names": (tuple(metadata.get("state_names", ())), tuple(functions)),
    }
    for label, (stored, current) in checks.items():
        if stored != current:
            raise ValueError(
                f"Transient checkpoint {label} differs: stored={stored!r}, "
                f"current={current!r}."
            )
    identity_error = None
    try:
        stored_identity = metadata["state_identity_by_rank"][comm.rank]
        for name, function in functions.items():
            current_identity = function_partition_identity(function)
            if stored_identity[name] != current_identity:
                raise ValueError(
                    f"state layout for {name!r} differs from the current "
                    "mesh partition/function space"
                )
    except Exception as exc:
        identity_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(
        comm,
        "validate state identity",
        identity_error,
    )
    shard = manifest.parent / metadata["shards"][comm.rank]
    restored = None
    shard_error = None
    try:
        with np.load(shard, allow_pickle=False) as data:
            restored = {}
            for name, function in functions.items():
                values = np.asarray(data[name]).copy()
                if values.shape != function.x.array.shape:
                    raise ValueError(
                        f"array for {name!r} has an incompatible shape"
                    )
                restored[name] = values
    except Exception as exc:
        shard_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(comm, "read state shard", shard_error)
    for name, function in functions.items():
        function.x.array[:] = restored[name]
        function.x.scatter_forward()
    metadata["manifest_path"] = str(manifest)
    return metadata


def _manifest_path(path) -> Path:
    selected = Path(path)
    text = selected.name
    if text.endswith(".checkpoint.json"):
        return selected
    if selected.suffix:
        selected = selected.with_suffix("")
    return selected.with_name(selected.name + ".checkpoint.json")


def function_partition_identity(function) -> dict[str, object]:
    """Return a JSON-safe identity for one field on one mesh partition."""

    function = fields.unwrap(function)
    V = function.function_space
    index_map = V.dofmap.index_map
    domain = V.mesh
    geometry = np.asarray(domain.geometry.x)
    geometry_map = np.asarray(domain.geometry.dofmaps[0])
    digest = sha256()
    digest.update(geometry.tobytes())
    digest.update(geometry_map.tobytes())
    return {
        "element": str(V.ufl_element()),
        "value_shape": list(function.ufl_shape),
        "block_size": int(V.dofmap.index_map_bs),
        "owned_dofs": int(index_map.size_local),
        "global_dofs": int(index_map.size_global),
        "local_range": [int(value) for value in index_map.local_range],
        "array_size": int(function.x.array.size),
        "mesh_topology_dimension": int(domain.topology.dim),
        "mesh_geometry_dimension": int(domain.geometry.dim),
        "mesh_partition_hash": digest.hexdigest(),
    }


def _raise_collective_checkpoint_error(comm, action: str, local_error) -> None:
    errors = comm.allgather(local_error)
    failures = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(
            f"Transient checkpoint could not {action}; " + "; ".join(failures)
        )


__all__ = [
    "TRANSIENT_CHECKPOINT_SCHEMA",
    "function_partition_identity",
    "load_transient_checkpoint",
    "save_transient_checkpoint",
]
