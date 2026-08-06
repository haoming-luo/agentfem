"""Restart envelopes shared by transient finite-element procedures.

The first schema deliberately stores rank-local state shards.  It is safe for
restart with the same mesh partition and MPI size, and records that boundary
explicitly instead of presenting partition-bound arrays as portable data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np

from . import fields


TRANSIENT_CHECKPOINT_SCHEMA = "agentfem.transient-checkpoint.v2"
_LEGACY_TRANSIENT_CHECKPOINT_SCHEMAS = {
    "agentfem.transient-checkpoint.v1",
    TRANSIENT_CHECKPOINT_SCHEMA,
}


@dataclass(frozen=True)
class CheckpointPolicy:
    """Automatic accepted-increment checkpoint cadence for transient steps."""

    every: int
    directory: Path
    final: bool = True
    prefix: str | None = None
    keep_last: int | None = None

    def __post_init__(self) -> None:
        interval = int(self.every)
        if interval <= 0:
            raise ValueError("CheckpointPolicy.every must be positive.")
        object.__setattr__(self, "every", interval)
        object.__setattr__(self, "directory", Path(self.directory))
        if self.prefix is not None and not str(self.prefix).strip():
            raise ValueError("CheckpointPolicy.prefix must be non-empty when supplied.")
        if self.keep_last is not None:
            selected = int(self.keep_last)
            if selected <= 0:
                raise ValueError("CheckpointPolicy.keep_last must be positive.")
            object.__setattr__(self, "keep_last", selected)

    def due(self, increment: int, total: int) -> bool:
        selected = int(increment)
        return selected % self.every == 0 or (self.final and selected == int(total))

    def path(self, *, step_name: str, increment: int) -> Path:
        base = self.prefix or _safe_name(step_name)
        return self.directory / f"{base}-inc-{int(increment):08d}"

    def summary(self) -> dict[str, object]:
        return {
            "kind": "checkpoint_policy",
            "every": self.every,
            "directory": str(self.directory),
            "final": bool(self.final),
            "prefix": self.prefix,
            "keep_last": self.keep_last,
            "retention": (
                "all_scheduled_checkpoints"
                if self.keep_last is None
                else f"latest_{self.keep_last}_scheduled_checkpoints"
            ),
        }


def every(
    increments: int,
    *,
    directory="checkpoints",
    final: bool = True,
    prefix: str | None = None,
    keep_last: int | None = None,
) -> CheckpointPolicy:
    """Create an automatic checkpoint policy for accepted time increments."""

    return CheckpointPolicy(
        every=increments,
        directory=Path(directory),
        final=final,
        prefix=prefix,
        keep_last=keep_last,
    )


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
    generation = comm.bcast(uuid4().hex[:16] if comm.rank == 0 else None, root=0)
    shard = manifest.with_name(
        f"{manifest.name.removesuffix('.checkpoint.json')}.{generation}."
        f"rank-{comm.rank:05d}.npz"
    )
    local_identity = None
    local_shard = None
    local_error = None
    try:
        atomic_savez(
            shard,
            **{name: value.x.array for name, value in functions.items()},
        )
        local_identity = {
            name: function_partition_identity(value)
            for name, value in functions.items()
        }
        local_shard = {
            "path": shard.name,
            "size": int(shard.stat().st_size),
            "sha256": _file_sha256(shard),
        }
    except Exception as exc:  # pragma: no cover - injected in MPI regression
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(comm, "write state shard", local_error)
    identities = comm.gather(local_identity, root=0)
    shards = comm.gather(local_shard, root=0)
    metadata = {
        "schema": TRANSIENT_CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": generation,
        "software": {"name": "AgentFEM", "version": _software_version()},
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
            metadata["shards"] = list(shards)
            metadata["state_identity_by_rank"] = list(identities)
            atomic_write_text(
                manifest,
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
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
    stored_schema = metadata.get("schema")
    if stored_schema not in _LEGACY_TRANSIENT_CHECKPOINT_SCHEMAS:
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
            current_identity = (
                _legacy_function_partition_identity(function)
                if stored_schema == "agentfem.transient-checkpoint.v1"
                else function_partition_identity(function)
            )
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
    shard_record = metadata["shards"][comm.rank]
    if isinstance(shard_record, str):
        shard = manifest.parent / shard_record
        expected_size = None
        expected_digest = None
    else:
        shard = manifest.parent / shard_record["path"]
        expected_size = int(shard_record["size"])
        expected_digest = str(shard_record["sha256"])
    restored = None
    shard_error = None
    try:
        if expected_size is not None and shard.stat().st_size != expected_size:
            raise ValueError("checkpoint shard size does not match its manifest")
        if expected_digest is not None and _file_sha256(shard) != expected_digest:
            raise ValueError("checkpoint shard checksum does not match its manifest")
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


def _remove_transient_checkpoint(path, *, comm) -> None:
    """Collectively remove one published manifest and exactly its shards.

    The manifest is removed last. This narrow operation is used only by an
    explicit retention policy after a newer checkpoint has been published.
    """

    manifest = _manifest_path(path)
    error = None
    if comm.rank == 0:
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            for record in metadata.get("shards", ()):
                name = record if isinstance(record, str) else record["path"]
                shard = manifest.parent / name
                if shard.exists():
                    shard.unlink()
            manifest.unlink()
        except Exception as exc:  # pragma: no cover - filesystem failure
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"Transient checkpoint removal failed: {error}")
    comm.barrier()


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
    topology = domain.topology
    topology.create_connectivity(topology.dim, 0)
    cell_vertices = topology.connectivity(topology.dim, 0)
    digest = sha256()
    digest.update(geometry.tobytes())
    digest.update(geometry_map.tobytes())
    digest.update(np.asarray(cell_vertices.array).tobytes())
    digest.update(np.asarray(cell_vertices.offsets).tobytes())
    cell_type = str(topology.cell_name())
    digest.update(cell_type.encode("utf-8"))
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
        "mesh_cell_type": cell_type,
        "mesh_partition_hash": digest.hexdigest(),
    }


def _legacy_function_partition_identity(function) -> dict[str, object]:
    """Reproduce the geometry-only identity written by checkpoint schema v1."""

    function = fields.unwrap(function)
    V = function.function_space
    index_map = V.dofmap.index_map
    domain = V.mesh
    digest = sha256()
    digest.update(np.asarray(domain.geometry.x).tobytes())
    digest.update(np.asarray(domain.geometry.dofmaps[0]).tobytes())
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


def atomic_savez(path, **arrays) -> Path:
    """Atomically publish one NumPy archive in its destination directory."""

    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=selected.parent,
            prefix=f".{selected.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(selected)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return selected


def atomic_write_text(path, content: str) -> Path:
    """Atomically publish UTF-8 text in its destination directory."""

    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=selected.parent,
            prefix=f".{selected.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(selected)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return selected


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _software_version() -> str:
    from . import __version__

    return str(__version__)


def _safe_name(value: str) -> str:
    selected = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value).strip()
    ).strip("-")
    return selected or "step"


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
    "CheckpointPolicy",
    "TRANSIENT_CHECKPOINT_SCHEMA",
    "atomic_savez",
    "atomic_write_text",
    "function_partition_identity",
    "every",
    "load_transient_checkpoint",
    "save_transient_checkpoint",
]
