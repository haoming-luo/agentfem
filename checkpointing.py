"""Restart envelopes shared by transient finite-element procedures.

Fast rank-local shards remain the default.  Schema v3 can additionally store
an explicit coordinate-keyed nodal state for restart across MPI partitions and
rank counts; constitutive integration-point state is not implied by that
portable nodal contract.
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
from mpi4py import MPI

from . import fields


TRANSIENT_CHECKPOINT_SCHEMA = "agentfem.transient-checkpoint.v3"
_LEGACY_TRANSIENT_CHECKPOINT_SCHEMAS = {
    "agentfem.transient-checkpoint.v1",
    "agentfem.transient-checkpoint.v2",
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
    portable: bool = False

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
            "portable": bool(self.portable),
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
    portable: bool = False,
) -> CheckpointPolicy:
    """Create an automatic checkpoint policy for accepted time increments."""

    return CheckpointPolicy(
        every=increments,
        directory=Path(directory),
        final=final,
        prefix=prefix,
        keep_last=keep_last,
        portable=portable,
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
    auxiliary_state: dict[str, object] | None = None,
    portable: bool = False,
):
    """Write a transient restart, optionally with partition-independent state."""

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
    portable_record = None
    portable_identities = None
    portability = "same mesh partition and MPI size"
    if portable:
        portable_record, portable_identities = _write_portable_state(
            manifest,
            generation=generation,
            functions=functions,
            comm=comm,
        )
        portability = (
            "nodal state portable across MPI partitions and rank counts; "
            "rank shards retained for same-partition restart"
        )
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
        "portable": bool(portable_record is not None),
        "portability": portability,
        "state_names": list(functions),
        "accepted_times": [float(value) for value in accepted_times],
        "execution_events": [
            event.as_dict() if hasattr(event, "as_dict") else dict(event)
            for event in execution_events
        ],
        "history_records": [dict(item) for item in history_records],
        "auxiliary_state": (
            None
            if auxiliary_state is None
            else json.loads(json.dumps(auxiliary_state, sort_keys=True))
        ),
    }
    root_error = None
    if comm.rank == 0:
        try:
            metadata["shards"] = list(shards)
            metadata["state_identity_by_rank"] = list(identities)
            if portable_record is not None:
                metadata["portable_state"] = portable_record
                metadata["portable_state_identity"] = portable_identities
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
        "state names": (tuple(metadata.get("state_names", ())), tuple(functions)),
    }
    for label, (stored, current) in checks.items():
        if stored != current:
            raise ValueError(
                f"Transient checkpoint {label} differs: stored={stored!r}, "
                f"current={current!r}."
            )
    same_rank_count = int(metadata.get("rank_count")) == int(comm.size)
    partition_compatible = same_rank_count
    if same_rank_count:
        try:
            stored_identity = metadata["state_identity_by_rank"][comm.rank]
            for name, function in functions.items():
                current_identity = (
                    _legacy_function_partition_identity(function)
                    if stored_schema == "agentfem.transient-checkpoint.v1"
                    else function_partition_identity(function)
                )
                if stored_identity[name] != current_identity:
                    partition_compatible = False
        except (KeyError, IndexError, TypeError):
            partition_compatible = False
    partition_compatible = bool(
        comm.allreduce(bool(partition_compatible), op=MPI.LAND)
    )
    if partition_compatible:
        _restore_partition_shard(
            manifest,
            metadata=metadata,
            functions=functions,
            comm=comm,
        )
        metadata["restart_mode"] = "same_partition_rank_shard"
    elif metadata.get("portable") and metadata.get("portable_state"):
        _restore_portable_state(
            manifest,
            metadata=metadata,
            functions=functions,
            comm=comm,
        )
        metadata["restart_mode"] = "portable_coordinate_keyed_state"
    else:
        raise ValueError(
            "Transient checkpoint MPI size or mesh partition differs and no "
            "portable state was written. Save with portable=True to permit "
            "cross-partition restart."
        )
    metadata["manifest_path"] = str(manifest)
    return metadata


def _write_portable_state(manifest, *, generation, functions, comm):
    """Write a coordinate-keyed nodal state alongside rank-local shards."""

    local_payload = None
    local_error = None
    try:
        local_payload = {
            name: _portable_local_field(function)
            for name, function in functions.items()
        }
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(
        comm,
        "prepare portable state",
        local_error,
    )
    identities = {
        name: function_portable_identity(function)
        for name, function in functions.items()
    }
    gathered = comm.gather(local_payload, root=0)
    response = None
    if comm.rank == 0:
        portable_path = manifest.with_name(
            f"{manifest.name.removesuffix('.checkpoint.json')}.{generation}."
            "portable.npz"
        )
        try:
            arrays = {}
            index = {}
            for field_index, name in enumerate(functions):
                modes = {item[name]["key_mode"] for item in gathered}
                if len(modes) != 1:
                    raise ValueError(
                        f"Portable field {name!r} has inconsistent key modes."
                    )
                coordinates = np.concatenate(
                    [item[name]["coordinates"] for item in gathered], axis=0
                )
                values = np.concatenate(
                    [item[name]["values"] for item in gathered], axis=0
                )
                order = _coordinate_order(coordinates)
                coordinates = coordinates[order]
                values = values[order]
                if _has_duplicate_coordinates(coordinates):
                    raise ValueError(
                        f"Portable field {name!r} has duplicate owned dof coordinates."
                    )
                coordinate_key = f"field_{field_index}_coordinates"
                value_key = f"field_{field_index}_values"
                arrays[coordinate_key] = coordinates
                arrays[value_key] = values
                index[name] = {
                    "coordinates": coordinate_key,
                    "values": value_key,
                    "rows": int(len(coordinates)),
                    "components": int(values.shape[1]),
                    "key_mode": modes.pop(),
                }
            atomic_savez(portable_path, **arrays)
            response = {
                "record": {
                    "path": portable_path.name,
                    "size": int(portable_path.stat().st_size),
                    "sha256": _file_sha256(portable_path),
                    "storage": "root_gathered_coordinate_keyed_npz",
                    "index": index,
                },
                "identities": identities,
                "error": None,
            }
        except Exception as exc:
            response = {
                "record": None,
                "identities": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    response = comm.bcast(response, root=0)
    if response["error"] is not None:
        raise RuntimeError(
            "Transient checkpoint portable state write failed: "
            f"{response['error']}"
        )
    return response["record"], response["identities"]


def _restore_partition_shard(manifest, *, metadata, functions, comm) -> None:
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
    error = None
    try:
        _validate_checkpoint_file(
            shard,
            expected_size=expected_size,
            expected_digest=expected_digest,
        )
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
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(comm, "read state shard", error)
    for name, function in functions.items():
        function.x.array[:] = restored[name]
        function.x.scatter_forward()


def _restore_portable_state(manifest, *, metadata, functions, comm) -> None:
    record = metadata["portable_state"]
    portable_path = manifest.parent / record["path"]
    error = None
    restored = None
    try:
        _validate_checkpoint_file(
            portable_path,
            expected_size=int(record["size"]),
            expected_digest=str(record["sha256"]),
        )
        expected_identity = metadata["portable_state_identity"]
        for name, function in functions.items():
            if expected_identity[name] != function_portable_identity(function):
                raise ValueError(
                    f"portable mesh/function identity for {name!r} differs"
                )
        with np.load(portable_path, allow_pickle=False) as data:
            restored = {}
            for name, function in functions.items():
                selected = record["index"][name]
                stored_coordinates = np.asarray(data[selected["coordinates"]])
                stored_values = np.asarray(data[selected["values"]])
                local = _portable_local_field(function)
                if selected.get("key_mode") != local["key_mode"]:
                    raise ValueError(
                        f"portable state key mode for {name!r} differs"
                    )
                lookup = {
                    row.tobytes(): index
                    for index, row in enumerate(stored_coordinates)
                }
                indices = []
                for row in local["coordinates"]:
                    key = row.tobytes()
                    if key not in lookup:
                        raise ValueError(
                            f"portable state for {name!r} lacks a local dof coordinate"
                        )
                    indices.append(lookup[key])
                restored[name] = stored_values[np.asarray(indices, dtype=np.int64)]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_checkpoint_error(comm, "read portable state", error)
    for name, function in functions.items():
        V = function.function_space
        owned = int(V.dofmap.index_map.size_local)
        block_size = int(V.dofmap.index_map_bs)
        function.x.array[: owned * block_size] = restored[name].reshape(-1)
        function.x.scatter_forward()


def _portable_local_field(function) -> dict[str, np.ndarray]:
    function = fields.unwrap(function)
    V = function.function_space
    index_map = V.dofmap.index_map
    owned = int(index_map.size_local)
    block_size = int(V.dofmap.index_map_bs)
    if int(V.dofmap.bs) != block_size:
        raise NotImplementedError(
            "Portable checkpoints currently require a blocked nodal space, "
            "not a mixed/subspace dof layout."
        )
    coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    if coordinates.shape[0] < owned:
        raise ValueError("Function space does not expose every owned dof coordinate.")
    coordinates = np.ascontiguousarray(
        coordinates[:owned, : int(V.mesh.geometry.dim)]
    )
    coordinate_keys = _quantized_coordinate_keys(coordinates, V.mesh)
    gathered_keys = V.mesh.comm.allgather(coordinate_keys)
    global_keys = np.concatenate(gathered_keys, axis=0)
    if _has_duplicate_coordinates(global_keys):
        source_ids = _owned_p1_input_node_ids(function)
        coordinate_keys = np.column_stack((coordinate_keys, source_ids))
        key_mode = "quantized_physical_dof_coordinate_and_input_node_id"
        gathered_augmented = V.mesh.comm.allgather(coordinate_keys)
        if _has_duplicate_coordinates(np.concatenate(gathered_augmented, axis=0)):
            raise NotImplementedError(
                "Portable state cannot distinguish coincident owned dofs even "
                "after adding their input-node identities."
            )
    else:
        key_mode = "quantized_physical_dof_coordinate_and_block_component"
    values = np.asarray(
        function.x.array[: owned * block_size]
    ).reshape((owned, block_size)).copy()
    return {
        "coordinates": coordinate_keys,
        "values": values,
        "key_mode": key_mode,
    }


def function_portable_identity(function) -> dict[str, object]:
    """Return an MPI-partition-independent identity for a nodal field."""

    function = fields.unwrap(function)
    V = function.function_space
    local = _portable_local_field(function)
    counts = V.mesh.comm.allgather(int(len(local["coordinates"])))
    return {
        "element": str(V.ufl_element()),
        "value_shape": list(function.ufl_shape),
        "block_size": int(V.dofmap.index_map_bs),
        "global_block_dofs": int(sum(counts)),
        "mesh": mesh_portable_identity(V.mesh),
        "key": local["key_mode"],
    }


def _owned_p1_input_node_ids(function) -> np.ndarray:
    """Map owned blocked P1 dofs to durable mesh-input node ids."""

    V = function.function_space
    domain = V.mesh
    owned = int(V.dofmap.index_map.size_local)
    if int(V.dofmap.bs) != int(V.dofmap.index_map_bs):
        raise NotImplementedError(
            "Coincident portable state requires a blocked nodal space."
        )
    geometry_maps = getattr(domain.geometry, "dofmaps", None)
    geometry_dofmap = (
        domain.geometry.dofmap if geometry_maps is None else geometry_maps[0]
    )
    input_indices = np.asarray(domain.geometry.input_global_indices, dtype=np.int64)
    source = np.full(owned, -1, dtype=np.int64)
    cell_map = domain.topology.index_map(domain.topology.dim)
    for cell in range(int(cell_map.size_local + cell_map.num_ghosts)):
        geometry_dofs = np.asarray(geometry_dofmap[cell], dtype=int)
        field_dofs = np.asarray(V.dofmap.cell_dofs(cell), dtype=int)
        if geometry_dofs.size != field_dofs.size:
            raise NotImplementedError(
                "Coincident portable keys require first-order nodal geometry "
                "and a first-order blocked field."
            )
        for geometry_dof, field_dof in zip(
            geometry_dofs, field_dofs, strict=True
        ):
            if int(field_dof) >= owned:
                continue
            node = int(input_indices[int(geometry_dof)])
            previous = int(source[int(field_dof)])
            if previous not in {-1, node}:
                raise RuntimeError(
                    "One owned field dof maps to inconsistent input-node ids."
                )
            source[int(field_dof)] = node
    missing = np.flatnonzero(source < 0)
    if missing.size:
        raise NotImplementedError(
            "Coincident portable state lacks input identity for owned dofs: "
            f"{missing.tolist()}."
        )
    return source


def mesh_portable_identity(domain) -> dict[str, object]:
    """Hash cell geometry independently of local numbering and partition."""

    topology = domain.topology
    cell_map = topology.index_map(topology.dim)
    geometry_dofmap = np.asarray(domain.geometry.dofmaps[0])
    geometry = np.asarray(domain.geometry.x)[:, : int(domain.geometry.dim)]
    coordinate_policy = _coordinate_key_policy(domain)
    local_signatures = []
    for cell in range(int(cell_map.size_local)):
        coordinates = np.asarray(geometry[geometry_dofmap[cell]], dtype=np.float64)
        coordinates = _quantized_coordinate_keys(
            coordinates,
            domain,
            policy=coordinate_policy,
        )
        coordinates = coordinates[_coordinate_order(coordinates)]
        digest = sha256()
        digest.update(np.ascontiguousarray(coordinates).tobytes())
        local_signatures.append(digest.hexdigest())
    signatures = []
    for rank_values in domain.comm.allgather(local_signatures):
        signatures.extend(rank_values)
    signatures.sort()
    digest = sha256()
    digest.update(str(topology.cell_name()).encode("utf-8"))
    for signature in signatures:
        digest.update(signature.encode("ascii"))
    return {
        "topology_dimension": int(topology.dim),
        "geometry_dimension": int(domain.geometry.dim),
        "cell_type": str(topology.cell_name()),
        "global_cells": int(len(signatures)),
        "geometry_connectivity_hash": digest.hexdigest(),
        "coordinate_key": "relative_bounds_scaled_int64",
    }


def _coordinate_key_policy(domain) -> tuple[np.ndarray, float]:
    geometry = np.asarray(domain.geometry.x)[:, : int(domain.geometry.dim)]
    gdim = int(domain.geometry.dim)
    local_min = np.min(geometry, axis=0) if len(geometry) else np.full(gdim, np.inf)
    local_max = np.max(geometry, axis=0) if len(geometry) else np.full(gdim, -np.inf)
    global_min = np.empty(gdim, dtype=np.float64)
    global_max = np.empty(gdim, dtype=np.float64)
    domain.comm.Allreduce(local_min, global_min, op=MPI.MIN)
    domain.comm.Allreduce(local_max, global_max, op=MPI.MAX)
    span = float(np.max(global_max - global_min))
    coordinate_scale = max(
        span,
        float(np.max(np.abs(global_min))),
        float(np.max(np.abs(global_max))),
        np.finfo(np.float64).tiny,
    )
    tolerance = max(
        np.finfo(np.float64).tiny,
        64.0 * np.finfo(np.float64).eps * coordinate_scale,
    )
    return global_min, tolerance


def _quantized_coordinate_keys(coordinates, domain, *, policy=None) -> np.ndarray:
    """Return partition-stable integer keys tolerant of mesh-build roundoff."""

    selected = np.asarray(coordinates, dtype=np.float64)
    gdim = int(domain.geometry.dim)
    global_min, tolerance = policy or _coordinate_key_policy(domain)
    return np.rint((selected[:, :gdim] - global_min) / tolerance).astype(np.int64)


def _coordinate_order(coordinates) -> np.ndarray:
    selected = np.asarray(coordinates)
    if selected.ndim != 2:
        raise ValueError("Coordinate keys require a two-dimensional array.")
    return np.lexsort(
        tuple(selected[:, axis] for axis in reversed(range(selected.shape[1])))
    )


def _has_duplicate_coordinates(coordinates) -> bool:
    selected = np.asarray(coordinates)
    if len(selected) < 2:
        return False
    ordered = selected[_coordinate_order(selected)]
    return bool(np.any(np.all(ordered[1:] == ordered[:-1], axis=1)))


def _validate_checkpoint_file(path, *, expected_size, expected_digest) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError("checkpoint shard size does not match its manifest")
    if expected_digest is not None and _file_sha256(path) != expected_digest:
        raise ValueError("checkpoint shard checksum does not match its manifest")


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
            portable = metadata.get("portable_state")
            if portable:
                portable_path = manifest.parent / portable["path"]
                if portable_path.exists():
                    portable_path.unlink()
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
    "function_portable_identity",
    "mesh_portable_identity",
    "every",
    "load_transient_checkpoint",
    "save_transient_checkpoint",
]
