"""Partition-independent state identity for cohesive interface points."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from .checkpointing import atomic_savez, atomic_write_text


COHESIVE_CHECKPOINT_SCHEMA = "agentfem.cohesive-checkpoint.v2"
LEGACY_COHESIVE_CHECKPOINT_SCHEMA = "agentfem.cohesive-checkpoint.v1"


@dataclass(frozen=True)
class FacetOwnership:
    """Deterministic owner selection for locally visible physical facets."""

    local_keys: tuple[str, ...]
    owner_by_key: dict[str, int]
    rank: int
    size: int

    @property
    def owned_mask(self) -> np.ndarray:
        return np.asarray(
            [self.owner_by_key[key] == self.rank for key in self.local_keys],
            dtype=bool,
        )

    @property
    def global_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.owner_by_key))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "deterministic_cohesive_facet_ownership",
            "rank": self.rank,
            "rank_count": self.size,
            "local_visible_facets": len(self.local_keys),
            "local_owned_facets": int(np.count_nonzero(self.owned_mask)),
            "global_facets": len(self.owner_by_key),
            "policy": "hash_preference_among_ranks_where_facet_is_visible",
        }


def deterministic_facet_ownership(topology, *, comm=MPI.COMM_WORLD) -> FacetOwnership:
    """Assign every physical facet to one rank that can assemble it.

    The preferred owner comes from the physical key hash, then indexes the
    sorted ranks on which that facet is visible.  This remains deterministic
    for one partition while allowing ghost overlap.  A changed partition may
    choose a different owner; state portability therefore follows the key,
    never the rank number.
    """

    identity = topology.identity()
    if identity.get("scope") != "ordered_reference_facet_geometry":
        raise ValueError(
            "Distributed cohesive ownership requires physical facet keys; "
            "construct topology with pair_coincident_line_facets()."
        )
    local = tuple(str(key) for key in topology.facet_keys)
    if len(local) != len(set(local)):
        raise ValueError("Local cohesive facet keys must be unique.")
    visible = comm.allgather(local)
    candidates: dict[str, list[int]] = {}
    for rank, keys in enumerate(visible):
        for key in keys:
            candidates.setdefault(str(key), []).append(int(rank))
    owners = {}
    for key, ranks in candidates.items():
        selected = tuple(sorted(set(ranks)))
        preferred = int(key[:16], 16) % len(selected)
        owners[key] = selected[preferred]
    return FacetOwnership(local, owners, int(comm.rank), int(comm.size))


def save_portable_cohesive_state(
    path,
    topology,
    transaction,
    *,
    comm=MPI.COMM_WORLD,
) -> Path:
    """Collectively save committed cohesive state once per physical facet."""

    state = getattr(transaction, "state", transaction)
    if int(state.size) != int(topology.number_of_points):
        raise ValueError("Cohesive transaction size does not match paired topology.")
    ownership = deterministic_facet_ownership(topology, comm=comm)
    points = int(topology.quadrature_points_per_facet)
    state_factory = getattr(state, "state_arrays", None)
    if state_factory is None or not callable(state_factory):
        local_state = {
            "maximum_opening": np.asarray(state.committed_maximum, dtype=float)
        }
    else:
        local_state = state_factory()
    if not local_state or "maximum_opening" not in local_state:
        raise ValueError("Cohesive state must include maximum_opening.")
    local_values = {}
    for name, values in sorted(local_state.items()):
        selected = np.asarray(values, dtype=float)
        if selected.shape != (int(state.size),) or np.any(~np.isfinite(selected)):
            raise ValueError(f"Cohesive state field {name!r} is invalid.")
        local_values[name] = selected.reshape((-1, points))
    owned = ownership.owned_mask
    local_keys = tuple(str(key) for key in topology.facet_keys)
    payload = {
        "visible": (
            local_keys,
            {name: values.copy() for name, values in local_values.items()},
        ),
        "owned": (
            tuple(np.asarray(local_keys, dtype=str)[owned].tolist()),
            {name: values[owned].copy() for name, values in local_values.items()},
        ),
    }
    gathered = comm.gather(payload, root=0)
    law_records = comm.allgather(state.law.summary())
    if any(record != law_records[0] for record in law_records[1:]):
        raise ValueError("All ranks must use the same cohesive law contract.")
    manifest, archive = _paths(path, create_parent=True)
    error = None
    if comm.rank == 0:
        try:
            field_names = tuple(sorted(local_values))
            visible_records: dict[str, dict[str, np.ndarray]] = {
                name: {} for name in field_names
            }
            for rank_payload in gathered:
                keys, fields = rank_payload["visible"]
                if tuple(sorted(fields)) != field_names:
                    raise ValueError("Cohesive state fields differ between ranks.")
                for name in field_names:
                    for key, value in zip(keys, fields[name], strict=True):
                        selected = np.asarray(value, dtype=float)
                        previous = visible_records[name].get(key)
                        if previous is not None and not np.array_equal(previous, selected):
                            raise ValueError(
                                "Cohesive state differs between ranks for physical "
                                f"facet {key}, field {name}; synchronize owner/ghost "
                                "state before save."
                            )
                        visible_records[name][key] = selected
            records: dict[str, dict[str, np.ndarray]] = {
                name: {} for name in field_names
            }
            for rank_payload in gathered:
                keys, fields = rank_payload["owned"]
                for name in field_names:
                    for key, value in zip(keys, fields[name], strict=True):
                        if key in records[name]:
                            raise ValueError(
                                f"Cohesive facet {key} has more than one owner."
                            )
                        records[name][key] = np.asarray(value, dtype=float)
            expected = set(ownership.global_keys)
            if any(set(records[name]) != expected for name in field_names):
                raise ValueError("Owned cohesive facets do not cover the global interface.")
            ordered = tuple(sorted(expected))
            archive_fields = {
                f"state__{name}": np.asarray(
                    [records[name][key] for key in ordered], dtype=float
                )
                for name in field_names
            }
            atomic_savez(
                archive,
                schema=np.asarray(COHESIVE_CHECKPOINT_SCHEMA),
                facet_keys=np.asarray(ordered, dtype="U64"),
                state_fields=np.asarray(field_names, dtype="U64"),
                law_json=np.asarray(json.dumps(law_records[0], sort_keys=True)),
                **archive_fields,
            )
            metadata = {
                "schema": COHESIVE_CHECKPOINT_SCHEMA,
                "parallel_contract": "physical_facet_keyed_state",
                "writer_rank_count": int(comm.size),
                "facets": len(ordered),
                "quadrature_points_per_facet": points,
                "state_fields": list(field_names),
                "global_interface_sha256": _keys_digest(ordered),
                "law": law_records[0],
                "archive": {
                    "path": archive.name,
                    "size": int(archive.stat().st_size),
                    "sha256": _file_digest(archive),
                },
            }
            atomic_write_text(
                manifest,
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
        except Exception as exc:  # pragma: no cover - MPI error propagation
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"Portable cohesive checkpoint write failed: {error}")
    comm.barrier()
    return manifest


def load_portable_cohesive_state(
    path,
    topology,
    transaction,
    *,
    comm=MPI.COMM_WORLD,
) -> dict[str, object]:
    """Collectively restore cohesive state across facet order and rank count."""

    state = getattr(transaction, "state", transaction)
    if int(state.size) != int(topology.number_of_points):
        raise ValueError("Cohesive transaction size does not match paired topology.")
    ownership = deterministic_facet_ownership(topology, comm=comm)
    manifest, _ = _paths(path)
    payload = None
    if comm.rank == 0:
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            schema = metadata.get("schema")
            if schema not in {
                COHESIVE_CHECKPOINT_SCHEMA,
                LEGACY_COHESIVE_CHECKPOINT_SCHEMA,
            }:
                raise ValueError("Unsupported cohesive checkpoint schema.")
            archive = manifest.parent / metadata["archive"]["path"]
            if int(archive.stat().st_size) != int(metadata["archive"]["size"]):
                raise ValueError("Cohesive checkpoint archive size differs.")
            if _file_digest(archive) != metadata["archive"]["sha256"]:
                raise ValueError("Cohesive checkpoint archive checksum differs.")
            with np.load(archive, allow_pickle=False) as stored:
                if str(stored["schema"]) != schema:
                    raise ValueError("Cohesive archive schema differs from its manifest.")
                keys = tuple(str(value) for value in stored["facet_keys"].tolist())
                law = json.loads(str(stored["law_json"]))
                if schema == LEGACY_COHESIVE_CHECKPOINT_SCHEMA:
                    state_fields = ("maximum_opening",)
                    values_by_field = {
                        "maximum_opening": np.asarray(
                            stored["maximum_opening"], dtype=float
                        )
                    }
                else:
                    state_fields = tuple(
                        str(value) for value in stored["state_fields"].tolist()
                    )
                    values_by_field = {
                        name: np.asarray(stored[f"state__{name}"], dtype=float)
                        for name in state_fields
                    }
            points = int(topology.quadrature_points_per_facet)
            if any(
                values.shape != (len(keys), points)
                or np.any(~np.isfinite(values))
                for values in values_by_field.values()
            ):
                raise ValueError("Cohesive checkpoint state array is invalid.")
            current_fields = tuple(sorted(state.state_arrays()))
            if tuple(sorted(state_fields)) != current_fields:
                raise ValueError(
                    "Cohesive checkpoint state fields differ from current law."
                )
            if int(metadata.get("quadrature_points_per_facet", -1)) != points:
                raise ValueError("Cohesive checkpoint quadrature contract differs.")
            if _keys_digest(keys) != metadata["global_interface_sha256"]:
                raise ValueError("Cohesive checkpoint interface identity differs.")
            if law != state.law.summary() or law != metadata["law"]:
                raise ValueError("Cohesive checkpoint law differs from current law.")
            current = set(ownership.global_keys)
            if set(keys) != current:
                missing = sorted(set(keys) - current)
                extra = sorted(current - set(keys))
                raise ValueError(
                    "Cohesive checkpoint physical interface differs: "
                    f"missing_current={missing}, extra_current={extra}."
                )
            payload = {
                "metadata": metadata,
                "values": {
                    name: {
                        key: values_by_field[name][index].tolist()
                        for index, key in enumerate(keys)
                    }
                    for name in state_fields
                },
            }
        except Exception as exc:  # pragma: no cover - collective error path
            payload = {"error": f"{type(exc).__name__}: {exc}"}
    payload = comm.bcast(payload, root=0)
    if "error" in payload:
        raise RuntimeError(f"Portable cohesive checkpoint read failed: {payload['error']}")
    local = {
        name: np.asarray(
            [records[key] for key in topology.facet_keys],
            dtype=float,
        ).reshape(-1)
        for name, records in payload["values"].items()
    }
    restore = getattr(state, "restore_state_arrays", None)
    if restore is None or not callable(restore):
        state.initialize(local["maximum_opening"])
    else:
        restore(local)
    metadata = dict(payload["metadata"])
    metadata["reader_rank_count"] = int(comm.size)
    metadata["local_facets"] = int(topology.number_of_facets)
    return metadata


def _paths(path, *, create_parent: bool = False) -> tuple[Path, Path]:
    selected = Path(path)
    if selected.name.endswith(".cohesive.json"):
        manifest = selected
        base = selected.name.removesuffix(".cohesive.json")
    else:
        manifest = selected.with_name(selected.name + ".cohesive.json")
        base = selected.name
    if create_parent:
        manifest.parent.mkdir(parents=True, exist_ok=True)
    return manifest, manifest.with_name(base + ".cohesive.npz")


def _keys_digest(keys) -> str:
    digest = sha256()
    for key in keys:
        digest.update(str(key).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "COHESIVE_CHECKPOINT_SCHEMA",
    "LEGACY_COHESIVE_CHECKPOINT_SCHEMA",
    "FacetOwnership",
    "deterministic_facet_ownership",
    "load_portable_cohesive_state",
    "save_portable_cohesive_state",
]
