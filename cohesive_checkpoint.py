"""Partition-independent state identity for cohesive interface points."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from .checkpointing import atomic_savez, atomic_write_text


COHESIVE_CHECKPOINT_SCHEMA = "agentfem.cohesive-checkpoint.v1"


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
    local_values = np.asarray(state.committed_maximum, dtype=float).reshape(
        (-1, points)
    )
    owned = ownership.owned_mask
    local_keys = tuple(str(key) for key in topology.facet_keys)
    payload = {
        "visible": (local_keys, local_values.copy()),
        "owned": (
            tuple(np.asarray(local_keys, dtype=str)[owned].tolist()),
            local_values[owned].copy(),
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
            visible_records: dict[str, np.ndarray] = {}
            for rank_payload in gathered:
                keys, values = rank_payload["visible"]
                for key, value in zip(keys, values, strict=True):
                    selected = np.asarray(value, dtype=float)
                    previous = visible_records.get(key)
                    if previous is not None and not np.array_equal(previous, selected):
                        raise ValueError(
                            "Cohesive state differs between ranks for physical "
                            f"facet {key}; synchronize owner/ghost state before save."
                        )
                    visible_records[key] = selected
            records: dict[str, np.ndarray] = {}
            for rank_payload in gathered:
                keys, values = rank_payload["owned"]
                for key, value in zip(keys, values, strict=True):
                    if key in records:
                        raise ValueError(f"Cohesive facet {key} has more than one owner.")
                    records[key] = np.asarray(value, dtype=float)
            expected = set(ownership.global_keys)
            if set(records) != expected:
                raise ValueError("Owned cohesive facets do not cover the global interface.")
            ordered = tuple(sorted(records))
            values = np.asarray([records[key] for key in ordered], dtype=float)
            atomic_savez(
                archive,
                schema=np.asarray(COHESIVE_CHECKPOINT_SCHEMA),
                facet_keys=np.asarray(ordered, dtype="U64"),
                maximum_opening=values,
                law_json=np.asarray(json.dumps(law_records[0], sort_keys=True)),
            )
            metadata = {
                "schema": COHESIVE_CHECKPOINT_SCHEMA,
                "parallel_contract": "physical_facet_keyed_state",
                "writer_rank_count": int(comm.size),
                "facets": len(ordered),
                "quadrature_points_per_facet": points,
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
            if metadata.get("schema") != COHESIVE_CHECKPOINT_SCHEMA:
                raise ValueError("Unsupported cohesive checkpoint schema.")
            archive = manifest.parent / metadata["archive"]["path"]
            if int(archive.stat().st_size) != int(metadata["archive"]["size"]):
                raise ValueError("Cohesive checkpoint archive size differs.")
            if _file_digest(archive) != metadata["archive"]["sha256"]:
                raise ValueError("Cohesive checkpoint archive checksum differs.")
            with np.load(archive, allow_pickle=False) as stored:
                if str(stored["schema"]) != COHESIVE_CHECKPOINT_SCHEMA:
                    raise ValueError("Cohesive archive schema differs from its manifest.")
                keys = tuple(str(value) for value in stored["facet_keys"].tolist())
                values = np.asarray(stored["maximum_opening"], dtype=float)
                law = json.loads(str(stored["law_json"]))
            points = int(topology.quadrature_points_per_facet)
            if values.shape != (len(keys), points) or np.any(~np.isfinite(values)):
                raise ValueError("Cohesive checkpoint state array is invalid.")
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
            payload = {key: values[index].tolist() for index, key in enumerate(keys)}
            payload = {"metadata": metadata, "values": payload}
        except Exception as exc:  # pragma: no cover - collective error path
            payload = {"error": f"{type(exc).__name__}: {exc}"}
    payload = comm.bcast(payload, root=0)
    if "error" in payload:
        raise RuntimeError(f"Portable cohesive checkpoint read failed: {payload['error']}")
    local = np.asarray(
        [payload["values"][key] for key in topology.facet_keys],
        dtype=float,
    )
    state.initialize(local.reshape(-1))
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
    "FacetOwnership",
    "deterministic_facet_ownership",
    "load_portable_cohesive_state",
    "save_portable_cohesive_state",
]
