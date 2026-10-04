# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable, lossless scientific output for integration-point fields.

XDMF is a visualization view and cannot truthfully represent several values
inside one cell as a single cell attribute.  This module therefore stores raw
quadrature values in a separately versioned HDF5 dataset keyed by stable
physical-cell identity and reference-rule point index.  The archive is
independent of the MPI partition and can be read without DOLFINx.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import h5py
import numpy as np


_SCHEMA = "agentfem.integration-point-dataset.v1"


@dataclass(frozen=True)
class IntegrationPointArtifacts:
    """Files and semantics of one raw integration-point dataset."""

    hdf5: Path
    field_names: tuple[str, ...]
    rule_count: int
    cell_count: int
    integration_point_count: int

    def summary(self) -> dict[str, object]:
        return {
            "status": "completed",
            "schema": _SCHEMA,
            "artifact": str(self.hdf5),
            "storage": "root_gathered_cell_keyed_hdf5",
            "partition_independent": True,
            "field_names": self.field_names,
            "rule_count": self.rule_count,
            "cell_count": self.cell_count,
            "integration_point_count": self.integration_point_count,
            "visualization_role": "scientific_source_not_direct_xdmf_view",
        }


def _rule_key(sampling) -> str:
    digest = sha256()
    for values in (sampling.points, sampling.weights):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _selected_records(result, names) -> tuple[object, ...]:
    requested = tuple(str(name) for name in names)
    if requested:
        missing = tuple(name for name in requested if name not in result.fields)
        if missing:
            raise KeyError(
                "Unknown integration-point fields requested for output: "
                f"{missing!r}."
            )
        records = tuple(result.fields[name] for name in requested)
        invalid = tuple(
            item.name for item in records if item.location != "quadrature_points"
        )
        if invalid:
            raise ValueError(
                "Integration-point output only accepts fields located at "
                f"quadrature_points; got {invalid!r}."
            )
    else:
        records = tuple(result.fields.values())
    selected = tuple(
        item
        for item in records
        if item.location == "quadrature_points" and item.field is not None
    )
    if not selected:
        raise ValueError("No live integration-point fields are available.")
    missing_sampling = tuple(item.name for item in selected if item.sampling is None)
    if missing_sampling:
        raise ValueError(
            "Raw integration-point output requires an explicit quadrature "
            f"sampling contract for {missing_sampling!r}."
        )
    return selected


def write_integration_point_fields(result, path, *, names=()) -> IntegrationPointArtifacts:
    """Collectively write raw integration-point fields to portable HDF5.

    Every rank contributes owned cells only.  Rank zero sorts rows by original
    physical-cell identity before publishing the file atomically.  Multiple
    quadrature rules are stored as separate groups, so a result may combine
    providers without inventing one global point layout.
    """

    records = _selected_records(result, names)
    selected_path = Path(path)
    if selected_path.suffix.lower() not in {".h5", ".hdf5"}:
        selected_path = selected_path.with_suffix(".h5")
    domain = records[0].sampling.function.function_space.mesh
    comm = domain.comm
    from ..checkpointing import mesh_portable_identity

    mesh_identity = mesh_portable_identity(domain)
    groups: dict[str, list[object]] = {}
    for item in records:
        sampling = item.sampling
        if sampling.function.function_space.mesh is not domain:
            raise ValueError(
                "One integration-point artifact currently requires one mesh."
            )
        groups.setdefault(_rule_key(sampling), []).append(item)

    gathered_groups = []
    for key in sorted(groups):
        items = groups[key]
        sampling = items[0].sampling
        local_problem = None
        local_payload = None
        try:
            keys = np.asarray(sampling.owned_cell_keys, dtype=np.int64)
            points = np.asarray(sampling.owned_physical_points, dtype=float)
            weights = np.asarray(
                sampling.owned_physical_weights(), dtype=float
            ).reshape((len(keys), len(sampling.points)))
            local_fields = {}
            for item in items:
                current = item.sampling
                if not (
                    np.array_equal(current.points, sampling.points)
                    and np.array_equal(current.weights, sampling.weights)
                    and np.array_equal(current.owned_cell_keys, keys)
                ):
                    raise ValueError(
                        f"Integration-point field {item.name!r} does not share "
                        "the declared rule and owned-cell ordering."
                    )
                if "/" in item.name:
                    raise ValueError(
                        f"Integration-point field name {item.name!r} contains '/'."
                    )
                local_fields[item.name] = np.asarray(current.owned_values).reshape(
                    (len(keys), len(current.points), *current.value_shape)
                )
            local_payload = (keys, points, weights, local_fields)
        except Exception as exc:
            local_problem = f"{type(exc).__name__}: {exc}"
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise ValueError(
                f"Rank {rank}: integration-point output preparation failed: "
                f"{problems[rank]}"
            )
        gathered = comm.gather(local_payload, root=0)
        gathered_groups.append((key, sampling, items, gathered))

    error = None
    totals = None
    if comm.rank == 0:
        temporary = selected_path.with_name(f".{selected_path.name}.tmp")
        try:
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            total_cells = None
            reference_cell_ids = None
            total_points = 0
            with h5py.File(temporary, "w") as output:
                output.attrs["schema"] = _SCHEMA
                output.attrs["result_name"] = str(result.name)
                output.attrs["field_count"] = len(records)
                output.attrs["mesh_identity"] = json.dumps(
                    mesh_identity, sort_keys=True
                )
                rules = output.create_group("rules")
                for index, (key, sampling, items, gathered) in enumerate(
                    gathered_groups
                ):
                    all_keys = np.concatenate([part[0] for part in gathered])
                    order = np.argsort(all_keys, kind="stable")
                    all_keys = all_keys[order]
                    if np.unique(all_keys).size != all_keys.size:
                        raise ValueError(
                            "Integration-point output contains duplicate physical "
                            "cell identities."
                        )
                    all_points = np.concatenate([part[1] for part in gathered], axis=0)[
                        order
                    ]
                    all_weights = np.concatenate(
                        [part[2] for part in gathered], axis=0
                    )[order]
                    group = rules.create_group(f"rule-{index:03d}")
                    group.attrs["identity_sha256"] = key
                    group.attrs["points_per_cell"] = len(sampling.points)
                    group.create_dataset("cell_id", data=all_keys)
                    group.create_dataset(
                        "coordinates",
                        data=all_points,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
                    group.create_dataset(
                        "physical_weights",
                        data=all_weights,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
                    group.create_dataset(
                        "reference_points", data=np.asarray(sampling.points, dtype=float)
                    )
                    group.create_dataset(
                        "reference_weights", data=np.asarray(sampling.weights, dtype=float)
                    )
                    fields = group.create_group("fields")
                    for item in items:
                        values = np.concatenate(
                            [part[3][item.name] for part in gathered], axis=0
                        )[order]
                        dataset = fields.create_dataset(
                            item.name,
                            data=values,
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                        )
                        dataset.attrs["metadata"] = json.dumps(
                            item.as_dict(), sort_keys=True
                        )
                    if total_cells is None:
                        total_cells = len(all_keys)
                        reference_cell_ids = all_keys.copy()
                    elif total_cells != len(all_keys):
                        raise ValueError(
                            "Integration-point rules cover different cell counts."
                        )
                    elif not np.array_equal(reference_cell_ids, all_keys):
                        raise ValueError(
                            "Integration-point rules cover different physical cells."
                        )
                    total_points += len(all_keys) * len(sampling.points)
            os.replace(temporary, selected_path)
            totals = (int(total_cells or 0), total_points)
        except Exception as exc:  # pragma: no cover - filesystem failure
            if temporary.exists():
                temporary.unlink()
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"Integration-point result write failed: {error}")
    totals = comm.bcast(totals, root=0)
    comm.barrier()
    return IntegrationPointArtifacts(
        hdf5=selected_path,
        field_names=tuple(item.name for item in records),
        rule_count=len(groups),
        cell_count=int(totals[0]),
        integration_point_count=int(totals[1]),
    )


def read_integration_point_fields(path) -> dict[str, object]:
    """Read a portable archive into NumPy arrays without importing DOLFINx."""

    selected = Path(path)
    with h5py.File(selected, "r") as source:
        schema = str(source.attrs.get("schema", ""))
        if schema != _SCHEMA:
            raise ValueError(f"Unsupported integration-point schema {schema!r}.")
        rules = []
        for name in sorted(source["rules"]):
            group = source["rules"][name]
            fields = {}
            field_metadata = {}
            for field_name in sorted(group["fields"]):
                dataset = group["fields"][field_name]
                fields[field_name] = np.asarray(dataset).copy()
                field_metadata[field_name] = json.loads(dataset.attrs["metadata"])
            rules.append(
                {
                    "name": name,
                    "identity_sha256": str(group.attrs["identity_sha256"]),
                    "cell_id": np.asarray(group["cell_id"]).copy(),
                    "coordinates": np.asarray(group["coordinates"]).copy(),
                    "physical_weights": np.asarray(group["physical_weights"]).copy(),
                    "reference_points": np.asarray(group["reference_points"]).copy(),
                    "reference_weights": np.asarray(group["reference_weights"]).copy(),
                    "fields": fields,
                    "field_metadata": field_metadata,
                }
            )
        return {
            "schema": schema,
            "result_name": str(source.attrs.get("result_name", "")),
            "mesh_identity": json.loads(source.attrs["mesh_identity"]),
            "rules": tuple(rules),
        }


__all__ = [
    "IntegrationPointArtifacts",
    "read_integration_point_fields",
    "write_integration_point_fields",
]
