# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Partition-independent accepted closest-point evidence for checkpoints.

The snapshot is keyed by stable slave-point identity rather than rank-local
array position.  It preserves the accepted master entity, local coordinates,
gap, normal, and status for audit and finite-sliding continuity.  Restoring
this evidence never replaces the mandatory closest-point search performed by
the next residual evaluation.
"""

from __future__ import annotations

import numpy as np

from .contact_state import ContactProjectionRecord
from .rigid import SurfaceProjection


_SCHEMA = "agentfem.global-contact-projection-state.v1"


def _metadata(record: ContactProjectionRecord) -> dict[str, object]:
    projection = record.projection
    return {
        "surface_name": projection.surface_name,
        "surface_kind": projection.surface_kind,
        "dimension": projection.dimension,
        "method": projection.method,
        "geometry_fingerprint": projection.geometry_fingerprint,
        "has_entity_ids": projection.entity_ids is not None,
        "has_local_coordinates": projection.local_coordinates is not None,
        "local_coordinate_size": (
            None
            if projection.local_coordinates is None
            else int(projection.local_coordinates.shape[1])
        ),
        "local_coordinate_system": projection.local_coordinate_system,
    }


def global_projection_state_snapshot(record, communicator) -> dict[str, object]:
    """Gather one accepted projection into a rank-canonical point snapshot."""

    if not isinstance(record, ContactProjectionRecord):
        raise TypeError("Projection checkpoint requires ContactProjectionRecord.")
    metadata = _metadata(record)
    metadata_copies = tuple(communicator.allgather(metadata))
    if any(item != metadata_copies[0] for item in metadata_copies[1:]):
        raise RuntimeError("Projection checkpoint surface metadata differs by rank.")

    projection = record.projection
    local = []
    for index, point_id in enumerate(record.point_ids):
        valid = bool(projection.valid[index])
        local.append(
            {
                "point_id": int(point_id),
                "query_point": projection.query_points[index].tolist(),
                "closest_point": (
                    projection.closest_points[index].tolist() if valid else None
                ),
                "normal": projection.normals[index].tolist() if valid else None,
                "signed_gap": (
                    float(projection.signed_gaps[index]) if valid else None
                ),
                "valid": valid,
                "status_code": str(projection.status_codes[index]),
                "entity_id": (
                    None
                    if projection.entity_ids is None
                    else int(projection.entity_ids[index])
                ),
                "local_coordinates": (
                    None
                    if projection.local_coordinates is None or not valid
                    else projection.local_coordinates[index].tolist()
                ),
            }
        )
    records = sorted(
        (item for shard in communicator.allgather(local) for item in shard),
        key=lambda item: item["point_id"],
    )
    point_ids = [item["point_id"] for item in records]
    if len(set(point_ids)) != len(point_ids):
        raise RuntimeError("Projection checkpoint repeats a global contact point ID.")
    return {
        "schema": _SCHEMA,
        "metadata": metadata_copies[0],
        "records": records,
    }


def local_projection_state_from_snapshot(
    snapshot,
    *,
    point_ids,
) -> ContactProjectionRecord:
    """Validate a global snapshot and select one rank's current point shard."""

    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"schema", "metadata", "records"}
        or snapshot.get("schema") != _SCHEMA
        or not isinstance(snapshot["metadata"], dict)
        or not isinstance(snapshot["records"], list)
    ):
        raise ValueError("Unsupported global contact-projection State.")
    metadata = snapshot["metadata"]
    metadata_fields = {
        "surface_name",
        "surface_kind",
        "dimension",
        "method",
        "geometry_fingerprint",
        "has_entity_ids",
        "has_local_coordinates",
        "local_coordinate_size",
        "local_coordinate_system",
    }
    if set(metadata) != metadata_fields:
        raise ValueError("Contact-projection metadata fields differ.")
    dimension = int(metadata["dimension"])
    if dimension not in {2, 3}:
        raise ValueError("Contact-projection dimension must be two or three.")
    if not isinstance(metadata["has_entity_ids"], bool) or not isinstance(
        metadata["has_local_coordinates"], bool
    ):
        raise TypeError("Contact-projection capability flags must be booleans.")
    has_entities = metadata["has_entity_ids"]
    has_local = metadata["has_local_coordinates"]
    local_size = metadata["local_coordinate_size"]
    if has_local:
        local_size = int(local_size)
        if not has_entities or local_size < 1:
            raise ValueError("Local projection coordinates require entity identity.")
        if not str(metadata["local_coordinate_system"] or "").strip():
            raise ValueError("Local projection coordinates require a named system.")
    elif local_size is not None or metadata["local_coordinate_system"] is not None:
        raise ValueError("Absent local coordinates cannot declare local metadata.")

    record_fields = {
        "point_id",
        "query_point",
        "closest_point",
        "normal",
        "signed_gap",
        "valid",
        "status_code",
        "entity_id",
        "local_coordinates",
    }
    by_id: dict[int, dict[str, object]] = {}
    for item in snapshot["records"]:
        if not isinstance(item, dict) or set(item) != record_fields:
            raise ValueError("Contact-projection point fields differ.")
        if isinstance(item["point_id"], (bool, np.bool_)) or not isinstance(
            item["point_id"], (int, np.integer)
        ):
            raise TypeError("Contact-projection point IDs must be integers.")
        if not isinstance(item["valid"], (bool, np.bool_)):
            raise TypeError("Contact-projection validity must be boolean.")
        if not isinstance(item["status_code"], str) or not item[
            "status_code"
        ]:
            raise ValueError("Contact-projection status codes must be non-empty.")
        point_id = int(item["point_id"])
        if point_id < 0 or point_id in by_id:
            raise ValueError("Contact-projection point identity is invalid or repeated.")
        by_id[point_id] = item

    local_ids = np.asarray(point_ids)
    if local_ids.ndim != 1 or (
        local_ids.size and local_ids.dtype.kind not in {"i", "u"}
    ):
        raise TypeError("Local contact point IDs must be one integer vector.")
    if local_ids.dtype.kind == "u" and np.any(
        local_ids > np.iinfo(np.int64).max
    ):
        raise ValueError("Local contact point IDs must fit signed 64-bit identity.")
    local_ids = local_ids.astype(np.int64, copy=False)
    missing = [int(point_id) for point_id in local_ids if int(point_id) not in by_id]
    if missing:
        raise ValueError(
            "Contact-projection State lacks local point IDs: " + repr(missing[:8])
        )
    selected = [by_id[int(point_id)] for point_id in local_ids]
    count = len(selected)
    query = np.asarray(
        [item["query_point"] for item in selected], dtype=float
    ).reshape(count, dimension)
    valid = np.asarray([item["valid"] for item in selected], dtype=bool)
    closest = np.full((count, dimension), np.nan, dtype=float)
    normals = np.full((count, dimension), np.nan, dtype=float)
    gaps = np.full(count, np.nan, dtype=float)
    for index, item in enumerate(selected):
        if valid[index]:
            if (
                item["closest_point"] is None
                or item["normal"] is None
                or item["signed_gap"] is None
            ):
                raise ValueError("Valid projection record lacks geometric values.")
            closest[index] = np.asarray(item["closest_point"], dtype=float)
            normals[index] = np.asarray(item["normal"], dtype=float)
            gaps[index] = float(item["signed_gap"])
        elif any(
            item[name] is not None
            for name in ("closest_point", "normal", "signed_gap")
        ):
            raise ValueError("Invalid projection record must omit geometric values.")

    entity_ids = None
    if has_entities:
        if any(item["entity_id"] is None for item in selected):
            raise ValueError("Discrete projection record lacks an entity ID.")
        entity_ids = np.asarray(
            [item["entity_id"] for item in selected], dtype=np.int64
        )
    elif any(item["entity_id"] is not None for item in selected):
        raise ValueError("Analytical projection record cannot contain entity IDs.")

    local_coordinates = None
    if has_local:
        local_coordinates = np.full((count, local_size), np.nan, dtype=float)
        for index, item in enumerate(selected):
            raw = item["local_coordinates"]
            if valid[index]:
                if raw is None:
                    raise ValueError("Valid projection lacks local coordinates.")
                local_coordinates[index] = np.asarray(raw, dtype=float)
            elif raw is not None:
                raise ValueError("Invalid projection cannot contain local coordinates.")
    elif any(item["local_coordinates"] is not None for item in selected):
        raise ValueError("Projection record cannot contain undeclared local values.")

    projection = SurfaceProjection(
        surface_name=metadata["surface_name"],
        surface_kind=metadata["surface_kind"],
        query_points=query,
        closest_points=closest,
        normals=normals,
        signed_gaps=gaps,
        valid=valid,
        status_codes=[item["status_code"] for item in selected],
        method=metadata["method"],
        entity_ids=entity_ids,
        geometry_fingerprint=metadata["geometry_fingerprint"],
        local_coordinates=local_coordinates,
        local_coordinate_system=metadata["local_coordinate_system"],
    )
    return ContactProjectionRecord(point_ids=local_ids, projection=projection)


__all__ = [
    "global_projection_state_snapshot",
    "local_projection_state_from_snapshot",
]
