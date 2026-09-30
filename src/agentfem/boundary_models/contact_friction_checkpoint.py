# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Partition-independent checkpoint helpers for tangential contact State."""

from __future__ import annotations

import numpy as np

from .contact_friction import (
    TangentialContactRecord,
    TangentialKinematicRecord,
)


_SCHEMA = "agentfem.global-explicit-friction-state.v1"


def global_friction_state_snapshot(friction, kinematics, communicator):
    """Gather point-keyed local State into one rank-canonical snapshot."""

    if friction is None or kinematics is None:
        raise RuntimeError("Friction checkpoint lacks an accepted local State.")
    if not np.array_equal(friction.point_ids, kinematics.point_ids):
        raise RuntimeError("Friction constitutive and kinematic identities differ.")
    local = []
    for index, point_id in enumerate(friction.point_ids):
        local.append(
            {
                "point_id": int(point_id),
                "normal": friction.normals[index].tolist(),
                "elastic_slip": friction.elastic_slips[index].tolist(),
                "cumulative_dissipation_density": float(
                    friction.cumulative_dissipation_densities[index]
                ),
                "cumulative_separation_release_density": float(
                    friction.cumulative_separation_release_densities[index]
                ),
                "slave_position": kinematics.slave_positions[index].tolist(),
            }
        )
    records = sorted(
        (item for shard in communicator.allgather(local) for item in shard),
        key=lambda item: item["point_id"],
    )
    point_ids = [item["point_id"] for item in records]
    if len(set(point_ids)) != len(point_ids):
        raise RuntimeError("Friction checkpoint contains duplicate global point IDs.")
    factors = tuple(
        float(value)
        for value in communicator.allgather(kinematics.motion_factor)
    )
    if not np.allclose(factors, factors[0], rtol=0.0, atol=1.0e-14):
        raise RuntimeError("Friction motion factor differs across MPI ranks.")
    return {"schema": _SCHEMA, "motion_factor": factors[0], "records": records}


def local_friction_state_from_snapshot(snapshot, *, point_ids, dimension: int):
    """Validate a global snapshot and select the current rank's point IDs."""

    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"schema", "motion_factor", "records"}
        or snapshot.get("schema") != _SCHEMA
        or not isinstance(snapshot["records"], list)
    ):
        raise ValueError("Unsupported global explicit-friction State.")
    by_id = {}
    required = {
        "point_id",
        "normal",
        "elastic_slip",
        "cumulative_dissipation_density",
        "cumulative_separation_release_density",
        "slave_position",
    }
    for item in snapshot["records"]:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("Explicit-friction point record fields differ.")
        point_id = int(item["point_id"])
        if point_id in by_id:
            raise ValueError("Explicit-friction State repeats one point ID.")
        by_id[point_id] = item
    local_ids = np.asarray(point_ids, dtype=np.int64)
    missing = [int(point_id) for point_id in local_ids if int(point_id) not in by_id]
    if missing:
        raise ValueError(
            "Explicit-friction State lacks local point IDs: " + repr(missing[:8])
        )
    selected = [by_id[int(point_id)] for point_id in local_ids]
    shape = (-1, int(dimension))
    friction = TangentialContactRecord(
        point_ids=local_ids,
        normals=np.asarray(
            [item["normal"] for item in selected], dtype=float
        ).reshape(shape),
        elastic_slips=np.asarray(
            [item["elastic_slip"] for item in selected], dtype=float
        ).reshape(shape),
        cumulative_dissipation_densities=np.asarray(
            [item["cumulative_dissipation_density"] for item in selected],
            dtype=float,
        ),
        cumulative_separation_release_densities=np.asarray(
            [
                item["cumulative_separation_release_density"]
                for item in selected
            ],
            dtype=float,
        ),
    )
    kinematics = TangentialKinematicRecord(
        point_ids=local_ids,
        slave_positions=np.asarray(
            [item["slave_position"] for item in selected], dtype=float
        ).reshape(shape),
        motion_factor=snapshot["motion_factor"],
    )
    return friction, kinematics


__all__ = [
    "global_friction_state_snapshot",
    "local_friction_state_from_snapshot",
]
