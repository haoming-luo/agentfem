# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.contact_projection_checkpoint import (
    global_projection_state_snapshot,
    local_projection_state_from_snapshot,
    numeric_projection_state_snapshot,
    numeric_projection_state_from_snapshot,
)


def _record(point_ids=(12, 4)):
    point_ids = np.asarray(point_ids, dtype=np.int64)
    count = point_ids.size
    query = np.column_stack(
        (np.linspace(0.1, 0.2, count), np.zeros(count), np.zeros(count))
    )
    projection = boundary_models.SurfaceProjection(
        surface_name="tool",
        surface_kind="triangulated_rigid_surface",
        query_points=query,
        closest_points=query.copy(),
        normals=np.tile((0.0, 0.0, 1.0), (count, 1)),
        signed_gaps=np.zeros(count),
        valid=np.ones(count, dtype=bool),
        status_codes=np.full(count, "ok"),
        method="reviewed_test_projection",
        entity_ids=np.arange(100, 100 + count, dtype=np.int64),
        geometry_fingerprint="1" * 64,
        local_coordinates=np.tile((0.2, 0.3, 0.5), (count, 1)),
        local_coordinate_system="triangle_barycentric_connectivity_order",
    )
    return boundary_models.ContactProjectionRecord(point_ids, projection)


def test_projection_checkpoint_roundtrips_by_stable_point_identity():
    accepted = _record()
    snapshot = global_projection_state_snapshot(accepted, MPI.COMM_SELF)
    restored = local_projection_state_from_snapshot(
        snapshot,
        point_ids=(12, 4),
    )

    np.testing.assert_array_equal(restored.point_ids, accepted.point_ids)
    np.testing.assert_array_equal(
        restored.projection.entity_ids,
        accepted.projection.entity_ids,
    )
    np.testing.assert_allclose(
        restored.projection.local_coordinates,
        accepted.projection.local_coordinates,
    )
    assert snapshot["records"][0]["point_id"] == 4
    assert snapshot["records"][0]["entity_id"] == 101


def test_projection_checkpoint_rejects_missing_and_corrupt_point_evidence():
    snapshot = global_projection_state_snapshot(_record(), MPI.COMM_SELF)
    missing = deepcopy(snapshot)
    missing["records"].pop()
    with pytest.raises(ValueError, match="lacks local point IDs"):
        local_projection_state_from_snapshot(missing, point_ids=(4, 12))

    corrupt = deepcopy(snapshot)
    corrupt["records"][0]["local_coordinates"] = None
    with pytest.raises(ValueError, match="lacks local coordinates"):
        local_projection_state_from_snapshot(corrupt, point_ids=(4, 12))

    ambiguous_flag = deepcopy(snapshot)
    ambiguous_flag["metadata"]["has_entity_ids"] = "yes"
    with pytest.raises(TypeError, match="flags must be booleans"):
        local_projection_state_from_snapshot(ambiguous_flag, point_ids=(4, 12))


def test_projection_checkpoint_is_rank_canonical_and_redistributable():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Projection checkpoint redistribution is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    accepted = _record((20 + comm.rank,))
    snapshot = global_projection_state_snapshot(accepted, comm)
    copies = comm.allgather(snapshot)

    assert all(item == copies[0] for item in copies)
    other_id = 20 + (1 - comm.rank)
    restored = local_projection_state_from_snapshot(
        snapshot,
        point_ids=(other_id,),
    )
    assert restored.point_ids.tolist() == [other_id]


def test_numeric_projection_roundtrip_through_common_array_codec():
    from agentfem._checkpoint_arrays import encode_tree, decode_tree

    record = _record()
    numeric = numeric_projection_state_snapshot(record)
    encoded, arrays = encode_tree(numeric)
    restored = numeric_projection_state_from_snapshot(
        decode_tree(encoded, arrays), point_ids=(12, 4)
    )
    np.testing.assert_array_equal(
        restored.projection.query_points, record.projection.query_points
    )
    np.testing.assert_array_equal(
        restored.projection.local_coordinates, record.projection.local_coordinates
    )
    assert "records" not in numeric
    assert len(arrays) > 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("status_indices", np.array([0, 42])),
        ("status_catalog", ["ok", "ok"]),
        ("point_ids", np.array([4, 99])),
        ("valid", np.array([1, 1])),
        ("normals", np.full((2, 3), np.nan)),
    ],
)
def test_numeric_projection_rejects_corruption(field, value):
    snapshot = numeric_projection_state_snapshot(_record())
    snapshot[field] = value
    with pytest.raises((ValueError, TypeError)):
        numeric_projection_state_from_snapshot(snapshot, point_ids=(12, 4))


def test_numeric_projection_preserves_invalid_geometry_without_nan_archive():
    cylinder = boundary_models.rigid_cylinder(
        axis_point=(0, 0, 0),
        axis_direction=(0, 0, 1),
        radius=1,
        axial_interval=(0, 2),
    )
    projection = cylinder.project(np.array([[2.0, 0.0, 1.0], [2.0, 0.0, 3.0]]))
    record = boundary_models.ContactProjectionRecord((0, 1), projection)
    snapshot = numeric_projection_state_snapshot(record)
    assert not snapshot["valid"][1]
    assert np.all(np.isfinite(snapshot["closest_points"]))
    restored = numeric_projection_state_from_snapshot(snapshot, point_ids=(0, 1))
    np.testing.assert_array_equal(restored.projection.valid, record.projection.valid)
    np.testing.assert_array_equal(
        restored.projection.status_codes, record.projection.status_codes
    )
    assert np.all(np.isnan(restored.projection.closest_points[1]))
    corrupt = deepcopy(snapshot)
    corrupt["closest_points"][1, 0] = 1
    with pytest.raises(ValueError, match="zero-filled"):
        numeric_projection_state_from_snapshot(corrupt, point_ids=(0, 1))
