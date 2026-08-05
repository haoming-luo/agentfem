from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI
import pytest

from agentfem import mesh, results


def test_point_and_path_sampling_are_partition_independent():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed point sampling requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (8, 4),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    field = fem.Function(V, name="response")
    field.interpolate(lambda x: np.vstack((x[0] + x[1], x[0] - x[1])))
    field.x.scatter_forward()

    points = np.array(
        ((0.0, 0.0), (0.5, 0.25), (1.0, 0.5), (1.5, 0.75), (2.0, 1.0))
    )
    expected = np.column_stack(
        (points[:, 0] + points[:, 1], points[:, 0] - points[:, 1])
    )
    np.testing.assert_allclose(
        results.sample_points(field, points),
        expected,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        results.probe(field, at=(1.0, 0.5)),
        [1.5, 0.5],
    )
    path = results.sample_path(
        field,
        start=(0.0, 0.0),
        end=(2.0, 1.0),
        count=5,
    )
    np.testing.assert_allclose(path.values, expected, atol=1.0e-14)


def test_point_sampling_rejects_rank_inconsistent_requests():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed point sampling requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    field = fem.Function(V, name="rank_safe")
    field.interpolate(lambda x: x[0])
    point = ((0.25 + 0.1 * MPI.COMM_WORLD.rank, 0.5),)

    with pytest.raises(ValueError, match="identical coordinates"):
        results.sample_points(field, point)
