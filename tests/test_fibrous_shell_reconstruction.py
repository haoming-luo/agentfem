# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import mechanics


def _cell_centroids(domain):
    cell_map = domain.topology.index_map(domain.topology.dim)
    cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    return dolfinx_mesh.compute_midpoints(
        domain,
        domain.topology.dim,
        cells,
    )[:, :2]


def _rotation():
    axis = np.array((1.0, 2.0, -0.5))
    axis /= np.linalg.norm(axis)
    angle = 0.7
    skew = np.array(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _curvature_case(n, *, rotation=None):
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        n,
        n,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    points = _cell_centroids(domain)
    rate = 0.8
    angle = rate * points[:, 0]
    directions = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    tangents = np.broadcast_to(
        np.array(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        (len(points), 3, 2),
    ).copy()
    if rotation is not None:
        directions = directions @ rotation.T
        tangents = np.einsum("ij,cjk->cik", rotation, tangents)
    result = mechanics.reconstruct_fiber_curvature(
        domain,
        directions,
        tangents,
        rings=2,
    )
    exact = rate * np.cos(angle[: result.in_plane_curvature.size])
    error = np.linalg.norm(result.in_plane_curvature - exact) / np.linalg.norm(exact)
    return result, error


def test_reconstructed_fiber_curvature_is_objective_and_separates_channels():
    baseline, _ = _curvature_case(8)
    rotated, _ = _curvature_case(8, rotation=_rotation())

    np.testing.assert_allclose(
        rotated.in_plane_curvature,
        baseline.in_plane_curvature,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(baseline.normal_curvature, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(rotated.normal_curvature, 0.0, atol=2.0e-13)
    assert baseline.as_dict()["maturity"].endswith("not_shell_equilibrium")


def test_reconstructed_fiber_curvature_converges_for_smooth_rotation_field():
    _, coarse = _curvature_case(4)
    _, medium = _curvature_case(8)
    _, fine = _curvature_case(16)

    assert medium < coarse
    assert fine < medium
    assert fine < 2.0e-3


def test_reconstructed_fiber_curvature_rejects_nonparameter_mesh_contract():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    with pytest.raises(ValueError, match="two-dimensional parameter mesh"):
        mechanics.reconstruct_fiber_curvature(
            domain,
            np.zeros((6, 3)),
            np.zeros((6, 3, 2)),
        )
