"""Batch topology validation retains scalar edge incidence semantics."""
import numpy as np
import pytest

from agentfem.boundary_models.rigid import TriangulatedRigidSurface


def test_large_open_grid_edge_inventory():
    n, m = 23, 17
    vertices = np.array([(i, j, 0.) for i in range(n) for j in range(m)])
    quads = np.array([(i*m+j, (i+1)*m+j, (i+1)*m+j+1, i*m+j+1)
                      for i in range(n-1) for j in range(m-1)])
    triangles = np.stack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), 1).reshape(-1, 3)
    surface = TriangulatedRigidSurface(vertices, triangles)
    uses = {}
    for row in triangles:
        for a, b in zip(row, np.roll(row, -1)):
            uses.setdefault(tuple(sorted((int(a), int(b)))), []).append(a < b)
    assert surface._topology_summary['edge_count'] == len(uses)
    assert surface._topology_summary['boundary_edge_count'] == sum(len(v) == 1 for v in uses.values())
    assert surface._topology_summary['closed'] is False
    assert all(len(v) == 1 or v[0] != v[1] for v in uses.values())


def test_first_input_conflicting_edge_is_retained():
    vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                         (1, 1, 1), (0, 0, 2), (1, 0, 2)], dtype=float)
    # The first encountered conflict is (3,4), not lexicographically first.
    triangles = np.array([(3, 4, 5), (3, 4, 2), (0, 1, 2), (0, 1, 5)])
    with pytest.raises(ValueError, match=r"orientation; first edge is \(3, 4\)"):
        TriangulatedRigidSurface(vertices, triangles)
