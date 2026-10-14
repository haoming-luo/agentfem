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


def test_cached_topology_does_not_cache_coordinate_validity():
    from agentfem.boundary_models._triangle_topology import _cached_check
    _cached_check.cache_clear()
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    triangles = np.array([[0, 1, 2]])
    original = TriangulatedRigidSurface(vertices, triangles)
    moved = TriangulatedRigidSurface(vertices + [1., 2., 3.], triangles)
    assert _cached_check.cache_info().hits == 1
    assert original.geometry_fingerprint != moved.geometry_fingerprint
    np.testing.assert_array_equal(original.facet_normals, moved.facet_normals)
    # The connectivity remains valid while the current geometry degenerates.
    invalid = vertices.copy(); invalid[2] = [2., 0, 0]
    with pytest.raises(ValueError, match="degenerate"):
        TriangulatedRigidSurface(invalid, triangles)
    invalid[2] = invalid[1]
    with pytest.raises(ValueError, match="duplicate vertices"):
        TriangulatedRigidSurface(invalid, triangles)


def test_cache_keys_connectivity_bytes_and_vertex_count():
    from agentfem.boundary_models._triangle_topology import reviewed_triangle_topology
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    assert reviewed_triangle_topology(3, indices) == (3, 3)
    indices[0, 2] = 3
    with pytest.raises(ValueError, match="outside"):
        reviewed_triangle_topology(3, indices)
    with pytest.raises(ValueError, match="Every surface vertex"):
        reviewed_triangle_topology(4, indices)


def test_q1_refit_preserves_first_occurrence_topology_under_rotation():
    from agentfem._closed_q1_surface import ClosedQ1Surface
    from agentfem.boundary_models._triangle_topology import _cached_check
    from test_closed_q1_surface import solid
    x, cells = solid()
    _cached_check.cache_clear()
    before = ClosedQ1Surface(x, cells)
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
    current = x @ rotation.T + [3., 4., 5.]
    after = before.refitted(current)
    assert _cached_check.cache_info().hits >= 1
    np.testing.assert_array_equal(before.search.tree.surface.triangles,
                                  after.search.tree.surface.triangles)
    fresh = ClosedQ1Surface(current, cells)
    points = np.array([[-.1, .3, .4], [.2, .3, .1]]) @ rotation.T + [3., 4., 5.]
    a, b = (s.project(points, maximum_distance=.5) for s in (after, fresh))
    for name in ("points", "normals", "signed_gaps", "weights"):
        np.testing.assert_array_equal(a[name], b[name])
