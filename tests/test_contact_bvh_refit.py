import numpy as np
import pytest

from agentfem._closed_q1_surface import ClosedQ1Surface
from agentfem.boundary_models.rigid import TriangulatedRigidSurface
from agentfem.boundary_models.search import TriangleSurfaceBVH
from test_rigid_surface_search import _grid_surface
from test_closed_q1_surface import solid


def test_refit_matches_rebuild_after_large_deformation_without_mutating_old_tree():
    surface = _grid_surface(9)
    old = TriangleSurfaceBVH(surface)
    bounds = tuple(a.copy() for a in old.bounds)
    xyz = surface.vertices.copy()
    xyz[:, 0] += .6*np.sin(3*xyz[:, 1])
    xyz[:, 2] += .4*xyz[:, 0]*xyz[:, 1]
    moved = TriangulatedRigidSurface(xyz, surface.triangles, facet_ids=surface.facet_ids)
    refit, fresh = old._refitted(moved), TriangleSurfaceBVH(moved)
    rng = np.random.default_rng(34)
    for _ in range(100):
        a, b = rng.uniform(-.5, 2, (2, 3))
        np.testing.assert_array_equal(refit.segment_overlapping_facets(a,b),
                                      fresh.segment_overlapping_facets(a,b))
        lo, hi = np.minimum(a,b), np.maximum(a,b)
        np.testing.assert_array_equal(refit.overlapping_facets(lo,hi),
                                      fresh.overlapping_facets(lo,hi))
    for a,b in zip(old.bounds,bounds,strict=True):
        np.testing.assert_array_equal(a,b)
    assert not refit._lower.flags.writeable
    assert not np.shares_memory(old._lower,refit._lower)
    with pytest.raises(ValueError,match='identities'):
        old._refitted(_grid_surface(2))


@pytest.mark.parametrize('concave',[False,True])
def test_closed_refit_current_geometry_and_rollback_match_fresh_search(concave):
    xyz, faces = solid(concave)
    old = ClosedQ1Surface(xyz,faces)
    queries = np.array([[.2,.3,.1],[-.1,.3,.4],[-.1,-.2,.4]])
    baseline = old.project(queries,maximum_distance=.5)
    moved = xyz.copy()
    moved[:,2] += .1*moved[:,0]*moved[:,1]
    moved += [4,-2,1]
    current = old.refitted(moved)
    fresh = ClosedQ1Surface(moved,faces)
    query = queries + [4,-2,1]
    a,b = current.project(query,maximum_distance=.5),fresh.project(query,maximum_distance=.5)
    for name in ('signed_gaps','points','normals','facet_indices','weights'):
        np.testing.assert_array_equal(a[name],b[name])
    restored = current.refitted(xyz).project(queries,maximum_distance=.5)
    for name in ('signed_gaps','points','normals','facet_indices','weights'):
        np.testing.assert_array_equal(restored[name],baseline[name])
    for bad in (xyz[:-1], xyz*complex(1,1), np.full_like(xyz,np.nan)):
        with pytest.raises(ValueError): old.refitted(bad)


def test_single_leaf_refit():
    surface = TriangulatedRigidSurface(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
    tree = TriangleSurfaceBVH(surface)
    moved = TriangulatedRigidSurface(surface.vertices+[0,0,2],surface.triangles)
    answer = tree._refitted(moved)
    np.testing.assert_array_equal(answer.bounds,[[0,0,2],[1,1,2]])


def test_q1_refit_reuses_only_unchanged_exact_weld_classes(monkeypatch):
    from agentfem import _q1_surface_search as module
    from test_q1_surface_search import square
    vertices = np.array([square(), square() + [1, 0, 0]])
    original_weld = module._weld_coordinates
    calls = []
    def tracked(points):
        calls.append(len(points))
        return original_weld(points)
    monkeypatch.setattr(module, '_weld_coordinates', tracked)
    original = module.Q1SurfaceSearch(vertices)
    moved = module.Q1SurfaceSearch(vertices + [3, 4, 5], _previous=original)
    assert calls == [8]
    assert not moved._weld_inverse.flags.writeable
    # Changing coordinate equivalence without changing facet count must not
    # silently retain the old welding. This generic open patch has no node IDs.
    split = vertices.copy(); split[1, :, 2] += .1
    split_search = module.Q1SurfaceSearch(split, _previous=original)
    assert calls == [8, 8]
    merged = module.Q1SurfaceSearch(vertices, _previous=split_search)
    assert calls == [8, 8, 8]
    query = np.array([[.2, .3, .01], [1.2, .3, .01]])
    for current, expected in ((merged, original),
                              (split_search, module.Q1SurfaceSearch(split))):
        a, b = (s.project(query, maximum_distance=.2) for s in (current, expected))
        for name in ('points', 'normals', 'weights', 'facet_indices', 'distances'):
            np.testing.assert_array_equal(a[name], b[name])


def test_cached_node_sort_does_not_hide_inconsistent_coordinates():
    from agentfem._q1_surface_search import Q1SurfaceSearch
    from test_q1_surface_search import square
    vertices = np.array([square(), square() + [1, 0, 0]])
    ids = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    initial = Q1SurfaceSearch(vertices, node_ids=ids)
    with pytest.raises(ValueError):
        initial.node_ids.setflags(write=True)
    moved = vertices.copy(); moved[1, :, 2] += .1
    with pytest.raises(ValueError, match='inconsistent coordinates'):
        Q1SurfaceSearch(moved, node_ids=ids, _previous=initial)
    changed = ids.copy(); changed[1, 0] = 6
    with pytest.raises(ValueError, match='unchanged ordered trace topology'):
        Q1SurfaceSearch(vertices, node_ids=changed, _previous=initial)
