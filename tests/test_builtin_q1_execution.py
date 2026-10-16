# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Installed generic Q1 algorithms retain shape, ties and path rejection."""
import numpy as np
import pytest
from agentfem import boundary_models
from agentfem._closed_q1_surface import ClosedQ1Surface
from agentfem._q1_swept_separation import separated_q1_paths
from test_closed_q1_surface import solid


@pytest.mark.parametrize('concave', [False, True])
@pytest.mark.parametrize('shift', [0., 10000.])
def test_native_projection_classification_bounds_and_refit(concave, shift):
    pytest.importorskip('agentfem._q1_native')
    x, cells = solid(concave)
    # Warped Q1 faces, not a triangle approximation; retain closed orientation.
    x[:, 2] += .08*x[:, 0]*x[:, 1]
    x += shift
    points = np.array([[.21,.33,.12],[-.13,.31,.41],[-.12,-.22,.42],[-.13,-.23,-.34]]) + shift
    plain = ClosedQ1Surface(x, cells)
    with boundary_models.q1_contact_runtime(backend='native') as runtime:
        assert runtime.execution.summary()['project_batch']
        fast = ClosedQ1Surface(x, cells, execution=runtime.execution).refitted(x)
        a, b = fast.project(points, maximum_distance=.6), plain.project(points, maximum_distance=.6)
        for name in ('points','signed_gaps','weights','normals'):
            np.testing.assert_allclose(a[name], b[name], atol=2e-10, rtol=2e-10)
        for targets in (None, np.full(len(points), .2)):
            lower = fast.search.distance_lower_bounds(points, refinement_distance=targets)
            reference = plain.search.distance_lower_bounds(points, refinement_distance=targets)
            np.testing.assert_allclose(lower, reference, atol=2e-10, rtol=2e-10)
            assert np.all(lower <= np.abs(b['signed_gaps']) + 2e-10)
        assert runtime.statistics['winding_native'] > 0


def test_native_ties_and_invalid_queries_keep_reference_rejection():
    pytest.importorskip('agentfem._q1_native')
    x, cells = solid()
    with boundary_models.q1_contact_runtime(backend='native') as runtime:
        fast = ClosedQ1Surface(x, cells, execution=runtime.execution)
        plain = ClosedQ1Surface(x, cells)
        for surface in (fast, plain):
            with pytest.raises(ValueError):
                surface.project([[.5,.5,.5]], maximum_distance=1.)
            with pytest.raises(ValueError):
                surface.project([[np.nan,0.,0.]], maximum_distance=1.)


def test_native_swept_does_not_certify_crossing():
    pytest.importorskip('agentfem._q1_native')
    x, cells = solid()
    start = np.array([[-.2,.3,.4], [-.2,-.3,.4], [2.,2.,2.]])
    end = np.array([[1.2,.3,.4], [-.1,-.3,.4], [3.,2.,2.]])
    with boundary_models.q1_contact_runtime(backend='native') as runtime:
        actual = runtime.execution.separated(x, x, cells, start, end, separated_q1_paths)
    np.testing.assert_array_equal(actual, separated_q1_paths(x,x,cells,start,end))
    assert not actual[0]
    assert actual[1:].all()


def test_reference_and_missing_native_fallback(monkeypatch):
    from agentfem.boundary_models import _native_library
    def missing(*args):
        raise ImportError('not installed')
    monkeypatch.setattr(_native_library.importlib, 'import_module', missing)
    with boundary_models.q1_contact_runtime() as runtime:
        assert runtime.execution.summary()['identity']['arithmetic'] == 'reference'
    with pytest.raises(RuntimeError, match='unavailable'):
        boundary_models.q1_contact_runtime(backend='native')
    with pytest.raises(ValueError):
        boundary_models.q1_contact_runtime(projection_workers=True)


def test_native_direct_projection_validates_buffer_and_lifetime():
    module = pytest.importorskip('agentfem._q1_native')
    assert module.projection_policy == 'scale-conditioned-margin+bernstein-gmp-v1'
    runtime = boundary_models.q1_contact_runtime(backend='native')
    for vertices in (np.zeros((3,3)), np.full((4,3),np.nan), np.ones((4,3),dtype=complex)):
        with pytest.raises(ValueError):
            runtime.closest(vertices,np.zeros(3),None)
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError,match='closed'):
        runtime.closest(np.zeros((4,3)),np.zeros(3),None)
    with pytest.raises(RuntimeError,match='closed'):
        runtime.__enter__()


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_winding_disjoint_slices_preserve_query_order_and_fallback(workers):
    pytest.importorskip('agentfem._q1_native')
    x, cells = solid(True)
    points = np.tile([[.21, .33, .12], [1.2, 1.3, .41]], (65, 1))
    surface = ClosedQ1Surface(x, cells)
    expected = surface._inside_batch_reference(points)
    with boundary_models.q1_contact_runtime(
        backend='native', projection_workers=workers
    ) as runtime:
        np.testing.assert_array_equal(
            runtime.inside(surface, points, surface._inside_batch_reference), expected)
        native = runtime.winding

        def failed_slice(n, *args):
            # A nonzero native status invalidates its whole slice, even if it
            # wrote some outputs. Other slices may still resolve normally.
            status = native(n, *args)
            if args[-5][0, 0] < 1:
                args[-1].fill(7)
                return -1
            return status

        runtime.winding = failed_slice
        np.testing.assert_array_equal(
            runtime.inside(surface, points, surface._inside_batch_reference), expected)
        assert runtime.statistics['winding_fallback'] > 0
