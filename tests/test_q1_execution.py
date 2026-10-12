# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem.boundary_models import Q1ContactExecution
from agentfem._closed_q1_surface import ClosedQ1Surface
from agentfem._q1_surface_search import Q1SurfaceSearch
from test_closed_q1_surface import solid


def test_local_callbacks_reference_equivalence_and_refit():
    calls = []
    def closest(vertices, point, reference):
        calls.append("closest")
        return reference(vertices, point)
    def inside(surface, points, reference):
        calls.append("inside")
        return reference(points)
    identity = {"name": "reference-adapter"}
    kernel = Q1ContactExecution(identity, closest, inside)
    identity["name"] = "changed"
    assert kernel.summary()["identity"]["name"] == "reference-adapter"
    x, cells = solid()
    plain = ClosedQ1Surface(x, cells)
    accelerated = ClosedQ1Surface(x, cells, execution=kernel)
    points = np.array([[.2, .3, .1], [-.1, .3, .4], [-.1, -.2, .4]])
    expected = plain.project(points, maximum_distance=.5)
    for surface in (accelerated, accelerated.refitted(x)):
        actual = surface.project(points, maximum_distance=.5)
        for key in ("points", "signed_gaps", "normals", "weights"):
            np.testing.assert_array_equal(actual[key], expected[key])
    assert "closest" in calls and "inside" in calls
    count = len(calls)
    plain.project(points, maximum_distance=.5)
    assert len(calls) == count


def test_invalid_kernel_outputs_rejected():
    x, cells = solid()
    def invalid(vertices, point, reference):
        values = list(reference(vertices, point))
        values[-1] = float("nan")
        return tuple(values)
    search = Q1SurfaceSearch(x[cells], execution=Q1ContactExecution({"bad": 1}, invalid))
    with pytest.raises(ValueError, match="invalid geometric"):
        search.project(np.array([[-.1, .3, .4]]), maximum_distance=.5)
    surface = ClosedQ1Surface(x, cells, execution=Q1ContactExecution(
        {"bad": 2}, classify_inside=lambda surface, points, reference: np.zeros(len(points))))
    with pytest.raises(ValueError, match="boolean"):
        surface.project(np.array([[.2, .3, .1]]), maximum_distance=.5)


def test_bounds_and_path_callbacks_preserve_instance_and_validate_transport():
    x, cells = solid()
    calls = []
    def bounds(surface, points, reference, **options):
        calls.append('bounds')
        return reference(points, **options)
    execution = Q1ContactExecution({'test':'bounds'}, distance_bounds=bounds)
    plain = ClosedQ1Surface(x, cells)
    fast = ClosedQ1Surface(x, cells, execution=execution).refitted(x)
    query = np.array([[2.,2.,2.],[-.1,.2,.3]])
    np.testing.assert_array_equal(fast.search.distance_lower_bounds(query),
                                  plain.search.distance_lower_bounds(query))
    assert calls == ['bounds']
    invalid = Q1ContactExecution({'test':'invalid'}, distance_bounds=lambda *args, **kw: [-1., 0.],
                                 separated_paths=lambda *args: np.ones(2))
    with pytest.raises(ValueError, match='nonnegative'):
        invalid.bounds(fast, query, plain.search.distance_lower_bounds, refinement_distance=None)
    with pytest.raises(ValueError, match='boolean certificate'):
        invalid.separated(x, x, cells, query, query, lambda *args: np.ones(2,dtype=bool))
