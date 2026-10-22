# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Current Q1 smooth-interior signs; edges retain the independent rays."""
import numpy as np
import pytest

from agentfem._closed_q1_surface import ClosedQ1Surface
from agentfem._q1_interior_sign import outward_volume
from agentfem._q1_surface_search import AmbiguousSurfaceProjection
from agentfem.boundary_models import Q1ContactExecution, q1_contact_runtime
from test_closed_q1_surface import solid


def policy():
    return Q1ContactExecution({"provider": "test"}, sign_policy="interior_normal")


def same(x, faces, points, radius=4):
    reference = ClosedQ1Surface(x, faces).project(points, maximum_distance=radius)
    fast = ClosedQ1Surface(x, faces, execution=policy())
    answer = fast.project(points, maximum_distance=radius)
    for name in ("points", "weights", "normals", "signed_gaps", "separation_lower_bounds"):
        np.testing.assert_array_equal(answer[name], reference[name])
    return fast


@pytest.mark.parametrize("concave,warped", [(False, False), (False, True),
                                          (True, False), (True, True)])
def test_random_current_geometry_matches_rays(concave, warped):
    x, faces = solid(concave)
    if warped:
        x[:, 2] += .19 * x[:, 0] * x[:, 1] * x[:, 2]
    points = np.random.default_rng(681).uniform(-.25, 2.25 if concave else 1.25, (240, 3))
    assert same(x, faces, points)._interior_sign_eligible


def test_fallback_and_refit_orientation(monkeypatch):
    x, faces = solid()
    points = np.array([[.2, .3, 3e-12], [.2, .3, -3e-12], [.2, .3, .07],
                       [-.1, -.1, .3], [0., .3, 0.]])
    surface = same(x, faces, points)
    reference = surface._inside_batch
    calls = []

    def record(query):
        calls.extend(query.tolist())
        return reference(query)

    monkeypatch.setattr(surface, "_inside_batch", record)
    surface.project(points, maximum_distance=1.)
    np.testing.assert_array_equal(calls, points[[0, 1, 3]])
    rotation, _ = np.linalg.qr(np.random.default_rng(74).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    y = x @ rotation.T + [4., -5., 7.]
    q = points @ rotation.T + [4., -5., 7.]
    answer = surface.refitted(y).project(q, maximum_distance=1.)
    ordinary = ClosedQ1Surface(y, faces).project(q, maximum_distance=1.)
    np.testing.assert_array_equal(answer["signed_gaps"], ordinary["signed_gaps"])
    for reversed_surface in (ClosedQ1Surface(x, faces[:, ::-1], execution=policy()),
                             surface.refitted(x * [-1., 1., 1.])):
        assert not reversed_surface._interior_sign_eligible
    with pytest.raises(ValueError, match="outward"):
        ClosedQ1Surface(x, faces[:, ::-1], execution=policy()).project(
            [[.2, .3, .1]], maximum_distance=1.)


def test_disconnected_nested_and_ambiguous_geometry():
    x, faces = solid()
    for second, reverse in ((x + [2., 0., 0.], False), (.2 + .6*x, True)):
        combined = np.vstack([x, second])
        connectivity = np.vstack([faces, (faces[:, ::-1] if reverse else faces) + len(x)])
        surface = same(combined, connectivity, [[.08, .3, .4], [1.2, .3, .4]], radius=2.)
        assert not surface._interior_sign_eligible
        if reverse:
            with pytest.raises(AmbiguousSurfaceProjection):
                surface.project([[.1, .3, .4]], maximum_distance=2.)


def test_bilinear_flux_matches_gauss_integral():
    vertices = np.random.default_rng(177).normal(size=(15, 4, 3))
    value, _ = outward_volume(vertices)
    face = vertices - .5 * (vertices.min(axis=(0, 1)) + vertices.max(axis=(0, 1)))
    quadrature = 0.
    for r in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for s in (-1/np.sqrt(3), 1/np.sqrt(3)):
            w = np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)]) / 4
            wr = np.array([-(1-s), 1-s, 1+s, -(1+s)]) / 4
            ws = np.array([-(1-r), -(1+r), 1+r, 1-r]) / 4
            p = np.einsum("q,fqc->fc", w, face)
            dr, ds = np.einsum("q,fqc->fc", wr, face), np.einsum("q,fqc->fc", ws, face)
            quadrature += np.einsum("fc,fc->", p, np.cross(dr, ds)) / 3
    np.testing.assert_allclose(value, quadrature, rtol=2e-14, atol=2e-14)


def test_connected_shell_with_handle_and_nonconvex_interior():
    # A toroidal shell is connected but not convex or topologically a sphere.
    # The current global-closest smooth-interior argument still applies.
    n, m = 8, 6
    a, b = np.meshgrid(np.arange(n)*2*np.pi/n, np.arange(m)*2*np.pi/m,
                       indexing="ij")
    x = np.stack(((2+.6*np.cos(b))*np.cos(a), (2+.6*np.cos(b))*np.sin(a),
                   .6*np.sin(b)), axis=-1).reshape(-1, 3)
    faces = np.array([[i*m+j, ((i+1)%n)*m+j, ((i+1)%n)*m+(j+1)%m,
                       i*m+(j+1)%m] for i in range(n) for j in range(m)])
    points = np.random.default_rng(712).uniform([-2.5, -2.5, -.8],
                                               [2.5, 2.5, .8], (40, 3))
    assert same(x, faces, points)._interior_sign_eligible


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_runtime_policy_and_current_refit(backend):
    with q1_contact_runtime(backend=backend, sign_policy="interior_normal") as runtime:
        assert runtime.execution.summary()["sign_policy"] == "interior_normal-v1"
        x, faces = solid()
        surface = ClosedQ1Surface(x, faces, execution=runtime.execution)
        answer = surface.refitted(x + [1., 0., 0.]).project([[1.1, .3, .4]], maximum_distance=1.)
        np.testing.assert_allclose(answer["signed_gaps"], [-.1], atol=1e-15)
        assert runtime.statistics["winding_native"] == 0
    with q1_contact_runtime(backend="reference") as runtime:
        assert "sign_policy" not in runtime.execution.summary()
    with pytest.raises(ValueError, match="sign_policy"):
        q1_contact_runtime(sign_policy="cached_old_normals")


def test_sign_reconstructs_normal_not_provider_orientation(monkeypatch):
    x, faces = solid()
    surface = ClosedQ1Surface(x, faces, execution=policy())
    project = surface.search.project

    def reversed_provider_normals(*args, **kwargs):
        found = project(*args, **kwargs)
        found["normals"] = -found["normals"]
        return found

    monkeypatch.setattr(surface.search, "project", reversed_provider_normals)
    answer = surface.project([[.1, .3, .4], [-.1, .3, .4]], maximum_distance=1.)
    np.testing.assert_allclose(answer["signed_gaps"], [-.1, .1], atol=1e-15)
