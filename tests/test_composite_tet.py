# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Algebraic element verification, independent of application simulations."""

import itertools
import numpy as np
import pytest
from agentfem.elements import composite_tet10

EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])


def coordinates():
    v = np.array([[0.0, 0, 0], [2.0, 0.1, 0], [0.2, 1.3, 0.1], [0.1, 0.3, 1.7]])
    return np.vstack((v, v[EDGES].mean(axis=1)))


def material():
    C = np.zeros((6, 6))
    C[:3, :3] = 2
    C += np.diag([6, 6, 6, 3, 3, 3])
    return C


def test_partition_gradient_and_linear_completeness():
    x = coordinates()
    e = composite_tet10(x)
    g = e.gradients(np.eye(4))
    np.testing.assert_allclose(g.sum(axis=1), 0, atol=2e-15)
    for gradient in g:
        np.testing.assert_allclose(x.T @ gradient, np.eye(3), atol=2e-15)
    G = np.array([[0.03, 0.2, -0.1], [0.1, 0.05, 0.3], [0, 0.1, -0.02]])
    u = x @ G.T + [0.2, -0.1, 0.5]
    expected = [
        G[0, 0],
        G[1, 1],
        G[2, 2],
        G[0, 1] + G[1, 0],
        G[1, 2] + G[2, 1],
        G[0, 2] + G[2, 0],
    ]
    for bbar in (False, True):
        np.testing.assert_allclose(
            e.strain_matrices(mean_dilatation=bbar) @ u.ravel(),
            np.tile(expected, (4, 1)),
            atol=2e-15,
        )


def test_mass_exact_positive_and_consistent():
    e = composite_tet10(coordinates())
    m = e.mass(2).reshape(10, 3)[:, 0]
    np.testing.assert_allclose(
        m / (2 * e.volume), [1 / 32] * 4 + [7 / 48] * 6, atol=2e-16
    )
    np.testing.assert_allclose(m.sum(), 2 * e.volume)
    assert np.linalg.eigvalsh(e.mass(2, lumped=False)).min() > 0
    # Consistent mass exactly integrates affine velocity fields on all subtets.
    v = np.arange(30).reshape(10, 3) * 0.0 + [1, 2, 3]
    np.testing.assert_allclose(
        v.ravel() @ e.mass(2, lumped=False) @ v.ravel(), 28 * e.volume
    )
    np.testing.assert_allclose(
        m @ e.coordinates / m.sum(), e.coordinates[:4].mean(axis=0)
    )


def test_rank_rigid_motion_and_energy_gradient():
    e = composite_tet10(coordinates())
    rng = np.random.default_rng(310)
    u, d = rng.normal(size=(2, 30))
    for bbar in (False, True):
        K = e.stiffness(material(), mean_dilatation=bbar)
        eig = np.linalg.eigvalsh(K)
        assert np.count_nonzero(eig < eig[-1] * 1e-11) == 6
        for w in np.eye(3):
            rigid = np.cross(w, e.coordinates).ravel()
            np.testing.assert_allclose(K @ rigid, 0, atol=2e-14)
            np.testing.assert_allclose(K @ np.tile(w, 10), 0, atol=2e-14)
        eps = 1e-5
        numerical = (
            (u + eps * d) @ K @ (u + eps * d) - (u - eps * d) @ K @ (u - eps * d)
        ) / (4 * eps)
        np.testing.assert_allclose(numerical, d @ K @ u, rtol=1e-9)


def test_vertex_permutations_and_length_scaling():
    x = coordinates()
    base = composite_tet10(x)
    K = base.stiffness(material())
    for perm in itertools.permutations(range(4)):
        ids = list(perm)
        for a, b in EDGES:
            pair = {perm[a], perm[b]}
            ids.append(4 + next(i for i, edge in enumerate(EDGES) if set(edge) == pair))
        dofs = (3 * np.array(ids)[:, None] + np.arange(3)).ravel()
        e = composite_tet10(x[ids])
        np.testing.assert_allclose(
            e.stiffness(material()), K[np.ix_(dofs, dofs)], atol=2e-14
        )
    scaled = composite_tet10(4 * x + [12, -3, 5])
    np.testing.assert_allclose(scaled.stiffness(material()), 4 * K, atol=1e-13)
    np.testing.assert_allclose(scaled.scalar_mass, 64 * base.scalar_mass, atol=1e-14)


def test_reject_unsupported_inputs():
    x = coordinates()
    e = composite_tet10(x)
    x[4, 0] += 0.01
    with pytest.raises(NotImplementedError, match="affine"):
        composite_tet10(x)
    with pytest.raises(ValueError, match="degenerate"):
        composite_tet10(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="positive-definite"):
        e.stiffness(-np.eye(6))
    with pytest.raises(ValueError, match="symmetric"):
        e.stiffness(np.triu(np.ones((6, 6))))
    with pytest.raises(ValueError, match="density"):
        e.mass(-1)
    with pytest.raises(ValueError, match="barycentric"):
        e.gradients([1, 1, 0, 0])
    with pytest.raises(ValueError):
        e.coordinates[0, 0] = 1


def test_projection_residual_stabilization_and_bulk_independence():
    e = composite_tet10(coordinates())
    C = material()
    raw = e.stiffness(C, mean_dilatation=True, stabilize=False)
    fixed = e.stiffness(C, mean_dilatation=True)
    eigenvalues = np.linalg.eigvalsh(raw)
    assert np.count_nonzero(eigenvalues < eigenvalues[-1] * 1e-11) == 9
    correction = fixed - raw
    assert np.linalg.eigvalsh(correction).min() > -1e-13
    # Stabilization must add no bulk penalty or affine-displacement energy.
    bulk = np.zeros((6, 6))
    bulk[:3, :3] = 1e5
    correction2 = e.stiffness(C + bulk, mean_dilatation=True) - e.stiffness(
        C + bulk, mean_dilatation=True, stabilize=False
    )
    np.testing.assert_allclose(correction2, correction, atol=1e-10)
    for i in range(3):
        for j in range(3):
            u = np.zeros((10, 3))
            u[:, i] = e.coordinates[:, j]
            np.testing.assert_allclose(correction @ u.ravel(), 0, atol=2e-14)


def test_consistent_mass_affine_inertia_and_time_step_scaling():
    x = coordinates()
    e = composite_tet10(x)
    # Exact integral of (w cross x)^2 from tetrahedron second moments.
    vertices = x[:4]
    second = (vertices.T @ vertices + np.outer(vertices.sum(0), vertices.sum(0))) / 20
    for w in np.eye(3):
        u = np.cross(w, x).ravel()
        target = e.volume * (np.dot(w, w) * np.trace(second) - w @ second @ w)
        np.testing.assert_allclose(u @ e.mass(1, lumped=False) @ u, target, rtol=1e-14)
    dt = e.critical_time_step(material(), 1)
    assert np.isfinite(dt) and dt > 0
    np.testing.assert_allclose(
        composite_tet10(3 * x).critical_time_step(material(), 1), 3 * dt
    )
    np.testing.assert_allclose(e.critical_time_step(4 * material(), 1), dt / 2)
    np.testing.assert_allclose(e.critical_time_step(material(), 4), 2 * dt)


def test_stabilized_energy_against_direct_subtetrahedron_integration():
    # Independent energy calculation: no projected gradients or kernel B matrices.
    x = coordinates()
    e = composite_tet10(x)
    rng = np.random.default_rng(927)
    u = rng.normal(size=(10, 3))
    xx = np.vstack((x, x[4:].mean(axis=0)))
    uu = np.vstack((u, u[4:].mean(axis=0)))
    corner = [[0, 4, 6, 7], [1, 5, 4, 8], [2, 6, 5, 9], [3, 8, 7, 9]]
    inner_faces = [
        [4, 8, 5],
        [5, 8, 9],
        [9, 8, 7],
        [7, 8, 4],
        [4, 5, 6],
        [5, 9, 6],
        [9, 7, 6],
        [7, 4, 6],
    ]
    energy_dev = 0.0
    divergence_integral = 0.0
    for ids in corner + [face + [10] for face in inner_faces]:
        p, v = xx[ids], uu[ids]
        volume = abs(np.linalg.det((p[1:] - p[0]).T)) / 6
        gradient = np.linalg.solve(p[1:] - p[0], v[1:] - v[0]).T
        strain = (gradient + gradient.T) / 2
        trace = np.trace(strain)
        dev = strain - np.eye(3) * trace / 3
        energy_dev += volume * 3 * np.sum(dev * dev)  # shear modulus = 3
        divergence_integral += volume * trace
    # material(): lambda = 2, mu = 3, bulk = 4.
    expected = energy_dev + 2 * divergence_integral**2 / e.volume
    actual = 0.5 * u.ravel() @ e.stiffness(material(), mean_dilatation=True) @ u.ravel()
    np.testing.assert_allclose(actual, expected, rtol=3e-14)
