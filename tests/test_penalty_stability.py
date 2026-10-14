# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._penalty_stability import (
    penalty_stencil_row_sums,
)
from agentfem._initial_nodal_bond import initial_nodal_q1_pairing
from test_initial_nodal_bond import trace


def test_signed_duplicate_stencils_bound_independent_dense_psd_operator():
    rng = np.random.default_rng(917)
    for _ in range(15):
        nodes = rng.integers(0, 11, size=(23, 8))
        shape = rng.normal(size=nodes.shape)
        weights = rng.uniform(0.1, 2, 23)
        stiffness = rng.uniform(1, 5, 23)
        mass = rng.uniform(0.1, 3, (11, 3))
        rows = penalty_stencil_row_sums(nodes, shape, weights, stiffness, mass)
        matrix = np.zeros((33, 33))
        for ids, coefficients, w, k in zip(nodes, shape, weights, stiffness):
            b = np.zeros((3, 33))
            for node, value in zip(ids, coefficients):
                b[:, 3 * node : 3 * node + 3] += value * np.eye(3)
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            d = q @ np.diag(rng.uniform(0, k, 3)) @ q.T
            matrix += w * b.T @ d @ b
        inv = 1 / np.sqrt(mass.ravel())
        exact = np.linalg.eigvalsh(inv[:, None] * matrix * inv[None, :])[-1]
        assert rows.max() >= exact * (1 - 1e-14)


def test_chunked_rows_preserve_duplicate_accumulation_and_mass_scaling():
    rng = np.random.default_rng(212)
    nodes = rng.integers(0, 30, (8201, 4))
    shape = rng.normal(size=nodes.shape)
    weights = rng.uniform(0.1, 2, len(nodes))
    mass = rng.uniform(0.5, 2, (30, 3))
    result = penalty_stencil_row_sums(nodes, shape, weights, 7.0, mass)
    expected = np.zeros_like(mass)
    for ids, values, w in zip(nodes, shape, weights):
        scaled = abs(values) / np.sqrt(mass[ids].min(axis=1))
        np.add.at(expected, ids, (w * 7 * scaled * scaled.sum())[:, None])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_allclose(
        penalty_stencil_row_sums(nodes, shape, weights, 7.0, 4 * mass),
        result / 4,
        rtol=1e-14,
    )


def test_fixed_q1_elastic_bound_dominates_exact_jump_tangent():
    a, b = trace(2), trace(1)
    pair = initial_nodal_q1_pairing(
        a, b, eligible_nodes=np.ones(9, dtype=bool), tolerance=1e-10
    )
    ma, mb = np.ones((9, 3)), np.full((4, 3), 2.0)
    nodes = np.concatenate((pair.negative_nodes, pair.positive_nodes + 9), axis=1)
    shape = np.concatenate((-pair.negative_weights, pair.positive_weights), axis=1)
    bound = penalty_stencil_row_sums(
        nodes, shape, pair.weights, 100.0, np.concatenate((ma, mb))
    ).max()
    columns = []
    for column in np.eye(39):
        da, db = column[:27].reshape(9, 3), column[27:].reshape(4, 3)
        ra, rb = pair.tangent_action(
            np.broadcast_to(100 * np.eye(3), (9, 3, 3)), da, db
        )
        columns.append(np.concatenate((ra.ravel(), rb.ravel())))
    matrix = np.array(columns).T
    inv = 1 / np.sqrt(np.concatenate((ma.ravel(), mb.ravel())))
    exact = np.linalg.eigvalsh(inv[:, None] * matrix * inv[None, :])[-1]
    assert bound >= exact * (1 - 1e-14)
    scaled = penalty_stencil_row_sums(
        nodes, shape, pair.weights, 100.0, 4 * np.concatenate((ma, mb))
    ).max()
    assert scaled == pytest.approx(bound / 4)


@pytest.mark.parametrize(
    "bad", ["negative_index", "zero_mass", "complex", "negative_weight", "nan"]
)
def test_bad_stencil_data_rejects(bad):
    nodes = np.array([[0, 1]])
    shape = np.array([[1.0, -1.0]])
    weights, mass = np.ones(1), np.ones((2, 3))
    if bad == "negative_index":
        nodes[0, 0] = -1
    if bad == "zero_mass":
        mass[1, 0] = 0
    if bad == "complex":
        shape = shape.astype(complex) + 1j
    if bad == "negative_weight":
        weights[0] = -1
    if bad == "nan":
        shape[0, 0] = np.nan
    with pytest.raises(ValueError):
        penalty_stencil_row_sums(nodes, shape, weights, 1.0, mass)
