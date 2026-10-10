# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent dense-matrix oracles for material-envelope geometry lowering."""

import basix
import numpy as np
import pytest

from agentfem.elements._finite_uniform_hex import FiniteUniformHexBatch
from test_finite_uniform_hex import _material, _operator


def dense_mass_scaled(operator, tangent):
    matrix = np.column_stack([
        operator.tangent_action(v.reshape(1, 8, 3), first_piola_tangent=tangent).ravel()
        for v in np.eye(24)
    ])
    mass = np.repeat(operator.lumped_mass[0], 3)
    return matrix, mass


@pytest.mark.parametrize("distorted", [False, True])
def test_signed_anisotropic_envelope_bounds_dense_element(distorted):
    op = _operator(distorted)
    rng = np.random.default_rng(348)
    for _ in range(12):
        basis, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        eigenvalues = rng.uniform(-30, 100, 9)
        tangent = ((basis * eigenvalues) @ basis.T).reshape(1, 3, 3, 3, 3)
        upper, negative = op.tangent_envelope_bounds(
            positive_modulus=100, negative_modulus=30
        )
        matrix, mass = dense_mass_scaled(op, tangent)
        actual = np.linalg.eigvalsh(matrix / np.sqrt(mass[:, None] * mass[None, :]))
        assert actual[-1] <= upper * (1 + 1e-12)
        assert actual[0] >= -negative * (1 + 1e-12)


def test_shared_node_assembly_is_bounded_by_maximum_element_envelope():
    x = basix.cell.geometry(basix.CellType.hexahedron)
    coordinates = np.stack((x, x + [1, 0, 0]))
    _, inverse = np.unique(coordinates.reshape(-1, 3), axis=0, return_inverse=True)
    dofs = (3 * inverse.reshape(2, 8, 1) + np.arange(3)).reshape(2, 24)
    stiffness = np.zeros((36, 36))
    masses = np.zeros(36)
    bounds, lower_bounds = [], []
    rng = np.random.default_rng(23)
    for index, (rho, positive, negative) in enumerate(((1, 40, 9), (5, 170, 23))):
        op = FiniteUniformHexBatch(coordinates[index:index + 1], density=rho,
                                  hourglass_modulus=30, hourglass_scale=0.1)
        basis, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        tangent = ((basis * np.linspace(-negative, positive, 9)) @ basis.T)
        matrix, mass = dense_mass_scaled(op, tangent.reshape(1, 3, 3, 3, 3))
        stiffness[np.ix_(dofs[index], dofs[index])] += matrix
        masses[dofs[index]] += mass
        upper, lower = op.tangent_envelope_bounds(
            positive_modulus=positive, negative_modulus=negative
        )
        bounds.append(upper)
        lower_bounds.append(lower)
    actual = np.linalg.eigvalsh(stiffness / np.sqrt(masses[:, None] * masses[None, :]))
    assert actual[-1] <= max(bounds) * (1 + 1e-12)
    assert actual[0] >= -max(lower_bounds) * (1 + 1e-12)


def test_neo_hookean_domain_envelope_bounds_rotated_stretched_states():
    # Analytically declared test provider, not an automatic core material guess.
    mu, lam, minimum, maximum = 30, 40, 0.65, 1.6
    coefficient = max(abs(3 * lam * np.log(s) - mu) for s in (minimum, maximum))
    positive = mu + (3 * lam + coefficient) / minimum**2
    negative = max(0, coefficient / minimum**2 - mu)
    rng = np.random.default_rng(20261010)
    op = _operator(True)
    upper, lower = op.tangent_envelope_bounds(
        positive_modulus=positive, negative_modulus=negative
    )
    for _ in range(24):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        r, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        f = q @ np.diag(rng.uniform(minimum, maximum, 3)) @ r.T
        _, _, tangent = _material(f[None])
        values = np.linalg.eigvalsh(tangent.reshape(9, 9))
        assert values[-1] <= positive
        assert values[0] >= -negative
        matrix, mass = dense_mass_scaled(op, tangent)
        actual = np.linalg.eigvalsh(matrix / np.sqrt(mass[:, None] * mass[None, :]))
        assert actual[-1] <= upper * (1 + 1e-12)
        assert actual[0] >= -lower * (1 + 1e-12)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, [1, 2], [[1]]])
@pytest.mark.parametrize("key", ["positive_modulus", "negative_modulus"])
def test_invalid_envelope_rejected(bad, key):
    options = {"positive_modulus": 10, "negative_modulus": 0, key: bad}
    with pytest.raises(ValueError, match=key):
        _operator().tangent_envelope_bounds(**options)


def test_envelope_batches_empty_partitions_and_hourglass_only():
    op = _operator(True, count=4)
    positive, negative = np.arange(4) * 30, np.arange(4) * 5
    batch = op.tangent_envelope_bounds(positive_modulus=positive, negative_modulus=negative)
    separate = []
    for index in range(4):
        single = FiniteUniformHexBatch(op.coordinates[index:index + 1], density=2,
                                      hourglass_modulus=30, hourglass_scale=0.1)
        separate.append(single.tangent_envelope_bounds(
            positive_modulus=positive[index], negative_modulus=negative[index]))
    np.testing.assert_allclose(batch, np.max(separate, axis=0))
    empty = FiniteUniformHexBatch(np.empty((0, 8, 3)), density=2,
                                 hourglass_modulus=30, hourglass_scale=0.1)
    assert empty.tangent_envelope_bounds(positive_modulus=10) == (0, 0)
    assert op.tangent_envelope_bounds(positive_modulus=0) == (
        np.max(op._hourglass_spectral_bound), 0
    )
