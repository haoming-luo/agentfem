# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from test_finite_uniform_hex import _operator


def test_distorted_hex_majorant_bounds_independent_dense_anisotropic_operators():
    op = _operator(True, count=7, chunk_size=2)
    modulus = np.arange(7) + 100.0
    majorant = op.tangent_envelope_cell_row_majorant(positive_modulus=modulus)
    rng = np.random.default_rng(52)
    for c in range(7):
        q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        a = q @ np.diag(rng.uniform(0, modulus[c], 9)) @ q.T
        g = op.average_gradient[c]
        b = np.zeros((9, 24))
        for i in range(3):
            for j in range(3):
                b[3 * i + j, i::3] = g[:, j]
        gamma = op.hourglass_modes[c]
        k = op.volume[c] * b.T @ a @ b + op.hourglass_coefficient[c] * np.kron(
            gamma @ gamma.T, np.eye(3)
        )
        diagonal = np.diag(np.repeat(majorant[c], 3))
        assert np.linalg.eigvalsh(diagonal - k).min() >= -1e-10
    assert not majorant.flags.writeable
    other = _operator(True, count=7, chunk_size=1)
    np.testing.assert_array_equal(
        majorant, other.tangent_envelope_cell_row_majorant(positive_modulus=modulus)
    )


@pytest.mark.parametrize("bad", [True, -1, np.nan, 1j, [1, 2]])
def test_invalid_material_majorant_rejects(bad):
    with pytest.raises(ValueError):
        _operator().tangent_envelope_cell_row_majorant(positive_modulus=bad)
