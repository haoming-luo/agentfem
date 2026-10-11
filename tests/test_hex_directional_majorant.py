"""Directional upper-tensor lowering checked through independent actions."""

import basix
import numpy as np
import pytest

from agentfem.elements._finite_uniform_hex import FiniteUniformHexBatch


def operator(count=3, chunk_size=2):
    x = np.tile(basix.cell.geometry(basix.CellType.hexahedron), (count, 1, 1))
    x += np.random.default_rng(37).uniform(-0.03, 0.03, x.shape)
    return FiniteUniformHexBatch(
        x, density=2, hourglass_modulus=3, hourglass_scale=0.1, chunk_size=chunk_size
    )


def test_directional_rows_bound_independent_dense_actions_and_are_chunk_invariant():
    op = operator()
    rng = np.random.default_rng(83)
    roots = rng.normal(size=(3, 9, 9))
    tensor = (roots @ roots.swapaxes(1, 2)).reshape(3, 3, 3, 3, 3)
    rows = op.directional_cell_row_majorant(positive_tensor=tensor)
    columns = []
    for i in range(24):
        v = np.zeros((3, 8, 3))
        v.reshape(3, 24)[:, i] = 1
        columns.append(op.tangent_action(v, first_piola_tangent=tensor).reshape(3, 24))
    dense = np.stack(columns, axis=-1)
    np.testing.assert_allclose(
        rows.reshape(3, 24), np.abs(dense).sum(axis=2), rtol=1e-12
    )
    for c in range(3):
        assert np.linalg.eigvalsh(np.diag(rows[c].ravel()) - dense[c])[0] > -1e-12
    np.testing.assert_array_equal(
        rows,
        operator(chunk_size=1).directional_cell_row_majorant(positive_tensor=tensor),
    )
    assert not rows.flags.writeable


@pytest.mark.parametrize(
    "kind", ["negative", "nonsymmetric", "nonfinite", "shape", "complex"]
)
def test_invalid_upper_tensor_rejected(kind):
    tensor = np.eye(9)
    if kind == "negative":
        tensor[0, 0] = -1
    if kind == "nonsymmetric":
        tensor[0, 1] = 1
    if kind == "nonfinite":
        tensor[0, 0] = np.nan
    tensor = tensor.reshape(3, 3, 3, 3)
    if kind == "shape":
        tensor = tensor[0]
    if kind == "complex":
        tensor = tensor.astype(complex)
    with pytest.raises(ValueError):
        operator().directional_cell_row_majorant(positive_tensor=tensor)


def test_zero_and_empty_material_tensors_are_legal():
    assert operator(0).directional_cell_row_majorant(
        positive_tensor=np.zeros((3, 3, 3, 3))
    ).shape == (0, 8, 3)
    assert np.all(
        operator().directional_cell_row_majorant(positive_tensor=np.zeros((3, 3, 3, 3)))
        >= 0
    )


def test_directional_bound_avoids_stiff_in_plane_modulus_in_thin_direction():
    x = basix.cell.geometry(basix.CellType.hexahedron)[None].copy()
    x[:, :, 2] *= 0.01
    op = FiniteUniformHexBatch(x, density=2, hourglass_modulus=1, hourglass_scale=0.01)
    matrix = np.eye(9)
    matrix[0, 0] = 1000
    directional = op.directional_cell_row_majorant(
        positive_tensor=matrix.reshape(3, 3, 3, 3)
    )
    isotropic = op.tangent_envelope_cell_row_majorant(positive_modulus=1000)
    assert directional.max() < isotropic.max() / 100
