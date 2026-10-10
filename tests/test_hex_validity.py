# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from itertools import product

import basix
import numpy as np
import pytest

from agentfem.elements._hex_validity import (
    _admit,
    _bernstein,
    _children,
    require_positive_hex_jacobian,
)
from agentfem.elements._uniform_hex import UniformHex8, _reference


def _evaluate(b, points):
    bases = [np.column_stack(((1 - t) ** 2, 2 * t * (1 - t), t * t)) for t in points.T]
    return np.einsum("ijk,pi,pj,pk->p", b, *bases)


def test_bernstein_transform_and_subdivision_reconstruct_polynomial():
    rng = np.random.default_rng(11)
    b = rng.normal(size=(3, 3, 3))
    grid = np.array(list(product((0.0, 0.5, 1.0), repeat=3)))
    np.testing.assert_allclose(_bernstein(_evaluate(b, grid).reshape(3, 3, 3)), b)
    points = rng.uniform(size=(43, 3))
    for child, offset in zip(_children(b), product((0, 1), repeat=3)):
        np.testing.assert_allclose(
            _evaluate(child, points), _evaluate(b, (points + offset) / 2), atol=2e-15
        )


def test_negative_control_value_is_not_reported_as_inversion():
    grid = np.array(list(product((0.0, 0.5, 1.0), repeat=3)))
    values = (grid[:, 0] - 0.5) ** 2 + 0.01
    b = _bernstein(values.reshape(3, 3, 3))
    assert b.min() < 0
    assert _admit(b, 1e-13) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="unresolved"):
        _admit(b, 1e-13, max_depth=0)
    with pytest.raises(ValueError, match="unresolved"):
        _admit(b, 1e-13, max_boxes=1)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_affine_geometry_scale_and_translation(scale):
    x = basix.cell.geometry(basix.CellType.hexahedron)
    mapping = np.array([[2.0, 0.2, 0.1], [0.0, 3.0, 0.4], [0.0, 0.0, 0.5]])
    coordinates = scale * (x @ mapping.T + [12, -8, 5])
    bound = require_positive_hex_jacobian(coordinates[None])[0]
    assert bound == pytest.approx(np.linalg.det(mapping) * scale**3, rel=1e-12)


def test_hidden_inversion_passes_old_samples_but_is_rejected():
    # Deterministic distorted cell found with RNG seed 172. Basix vertex order.
    x = np.array(
        [
            [-0.5304872023562772, -0.36923279272380827, -0.21716400467083402],
            [0.6421403270759551, -0.023839202735277176, -0.3648881652300784],
            [-0.759106142975849, 0.19640953627839142, 0.09295468823206188],
            [-0.06883861221129184, 0.9907653469330054, -0.8636411701860751],
            [-0.8151812483955128, -0.8184564396160648, 0.48631842140531245],
            [0.8652824627624074, -0.5842759901940852, 0.5894064515499378],
            [0.03356305616380029, 0.6224293184236763, 0.44273102689190724],
            [0.6326273599367684, -0.013799776365838068, -0.16571724799223186],
        ]
    )
    old_gradients = np.concatenate((_reference()[2], _reference()[3]))
    assert np.linalg.det(np.einsum("ai,qaj->qij", x, old_gradients)).min() > 0
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        1,
        basix.LagrangeVariant.equispaced,
    )
    points = np.array(list(product(np.linspace(0, 1, 11), repeat=3)))
    gradients = element.tabulate(1, points)[1:4, :, :, 0].transpose(1, 2, 0)
    assert np.linalg.det(np.einsum("ai,qaj->qij", x, gradients)).min() < -0.01
    with pytest.raises(ValueError, match="inverted"):
        UniformHex8(x, np.eye(6), density=1, hourglass_modulus=1, hourglass_scale=0.1)


def test_batched_triple_product_matches_lapack_on_distorted_cells():
    from agentfem.elements._hex_validity import _derivatives

    rng = np.random.default_rng(274)
    vertices = basix.cell.geometry(basix.CellType.hexahedron)
    coordinates = vertices + rng.uniform(-0.08, 0.08, (127, 8, 3))
    jacobian = np.einsum(
        "cai,qaj->cqij", coordinates - coordinates.mean(axis=1, keepdims=True),
        _derivatives(),
    )
    expected = _bernstein(np.linalg.det(jacobian).reshape(-1, 3, 3, 3)).min((1, 2, 3))
    assert np.all(expected > 0)
    np.testing.assert_allclose(
        require_positive_hex_jacobian(coordinates), expected, rtol=5e-14, atol=0,
    )


@pytest.mark.parametrize("scale", [1e-8, 1, 1e8])
@pytest.mark.parametrize("height", [-1e-14, 0, 1e-14])
def test_cancellation_does_not_admit_near_degenerate_affine_cell(scale, height):
    vertices = basix.cell.geometry(basix.CellType.hexahedron)
    mapping = np.array([[1, 1, 0], [1, 1 + height, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="near-singular"):
        require_positive_hex_jacobian((scale * (vertices @ mapping.T))[None])
