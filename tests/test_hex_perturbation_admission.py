# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent derivative checks for sufficient Q1 geometry admission."""

import basix
import numpy as np
import pytest

from agentfem.elements._hex_validity import (
    _jacobian_frobenius_bound,
    admits_reference_perturbation,
    reference_perturbation_radius,
    require_positive_hex_jacobian,
)


def test_edge_bound_dominates_independent_basix_derivatives():
    rng = np.random.default_rng(132)
    values = rng.normal(size=(80, 8, 3))
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        1,
        basix.LagrangeVariant.equispaced,
    )
    points = np.vstack(
        (rng.uniform(size=(200, 3)), basix.cell.geometry(basix.CellType.hexahedron))
    )
    d = element.tabulate(1, points)[1:4, :, :, 0]
    jacobian = np.einsum("cni,jpn->cpij", values, d)
    actual = np.linalg.norm(jacobian, axis=(2, 3)).max(axis=1)
    assert np.all(actual <= _jacobian_frobenius_bound(values) * (1 + 1e-14))


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_admitted_warped_cells_pass_full_bernstein_check(scale):
    rng = np.random.default_rng(82)
    x = np.tile(basix.cell.geometry(basix.CellType.hexahedron), (100, 1, 1))
    x += rng.uniform(-0.08, 0.08, x.shape)
    x *= scale
    radius = reference_perturbation_radius(x)
    u = rng.normal(size=x.shape)
    u *= (0.8 * radius / _jacobian_frobenius_bound(u))[:, None, None]
    assert np.all(admits_reference_perturbation(x, u, radius))
    assert np.all(require_positive_hex_jacobian(x + u) > 0)
    # An inversion is never admitted, even on highly scaled meshes.
    u = np.zeros_like(x)
    u[:, :, 0] = -2 * x[:, :, 0]
    assert not np.any(admits_reference_perturbation(x, u, radius))
    with pytest.raises(ValueError, match="Jacobian"):
        require_positive_hex_jacobian(x + u)


def test_unresolved_coordinate_roundoff_requests_fallback():
    x = basix.cell.geometry(basix.CellType.hexahedron)[None]
    radius = reference_perturbation_radius(x)
    assert not admits_reference_perturbation(x, np.full_like(x, 1e15), radius)[0]
