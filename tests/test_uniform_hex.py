# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import basix
import numpy as np
import pytest

from agentfem.elements._uniform_hex import UniformHex8, UniformHexBatch


def _cell(distorted=False):
    x = basix.cell.geometry(basix.CellType.hexahedron).copy()
    if distorted:
        x += np.random.default_rng(13).uniform(-0.12, 0.12, x.shape)
    return x


def _material():
    c = np.zeros((6, 6))
    c[:3, :3] = 40
    c[np.arange(3), np.arange(3)] += 60
    c[3:, 3:] = np.eye(3) * 30
    return c


def _operator(x):
    return UniformHex8(
        x, _material(), density=2.0, hourglass_modulus=30.0, hourglass_scale=0.1
    )


@pytest.mark.parametrize("distorted", [False, True])
def test_uniform_hex_affine_patch_and_infinitesimal_rigid_modes(distorted):
    x = _cell(distorted)
    op = _operator(x)
    g = np.array([[0.1, 0.2, -0.3], [0.4, 0.3, 0.1], [-0.2, 0.5, -0.1]])
    displacement = x @ g.T + [0.1, -0.3, 0.5]
    result = op.response(displacement)
    expected = [
        g[0, 0],
        g[1, 1],
        g[2, 2],
        g[0, 1] + g[1, 0],
        g[1, 2] + g[2, 1],
        g[0, 2] + g[2, 0],
    ]
    np.testing.assert_allclose(result.strain, expected, atol=1e-14)
    assert result.hourglass_energy < 1e-26
    assert op.lumped_mass.sum() == pytest.approx(2 * op.volume)
    assert np.all(op.lumped_mass > 0)
    for axis in np.eye(3):
        for rigid in (np.tile(axis, (8, 1)), np.cross(np.tile(axis, (8, 1)), x)):
            response = op.response(rigid)
            np.testing.assert_allclose(response.internal_force, 0, atol=1e-12)
            assert response.physical_energy < 1e-25
            assert response.hourglass_energy < 1e-25
    np.testing.assert_allclose(x.T @ op.average_gradient, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(x.T @ op.hourglass_modes, 0, atol=1e-14)


@pytest.mark.parametrize("distorted", [False, True])
def test_uniform_hex_controls_twelve_modes_without_removing_rigid_modes(distorted):
    op = _operator(_cell(distorted))
    physical = np.linalg.eigvalsh(op.physical_matrix)
    hourglass = np.linalg.eigvalsh(op.hourglass_matrix)
    combined = np.linalg.eigvalsh(op.physical_matrix + op.hourglass_matrix)
    assert np.count_nonzero(physical > 1e-9) == 6
    assert np.count_nonzero(hourglass > 1e-9) == 12
    assert np.count_nonzero(combined > 1e-9) == 18
    assert combined.min() > -1e-10
    mass = np.repeat(op.lumped_mass, 3)
    scaled = (op.physical_matrix + op.hourglass_matrix) / np.sqrt(
        mass[:, None] * mass[None, :]
    )
    assert np.linalg.eigvalsh(scaled).max() <= op.stability_bound() * (1 + 1e-14)


def test_uniform_gradient_is_not_distorted_center_gradient():
    x = _cell(True)
    op = _operator(x)
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        1,
        basix.LagrangeVariant.equispaced,
    )
    gradient = element.tabulate(1, np.full((1, 3), 0.5))[1:4, 0, :, 0].T
    center = gradient @ np.linalg.inv(x.T @ gradient)
    assert np.linalg.norm(center - op.average_gradient) > 1e-3


def test_uniform_hex_tangent_energy_gradient_and_scaling():
    x = _cell(True)
    op = _operator(x)
    rng = np.random.default_rng(8)
    u, v = rng.normal(size=(2, 8, 3))
    h = 1e-6
    plus, minus = op.response(u + h * v), op.response(u - h * v)
    action = (op.physical_matrix + op.hourglass_matrix) @ v.ravel()
    np.testing.assert_allclose(
        (plus.internal_force - minus.internal_force).ravel() / (2 * h),
        action,
        rtol=1e-8,
        atol=1e-7,
    )
    derivative = (
        plus.physical_energy
        + plus.hourglass_energy
        - minus.physical_energy
        - minus.hourglass_energy
    ) / (2 * h)
    assert derivative == pytest.approx(
        np.sum(op.response(u).internal_force * v), rel=1e-8
    )
    scaled = _operator(3 * x + [100, -200, 50])
    np.testing.assert_allclose(
        scaled.physical_matrix, 3 * op.physical_matrix, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        scaled.hourglass_matrix, 3 * op.hourglass_matrix, rtol=1e-10, atol=1e-11
    )


def test_uniform_hex_rejects_invalid_geometry_and_material():
    x = _cell()
    x[:, 2] *= -1
    with pytest.raises(ValueError, match="Jacobian"):
        _operator(x)
    with pytest.raises(ValueError, match="positive definite"):
        UniformHex8(
            _cell(), -np.eye(6), density=1, hourglass_modulus=1, hourglass_scale=0.1
        )


def test_uniform_hex_orthotropic_energy_and_mass_against_independent_quadrature():
    x = _cell(True)
    c = np.diag([120.0, 90.0, 70.0, 25.0, 18.0, 21.0])
    c[:3, :3] += 10
    op = UniformHex8(x, c, density=3.0, hourglass_modulus=20.0, hourglass_scale=0.2)
    points, weights = basix.make_quadrature(basix.CellType.hexahedron, 8)
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        1,
        basix.LagrangeVariant.equispaced,
    )
    table = element.tabulate(1, points)
    integrated_gradient = np.zeros((8, 3))
    mass = np.zeros(8)
    volume = 0.0
    for q, weight in enumerate(weights):
        gradient = table[1:4, q, :, 0].T
        jacobian = x.T @ gradient
        measure = weight * np.linalg.det(jacobian)
        integrated_gradient += measure * gradient @ np.linalg.inv(jacobian)
        mass += 3 * measure * table[0, q, :, 0]
        volume += measure
    np.testing.assert_allclose(
        op.average_gradient, integrated_gradient / volume, atol=1e-14
    )
    np.testing.assert_allclose(op.lumped_mass, mass, atol=1e-14)
    u = x * [0.1, 0.2, 0.3]
    strain = np.array([0.1, 0.2, 0.3, 0, 0, 0])
    assert op.response(u).physical_energy == pytest.approx(
        0.5 * volume * strain @ c @ strain
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 20])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_compact_hex_batch_matches_cell_oracle(chunk_size, heterogeneous):
    count = 7
    x = np.stack([_cell(True) * (1 + k / 10) + k for k in range(count)])
    c = (
        np.stack([_material() * (1 + k / 5) for k in range(count)])
        if heterogeneous
        else _material()
    )
    rho = np.linspace(1, 2, count)
    modulus = np.linspace(20, 30, count)
    op = UniformHexBatch(
        x,
        c,
        density=rho,
        hourglass_modulus=modulus,
        hourglass_scale=0.1,
        chunk_size=chunk_size,
    )
    u = np.random.default_rng(30).normal(size=(count, 8, 3))
    seen = []
    for region, result in op.iter_responses(u):
        for local, index in enumerate(range(region.start, region.stop)):
            seen.append(index)
            single = UniformHex8(
                x[index],
                c[index] if heterogeneous else c,
                density=rho[index],
                hourglass_modulus=modulus[index],
                hourglass_scale=0.1,
            )
            reference = single.response(u[index])
            for name in (
                "strain",
                "stress",
                "internal_force",
                "physical_energy",
                "hourglass_energy",
            ):
                np.testing.assert_allclose(
                    getattr(result, name)[local],
                    getattr(reference, name),
                    rtol=1e-12,
                    atol=1e-12,
                )
            np.testing.assert_allclose(op.lumped_mass[index], single.lumped_mass)
            mass = np.repeat(single.lumped_mass, 3)
            scaled = (single.physical_matrix + single.hourglass_matrix) / np.sqrt(
                mass[:, None] * mass[None, :]
            )
            assert op.stability_bound() >= np.linalg.eigvalsh(scaled).max()
    assert seen == list(range(count))
    assert op.storage_bytes < count * 2 * 24 * 24 * 8 / 10
    assert not op.average_gradient.flags.writeable


def test_compact_hex_batch_rejects_invalid_inputs():
    args = dict(density=1, hourglass_modulus=30, hourglass_scale=0.1)
    x = np.stack([_cell(), _cell()])
    with pytest.raises(ValueError, match="chunk_size"):
        UniformHexBatch(x, _material(), chunk_size=True, **args)
    with pytest.raises(ValueError, match="Stiffness"):
        UniformHexBatch(x, np.zeros((3, 6, 6)), **args)
    x[1, :, 2] *= -1
    with pytest.raises(ValueError, match="starting at cell 1"):
        UniformHexBatch(x, _material(), chunk_size=1, **args)
    op = UniformHexBatch(x[:1], _material(), **args)
    with pytest.raises(ValueError, match="displacement"):
        list(op.iter_responses(np.full((1, 8, 3), np.nan)))
