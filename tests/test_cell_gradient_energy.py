# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import assembly, mesh, operators


def _operator(*, components=()):
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        3,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    gradient = mesh.cell_gradient_operator(domain, rings=2)
    rng = np.random.default_rng(20260920)
    values = rng.normal(size=(gradient.total_cells, *components))
    return gradient, values


@pytest.mark.parametrize("components", [(), (3,)])
def test_cell_gradient_energy_residual_is_exact_directional_derivative(components):
    gradient, values = _operator(components=components)
    rng = np.random.default_rng(12)
    direction = rng.normal(size=values.shape)
    energy = operators.cell_gradient_energy(
        gradient,
        cell_weights=np.linspace(0.5, 1.5, gradient.owned_cells),
        stiffness=np.linspace(2.0, 3.0, gradient.owned_cells),
    )

    epsilon = 1.0e-5
    finite_difference = (
        energy.energy(values + epsilon * direction)
        - energy.energy(values - epsilon * direction)
    ) / (2.0 * epsilon)
    exact = np.vdot(energy.residual(values), direction)

    assert finite_difference == pytest.approx(exact, rel=2.0e-9, abs=2.0e-9)


@pytest.mark.parametrize("components", [(), (2,)])
def test_cell_gradient_energy_tangent_is_symmetric_semidefinite(components):
    gradient, first = _operator(components=components)
    rng = np.random.default_rng(53)
    second = rng.normal(size=first.shape)
    energy = operators.cell_gradient_energy(
        gradient,
        cell_weights=np.ones(gradient.owned_cells),
        stiffness=4.0,
    )

    first_action = energy.tangent_action(first)
    second_action = energy.tangent_action(second)

    assert np.vdot(first, second_action) == pytest.approx(
        np.vdot(first_action, second), rel=1.0e-13, abs=1.0e-13
    )
    assert np.vdot(first, first_action) == pytest.approx(
        2.0 * energy.energy(first), rel=1.0e-13, abs=1.0e-13
    )
    assert np.vdot(first, first_action) >= 0.0


def test_cell_gradient_energy_annihilates_constant_field():
    gradient, _ = _operator(components=(3,))
    values = np.broadcast_to((3.0, -2.0, 7.0), (gradient.total_cells, 3)).copy()
    energy = operators.cell_gradient_energy(
        gradient,
        cell_weights=np.ones(gradient.owned_cells),
    )

    assert energy.energy(values) == pytest.approx(0.0, abs=1.0e-25)
    np.testing.assert_allclose(energy.residual(values), 0.0, atol=2.0e-14)
    assert energy.as_dict()["mpi_requirement"].startswith("reverse_scatter")


def test_cell_gradient_energy_uses_physical_cell_measures():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        3,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    gradient = mesh.cell_gradient_operator(domain, rings=2)
    weights = mesh.owned_cell_measures(domain)
    cells = np.arange(gradient.total_cells, dtype=np.int32)
    points = dolfinx_mesh.compute_midpoints(domain, domain.topology.dim, cells)[:, :2]
    values = 2.0 * points[:, 0] - 3.0 * points[:, 1]
    energy = operators.cell_gradient_energy(
        gradient,
        cell_weights=weights,
        stiffness=4.0,
    )

    assert np.sum(weights) == pytest.approx(1.0, rel=1.0e-13)
    assert energy.energy(values) == pytest.approx(0.5 * 4.0 * (2.0**2 + 3.0**2))


def test_cell_residual_assembly_preserves_scalar_and_vector_cell_identity():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    cell_count = domain.topology.index_map(domain.topology.dim).size_local
    for shape in ((), (3,)):
        space = fem.functionspace(domain, ("DG", 0, shape)) if shape else fem.functionspace(domain, ("DG", 0))
        contributions = np.arange(
            cell_count * (shape[0] if shape else 1), dtype=float
        ).reshape((cell_count, *shape))
        vector = assembly.assemble_cell_residual(space, contributions)
        expected = np.empty_like(vector.array_r)
        for cell in range(cell_count):
            dof = int(space.dofmap.cell_dofs(cell)[0])
            block_size = int(space.dofmap.index_map_bs)
            expected[dof * block_size : (dof + 1) * block_size] = np.asarray(
                contributions[cell]
            ).reshape(-1)
        np.testing.assert_allclose(vector.array_r, expected)
        vector.destroy()


def test_cell_residual_assembly_rejects_non_dg0_space():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    with pytest.raises(ValueError, match="DG0"):
        assembly.assemble_cell_residual(space, np.zeros((2, 3)))


@pytest.mark.parametrize(
    ("weights", "stiffness", "message"),
    [
        ([1.0], 1.0, "cell_weights"),
        ([-1.0] * 12, 1.0, "strictly positive"),
        ([1.0] * 12, -1.0, "nonnegative"),
    ],
)
def test_cell_gradient_energy_rejects_invalid_coefficients(
    weights, stiffness, message
):
    gradient, _ = _operator()
    with pytest.raises(ValueError, match=message):
        operators.cell_gradient_energy(
            gradient,
            cell_weights=weights,
            stiffness=stiffness,
        )
