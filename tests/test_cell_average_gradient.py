# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import operators


@pytest.mark.parametrize("shape", [(), (3,)])
def test_cell_average_gradient_is_affine_exact_and_has_exact_adjoint(shape):
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        3,
        2,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, shape)) if shape else fem.functionspace(domain, ("Lagrange", 1))
    field = fem.Function(space, name="U")
    if shape:
        gradient = np.array(((2.0, -3.0), (0.5, 1.25), (-1.0, 4.0)))
        offset = np.array((4.0, -2.0, 1.0))
        field.interpolate(lambda x: gradient @ x[:2] + offset[:, None])
    else:
        gradient = np.array((2.0, -3.0))
        field.interpolate(lambda x: gradient @ x[:2] + 4.0)
    transfer = operators.cell_average_gradient(space)
    result = transfer.apply(field)
    block_size = int(result.function_space.dofmap.index_map_bs)
    owned = domain.topology.index_map(domain.topology.dim).size_local
    cell_gradients = np.empty((owned, *transfer.value_shape))
    for cell in range(owned):
        dof = int(result.function_space.dofmap.cell_dofs(cell)[0])
        cell_gradients[cell] = result.x.array[
            dof * block_size : (dof + 1) * block_size
        ].reshape(transfer.value_shape)
    np.testing.assert_allclose(
        cell_gradients,
        np.broadcast_to(gradient, cell_gradients.shape),
        atol=2.0e-14,
    )

    rng = np.random.default_rng(20260920)
    cell_map = domain.topology.index_map(domain.topology.dim)
    duals = rng.normal(
        size=(cell_map.size_local + cell_map.num_ghosts, *transfer.value_shape)
    )
    adjoint = transfer.apply_adjoint(duals)
    lhs = np.vdot(result.x.array.reshape((-1, *transfer.value_shape)), duals)
    rhs = np.vdot(field.x.petsc_vec.array_r, adjoint.array_r)
    assert lhs == pytest.approx(rhs, rel=2.0e-13, abs=2.0e-13)
    adjoint.destroy()


def test_cell_average_gradient_rejects_field_from_another_space():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    source = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    other = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    transfer = operators.cell_average_gradient(source)
    with pytest.raises(ValueError, match="source_space"):
        transfer.apply(fem.Function(other))
