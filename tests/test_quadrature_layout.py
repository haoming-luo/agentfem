# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Backend-owned quadrature layout agrees with cell-wise indexing, also MPI."""

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem.constitutive.quadrature import QuadratureField


@pytest.mark.parametrize("shape", [(), (6,), (3, 3)])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("cell", [mesh.CellType.tetrahedron, mesh.CellType.hexahedron])
def test_bulk_layout_matches_backend_cell_dofs(shape, degree, cell):
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 2, 2, cell_type=cell)
    field = QuadratureField.create(
        domain, name="history", degree=degree, value_shape=shape
    )
    V = field.function.function_space
    cell_map = domain.topology.index_map(3)
    n = cell_map.size_local + cell_map.num_ghosts
    expected = np.concatenate(
        [
            (
                np.asarray(V.dofmap.cell_dofs(c), dtype=np.int64)[:, None] * V.dofmap.bs
                + np.arange(V.dofmap.bs)
            ).ravel()
            for c in range(n)
        ]
    )
    actual = field._array_indices()
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert field._array_indices() is actual

    # Distinct rank-local values are scattered first; compare with the actual
    # ghosted backend array, not an assumption about local cell numbering.
    field.function.x.array[:] = (
        np.arange(field.function.x.array.size) + 10000 * MPI.COMM_WORLD.rank
    )
    field.function.x.scatter_forward()
    np.testing.assert_array_equal(
        field.values.ravel(), field.function.x.array[expected]
    )
    original = field.values.copy()
    field.assign(original)
    np.testing.assert_array_equal(field.values, original)
