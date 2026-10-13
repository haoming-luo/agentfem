# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI
from agentfem.results import cell_field_from_values


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_cell_values_preserve_order_identity_and_input(shape):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2, cell_type=mesh.CellType.hexahedron)
    count = domain.topology.index_map(3).size_local
    values = np.arange(count*int(np.prod(shape)), dtype=float).reshape((count, *shape))
    original = domain.topology.original_cell_index.copy()
    permutation = np.random.default_rng(81).permutation(count)
    saved = values.copy()
    direct = cell_field_from_values(domain, values, name="direct")
    keyed = cell_field_from_values(domain, values[permutation], name="keyed",
                                   cell_keys=original[permutation])
    np.testing.assert_array_equal(direct.x.array, keyed.x.array)
    dofs = np.asarray(keyed.function_space.dofmap.list)[:, 0]
    np.testing.assert_array_equal(keyed.x.array.reshape(count, -1)[dofs], values.reshape(count, -1))
    keyed.x.array[:] = 0
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize("fault", ["nan", "complex", "count", "duplicate", "unknown", "float_keys"])
def test_cell_values_reject_wrong_data(fault):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1, cell_type=mesh.CellType.hexahedron)
    values = np.ones(2)
    keys = domain.topology.original_cell_index.copy()
    if fault == "nan": values[0] = np.nan
    elif fault == "complex": values = values.astype(complex)
    elif fault == "count": values = values[:1]
    elif fault == "duplicate": keys[:] = keys[0]
    elif fault == "unknown": keys += 1000
    else: keys = keys.astype(float)
    with pytest.raises(ValueError):
        cell_field_from_values(domain, values, name="data", cell_keys=keys)
