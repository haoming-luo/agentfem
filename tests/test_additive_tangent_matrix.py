# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from agentfem.backends import create_additive_tangent_matrix


def _dense_matrix(values, comm=MPI.COMM_SELF):
    values = np.asarray(values, dtype=PETSc.ScalarType)
    matrix = PETSc.Mat().createDense(values.shape, array=values.copy(), comm=comm)
    matrix.assemble()
    return matrix


def test_additive_tangent_combines_local_matrix_and_matrix_free_actions():
    local_values = np.array(
        ((4.0, -1.0, 0.0), (-1.0, 3.0, -0.5), (0.0, -0.5, 2.0))
    )
    nonlocal_values = np.array(
        ((0.2, 0.1, -0.3), (0.1, 0.4, 0.2), (-0.3, 0.2, 0.6))
    )
    local = _dense_matrix(local_values)

    def action(source, target):
        target.array[:] = nonlocal_values @ source.array_r

    additive = create_additive_tangent_matrix(local, (action,))
    source = local.createVecRight()
    target = local.createVecLeft()
    source.array[:] = (0.3, -0.7, 1.1)

    additive.operator.mult(source, target)

    np.testing.assert_allclose(
        target.array_r,
        (local_values + nonlocal_values) @ source.array_r,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert additive.preconditioner is local
    assert additive.summary()["matrix_free_action_count"] == 1
    additive.close()
    assert additive.closed
    assert local.getSize() == (3, 3)
    source.destroy()
    target.destroy()
    local.destroy()


def test_additive_tangent_applies_constraints_once_at_composition_boundary():
    local_values = np.diag((1.0, 3.0, 2.0))
    nonlocal_values = np.ones((3, 3))
    local = _dense_matrix(local_values)
    observed = []

    def action(source, target):
        observed.append(source.array_r.copy())
        target.array[:] = nonlocal_values @ source.array_r

    with create_additive_tangent_matrix(
        local,
        (action,),
        constrained_local_dofs=(0,),
    ) as additive:
        source = local.createVecRight()
        target = local.createVecLeft()
        source.array[:] = (2.0, -0.5, 0.25)
        additive.operator.mult(source, target)
        np.testing.assert_allclose(observed[0], (0.0, -0.5, 0.25))
        np.testing.assert_allclose(target.array_r, (2.0, -1.75, 0.25))
        source.destroy()
        target.destroy()

    local.destroy()


def test_additive_tangent_rejects_invalid_contracts():
    local = _dense_matrix(np.eye(2))
    with pytest.raises(TypeError, match="callable"):
        create_additive_tangent_matrix(local, (object(),))
    with pytest.raises(ValueError, match="locally owned"):
        create_additive_tangent_matrix(local, constrained_local_dofs=(2,))
    local.destroy()
