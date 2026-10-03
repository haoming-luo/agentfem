"""Regression for spurious periodic masters from high-order basis roundoff."""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from agentfem import constraints

pytest.importorskip("dolfinx_mpc")


@pytest.mark.parametrize("length", [1.0, 2.0])
def test_cubic_periodic_nodes_have_one_unit_master(length):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [np.zeros(2), np.full(2, length)], [32, 32]
    )
    space = fem.functionspace(domain, ("Lagrange", 3))
    periodicity = constraints.rectangular_periodic_mpc(space)
    diagnostics = periodicity.diagnostics()
    # A cubic 32x32 grid has 97 nodes per edge, with one shared corner.
    assert diagnostics["global_slave_dofs"] == 193
    assert diagnostics["global_master_relations"] == 193
    assert diagnostics["multiply_matched_slave_dofs"] == 0
    assert diagnostics["nonunit_coefficients_detected"] is False
    assert diagnostics["status"] == "valid"
