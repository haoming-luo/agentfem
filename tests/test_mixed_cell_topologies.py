from __future__ import annotations

import basix
import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import fem
from mpi4py import MPI

from agentfem import mesh


def _reference_domain(cell_name: str):
    cell_type = getattr(basix.CellType, cell_name)
    coordinates = np.asarray(basix.cell.geometry(cell_type), dtype=float)
    coordinate_element = ufl.Mesh(
        element("Lagrange", cell_name, 1, shape=(coordinates.shape[1],))
    )
    return mesh.from_arrays(
        cells=[list(range(coordinates.shape[0]))],
        coordinates=coordinates,
        coordinate_element=coordinate_element,
        comm=MPI.COMM_SELF,
    )


@pytest.mark.parametrize("cell_name", ("prism", "pyramid"))
def test_mixed_facet_topology_passes_conforming_p1_affine_patch(cell_name):
    domain = _reference_domain(cell_name)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space)
    exact_gradient = np.asarray((1.0, 2.0, -0.5))
    solution.interpolate(
        lambda x: x[0] + 2.0 * x[1] - 0.5 * x[2] + 0.25
    )
    exact_gradient_expression = ufl.as_vector(tuple(exact_gradient))
    gradient_difference = ufl.grad(solution) - exact_gradient_expression
    gradient_error = fem.assemble_scalar(
        fem.form(
            ufl.inner(gradient_difference, gradient_difference) * ufl.dx
        )
    )
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    operator = fem.petsc.assemble_matrix(
        fem.form(
            (
                ufl.inner(trial, test)
                + ufl.inner(ufl.grad(trial), ufl.grad(test))
            )
            * ufl.dx
        )
    )
    operator.assemble()

    assert gradient_error == pytest.approx(0.0, abs=2.0e-14)
    assert operator.getSize() == (space.dofmap.index_map.size_global,) * 2
    assert operator.norm() > 0.0
    operator.destroy()

    topology = mesh.describe_topology_compatibility(cell_name)
    capability = {
        item.name: item for item in topology.capabilities
    }["conforming_p1_patch"]
    assert capability.verified
    assert "affine-gradient" in capability.scope
