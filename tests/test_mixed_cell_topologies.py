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


def test_linear_prism_partition_preserves_quality_and_affine_patch():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed prism evidence requires at least two MPI ranks")
    rank = MPI.COMM_WORLD.rank
    if rank == 0:
        base = [(i / 2.0, j / 2.0) for j in range(3) for i in range(3)]
        coordinates = np.asarray(
            [(x, y, z) for z in (0.0, 1.0) for x, y in base],
            dtype=float,
        )
        cells = []
        for j in range(2):
            for i in range(2):
                lower_left = j * 3 + i
                lower_right = lower_left + 1
                upper_left = (j + 1) * 3 + i
                upper_right = upper_left + 1
                cells.extend(
                    (
                        (
                            lower_left,
                            lower_right,
                            upper_right,
                            lower_left + 9,
                            lower_right + 9,
                            upper_right + 9,
                        ),
                        (
                            lower_left,
                            upper_right,
                            upper_left,
                            lower_left + 9,
                            upper_right + 9,
                            upper_left + 9,
                        ),
                    )
                )
        cells = np.asarray(cells, dtype=np.int64)
    else:
        coordinates = np.empty((0, 3), dtype=float)
        cells = np.empty((0, 6), dtype=np.int64)

    coordinate_element = ufl.Mesh(
        element("Lagrange", "prism", 1, shape=(3,))
    )
    domain = mesh.from_arrays(
        cells=cells,
        coordinates=coordinates,
        coordinate_element=coordinate_element,
        comm=MPI.COMM_WORLD,
    )
    cell_map = domain.topology.index_map(domain.topology.dim)
    quality = mesh.audit_quality(domain, threshold=0.6, strict=True)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space)
    solution.interpolate(
        lambda x: x[0] + 2.0 * x[1] - 0.5 * x[2] + 0.25
    )
    difference = ufl.grad(solution) - ufl.as_vector((1.0, 2.0, -0.5))
    local_error = fem.assemble_scalar(
        fem.form(ufl.inner(difference, difference) * ufl.dx)
    )
    global_error = MPI.COMM_WORLD.allreduce(local_error, op=MPI.SUM)

    assert cell_map.size_global == 8
    assert cell_map.size_local > 0
    assert quality.acceptable
    assert quality.minimum > 0.7
    assert global_error == pytest.approx(0.0, abs=5.0e-14)
    assert mesh.describe_topology_compatibility("prism").release_ready
