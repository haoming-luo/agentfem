from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, constraints, results
from agentfem.mesh import abaqus


pytest.importorskip("dolfinx_mpc")


def test_distributed_abaqus_equation_mapping_and_source_order():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(space, name="Displacement")
    nodes = abaqus.AbaqusNodeTable(
        labels=np.arange(1, 10),
        coordinates=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ]
        ),
    )
    equations = abaqus.AbaqusEquationSet(
        tuple(
            abaqus.LinearEquation(
                (
                    abaqus.EquationTerm(5, component, -1.0),
                    abaqus.EquationTerm(4, component, 1.0),
                    abaqus.EquationTerm(2, component, 1.0),
                    abaqus.EquationTerm(1, component, -1.0),
                )
            )
            for component in (1, 2)
        )
    )
    target_f = np.diag([1.10, 1.0 / 1.10])
    periodicity = constraints.abaqus_periodic_cell(
        displacement,
        nodes=nodes,
        equations=equations,
        deformation_gradient=target_f,
        anchor_node=1,
        reference_nodes=(2, 3),
    )

    reduction = periodicity.distributed_reduction()
    correction = reduction.correction()
    reduction.validate_prefix_layout(correction)
    periodicity.apply_affine_increment(0.0, 1.0)

    global_slaves = domain.comm.allreduce(
        reduction.mpc.num_local_slaves,
        op=MPI.SUM,
    )
    source_values = abaqus.displacement_in_source_order(
        displacement,
        nodes,
    )

    assert global_slaves == 2
    assert reduction.reduced_size == space.dofmap.index_map.size_global * 2 - 8
    assert periodicity.mismatch() < 1.0e-13
    np.testing.assert_allclose(
        source_values,
        nodes.coordinates @ (target_f - np.eye(2)).T,
        atol=1.0e-13,
    )


def test_parallel_element_volume_writes_owned_then_ghost_values():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(space, name="Displacement")
    material = constitutive.neo_hookean(young=1000.0, poisson=0.3)

    (element_volume,) = results.finite_strain_cell_fields(
        displacement,
        material,
        variables=("EVOL",),
    )
    owned = element_volume.function_space.dofmap.index_map.size_local
    total = domain.comm.allreduce(
        float(np.sum(element_volume.x.array[:owned])),
        op=MPI.SUM,
    )

    assert total == pytest.approx(1.0)
