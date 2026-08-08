from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI
import pytest
import ufl

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    fields,
    fracture,
    interfaces,
    models,
    operators,
    studies,
)


def _split_strip():
    coordinates = np.asarray(
        [(x / 3.0, y / 2.0) for y in range(3) for x in range(4)],
        dtype=float,
    )
    cells = np.asarray(
        [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [4, 5, 9, 8],
            [5, 6, 10, 9],
            [6, 7, 11, 10],
        ],
        dtype=int,
    )
    return interfaces.split_conforming_line_interface(
        coordinates,
        cells,
        [[4, 5], [5, 6], [6, 7]],
        positive_cells=[3, 4, 5],
    )


def _law():
    return interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )


def _distributed_force(split, displacement):
    return fracture.mode_i_cohesive_force(
        split,
        displacement,
        _law(),
        normal_hint=(0.0, 1.0),
    )


def test_distributed_split_interface_assembles_each_physical_facet_once():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed cohesive acceptance requires exactly two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    cohesive = _distributed_force(split, displacement)
    assert isinstance(cohesive, fracture.DistributedDofMappedCohesiveForce)

    values = displacement.value.x.array.reshape((-1, 2))
    positive_nodes = set(int(value) for value in split.positive_facets.reshape(-1))
    for node in np.flatnonzero(cohesive.input_node_owned):
        if int(node) in positive_nodes:
            values[int(cohesive.node_to_block_dof[node]), 1] = _law().peak_opening
    displacement.value.x.scatter_forward()

    zero = operators.OperatorForm(
        name="zero_bulk",
        expression=ufl.inner(
            fem.Constant(domain, np.asarray((0.0, 0.0))), displacement.test
        )
        * ufl.dx,
        kind="zero_bulk_force",
        role="vector",
        family="test",
    )
    residual = fracture.FiniteStrainCohesiveResidual(zero, cohesive)
    vector = residual.assemble_vector()
    try:
        array = vector.array.reshape((-1, 2))
        local = {}
        interface_nodes = set(
            int(value)
            for value in np.concatenate(
                (split.negative_facets.reshape(-1), split.positive_facets.reshape(-1))
            )
        )
        for node in np.flatnonzero(cohesive.input_node_owned):
            if int(node) in interface_nodes:
                local[int(node)] = array[int(cohesive.node_to_block_dof[node])].copy()
        assembled = {}
        for payload in comm.allgather(local):
            assembled.update(payload)
        total = np.sum(np.asarray(list(assembled.values())), axis=0)
        positive_force = sum(assembled[node][1] for node in positive_nodes)
        np.testing.assert_allclose(total, 0.0, atol=1.0e-12)
        assert positive_force == pytest.approx(_law().strength)
        residual.commit()
        response = cohesive.current_response()
        assert response.stored_energy == pytest.approx(
            0.5 * _law().strength * _law().peak_opening
        )
        assert sum(comm.allgather(cohesive.assembler.topology.number_of_facets)) == 3
        snapshot = residual.snapshot()
        assert len(snapshot["cohesive"]["maximum_opening_by_key"]) == 3
        assert all(item == snapshot for item in comm.allgather(snapshot))
    finally:
        vector.destroy()


def test_distributed_cohesive_force_runs_through_public_explicit_step():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed cohesive Explicit acceptance requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(young=1000.0, poisson=0.25, density=1.0)
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 0.0),
            value=0.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda x: np.isclose(x[0], 0.0),
            value=0.0,
        )
    )
    model.constraint(
        constraints.time_dependent_component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 1.0),
            amplitude=amplitudes.ramp(
                0.0,
                1.0e-4,
                start_time=0.0,
                end_time=1.0e-5,
            ),
        )
    )
    cohesive = _distributed_force(split, displacement)
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=1.0e-5,
        steps=1,
        progress=False,
    )
    step.run()

    assert step.completed_steps == 1
    assert step.residual.cohesive.summary()["global_facets"] == 3
    assert step.residual.summary()["maturity"] == (
        "experimental_mpi_reference_consumer"
    )
    assert np.isfinite(step.history_records[-1]["energy_balance_error"])
    histories = comm.allgather(step.history_records)
    assert all(item == histories[0] for item in histories[1:])


def test_distributed_cohesive_contract_rejects_rank_inconsistent_orientation():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed cohesive contract requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    normal = (0.0, 1.0) if comm.rank == 0 else (0.0, -1.0)

    with pytest.raises(ValueError, match="Every MPI rank must declare the same"):
        fracture.mode_i_cohesive_force(
            split,
            displacement,
            _law(),
            normal_hint=normal,
        )
