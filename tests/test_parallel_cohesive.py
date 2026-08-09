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
    return interfaces.split_conforming_cell_interface(
        coordinates,
        cells,
        positive_cells=[3, 4, 5],
    )


def _law():
    return interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )


def _split_surface_3d():
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, -1.0],
            [0.5, 0.5, 1.0],
        ]
    )
    cells = np.asarray(
        [
            [0, 1, 2, 4],
            [0, 2, 3, 4],
            [0, 2, 1, 5],
            [0, 3, 2, 5],
        ],
        dtype=int,
    )
    return interfaces.split_conforming_cell_interface(
        coordinates, cells, positive_cells=[2, 3]
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


def test_three_dimensional_cohesive_surface_uses_shared_sparse_mpi_contract():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("3D cohesive surface MPI acceptance requires two ranks")
    split = _split_surface_3d()
    domain = interfaces.create_dolfinx_split_mesh(
        split, comm=comm, cell_type="tetrahedron"
    )
    displacement = fields.displacement(domain)
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        _law(),
        normal_hint=(0.0, 0.0, 1.0),
    )
    values = displacement.value.x.array.reshape((-1, 3))
    positive_nodes = set(int(value) for value in split.positive_facets.reshape(-1))
    for node in np.flatnonzero(cohesive.input_node_owned):
        if int(node) in positive_nodes:
            values[int(cohesive.node_to_block_dof[node]), 2] = _law().peak_opening
    displacement.value.x.scatter_forward()
    zero = operators.OperatorForm(
        name="zero_bulk_3d",
        expression=ufl.inner(
            fem.Constant(domain, np.asarray((0.0, 0.0, 0.0))), displacement.test
        )
        * ufl.dx,
        kind="zero_bulk_force",
        role="vector",
        family="test",
    )
    residual = fracture.FiniteStrainCohesiveResidual(zero, cohesive)
    vector = residual.assemble_vector()
    try:
        local_z = 0.0
        array = vector.array.reshape((-1, 3))
        for node in np.flatnonzero(cohesive.input_node_owned):
            if int(node) in positive_nodes:
                local_z += float(array[int(cohesive.node_to_block_dof[node]), 2])
        assert comm.allreduce(local_z, op=MPI.SUM) == pytest.approx(_law().strength)
        assert cohesive.summary()["interface"]["kind"] == (
            "paired_triangular_surface_facets"
        )
        assert sum(
            comm.allgather(cohesive.assembler.topology.number_of_facets)
        ) == 2
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
        "experimental_mpi_sparse_consumer"
    )
    assert np.isfinite(step.history_records[-1]["energy_balance_error"])
    histories = comm.allgather(step.history_records)
    assert all(item == histories[0] for item in histories[1:])


def test_distributed_cohesive_exchange_is_interface_sparse():
    comm = MPI.COMM_WORLD
    if comm.size not in {2, 3}:
        pytest.skip("distributed cohesive schedule requires two or three ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    cohesive = _distributed_force(split, displacement)

    summary = cohesive.summary()
    communication = summary["communication"]
    assert summary["parallel_scope"] == "mpi_sparse_owner_exchange"
    assert summary["force_exchange"] == "physical_input_nodes+sparse_alltoallv"
    assert summary["local_assembly_layout"] == "interface_compact"
    assert communication["rank_required_nodes"] <= len(cohesive.interface_nodes)
    assert cohesive.assembler.number_of_nodes == communication["rank_required_nodes"]
    assert cohesive.assembler.number_of_nodes < split.coordinates.shape[0]
    assert communication["rank_remote_trace_nodes"] <= communication[
        "rank_required_nodes"
    ]
    assert communication["rank_trace_values_per_exchange"] == (
        communication["rank_remote_trace_nodes"] * 2
    )
    assert communication["rank_force_values_per_exchange"] == (
        communication["rank_remote_trace_nodes"] * 2
    )
    assert communication["rank_trace_values_per_exchange"] < (
        split.coordinates.shape[0] * 2
    )
    records = comm.allgather(communication)
    assert any(item["rank_remote_trace_nodes"] > 0 for item in records)


def test_sparse_exchange_matches_serial_nonuniform_interface_response():
    comm = MPI.COMM_WORLD
    if comm.size not in {2, 3}:
        pytest.skip("distributed cohesive equivalence requires two or three ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    cohesive = _distributed_force(split, displacement)
    values_by_x = {
        0.0: -0.25 * _law().peak_opening,
        1.0 / 3.0: 0.50 * _law().peak_opening,
        2.0 / 3.0: 1.50 * _law().peak_opening,
        1.0: 0.75 * _law().failure_opening,
    }
    positive_nodes = set(int(value) for value in split.positive_facets.reshape(-1))
    local_values = displacement.value.x.array.reshape((-1, 2))
    prescribed = np.zeros((split.coordinates.shape[0], 2), dtype=float)
    for node in positive_nodes:
        x = float(split.coordinates[node, 0])
        opening = values_by_x[min(values_by_x, key=lambda value: abs(value - x))]
        prescribed[node, 1] = opening
    for node in np.flatnonzero(cohesive.input_node_owned):
        local_values[int(cohesive.node_to_block_dof[node])] = prescribed[int(node)]
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
        local_force = {
            int(node): array[int(cohesive.node_to_block_dof[node])].copy()
            for node in np.flatnonzero(cohesive.input_node_owned)
            if int(node) in cohesive.interface_nodes
        }
        distributed = {}
        for payload in comm.allgather(local_force):
            distributed.update(payload)

        topology = interfaces.pair_coincident_line_facets(
            split.coordinates,
            split.negative_facets,
            split.positive_facets,
            normal_hint=(0.0, 1.0),
        )
        reference_assembler = interfaces.ModeICohesiveFacetAssembler(
            topology,
            _law(),
            number_of_nodes=split.coordinates.shape[0],
        )
        reference = reference_assembler.begin(prescribed)
        for node in cohesive.interface_nodes:
            np.testing.assert_allclose(
                distributed[int(node)],
                reference.internal_force[int(node)],
                rtol=1.0e-13,
                atol=1.0e-13,
            )
        response = cohesive.current_response()
        assert response.stored_energy == pytest.approx(reference.stored_energy)
        assert response.dissipated_energy == pytest.approx(
            reference.dissipated_energy
        )
    finally:
        vector.destroy()
        residual.rollback()


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
