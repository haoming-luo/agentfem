from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
import pytest
import ufl

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    fatigue_fracture,
    fields,
    fracture,
    interfaces,
    models,
    operators,
    results,
    solvers,
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


def test_distributed_cohesive_tangent_matches_global_force_derivative():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed cohesive tangent acceptance requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    cohesive = _distributed_force(split, displacement)
    positive_nodes = set(int(value) for value in split.positive_facets.reshape(-1))
    values = displacement.value.x.array.reshape((-1, 2))
    for node in np.flatnonzero(cohesive.input_node_owned):
        if int(node) in positive_nodes:
            values[int(cohesive.node_to_block_dof[node]), 1] = 0.5 * _law().peak_opening
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
    matrix = operators.assemble_matrix(
        ufl.inner(displacement.trial, displacement.test) * ufl.dx
    )
    matrix.zeroEntries()
    cohesive.add_to_matrix(matrix)
    matrix.assemble()
    direction = displacement.value.x.petsc_vec.duplicate()
    action = displacement.value.x.petsc_vec.duplicate()
    coordinates = displacement.space.tabulate_dof_coordinates()
    direction_values = direction.array.reshape((-1, 2))
    direction_values[:, 0] = 0.1 + 0.2 * coordinates[:, 0]
    direction_values[:, 1] = -0.3 + 0.4 * coordinates[:, 1]
    direction.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    matrix.mult(direction, action)
    base = displacement.value.x.array.copy()
    epsilon = 1.0e-7
    displacement.value.x.array[:] = base + epsilon * direction.array
    displacement.value.x.scatter_forward()
    plus = residual.assemble_vector()
    residual.rollback()
    displacement.value.x.array[:] = base - epsilon * direction.array
    displacement.value.x.scatter_forward()
    minus = residual.assemble_vector()
    residual.rollback()
    derivative = plus.copy()
    derivative.axpy(-1.0, minus)
    derivative.scale(0.5 / epsilon)
    derivative.axpy(-1.0, action)
    try:
        assert derivative.norm() < 1.0e-8
    finally:
        displacement.value.x.array[:] = base
        displacement.value.x.scatter_forward()
        derivative.destroy()
        minus.destroy()
        plus.destroy()
        action.destroy()
        direction.destroy()
        matrix.destroy()


def test_distributed_native_cohesive_newton_force_control_and_reaction():
    from dolfinx import mesh as dolfinx_mesh

    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed cohesive Newton acceptance requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    material = constitutive.neo_hookean(young=100.0, poisson=0.25)
    internal = fracture.finite_strain_internal_force(
        displacement.value, displacement.test, material
    )
    facet_dimension = domain.topology.dim - 1
    top_facets = dolfinx_mesh.locate_entities_boundary(
        domain, facet_dimension, lambda x: np.isclose(x[1], 1.0)
    )
    tags = dolfinx_mesh.meshtags(
        domain,
        facet_dimension,
        np.sort(top_facets),
        np.ones(top_facets.size, dtype=np.int32),
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    load_parameter = fem.Constant(domain, 0.0)
    external = load_parameter * displacement.test[1] * ds(1)
    bulk = operators.residual_operator(
        internal.expression - external,
        name="R_bulk",
        family="total_lagrangian_neo_hookean",
    )
    tangent = operators.linearize(bulk, displacement)
    cohesive = _distributed_force(split, displacement)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)
    bcs = [
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 0.0),
            value=0.0,
        ).bc,
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda x: np.isclose(x[0], 0.0),
            value=0.0,
        ).bc,
    ]
    equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
        residual,
        tangent,
        displacement,
        bcs=bcs,
        load_parameter=load_parameter,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            maximum_iterations=20,
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
        reaction=lambda _function: results.reaction_resultant(
            residual,
            on=lambda x: np.isclose(x[1], 0.0),
            component=1,
        ),
    )
    point = equilibrium(load=0.5, branch="maximum", cycle=1)

    assert point["converged"] is True
    assert point["reaction"] == pytest.approx(-0.5, rel=1.0e-8, abs=1.0e-10)
    assert equilibrium.last_info.residual_norm < 1.0e-8
    assert all(item == point for item in comm.allgather(point))


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


def test_distributed_vector_cohesive_interface_transfers_shear_and_restarts():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed vector cohesive acceptance requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=500.0,
        interaction="power",
        interaction_exponent=1.5,
    )
    cohesive = fracture.cohesive_force(
        split, displacement, law, normal_hint=(0.0, 1.0)
    )
    values = displacement.value.x.array.reshape((-1, 2))
    positive_nodes = set(int(value) for value in split.positive_facets.reshape(-1))
    for node in np.flatnonzero(cohesive.input_node_owned):
        if int(node) in positive_nodes:
            values[int(cohesive.node_to_block_dof[node])] = (0.002, 0.005)
    displacement.value.x.scatter_forward()
    zero = operators.OperatorForm(
        name="zero_bulk_vector",
        expression=ufl.inner(
            fem.Constant(domain, np.asarray((0.0, 0.0))), displacement.test
        ) * ufl.dx,
        kind="zero_bulk_force",
        role="vector",
        family="test",
    )
    residual = fracture.FiniteStrainCohesiveResidual(zero, cohesive)
    vector = residual.assemble_vector()
    try:
        local = np.zeros(2)
        array = vector.array.reshape((-1, 2))
        for node in np.flatnonzero(cohesive.input_node_owned):
            if int(node) in positive_nodes:
                local += array[int(cohesive.node_to_block_dof[node])]
        resultant = np.asarray(comm.allreduce(local, op=MPI.SUM))
        np.testing.assert_allclose(resultant, [1.0, 5.0], rtol=1.0e-12)
        residual.commit()
        assert set(cohesive.interface_quantities()) == {
            "JUMP_N", "JUMP_T", "TRACTION_N", "TRACTION_T", "DAMAGE", "MODE_MIXITY"
        }
        snapshot = cohesive.snapshot()
        assert set(snapshot["state_by_field_and_key"]) == {
            "failure_separation",
            "initiation_separation",
            "initiation_stiffness",
            "maximum_effective_separation",
        }
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

    profile = cohesive.performance_profile()
    assert profile["schema"] == "agentfem.cohesive-mpi-profile.v1"
    assert profile["global_facets"] == 3
    assert profile["rank_count"] == comm.size
    assert profile["global_trace_values_per_exchange"] == sum(
        item["rank_trace_values_per_exchange"] for item in records
    )
    assert profile["global_force_values_per_exchange"] == sum(
        item["rank_force_values_per_exchange"] for item in records
    )
    assert profile["maximum_facet_imbalance"] >= 1.0
    assert set(profile["petsc_events"]) == {"constitutive", "vector", "matrix"}
    assert all(item == profile for item in comm.allgather(profile))


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


def test_distributed_cyclic_cohesive_snapshot_retains_every_state_field():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("distributed cyclic cohesive state requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    law = fatigue_fracture.cyclic_cohesive(
        monotonic=_law(),
        fatigue_coefficient=2.0e-3,
        fatigue_exponent=2.0,
        range_threshold=0.05,
        residual_exponent=1.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    count = cohesive.assembler.state.size
    cohesive.assembler.state.begin_cycle(
        np.full(count, 0.01),
        np.full(count, 0.20),
        cycles=123,
    )
    cohesive.assembler.state.commit_cycle()
    snapshot = cohesive.snapshot()
    assert set(snapshot["state_by_field_and_key"]) == set(
        cohesive.assembler.state.state_arrays()
    )
    assert all(item == snapshot for item in comm.allgather(snapshot))

    restored = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    restored.restore(snapshot)
    for name, values in cohesive.assembler.state.state_arrays().items():
        np.testing.assert_allclose(
            restored.assembler.state.state_arrays()[name], values
        )


def test_cyclic_bulk_field_checkpoint_roundtrips_on_same_two_rank_partition(tmp_path):
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("cyclic bulk checkpoint acceptance requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    displacement.value.x.array[:] = float(comm.rank + 1)
    expected = displacement.value.x.array.copy()
    state = fatigue_fracture.field_state(displacement=displacement)
    root_path = str(tmp_path / "mpi-cycle") if comm.rank == 0 else None
    path = Path(comm.bcast(root_path, root=0))
    manifest = state.save_checkpoint(path)
    displacement.value.x.array[:] = -1.0
    state.load_checkpoint(manifest)
    np.testing.assert_allclose(displacement.value.x.array, expected)


def test_global_cycle_controller_keeps_two_rank_interface_state_in_lockstep():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("global cyclic lifecycle acceptance requires two ranks")
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=comm)
    displacement = fields.displacement(domain)
    law = fatigue_fracture.cyclic_cohesive(
        monotonic=_law(),
        fatigue_coefficient=2.0e-3,
        fatigue_exponent=2.0,
        range_threshold=0.05,
        residual_exponent=1.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    collection = fracture.named_cohesive_forces(crack=cohesive)
    positive_nodes = set(int(value) for value in split.positive_facets.reshape(-1))

    def solve_equilibrium(*, load, branch, cycle):
        damage = float(
            np.max(cohesive.assembler.state.fatigue_damage, initial=0.0)
        )
        opening = 0.2 * float(load) * (1.0 + 0.1 * damage)
        values = displacement.value.x.array.reshape((-1, 2))
        values[:] = 0.0
        for node in np.flatnonzero(cohesive.input_node_owned):
            if int(node) in positive_nodes:
                values[int(cohesive.node_to_block_dof[node]), 1] = opening
        displacement.value.x.scatter_forward()
        return {"iterations": 1, "energy_balance_error": 0.0}

    step = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=fatigue_fracture.force_cycle(fmin=0.1, fmax=1.0),
        stop_cycle=5,
        interfaces=collection,
        state=fatigue_fracture.field_state(displacement=displacement),
        solve_equilibrium=solve_equilibrium,
        landing_cycles=(5,),
    )
    step.run()
    assert step.current_cycle == 5
    assert all(item == step.ledger.snapshot() for item in comm.allgather(step.ledger.snapshot()))
    np.testing.assert_allclose(
        cohesive.assembler.state.cumulative_cycles,
        5.0,
    )
