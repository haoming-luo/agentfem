import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    fields,
    fracture,
    interfaces,
    mesh,
    models,
    operators,
    problems,
    studies,
    time,
)


def test_serial_dof_adapter_adds_cohesive_force_to_one_global_residual():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    block_size = displacement.space.dofmap.index_map_bs
    number_of_blocks = displacement.value.x.array.size // block_size
    assert number_of_blocks == 4

    interface_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    )
    topology = interfaces.pair_coincident_line_facets(
        interface_coordinates,
        [[0, 1]],
        [[2, 3]],
        normal_hint=(0.0, 1.0),
    )
    law = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )
    local = interfaces.ModeICohesiveFacetAssembler(
        topology,
        law,
        number_of_nodes=4,
    )
    mapping = np.arange(4, dtype=int)
    cohesive = fracture.DofMappedCohesiveForce(
        local,
        displacement,
        node_to_block_dof=mapping,
    )
    values = displacement.value.x.array.reshape((-1, block_size))
    values[mapping[2:], 1] = law.peak_opening
    displacement.value.x.scatter_forward()

    zero = operators.OperatorForm(
        name="zero_bulk",
        expression=ufl.inner(
            fem.Constant(domain, np.array((0.0, 0.0))),
            displacement.test,
        )
        * ufl.dx,
        kind="zero_bulk_force",
        role="vector",
        family="test",
    )
    residual = fracture.FiniteStrainCohesiveResidual(zero, cohesive)
    vector = residual.assemble_vector()
    try:
        assembled = vector.array.reshape((-1, block_size))
        np.testing.assert_allclose(np.sum(assembled, axis=0), 0.0, atol=1.0e-12)
        assert np.linalg.norm(assembled) > 0.0
        residual.commit()
        np.testing.assert_allclose(
            local.state.committed_maximum,
            law.peak_opening,
        )
    finally:
        vector.destroy()


def test_split_mesh_builds_executable_domain_and_recovers_coincident_dofs():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [3, 2, 4, 5]])
    split = interfaces.split_conforming_line_interface(
        coordinates,
        cells,
        [[3, 2]],
        positive_cells=[1],
    )
    domain = interfaces.create_dolfinx_split_mesh(split)
    displacement = fields.displacement(domain)
    law = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    mapping = cohesive.node_to_block_dof
    assert mapping.size == split.coordinates.shape[0]
    assert np.unique(mapping).size == mapping.size
    negative = mapping[split.negative_facets.reshape(-1)]
    positive = mapping[split.positive_facets.reshape(-1)]
    assert not set(negative).intersection(positive)

    values = displacement.value.x.array.reshape((-1, 2))
    values[positive, 1] = law.peak_opening
    displacement.value.x.scatter_forward()
    response = cohesive.begin()
    np.testing.assert_allclose(response.opening, law.peak_opening)
    np.testing.assert_allclose(np.sum(response.internal_force, axis=0), 0.0)
    cohesive.rollback()


def test_split_interface_runs_through_prescribed_dynamic_energy_lifecycle():
    coordinates = np.array(
        [
            [0.0, 0.0], [1.0, 0.0],
            [1.0, 0.5], [0.0, 0.5],
            [1.0, 1.0], [0.0, 1.0],
        ]
    )
    split = interfaces.split_conforming_line_interface(
        coordinates,
        np.array([[0, 1, 2, 3], [3, 2, 4, 5]]),
        [[3, 2]],
        positive_cells=[1],
    )
    domain = interfaces.create_dolfinx_split_mesh(split)
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2, assumption="plane_strain", method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(young=1000.0, poisson=0.25, density=1.0)
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement, 1, on=lambda x: np.isclose(x[1], 0.0), value=0.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement, 0, on=lambda x: np.isclose(x[0], 0.0), value=0.0,
        )
    )
    top_motion = constraints.time_dependent_component_dirichlet(
        displacement,
        1,
        on=lambda x: np.isclose(x[1], 1.0),
        amplitude=amplitudes.ramp(
            0.0, 1.0e-3, start_time=0.0, end_time=2.0e-4,
        ),
    )
    model.constraint(top_motion)
    law = interfaces.bilinear_cohesive(
        strength=1.0,
        fracture_energy=0.01,
        initial_stiffness=1000.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split, displacement, law, normal_hint=(0.0, 1.0),
    )
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=1.0e-4,
        steps=2,
        progress=False,
    )
    assert step.residual.cohesive.displacement is step.state.u.value
    step.run()
    last = step.history_records[-1]
    assert last["prescribed_motion_work"] != 0.0
    assert np.isfinite(last["energy_balance_error"])
    assert last["cohesive_stored_energy"] >= 0.0
    dofs, first_ghost = top_motion.bc.dof_indices()
    np.testing.assert_allclose(
        step.state.u.value.x.array[dofs[:first_ghost]],
        top_motion.amplitude(2.0e-4),
    )


def test_custom_residual_assembly_dispatch_remains_operator_compatible():
    class Residual:
        def __init__(self):
            self.called = False

        def assemble_vector(self):
            self.called = True
            return "assembled"

    residual = Residual()
    assert operators.assemble_vector(residual) == "assembled"
    assert residual.called


def test_model_finite_strain_explicit_consumes_cohesive_force_and_energy():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
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
        constitutive.neo_hookean(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    interface_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    )
    topology = interfaces.pair_coincident_line_facets(
        interface_coordinates,
        [[0, 1]],
        [[2, 3]],
        normal_hint=(0.0, 1.0),
    )
    law = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        law,
        number_of_nodes=4,
    )
    cohesive = fracture.DofMappedCohesiveForce(
        assembler,
        displacement,
        node_to_block_dof=np.arange(4),
    )
    values = displacement.value.x.array.reshape((-1, 2))
    values[2:, 1] = 0.5 * law.peak_opening
    displacement.value.x.scatter_forward()

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=1.0e-8,
        steps=1,
        progress=False,
    )
    step.run()
    assert step.summary()["stability"]["interface_limit"] is not None
    assert step.residual.summary()["kind"] == "finite_strain_cohesive_residual"
    assert "cohesive_stored_energy" in step.history_records[0]
    assert assembler.state.committed_maximum.max() > 0.0


def test_cohesive_residual_snapshot_restores_state_and_rejects_other_topology():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    )
    topology = interfaces.pair_coincident_line_facets(
        coordinates, [[0, 1]], [[2, 3]], normal_hint=(0.0, 1.0),
    )
    law = interfaces.bilinear_cohesive(
        strength=10.0, fracture_energy=2.0, initial_stiffness=1000.0,
    )

    def make(mapping):
        assembler = interfaces.ModeICohesiveFacetAssembler(
            topology, law, number_of_nodes=4,
        )
        force = fracture.DofMappedCohesiveForce(
            assembler, displacement, node_to_block_dof=mapping,
        )
        zero = operators.OperatorForm(
            name="zero_bulk",
            expression=ufl.inner(
                fem.Constant(domain, np.array((0.0, 0.0))), displacement.test,
            ) * ufl.dx,
            kind="zero_bulk_force", role="vector", family="test",
        )
        return assembler, fracture.FiniteStrainCohesiveResidual(zero, force)

    assembler, residual = make(np.arange(4))
    assembler.state.initialize([0.1, 0.2])
    snapshot = residual.snapshot()
    restored_assembler, restored = make(np.arange(4))
    restored.restore(snapshot)
    np.testing.assert_allclose(
        restored_assembler.state.committed_maximum,
        [0.1, 0.2],
    )

    _, incompatible = make(np.array([1, 0, 2, 3]))
    import pytest
    with pytest.raises(ValueError, match="node-to-dof map differs"):
        incompatible.restore(snapshot)


def test_shared_transient_checkpoint_restores_cohesive_auxiliary_state(tmp_path):
    def make_step():
        domain = mesh.rectangle(
            (0.0, 0.0), (1.0, 1.0), (1, 1),
            comm=MPI.COMM_SELF, cell_type="quadrilateral",
        )
        displacement = fields.displacement(domain)
        state = problems.second_order_state(displacement)
        topology = interfaces.pair_coincident_line_facets(
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
            ),
            [[0, 1]], [[2, 3]], normal_hint=(0.0, 1.0),
        )
        law = interfaces.bilinear_cohesive(
            strength=10.0, fracture_energy=2.0, initial_stiffness=1000.0,
        )
        assembler = interfaces.ModeICohesiveFacetAssembler(
            topology, law, number_of_nodes=4,
        )
        cohesive = fracture.DofMappedCohesiveForce(
            assembler, state.u, node_to_block_dof=np.arange(4),
        )
        zero = operators.OperatorForm(
            name="zero_bulk",
            expression=ufl.inner(
                fem.Constant(domain, np.array((0.0, 0.0))), displacement.test,
            ) * ufl.dx,
            kind="zero_bulk_force", role="vector", family="test",
        )
        residual = fracture.FiniteStrainCohesiveResidual(zero, cohesive)
        mass = problems.LumpedMassOperator.assemble(displacement.space, density=1.0)
        integrator = time.explicit.central_difference(state=state, mass=mass)
        step = problems.explicit_dynamics(
            state=state,
            integrator=integrator,
            residual=residual,
            dt=1.0e-6,
            steps=2,
            progress=False,
            name="cohesive_restart",
        )
        return step, assembler

    partial, partial_assembler = make_step()
    partial.state.u.value.x.array.reshape((-1, 2))[2:, 1] = 0.01
    partial.state.u_next.assign(partial.state.u)
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(tmp_path / "cohesive")
    expected = partial_assembler.state.committed_maximum.copy()

    restarted, restarted_assembler = make_step()
    restarted.load_checkpoint(checkpoint)
    np.testing.assert_allclose(
        restarted_assembler.state.committed_maximum,
        expected,
    )
    assert restarted.completed_steps == 1
