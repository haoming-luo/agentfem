import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
import ufl

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    fields,
    fatigue_fracture,
    fracture,
    interfaces,
    mesh,
    models,
    operators,
    problems,
    results,
    solvers,
    studies,
    time,
    upgrades,
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


def test_serial_dof_adapter_adds_consistent_cohesive_tangent_to_petsc_matrix():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    )
    topology = interfaces.pair_coincident_line_facets(
        coordinates,
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
        topology, law, number_of_nodes=4
    )
    cohesive = fracture.DofMappedCohesiveForce(
        assembler,
        displacement,
        node_to_block_dof=np.arange(4, dtype=int),
    )
    values = displacement.value.x.array.reshape((-1, 2))
    values[2:, 1] = 0.5 * law.peak_opening
    displacement.value.x.scatter_forward()
    bulk_graph = fem.form(
        ufl.inner(displacement.trial, displacement.test) * ufl.dx
    )
    from dolfinx.fem import petsc as fem_petsc

    matrix = fem_petsc.assemble_matrix(bulk_graph)
    matrix.assemble()
    matrix.zeroEntries()
    cohesive.add_to_matrix(matrix)
    matrix.assemble()
    direction = displacement.value.x.petsc_vec.duplicate()
    action = displacement.value.x.petsc_vec.duplicate()
    try:
        direction.array[:] = np.arange(direction.array.size, dtype=float) - 2.5
        matrix.mult(direction, action)
        element = assembler.tangent_elements(values)
        expected = element.matrices[0] @ direction.array.reshape((-1, 2))[
            element.nodes[0]
        ].reshape(-1)
        np.testing.assert_allclose(action.array, expected, atol=1.0e-12)
    finally:
        action.destroy()
        direction.destroy()
        matrix.destroy()


def test_native_cohesive_newton_solves_force_controlled_finite_strain_strip():
    from dolfinx import mesh as dolfinx_mesh

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
    split = interfaces.split_conforming_line_interface(
        coordinates,
        np.array([[0, 1, 2, 3], [3, 2, 4, 5]]),
        [[3, 2]],
        positive_cells=[1],
    )
    domain = interfaces.create_dolfinx_split_mesh(split)
    displacement = fields.displacement(domain)
    material = constitutive.neo_hookean(young=100.0, poisson=0.25)
    internal = fracture.finite_strain_internal_force(
        displacement.value,
        displacement.test,
        material,
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
    external = operators.OperatorForm(
        name="top_force",
        expression=load_parameter * displacement.test[1] * ds(1),
        kind="surface_force",
        role="vector",
        family="mechanical_load",
    )
    bulk_vector = internal - external
    bulk = operators.residual_operator(
        bulk_vector.expression,
        name="R_bulk",
        family="total_lagrangian_neo_hookean",
    )
    tangent = operators.linearize(bulk, displacement)
    monotonic = interfaces.bilinear_cohesive(
        strength=10.0, fracture_energy=0.2, initial_stiffness=1000.0
    )
    law = fatigue_fracture.cyclic_cohesive(
        monotonic=monotonic,
        fatigue_coefficient=1.0e-2,
        fatigue_exponent=1.0,
        range_threshold=0.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    collection = fracture.named_cohesive_forces(crack=cohesive)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, collection)
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
            linear_solver=solvers.direct_solver(),
        ),
        control_displacement=lambda function: float(
            np.mean(
                function.x.array.reshape((-1, 2))[
                    np.isclose(
                        function.function_space.tabulate_dof_coordinates()[:, 1],
                        1.0,
                    ),
                    1,
                ]
            )
        ),
        reaction=lambda _function: results.reaction_resultant(
            residual,
            on=lambda x: np.isclose(x[1], 0.0),
            component=1,
        ),
    )
    point = equilibrium(load=0.5, branch="maximum", cycle=1)

    assert point["converged"] is True
    assert point["iterations"] < 10
    assert point["reaction"] == pytest.approx(-0.5, rel=1.0e-8, abs=1.0e-10)
    assert point["control_displacement"] > 0.0
    assert equilibrium.last_info.residual_norm < 1.0e-8
    values = displacement.value.x.array.reshape((-1, 2))
    dof_coordinates = displacement.space.tabulate_dof_coordinates()
    top = np.isclose(dof_coordinates[:, 1], 1.0)
    assert np.mean(values[top, 1]) > 0.0
    opening = cohesive.cycle_opening()
    assert 0.0 < float(np.mean(opening)) < law.peak_opening
    assert equilibrium.summary()["cohesive_state_commit"] == "owned_by_step_lifecycle"

    state = fatigue_fracture.field_state(displacement=displacement)
    cycle = fatigue_fracture.force_cycle(fmin=0.05, fmax=0.5)
    jump = fatigue_fracture.CycleJumpPolicy(
        maximum_damage_increment=0.1,
        maximum_cycles=10,
    )
    cycle_step = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle,
        stop_cycle=3,
        interfaces=collection,
        state=state,
        solve_equilibrium=equilibrium,
        jump=jump,
    )
    cycle_step.run(until_cycle=1)
    assert cycle_step.current_cycle == 1
    assert (
        cycle_step.history[0].closing.control_displacement
        < cycle_step.history[0].maximum.control_displacement
    )
    assert np.max(cohesive.assembler.state.fatigue_damage) > 0.0
    np.testing.assert_allclose(
        cohesive.assembler.state.cumulative_cycles,
        np.ones(cohesive.assembler.state.size),
    )
    checkpoint = cycle_step.snapshot()
    cycle_step.run()
    continuous_damage = cohesive.assembler.state.fatigue_damage.copy()
    continuous_displacement = displacement.value.x.array.copy()

    restarted = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle,
        stop_cycle=3,
        interfaces=collection,
        state=state,
        solve_equilibrium=equilibrium,
        jump=jump,
    )
    restarted.restore(checkpoint)
    restarted.run()
    assert restarted.current_cycle == cycle_step.current_cycle == 3
    np.testing.assert_allclose(
        cohesive.assembler.state.fatigue_damage,
        continuous_damage,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        displacement.value.x.array,
        continuous_displacement,
        rtol=1.0e-12,
        atol=1.0e-14,
    )


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


def test_public_vector_cohesive_force_exposes_standard_interface_fields():
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
    displacement = fields.displacement(domain)
    law = interfaces.bilinear_cohesive(
        strength=10.0, fracture_energy=2.0, initial_stiffness=1000.0
    )
    cohesive = fracture.cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
        tangential_stiffness=500.0,
    )
    mapping = cohesive.node_to_block_dof
    values = displacement.value.x.array.reshape((-1, 2))
    positive = mapping[split.positive_facets.reshape(-1)]
    values[positive, 0] = 0.002
    values[positive, 1] = 0.005
    displacement.value.x.scatter_forward()
    quantities = cohesive.interface_quantities()
    assert set(quantities) == {
        "JUMP_N", "JUMP_T", "TRACTION_N", "TRACTION_T", "DAMAGE", "MODE_MIXITY"
    }
    assert np.max(np.linalg.norm(quantities["TRACTION_T"], axis=-1)) > 0.0
    assert cohesive.summary()["interface_kinematics"] == "tie"
    snapshot = cohesive.snapshot()
    assert snapshot["schema"] == "agentfem.dof-mapped-cohesive-force.v5"
    legacy = dict(snapshot)
    legacy["schema"] = "agentfem.dof-mapped-cohesive-force.v4"
    legacy.pop("interface_kinematics")
    legacy.pop("tangential_stiffness")
    with pytest.raises(ValueError, match="Legacy cohesive checkpoints"):
        cohesive.restore(legacy)
    with pytest.raises(ValueError, match="acknowledge_physics_change"):
        upgrades.migrate_cohesive_checkpoint(legacy, tangential="tie")
    migrated = upgrades.migrate_cohesive_checkpoint(
        legacy,
        tangential="tie",
        tangential_stiffness=500.0,
        acknowledge_physics_change=True,
    )
    cohesive.restore(migrated)
    assert migrated["migration"]["source_schema"].endswith(".v4")
    assert len(migrated["migration"]["source_sha256"]) == 64
    free = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
        tangential="free",
    )
    with pytest.raises(ValueError, match="interface kinematics"):
        free.restore(snapshot)
    free.restore(legacy)


def test_public_mixed_mode_cyclic_force_consumes_vector_extrema_and_restarts():
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
    displacement = fields.displacement(domain)
    monotonic = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
        interaction="bk",
        interaction_exponent=1.6,
    )
    law = fatigue_fracture.cyclic_cohesive(
        monotonic=monotonic,
        driver=fatigue_fracture.mixed_mode_energy_range_driver(
            mode_i_threshold_fraction=0.01,
            mode_ii_threshold_fraction=0.01,
        ),
        fatigue_coefficient=0.1,
        fatigue_exponent=1.0,
        residual_exponent=1.0,
        range_threshold=0.0,
    )
    cohesive = fracture.cohesive_force(
        split, displacement, law, normal_hint=(0.0, 1.0)
    )
    collection = fracture.named_cohesive_forces(crack=cohesive)
    values = displacement.value.x.array.reshape((-1, 2))
    positive = cohesive.node_to_block_dof[split.positive_facets.reshape(-1)]

    values[positive] = (0.002, 0.002)
    displacement.value.x.scatter_forward()
    valley = collection.cycle_kinematics()
    values[positive] = (0.008, 0.008)
    displacement.value.x.scatter_forward()
    peak = collection.cycle_kinematics()
    assert valley["crack"].shape[1] == 2
    np.testing.assert_allclose(peak["crack"], 4.0 * valley["crack"])
    cohesive.begin()
    cohesive.commit()

    collection.begin_cycle(valley, peak, cycles=5)
    collection.commit_cycle()
    quantities = cohesive.interface_quantities()
    assert {
        "JUMP_MIN_LOCAL",
        "JUMP_MAX_LOCAL",
        "GI_COH_RANGE",
        "GII_COH_RANGE",
        "G_COH_RANGE_NORM",
        "LOCAL_LOAD_RATIO",
        "FATIGUE_DAMAGE",
        "CYCLES",
        "DISSIPATION_FATIGUE",
        "CYCLE_PATH_LENGTH",
        "CYCLE_REVERSALS",
        "CYCLE_STATIONS",
    } <= set(quantities)
    assert np.max(quantities["FATIGUE_DAMAGE"]) > 0.0
    assert np.all(quantities["DAMAGE"] >= quantities["FATIGUE_DAMAGE"])
    np.testing.assert_allclose(quantities["CYCLES"], 5.0)
    np.testing.assert_allclose(quantities["JUMP_MIN_LOCAL"], valley["crack"])
    np.testing.assert_allclose(quantities["JUMP_MAX_LOCAL"], peak["crack"])

    snapshot = cohesive.snapshot()
    state_before = cohesive.assembler.state.state_arrays()
    collection.begin_cycle(valley, peak, cycles=2)
    collection.commit_cycle()
    cohesive.restore(snapshot)
    for name, expected in state_before.items():
        np.testing.assert_allclose(
            cohesive.assembler.state.state_arrays()[name], expected
        )


def test_mixed_mode_public_force_restart_and_arc_length_consumer():
    from dolfinx import mesh as dolfinx_mesh

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
    external = operators.OperatorForm(
        name="top_force",
        expression=load_parameter * displacement.test[1] * ds(1),
        kind="surface_force",
        role="vector",
        family="mechanical_load",
    )
    bulk_vector = internal - external
    bulk = operators.residual_operator(
        bulk_vector.expression,
        name="R_bulk",
        family="total_lagrangian_neo_hookean",
    )
    tangent = operators.linearize(bulk, displacement)
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=0.2,
        shear_fracture_energy=0.3,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
        interaction="bk",
    )
    cohesive = fracture.cohesive_force(
        split, displacement, law, normal_hint=(0.0, 1.0)
    )
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)
    bcs = [
        constraints.component_dirichlet(
            displacement, 1, on=lambda x: np.isclose(x[1], 0.0), value=0.0
        ).bc,
        constraints.component_dirichlet(
            displacement, 0, on=lambda x: np.isclose(x[0], 0.0), value=0.0
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
            linear_solver=solvers.direct_solver(),
        ),
    )
    continuation = fracture.FiniteStrainCohesiveArcLength(
        equilibrium,
        fracture.ArcLengthOptions(
            radius=0.05,
            load_scale=0.1,
            maximum_iterations=15,
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            constraint_tolerance=1.0e-7,
        ),
    )
    info = continuation.advance()
    assert info.converged
    assert info.load > 0.0
    snapshot = continuation.snapshot()
    assert set(snapshot["cohesive"]["cohesive"]["state_by_field_and_key"]) == {
        "failure_separation",
        "initiation_separation",
        "initiation_stiffness",
        "maximum_effective_separation",
    }


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

    # Version 2 follows physical interface identity, not execution-local dof
    # numbering, so a valid renumbering restores the same irreversible state.
    renumbered_assembler, renumbered = make(np.array([1, 0, 2, 3]))
    renumbered.restore(snapshot)
    np.testing.assert_allclose(
        renumbered_assembler.state.committed_maximum,
        [0.1, 0.2],
    )

    # Legacy version 1 remains readable and retains its original strict map.
    legacy = dict(snapshot)
    legacy["cohesive"] = dict(snapshot["cohesive"])
    legacy["cohesive"]["schema"] = "agentfem.dof-mapped-cohesive-force.v1"
    import pytest
    with pytest.raises(ValueError, match="node-to-dof map differs"):
        renumbered.restore(legacy)


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
