import numpy as np
import pytest
from mpi4py import MPI

from agentfem import interfaces


def _law():
    return interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )


def test_bilinear_cohesive_envelope_has_declared_peak_and_exact_area():
    law = _law()
    assert law.peak_opening == pytest.approx(0.01)
    assert law.failure_opening == pytest.approx(0.4)
    assert law.envelope_traction(law.peak_opening) == pytest.approx(law.strength)
    assert law.envelope_traction(law.failure_opening) == pytest.approx(0.0)

    opening = np.linspace(0.0, law.failure_opening, 200_001)
    work = np.trapezoid(law.envelope_traction(opening), opening)
    assert work == pytest.approx(law.fracture_energy, rel=1.0e-9)
    assert law.envelope_work(law.failure_opening) == pytest.approx(
        law.fracture_energy
    )


def test_bilinear_cohesive_unloads_without_healing_and_closes_in_compression():
    law = _law()
    maximum = 0.2
    loaded = law.update(maximum)
    unloaded = law.update(0.05, loaded.maximum_opening)
    reloaded = law.update(maximum, loaded.maximum_opening)
    closed = law.update(-0.001, loaded.maximum_opening)

    assert unloaded.damage == pytest.approx(loaded.damage)
    assert reloaded.traction == pytest.approx(loaded.traction)
    assert unloaded.dissipated_energy == pytest.approx(loaded.dissipated_energy)
    assert closed.traction == pytest.approx(-1.0)
    assert closed.maximum_opening == pytest.approx(maximum)
    assert closed.damage == pytest.approx(loaded.damage)


def test_bilinear_trial_tangent_distinguishes_softening_from_unloading():
    law = _law()
    loading = law.update(0.2, committed_maximum=0.1)
    unloading = law.update(0.05, committed_maximum=0.2)
    assert loading.tangent == pytest.approx(
        -law.strength / (law.failure_opening - law.peak_opening)
    )
    assert unloading.tangent > 0.0


def test_complete_decohesion_dissipates_exact_fracture_energy():
    law = _law()
    response = law.update(law.failure_opening)
    assert response.damage == pytest.approx(1.0)
    assert response.traction == pytest.approx(0.0)
    assert response.stored_energy == pytest.approx(0.0)
    assert response.dissipated_energy == pytest.approx(law.fracture_energy)


def test_cohesive_transaction_commit_rollback_and_restart_are_atomic():
    transaction = interfaces.CohesiveTransaction(_law(), size=2)
    transaction.begin(np.array([0.1, 0.2]))
    transaction.rollback()
    np.testing.assert_allclose(transaction.committed_maximum, 0.0)

    transaction.begin(np.array([0.1, 0.2]))
    transaction.commit()
    snapshot = transaction.snapshot()

    transaction.begin(np.array([0.3, 0.3]))
    transaction.rollback()
    np.testing.assert_allclose(transaction.committed_maximum, [0.1, 0.2])

    restored = interfaces.CohesiveTransaction(_law(), size=2)
    restored.restore(snapshot)
    np.testing.assert_allclose(restored.committed_maximum, [0.1, 0.2])


def test_portable_cohesive_state_round_trip_uses_physical_facet_identity(tmp_path):
    _, topology = _one_segment_interface()
    source = interfaces.CohesiveTransaction(_law(), size=2)
    source.initialize([0.1, 0.2])
    ownership = interfaces.deterministic_facet_ownership(
        topology,
        comm=MPI.COMM_SELF,
    )
    manifest = interfaces.save_portable_cohesive_state(
        tmp_path / "interface",
        topology,
        source,
        comm=MPI.COMM_SELF,
    )
    restored = interfaces.CohesiveTransaction(_law(), size=2)
    metadata = interfaces.load_portable_cohesive_state(
        manifest,
        topology,
        restored,
        comm=MPI.COMM_SELF,
    )

    np.testing.assert_allclose(restored.committed_maximum, [0.1, 0.2])
    assert ownership.summary()["global_facets"] == 1
    assert metadata["reader_rank_count"] == 1
    assert metadata["parallel_contract"] == "physical_facet_keyed_state"


def test_cohesive_parameters_reject_an_impossible_bilinear_envelope():
    with pytest.raises(ValueError, match="too small"):
        interfaces.bilinear_cohesive(
            strength=10.0,
            fracture_energy=0.01,
            initial_stiffness=1000.0,
        )


def test_cohesive_surface_reports_independent_dof_topology_requirement():
    surface = interfaces.cohesive_surface(law=_law())
    summary = surface.summary()
    assert summary["maturity"] == "experimental"
    assert "independent dofs" in summary["topology_requirement"]
    assert interfaces.cohesive_characteristic_length(
        young=100.0,
        fracture_energy=2.0,
        strength=10.0,
    ) == pytest.approx(2.0)


def _one_segment_interface():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    topology = interfaces.pair_coincident_line_facets(
        coordinates,
        negative_facets=np.array([[0, 1]]),
        positive_facets=np.array([[3, 2]]),
        normal_hint=(0.0, 1.0),
    )
    return coordinates, topology


def test_paired_facets_recover_node_permutation_and_declared_normal():
    _, topology = _one_segment_interface()
    np.testing.assert_array_equal(topology.negative_nodes, [[0, 1]])
    np.testing.assert_array_equal(topology.positive_nodes, [[2, 3]])
    np.testing.assert_allclose(topology.normals, [[0.0, 1.0]])
    np.testing.assert_allclose(topology.lengths, [1.0])
    identity = topology.identity()
    assert identity["scope"] == "ordered_reference_facet_geometry"
    assert identity["orientation_sensitive"] is True
    assert len(identity["sha256"]) == 64


def test_paired_facet_identity_uses_physical_geometry_not_node_numbers():
    coordinates, topology = _one_segment_interface()
    permutation = np.array([2, 3, 0, 1])
    inverse = np.argsort(permutation)
    renumbered = interfaces.pair_coincident_line_facets(
        coordinates[permutation],
        negative_facets=inverse[np.array([[0, 1]])],
        positive_facets=inverse[np.array([[2, 3]])],
        normal_hint=(0.0, 1.0),
    )
    assert renumbered.identity() == topology.identity()


def test_mode_i_facet_kernel_produces_equal_opposite_force_and_exact_energy():
    coordinates, topology = _one_segment_interface()
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=coordinates.shape[0],
        thickness=2.0,
    )
    displacement = np.zeros_like(coordinates)
    displacement[2:, 1] = _law().peak_opening
    response = assembler.begin(displacement)

    np.testing.assert_allclose(
        np.sum(response.internal_force, axis=0), 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(response.internal_force[:2, 1], -10.0)
    np.testing.assert_allclose(response.internal_force[2:, 1], 10.0)
    assert response.stored_energy == pytest.approx(
        0.5 * _law().strength * _law().peak_opening * 2.0
    )
    assert response.dissipated_energy == pytest.approx(0.0)


def test_mode_i_facet_consistent_tangent_matches_force_directional_derivative():
    coordinates, topology = _one_segment_interface()
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=coordinates.shape[0],
        thickness=1.7,
    )
    displacement = np.zeros_like(coordinates)
    displacement[2:, 1] = 0.5 * _law().peak_opening
    direction = np.array(
        [[0.2, -0.4], [-0.1, 0.3], [0.5, 0.7], [-0.2, -0.6]],
        dtype=float,
    )
    tangent = assembler.tangent_elements(displacement)
    epsilon = 1.0e-7
    plus = assembler.begin(displacement + epsilon * direction).internal_force.copy()
    assembler.rollback()
    minus = assembler.begin(displacement - epsilon * direction).internal_force.copy()
    assembler.rollback()
    derivative = (plus - minus) / (2.0 * epsilon)

    nodes = tangent.nodes[0]
    predicted = tangent.matrices[0] @ direction[nodes].reshape(-1)
    np.testing.assert_allclose(
        predicted.reshape((-1, 2)),
        derivative[nodes],
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_mode_i_facet_kernel_is_invariant_to_common_rigid_translation():
    coordinates, topology = _one_segment_interface()
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=coordinates.shape[0],
    )
    displacement = np.tile([2.5, -1.2], (coordinates.shape[0], 1))
    response = assembler.begin(displacement)
    np.testing.assert_allclose(response.opening, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(response.internal_force, 0.0, atol=1.0e-14)


def test_vector_interface_modes_make_tangential_physics_explicit():
    coordinates, topology = _one_segment_interface()
    displacement = np.zeros_like(coordinates)
    displacement[2:, 0] = 0.02
    displacement[2:, 1] = 0.5 * _law().peak_opening

    free = interfaces.ModeICohesiveFacetAssembler(
        topology, _law(), number_of_nodes=4, tangential="free"
    ).begin(displacement)
    tied = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=4,
        tangential="tie",
        tangential_stiffness=250.0,
    ).begin(displacement)

    np.testing.assert_allclose(free.tangential_traction, 0.0)
    np.testing.assert_allclose(tied.tangential_traction[:, :, 0], 5.0)
    audit = interfaces.audit_mode_i_kinematics(tied, ratio_limit=1.0)
    assert audit.tangential_to_normal_ratio == pytest.approx(4.0)
    assert audit.accepted is False
    with pytest.raises(RuntimeError, match="excessive tangential jump"):
        interfaces.audit_mode_i_kinematics(
            tied, ratio_limit=1.0, error_if_exceeded=True
        )


def test_normal_driven_tangential_connection_releases_with_precrack():
    coordinates, topology = _one_segment_interface()
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=4,
        tangential="degraded",
        tangential_stiffness=250.0,
    )
    assembler.initialize_precrack([0])
    displacement = np.zeros_like(coordinates)
    displacement[2:, 0] = 0.02
    displacement[2:, 1] = 0.05
    response = assembler.begin(displacement)
    np.testing.assert_allclose(response.internal_force, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(response.tangential_traction, 0.0, atol=1.0e-14)


def test_degraded_vector_interface_tangent_matches_force_derivative():
    coordinates, topology = _one_segment_interface()
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=4,
        tangential="degraded",
        tangential_stiffness=400.0,
    )
    displacement = np.zeros_like(coordinates)
    displacement[2:, 0] = 0.015
    displacement[2:, 1] = 0.08
    direction = np.array(
        [[0.2, -0.4], [-0.1, 0.3], [0.5, 0.7], [-0.2, -0.6]],
        dtype=float,
    )
    tangent = assembler.tangent_elements(displacement)
    epsilon = 1.0e-7
    plus = assembler.begin(displacement + epsilon * direction).internal_force.copy()
    assembler.rollback()
    minus = assembler.begin(displacement - epsilon * direction).internal_force.copy()
    assembler.rollback()
    derivative = (plus - minus) / (2.0 * epsilon)
    nodes = tangent.nodes[0]
    predicted = tangent.matrices[0] @ direction[nodes].reshape(-1)
    np.testing.assert_allclose(
        predicted.reshape((-1, 2)), derivative[nodes], rtol=2.0e-7, atol=2.0e-7
    )


@pytest.mark.parametrize("interaction", ["bk", "power"])
def test_mixed_mode_law_recovers_pure_mode_fracture_energies(interaction):
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
        interaction=interaction,
        interaction_exponent=1.7,
    )
    normal = law.transaction(1)
    normal_failure = 2.0 * law.normal_fracture_energy / law.normal_strength
    response = normal.begin([[normal_failure, 0.0]])
    assert response.damage == pytest.approx([1.0])
    assert response.dissipated_energy == pytest.approx(
        [law.normal_fracture_energy]
    )
    normal.rollback()

    shear = law.transaction(1)
    shear_failure = 2.0 * law.shear_fracture_energy / law.shear_strength
    response = shear.begin([[0.0, shear_failure]])
    assert response.damage == pytest.approx([1.0])
    assert response.dissipated_energy == pytest.approx(
        [law.shear_fracture_energy]
    )


def test_mixed_mode_transaction_restart_tangent_and_compression_contact():
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
        interaction="bk",
        friction_coefficient=0.25,
        friction_regularization=1.0e-4,
    )
    state = law.transaction(1)
    initiation = np.array([[0.015, 0.012]])
    state.begin(initiation)
    state.commit()
    point = np.array([[0.06, 0.04]])
    response = state.evaluate(point)
    epsilon = 1.0e-7
    numerical = np.column_stack(
        [
            (
                state.evaluate(point + epsilon * np.eye(2)[component]).traction[0]
                - state.evaluate(point - epsilon * np.eye(2)[component]).traction[0]
            )
            / (2.0 * epsilon)
            for component in range(2)
        ]
    )
    np.testing.assert_allclose(response.tangent[0], numerical, rtol=2.0e-6, atol=2.0e-6)

    snapshot = state.snapshot()
    restored = law.transaction(1)
    restored.restore(snapshot)
    np.testing.assert_allclose(
        restored.state_arrays()["maximum_effective_separation"],
        state.state_arrays()["maximum_effective_separation"],
    )
    committed_damage = restored.evaluate(initiation).damage.copy()
    contact = restored.evaluate([[-0.01, 0.005]])
    assert contact.damage == pytest.approx(committed_damage)
    assert contact.traction[0, 0] < 0.0
    assert contact.traction[0, 1] > 0.0
    contact_point = np.array([[-0.01, 0.005]])
    numerical_contact = np.column_stack(
        [
            (
                restored.evaluate(
                    contact_point + epsilon * np.eye(2)[component]
                ).traction[0]
                - restored.evaluate(
                    contact_point - epsilon * np.eye(2)[component]
                ).traction[0]
            )
            / (2.0 * epsilon)
            for component in range(2)
        ]
    )
    np.testing.assert_allclose(
        contact.tangent[0], numerical_contact, rtol=2.0e-6, atol=2.0e-6
    )


def test_mixed_mode_initialization_requires_a_complete_physical_state():
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
    )
    state = law.transaction(1)
    state.begin([[0.0, 0.0]])
    with pytest.raises(RuntimeError, match="Rollback"):
        state.initialize(0.0)
    state.rollback()
    with pytest.raises(ValueError, match="does not determine"):
        state.initialize(0.01)

    incomplete = state.state_arrays()
    incomplete["maximum_effective_separation"][:] = 0.01
    with pytest.raises(ValueError, match="incomplete uninitiated"):
        state.restore_state_arrays(incomplete)


def test_residual_tangential_branch_stores_work_without_changing_giic():
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
        residual_tangential_fraction=0.1,
    )
    state = law.transaction(1)
    failure = 2.0 * law.shear_fracture_energy / law.shear_strength
    separations = np.linspace(0.0, failure, 10_001)
    tractions = np.empty_like(separations)
    response = None
    for index, separation in enumerate(separations):
        response = state.begin([[0.0, separation]])
        tractions[index] = response.traction[0, 1]
        state.commit()
    assert response is not None
    penalty_stored = (
        0.5
        * law.residual_tangential_fraction
        * law.tangential_stiffness
        * failure**2
    )
    assert response.dissipated_energy == pytest.approx(
        [law.shear_fracture_energy]
    )
    assert response.stored_energy == pytest.approx([penalty_stored])
    work = np.trapezoid(tractions, separations)
    assert work == pytest.approx(
        law.shear_fracture_energy + penalty_stored,
        rel=2.0e-5,
    )


def test_multi_interface_rigid_mode_audit_detects_free_middle_bodies():
    coordinates = np.array(
        [
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [1.0, 1.0],
            [0.0, 2.0], [1.0, 2.0],
            [0.0, 3.0], [1.0, 3.0],
        ]
    )
    cells = np.array([[0, 1, 3, 2], [2, 3, 5, 4], [4, 5, 7, 6]])
    split = interfaces.split_conforming_named_interfaces(
        coordinates,
        cells,
        {
            "lower": {"interface_facets": [[2, 3]], "positive_cells": [1, 2]},
            "upper": {"interface_facets": [[4, 5]], "positive_cells": [2]},
        },
    )
    constraints = {0: (0, 1), 1: (0, 1)}
    free = interfaces.audit_split_interface_rigid_modes(
        split,
        constrained_components=constraints,
        tangential="free",
    )
    tied = interfaces.audit_split_interface_rigid_modes(
        split,
        constrained_components=constraints,
        tangential="degraded",
    )
    assert free.nullity >= 2
    assert tied.well_posed
    with pytest.raises(ValueError, match="rank_tolerance"):
        interfaces.audit_split_interface_rigid_modes(
            split,
            constrained_components=constraints,
            tangential="tie",
            rank_tolerance=-1.0,
        )
    with pytest.raises(RuntimeError, match="unconstrained rigid-body"):
        interfaces.audit_split_interface_rigid_modes(
            split,
            constrained_components=constraints,
            tangential="free",
            error_if_singular=True,
        )


def test_three_layer_3d_axial_patch_transfers_normal_force_without_shear():
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    coordinates = np.vstack(
        [
            np.column_stack((triangle, np.full(3, height)))
            for height in (0.0, 1.0, 2.0, 3.0)
        ]
    )
    cells = []
    for layer in range(3):
        a, b, c = 3 * layer + np.arange(3)
        upper_a, upper_b, upper_c = 3 * (layer + 1) + np.arange(3)
        cells.extend(
            [
                [a, b, c, upper_a],
                [b, c, upper_a, upper_b],
                [c, upper_a, upper_b, upper_c],
            ]
        )
    split = interfaces.split_conforming_named_interfaces(
        coordinates,
        np.asarray(cells, dtype=int),
        {
            "lower": {
                "interface_facets": [[3, 4, 5]],
                "positive_cells": np.arange(3, 9),
            },
            "upper": {
                "interface_facets": [[6, 7, 8]],
                "positive_cells": np.arange(6, 9),
            },
        },
    )
    constraints = {node: (0, 1, 2) for node in (0, 1, 2)}
    audit = interfaces.audit_split_interface_rigid_modes(
        split,
        constrained_components=constraints,
        tangential="degraded",
    )
    assert audit.well_posed

    displacement = np.zeros_like(split.coordinates)
    opening = 0.005
    lower = split["lower"]
    upper = split["upper"]
    displacement[np.unique(lower.positive_facets), 2] = opening
    displacement[np.unique(upper.negative_facets), 2] = opening
    displacement[np.unique(upper.positive_facets), 2] = 2.0 * opening

    for surface in (lower, upper):
        topology = interfaces.pair_coincident_surface_facets(
            split.coordinates,
            surface.negative_facets,
            surface.positive_facets,
            normal_hint=(0.0, 0.0, 1.0),
        )
        assembler = interfaces.ModeICohesiveSurfaceAssembler(
            topology,
            _law(),
            number_of_nodes=split.coordinates.shape[0],
            tangential="degraded",
        )
        response = assembler.begin(displacement)
        positive_force = np.sum(
            response.internal_force[np.unique(surface.positive_facets)], axis=0
        )
        np.testing.assert_allclose(positive_force, [0.0, 0.0, 2.5])
        np.testing.assert_allclose(response.tangential_jump, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(response.tangential_traction, 0.0, atol=1.0e-14)


def test_precracked_facet_has_no_tensile_force_but_retains_closure_penalty():
    coordinates, topology = _one_segment_interface()
    assembler = interfaces.ModeICohesiveFacetAssembler(
        topology,
        _law(),
        number_of_nodes=coordinates.shape[0],
    )
    assembler.initialize_precrack([0])

    opening = np.zeros_like(coordinates)
    opening[2:, 1] = 0.05
    response = assembler.begin(opening)
    np.testing.assert_allclose(response.internal_force, 0.0)
    assembler.rollback()

    closure = np.zeros_like(coordinates)
    closure[2:, 1] = -0.001
    response = assembler.begin(closure)
    assert np.linalg.norm(response.internal_force) > 0.0


def test_pairing_rejects_ambiguous_or_missing_partners():
    coordinates, _ = _one_segment_interface()
    missing_coordinates = np.vstack((coordinates, [[0.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="exactly one"):
        interfaces.pair_coincident_line_facets(
            missing_coordinates,
            negative_facets=np.array([[0, 1]]),
            positive_facets=np.array([[4, 5]]),
            normal_hint=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="independent node identities"):
        interfaces.pair_coincident_line_facets(
            coordinates,
            negative_facets=np.array([[0, 1]]),
            positive_facets=np.array([[0, 1]]),
            normal_hint=(0.0, 1.0),
        )


def test_split_conforming_interface_duplicates_only_the_declared_side():
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
        np.array([[3, 2]]),
        positive_cells=[1],
    )
    assert split.coordinates.shape == (8, 2)
    np.testing.assert_array_equal(split.cells[0], cells[0])
    assert not set(split.cells[0]).intersection(split.positive_facets.reshape(-1))
    np.testing.assert_allclose(
        split.coordinates[split.negative_facets],
        split.coordinates[split.positive_facets],
    )
    topology = interfaces.pair_coincident_line_facets(
        split.coordinates,
        split.negative_facets,
        split.positive_facets,
        normal_hint=(0.0, 1.0),
    )
    assert not set(topology.negative_nodes.reshape(-1)).intersection(
        topology.positive_nodes.reshape(-1)
    )


def test_split_interface_rejects_a_path_that_does_not_separate_declared_sides():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(ValueError, match="cells on both declared sides"):
        interfaces.split_conforming_line_interface(
            coordinates,
            cells,
            np.array([[0, 2]]),
            positive_cells=[],
        )


def test_split_cell_interface_recovers_partition_boundary_without_manual_facets():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])

    split = interfaces.split_conforming_cell_interface(
        coordinates,
        cells,
        positive_cells=[1],
    )

    np.testing.assert_array_equal(split.negative_facets, [[1, 4]])
    np.testing.assert_allclose(
        split.coordinates[split.negative_facets],
        split.coordinates[split.positive_facets],
    )
    assert set(split.cells[0]).isdisjoint(split.positive_facets.reshape(-1))
    assert set(split.positive_facets.reshape(-1)).issubset(set(split.cells[1]))


def test_split_cell_interface_rejects_non_manifold_connectivity():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, -1.0],
            [0.5, 0.5],
        ]
    )
    cells = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])

    with pytest.raises(ValueError, match="manifold mesh"):
        interfaces.split_conforming_cell_interface(
            coordinates,
            cells,
            positive_cells=[1],
        )


def test_three_dimensional_surface_split_pair_and_mode_i_resultant():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    split = interfaces.split_conforming_cell_interface(
        coordinates, cells, positive_cells=[1]
    )
    topology = interfaces.pair_coincident_surface_facets(
        split.coordinates,
        split.negative_facets,
        split.positive_facets,
        normal_hint=(0.0, 0.0, 1.0),
    )
    assembler = interfaces.ModeICohesiveSurfaceAssembler(
        topology, _law(), number_of_nodes=split.coordinates.shape[0]
    )
    displacement = np.zeros_like(split.coordinates)
    displacement[split.positive_facets.reshape(-1), 2] = 0.005
    response = assembler.begin(displacement)

    assert split.negative_facets.shape == (1, 3)
    assert topology.areas == pytest.approx([0.5])
    assert topology.number_of_points == 3
    np.testing.assert_allclose(response.opening, [[0.005, 0.005, 0.005]])
    np.testing.assert_allclose(
        np.sum(response.internal_force, axis=0), 0.0, atol=1.0e-14
    )
    positive_force = np.sum(
        response.internal_force[np.unique(split.positive_facets)], axis=0
    )
    np.testing.assert_allclose(positive_force, [0.0, 0.0, 2.5])


def test_mode_i_surface_consistent_tangent_matches_force_derivative():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    split = interfaces.split_conforming_cell_interface(
        coordinates, cells, positive_cells=[1]
    )
    topology = interfaces.pair_coincident_surface_facets(
        split.coordinates,
        split.negative_facets,
        split.positive_facets,
        normal_hint=(0.0, 0.0, 1.0),
    )
    assembler = interfaces.ModeICohesiveSurfaceAssembler(
        topology, _law(), number_of_nodes=split.coordinates.shape[0]
    )
    displacement = np.zeros_like(split.coordinates)
    displacement[split.positive_facets.reshape(-1), 2] = 0.5 * _law().peak_opening
    direction = np.arange(displacement.size, dtype=float).reshape(displacement.shape)
    direction = 0.01 * (direction - np.mean(direction))
    tangent = assembler.tangent_elements(displacement)
    epsilon = 1.0e-7
    plus = assembler.begin(displacement + epsilon * direction).internal_force.copy()
    assembler.rollback()
    minus = assembler.begin(displacement - epsilon * direction).internal_force.copy()
    assembler.rollback()
    derivative = (plus - minus) / (2.0 * epsilon)

    nodes = tangent.nodes[0]
    predicted = tangent.matrices[0] @ direction[nodes].reshape(-1)
    np.testing.assert_allclose(
        predicted.reshape((-1, 3)),
        derivative[nodes],
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_split_cell_interface_supports_triangle_partitions():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]])

    split = interfaces.split_conforming_cell_interface(
        coordinates,
        cells,
        positive_cells=np.array([False, True]),
    )

    np.testing.assert_array_equal(split.negative_facets, [[0, 2]])
    assert split.summary()["number_of_interface_facets"] == 1


def test_split_cell_interface_rejects_disconnected_cell_partitions():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2], [3, 4, 5]])

    with pytest.raises(ValueError, match="do not share"):
        interfaces.split_conforming_cell_interface(
            coordinates,
            cells,
            positive_cells=[1],
        )
