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

    np.testing.assert_allclose(np.sum(response.internal_force, axis=0), 0.0)
    np.testing.assert_allclose(response.internal_force[:2, 1], -10.0)
    np.testing.assert_allclose(response.internal_force[2:, 1], 10.0)
    assert response.stored_energy == pytest.approx(
        0.5 * _law().strength * _law().peak_opening * 2.0
    )
    assert response.dissipated_energy == pytest.approx(0.0)


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
