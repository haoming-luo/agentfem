import numpy as np
import pytest
from mpi4py import MPI
from types import SimpleNamespace

from agentfem import fatigue_fracture, fields, fracture, interfaces, mesh, procedures


def _monotonic():
    return interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )


def _cyclic(**kwargs):
    options = {
        "monotonic": _monotonic(),
        "fatigue_coefficient": 2.0e-3,
        "fatigue_exponent": 2.0,
        "range_threshold": 0.05,
    }
    options.update(kwargs)
    return fatigue_fracture.cyclic_cohesive(**options)


def test_force_cycle_uses_test_parameters_and_has_exact_extrema():
    cycle = fatigue_fracture.force_cycle(
        fmin=226.0,
        fmax=2262.0,
        frequency=5.0,
        waveform="sine",
    )
    assert cycle.load_ratio == pytest.approx(226.0 / 2262.0)
    assert cycle.at_phase(0.0) == pytest.approx(226.0)
    assert cycle.at_phase(0.5) == pytest.approx(2262.0)
    assert cycle.at_phase(1.0) == pytest.approx(226.0)
    amplitude = cycle.normalized_amplitude()
    assert amplitude(0.1) == pytest.approx(1.0)
    assert amplitude.summary()["metadata"]["cycle_coordinate"] == (
        "frequency * physical_time"
    )


def test_force_cycle_supports_holds_and_experimental_tabular_waveform():
    held = fatigue_fracture.force_cycle(
        fmin=10.0,
        fmax=100.0,
        waveform="triangle",
        hold_minimum_fraction=0.2,
        hold_maximum_fraction=0.2,
    )
    assert held.at_phase(0.05) == pytest.approx(10.0)
    assert held.at_phase(0.5) == pytest.approx(100.0)
    table = fatigue_fracture.force_cycle(
        fmin=10.0,
        fmax=100.0,
        waveform="tabular",
        table=((0.0, 10.0), (0.4, 100.0), (0.6, 100.0), (1.0, 10.0)),
    )
    assert table.at_phase(0.5) == pytest.approx(100.0)
    assert table.at_phase(0.8) == pytest.approx(55.0)


def test_cycle_jump_is_damage_limited_and_lands_exactly_on_requested_output():
    policy = fatigue_fracture.CycleJumpPolicy(
        maximum_damage_increment=0.01,
        maximum_cycles=10_000,
        safety_factor=0.8,
    )
    limited = policy.propose(
        start_cycle=0,
        damage_rate=np.array([1.0e-5, 2.0e-5]),
        stop_cycle=20_000,
    )
    assert limited.cycles == 400
    assert limited.reason == "damage_increment_limit"

    landed = policy.propose(
        start_cycle=400,
        damage_rate=2.0e-5,
        stop_cycle=20_000,
        landing_cycles=(500, 1_000),
    )
    assert landed.end_cycle == 500
    assert landed.reason == "exact_landing"
    assert landed.exact_landing_target == 500


def test_cycle_jump_can_be_front_limited_and_ledger_records_cutback_and_restart():
    policy = fatigue_fracture.CycleJumpPolicy(
        maximum_damage_increment=0.01,
        maximum_front_advance=0.1,
        maximum_cycles=10_000,
        safety_factor=0.8,
    )
    decision = policy.propose(
        start_cycle=0,
        damage_rate=1.0e-8,
        front_advance_rate=1.0e-3,
        stop_cycle=1_000,
    )
    assert decision.reason == "front_advance_limit"
    assert decision.cycles == 80
    ledger = fatigue_fracture.CycleJumpLedger()
    ledger.begin(decision)
    ledger.rollback(error_estimate=0.2, message="front error; cut back")
    smaller = policy.propose(
        start_cycle=0,
        damage_rate=1.0e-4,
        front_advance_rate=1.0e-3,
        stop_cycle=20,
    )
    ledger.begin(smaller)
    ledger.commit(error_estimate=0.005)
    restored = fatigue_fracture.CycleJumpLedger()
    restored.restore(ledger.snapshot())
    assert restored.current_cycle == smaller.end_cycle
    assert restored.summary()["rejected_blocks"] == 1


def test_below_threshold_or_static_hold_does_not_accumulate_fatigue_damage():
    transaction = _cyclic().transaction(2)
    transaction.begin_cycle(
        np.array([0.10, 0.20]),
        np.array([0.10, 0.205]),
        cycles=10_000,
    )
    transaction.commit_cycle()
    np.testing.assert_allclose(transaction.fatigue_damage, 0.0)
    np.testing.assert_allclose(transaction.cumulative_cycles, 10_000.0)


def test_monotonic_path_exactly_recovers_wrapped_bilinear_law():
    law = _cyclic()
    transaction = law.transaction(1)
    for opening in (0.005, 0.1, 0.03, -0.001):
        response = transaction.begin(np.array([opening]))
        reference = law.monotonic.update(
            np.array([opening]), transaction.committed_maximum
        )
        np.testing.assert_allclose(response.traction, reference.traction)
        np.testing.assert_allclose(response.damage, reference.damage)
        np.testing.assert_allclose(
            response.dissipated_energy, reference.dissipated_energy
        )
        transaction.commit()


def test_cycle_damage_is_irreversible_and_compression_does_not_heal_or_dissipate():
    transaction = _cyclic().transaction(1)
    transaction.begin_cycle(np.array([0.0]), np.array([0.2]), cycles=100)
    transaction.commit_cycle()
    damage = transaction.fatigue_damage.copy()
    dissipation = transaction.state_arrays()["fatigue_dissipated_energy"].copy()

    closed = transaction.begin(np.array([-0.01]))
    assert closed.traction == pytest.approx([-10.0])
    transaction.commit()
    np.testing.assert_allclose(transaction.fatigue_damage, damage)
    np.testing.assert_allclose(
        transaction.state_arrays()["fatigue_dissipated_energy"], dissipation
    )


def test_cycle_block_commit_rollback_and_restart_are_atomic():
    law = _cyclic(residual_exponent=1.0)
    transaction = law.transaction(2)
    transaction.begin_cycle([0.0, 0.02], [0.2, 0.25], cycles=25)
    transaction.rollback()
    np.testing.assert_allclose(transaction.fatigue_damage, 0.0)
    np.testing.assert_allclose(transaction.cumulative_cycles, 0.0)

    transaction.begin_cycle([0.0, 0.02], [0.2, 0.25], cycles=25)
    transaction.commit_cycle()
    snapshot = transaction.snapshot()
    restored = law.transaction(2)
    restored.restore(snapshot)
    for name, values in transaction.state_arrays().items():
        np.testing.assert_allclose(restored.state_arrays()[name], values)


def test_cyclic_state_portable_checkpoint_preserves_every_history_field(tmp_path):
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    )
    topology = interfaces.pair_coincident_line_facets(
        coordinates,
        negative_facets=np.array([[0, 1]]),
        positive_facets=np.array([[2, 3]]),
        normal_hint=(0.0, 1.0),
    )
    law = _cyclic(residual_exponent=1.0)
    source = law.transaction(topology.number_of_points)
    source.begin_cycle([0.01, 0.02], [0.20, 0.25], cycles=123)
    source.commit_cycle()
    manifest = interfaces.save_portable_cohesive_state(
        tmp_path / "cyclic",
        topology,
        source,
        comm=MPI.COMM_SELF,
    )
    restored = law.transaction(topology.number_of_points)
    metadata = interfaces.load_portable_cohesive_state(
        manifest,
        topology,
        restored,
        comm=MPI.COMM_SELF,
    )
    for name, values in source.state_arrays().items():
        np.testing.assert_allclose(restored.state_arrays()[name], values)
    assert set(metadata["state_fields"]) == set(source.state_arrays())


def test_constant_extrema_cycle_jump_matches_exact_cycle_by_cycle_solution():
    law = _cyclic(residual_exponent=1.0)
    exact = law.transaction(1)
    for _ in range(100):
        exact.begin_cycle([0.0], [0.2], cycles=1)
        exact.commit_cycle()
    jumped = law.transaction(1)
    jumped.begin_cycle([0.0], [0.2], cycles=100)
    jumped.commit_cycle()
    np.testing.assert_allclose(
        jumped.fatigue_damage,
        exact.fatigue_damage,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def test_existing_3d_surface_assembler_consumes_cyclic_transaction():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    topology = interfaces.pair_coincident_surface_facets(
        coordinates,
        negative_facets=np.array([[0, 1, 2]]),
        positive_facets=np.array([[3, 4, 5]]),
        normal_hint=(0.0, 0.0, 1.0),
    )
    assembler = interfaces.ModeICohesiveSurfaceAssembler(
        topology,
        _cyclic(),
        number_of_nodes=6,
    )
    displacement = np.zeros_like(coordinates)
    displacement[3:, 2] = 0.2
    first = assembler.begin(displacement)
    assembler.commit()
    assembler.state.begin_cycle(
        np.zeros(topology.number_of_points),
        np.full(topology.number_of_points, 0.2),
        cycles=100,
    )
    assembler.state.commit_cycle()
    degraded = assembler.begin(displacement)
    assert np.linalg.norm(degraded.internal_force) < np.linalg.norm(
        first.internal_force
    )
    assert degraded.dissipated_energy > first.dissipated_energy


def test_cyclic_fatigue_procedure_keeps_cycles_distinct_from_time():
    procedure = procedures.cyclic_fatigue()
    assert procedure.control == "cycle_increments"
    assert procedure.equation_order == "static"
    assert procedure.stateful is True
    assert procedure.requires_global_solve is True


def test_3d_surface_observer_recovers_failed_area_front_and_components():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    facets = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    observation = fatigue_fracture.observe_surface_crack(
        coordinates,
        facets,
        damage=[1.0, 1.0, 0.0, 0.0],
        opening=[0.2, 0.3, 0.0, 0.0],
        cycle=100,
        name="lower",
    )
    assert observation.failed_area == pytest.approx(1.0)
    assert observation.front_length == pytest.approx(1.0)
    assert observation.component_count == 1
    assert observation.maximum_cod == pytest.approx(0.3)
    assert observation.mean_cod == pytest.approx(0.25)


def test_two_crack_interaction_reports_ligament_and_single_crack_baseline_ratio():
    first = fatigue_fracture.SurfaceCrackObservation(
        cycle=50,
        name="lower",
        failed_facets=np.array([True]),
        component_labels=np.array([0]),
        front_segments=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        failed_area=1.0,
        front_length=1.0,
        maximum_cod=0.2,
        mean_cod=0.1,
        damage_threshold=0.95,
    )
    second = fatigue_fracture.SurfaceCrackObservation(
        cycle=50,
        name="upper",
        failed_facets=np.array([True]),
        component_labels=np.array([0]),
        front_segments=np.array([[[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]]),
        failed_area=1.0,
        front_length=1.0,
        maximum_cod=0.2,
        mean_cod=0.1,
        damage_threshold=0.95,
    )
    interaction = fatigue_fracture.surface_crack_interaction(
        first,
        second,
        first_single_growth_rate=1.0e-4,
        first_double_growth_rate=0.8e-4,
        second_single_growth_rate=1.0e-4,
        second_double_growth_rate=1.2e-4,
    )
    assert interaction.minimum_ligament == pytest.approx(2.0)
    assert interaction.first_growth_ratio == pytest.approx(0.8)
    assert interaction.second_growth_ratio == pytest.approx(1.2)
    assert interaction.coalesced is False


def _two_component_surface_observation(*, cycle, damage):
    coordinates = np.array(
        [(float(x), float(y), 0.0) for y in range(2) for x in range(4)]
    )
    facets = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
        ]
    )
    return fatigue_fracture.observe_surface_crack(
        coordinates,
        facets,
        damage=damage,
        opening=np.asarray(damage) * 0.2,
        cycle=cycle,
        name="cylinder_surface",
        facet_ids=[("surface", index) for index in range(6)],
    )


def test_same_surface_crack_tracker_preserves_identity_and_records_merge_restart():
    separated = _two_component_surface_observation(
        cycle=10,
        damage=[1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    )
    assert separated.component_count == 2
    assert separated.facet_identity == "declared_physical_key"
    tracker = fatigue_fracture.SurfaceCrackTracker(
        interface_name="cylinder_surface"
    )
    first = tracker.observe(separated)
    assert len(first.cracks) == 2
    assert [event.kind for event in first.events] == ["birth", "birth"]
    assert first.interactions()[0].minimum_ligament == pytest.approx(1.0)

    continued = tracker.observe(
        _two_component_surface_observation(
            cycle=15,
            damage=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        )
    )
    assert not continued.events
    assert {crack.crack_id for crack in continued.cracks} == {
        crack.crack_id for crack in first.cracks
    }
    assert sorted(
        crack.area_growth_rate for crack in continued.cracks
    ) == pytest.approx([0.0, 0.1])

    restored = fatigue_fracture.SurfaceCrackTracker(
        interface_name="cylinder_surface"
    )
    restored.restore(tracker.snapshot())
    merged = restored.observe(
        _two_component_surface_observation(
            cycle=20,
            damage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    )
    assert len(merged.cracks) == 1
    assert [event.kind for event in merged.events] == ["merge"]
    assert set(merged.cracks[0].parent_ids) == {
        continued.cracks[0].crack_id,
        continued.cracks[1].crack_id,
    }
    assert merged.cracks[0].crack_id not in {
        continued.cracks[0].crack_id,
        continued.cracks[1].crack_id,
    }


def test_paris_evidence_is_postprocessing_and_recovers_declared_power_relation():
    cycles = np.arange(21, dtype=float)
    coefficient = 2.0e-6
    driving_force = np.sqrt(cycles + 1.0)
    crack_size = 0.01 + coefficient * (0.5 * cycles**2 + cycles)
    fit_mask = np.ones(cycles.size, dtype=bool)
    fit_mask[[0, -1]] = False

    evidence = fatigue_fracture.paris_evidence(
        cycles,
        crack_size,
        driving_force,
        fit_mask=fit_mask,
        driving_force_name="Delta K",
        driving_force_unit="MPa sqrt(m)",
        crack_size_unit="m",
    )

    assert evidence.coefficient == pytest.approx(coefficient, rel=1.0e-12)
    assert evidence.exponent == pytest.approx(2.0, rel=1.0e-12)
    assert evidence.coefficient_of_determination == pytest.approx(1.0)
    assert evidence.summary()["role"] == "postprocessing_not_solver_input"
    np.testing.assert_allclose(
        evidence.predict(driving_force[fit_mask]),
        coefficient * driving_force[fit_mask] ** 2,
    )


def test_paris_evidence_rejects_a_decreasing_crack_history():
    with pytest.raises(ValueError, match="cannot decrease"):
        fatigue_fracture.paris_evidence(
            [0.0, 1.0, 2.0],
            [0.0, 0.1, 0.05],
            [1.0, 1.1, 1.2],
        )


def test_named_cohesive_collection_keeps_independent_restart_identity():
    class FakeForce:
        def __init__(self, value):
            self.value = value

        def add_to_vector(self, vector):
            vector[:] += self.value

        def commit(self):
            pass

        def rollback(self):
            pass

        def snapshot(self):
            return {"value": self.value}

        def restore(self, snapshot):
            self.value = snapshot["value"]

        def summary(self):
            return {"value": self.value}

        def current_response(self):
            return type("Response", (), {"stored_energy": 1.0, "dissipated_energy": self.value})()

        def for_displacement(self, _displacement):
            return self

        def stability_inputs(self, _mass):
            return {
                "interface_stiffness": self.value,
                "interface_area": 1.0,
                "negative_mass": 1.0,
                "positive_mass": 1.0,
            }

    collection = fracture.named_cohesive_forces(
        lower=FakeForce(1.0),
        upper=FakeForce(2.0),
    )
    vector = np.zeros(2)
    collection.add_to_vector(vector)
    np.testing.assert_allclose(vector, 3.0)
    assert collection.names == ("lower", "upper")
    assert collection.current_response().dissipated_energy == pytest.approx(3.0)
    snapshot = collection.snapshot()
    collection["lower"].value = 9.0
    collection.restore(snapshot)
    assert collection["lower"].value == pytest.approx(1.0)


class _FakeArray:
    def __init__(self, values):
        self.array = np.asarray(values, dtype=float)

    def scatter_forward(self):
        pass


class _FakeField:
    def __init__(self):
        self.x = _FakeArray([0.0])
        self.function_space = SimpleNamespace(
            mesh=SimpleNamespace(comm=MPI.COMM_SELF)
        )


class _FakeCyclicForce:
    def __init__(self, law, displacement):
        self.displacement = displacement
        self.assembler = SimpleNamespace(
            law=law,
            state=law.transaction(1),
        )
        self.opening = np.zeros(1)

    def add_to_vector(self, vector):
        vector[:] += 0.0

    def commit(self):
        pass

    def rollback(self):
        self.assembler.state.rollback()

    def snapshot(self):
        return {
            "state": self.assembler.state.snapshot(),
            "opening": self.opening.tolist(),
        }

    def restore(self, snapshot):
        self.assembler.state.restore(snapshot["state"])
        self.opening = np.asarray(snapshot["opening"], dtype=float)

    def summary(self):
        return {"law": self.assembler.law.summary()}

    def current_response(self):
        response = self.material_point_response()
        return SimpleNamespace(
            stored_energy=float(np.sum(response.stored_energy)),
            dissipated_energy=float(np.sum(response.dissipated_energy)),
        )

    def material_point_response(self):
        return self.assembler.state.evaluate(self.opening)

    def cycle_opening(self):
        return self.opening.copy()

    def for_displacement(self, displacement):
        self.displacement = displacement
        return self

    def stability_inputs(self, _mass):
        return {
            "interface_stiffness": self.assembler.law.initial_stiffness,
            "interface_area": 1.0,
            "negative_mass": 1.0,
            "positive_mass": 1.0,
        }


def _global_fatigue_fixture(*, feedback=0.1):
    field = _FakeField()
    force = _FakeCyclicForce(_cyclic(residual_exponent=1.0), field)
    collection = fracture.named_cohesive_forces(crack=force)
    state = fatigue_fracture.field_state(displacement=field)

    def solve_equilibrium(*, load, branch, cycle):
        damage = force.assembler.state.fatigue_damage[0]
        opening = 0.2 * load * (1.0 + feedback * damage)
        field.x.array[0] = opening
        force.opening[:] = opening
        return {
            "iterations": 3,
            "reaction": load,
            "control_displacement": opening,
            "energy_balance_error": 1.0e-8,
        }

    cycle = fatigue_fracture.force_cycle(fmin=0.1, fmax=1.0)
    return field, force, collection, state, solve_equilibrium, cycle


def test_global_cycle_step_closes_at_valley_and_lands_on_requested_cycles():
    field, force, collection, state, solve, cycle = _global_fatigue_fixture()
    step = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle,
        stop_cycle=20,
        interfaces=collection,
        state=state,
        solve_equilibrium=solve,
        jump=fatigue_fracture.CycleJumpPolicy(
            maximum_damage_increment=0.01,
            maximum_cycles=100,
        ),
        landing_cycles=(5, 20),
        maximum_energy_balance_error=1.0e-6,
    )
    step.run()
    assert step.current_cycle == 20
    accepted_ends = [
        record.decision.end_cycle
        for record in step.ledger.records
        if record.accepted
    ]
    assert 5 in accepted_ends
    assert accepted_ends[-1] == 20
    assert step.history[-1].closing.branch == "closing"
    assert field.x.array[0] < step.history[-1].maximum.control_displacement
    assert force.assembler.state.cumulative_cycles == pytest.approx([20.0])


def test_global_cycle_step_cuts_back_structural_feedback_atomically():
    field, force, collection, state, solve, cycle = _global_fatigue_fixture(
        feedback=20.0
    )
    step = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle,
        stop_cycle=30,
        interfaces=collection,
        state=state,
        solve_equilibrium=solve,
        jump=fatigue_fracture.CycleJumpPolicy(
            maximum_damage_increment=0.02,
            maximum_cycles=100,
        ),
        maximum_opening_feedback=0.01,
    )
    step.run()
    assert step.current_cycle == 30
    assert step.ledger.summary()["rejected_blocks"] > 0
    assert force.assembler.state.cumulative_cycles == pytest.approx([30.0])
    assert all(
        block.opening_feedback_error <= 0.01
        for block in step.history
    )


def test_global_cycle_restart_matches_continuous_cycle_history():
    field, force, collection, state, solve, cycle = _global_fatigue_fixture()
    interrupted = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle,
        stop_cycle=25,
        interfaces=collection,
        state=state,
        solve_equilibrium=solve,
        landing_cycles=(5,),
    )
    interrupted.run(until_cycle=5)
    checkpoint = interrupted.snapshot()

    field2, force2, collection2, state2, solve2, cycle2 = _global_fatigue_fixture()
    restarted = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle2,
        stop_cycle=25,
        interfaces=collection2,
        state=state2,
        solve_equilibrium=solve2,
        landing_cycles=(5,),
    )
    restarted.restore(checkpoint)
    restarted.run()

    field3, force3, collection3, state3, solve3, cycle3 = _global_fatigue_fixture()
    continuous = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle3,
        stop_cycle=25,
        interfaces=collection3,
        state=state3,
        solve_equilibrium=solve3,
        landing_cycles=(5,),
    )
    continuous.run()
    np.testing.assert_allclose(
        force2.assembler.state.fatigue_damage,
        force3.assembler.state.fatigue_damage,
    )
    np.testing.assert_allclose(field2.x.array, field3.x.array)
    assert restarted.current_cycle == continuous.current_cycle == 25


def test_global_cycle_durable_checkpoint_restores_fields_interfaces_and_ledger(tmp_path):
    field, force, collection, state, solve, cycle = _global_fatigue_fixture()
    partial = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle,
        stop_cycle=12,
        interfaces=collection,
        state=state,
        solve_equilibrium=solve,
        landing_cycles=(4,),
    )
    partial.run(until_cycle=4)
    manifest = partial.save_checkpoint(tmp_path / "fatigue")
    assert manifest.name.endswith(".cyclic-fatigue.json")

    field2, force2, collection2, state2, solve2, cycle2 = _global_fatigue_fixture()
    restarted = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=cycle2,
        stop_cycle=12,
        interfaces=collection2,
        state=state2,
        solve_equilibrium=solve2,
        landing_cycles=(4,),
    )
    restarted.load_checkpoint(manifest)
    assert restarted.current_cycle == 4
    np.testing.assert_allclose(
        force2.assembler.state.fatigue_damage,
        force.assembler.state.fatigue_damage,
    )
    np.testing.assert_allclose(field2.x.array, field.x.array)
    restarted.run()
    assert restarted.current_cycle == 12


def test_field_state_durable_checkpoint_validates_real_fem_partition(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain)
    displacement.value.x.array[:] = np.arange(displacement.value.x.array.size)
    expected = displacement.value.x.array.copy()
    state = fatigue_fracture.field_state(displacement=displacement)
    manifest = state.save_checkpoint(tmp_path / "field")
    displacement.value.x.array[:] = -1.0
    state.load_checkpoint(manifest)
    np.testing.assert_allclose(displacement.value.x.array, expected)


def test_two_named_3d_interfaces_are_split_atomically_on_one_solver_mesh():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
        ]
    )
    cells = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [5, 6, 7, 8],
            [5, 6, 7, 9],
        ]
    )
    split = interfaces.split_conforming_named_interfaces(
        coordinates,
        cells,
        {
            "lower": {
                "interface_facets": [[0, 1, 2]],
                "positive_cells": [1],
            },
            "upper": {
                "interface_facets": [[5, 6, 7]],
                "positive_cells": [3],
            },
        },
    )
    assert split.names == ("lower", "upper")
    assert split.coordinates.shape == (16, 3)
    assert set(split["lower"].original_to_duplicate).isdisjoint(
        split["upper"].original_to_duplicate
    )
    assert split.combined().negative_facets.shape == (2, 3)

    domain = interfaces.create_dolfinx_split_mesh(
        split,
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    displacement = fields.displacement(domain)
    forces = fracture.named_mode_i_cohesive_forces(
        split,
        displacement,
        laws={"lower": _cyclic(), "upper": _cyclic()},
        normal_hints={"lower": (0.0, 0.0, 1.0), "upper": (0.0, 0.0, 1.0)},
    )
    assert forces.names == ("lower", "upper")
    assert set(forces.cycle_openings()) == {"lower", "upper"}

    mixed = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
    )
    recommended = fracture.cohesive_forces(
        split,
        displacement,
        laws={
            "lower": interfaces.bilinear_cohesive(
                strength=10.0,
                fracture_energy=2.0,
                initial_stiffness=1000.0,
            ),
            "upper": mixed,
        },
        normal_hints={"lower": (0.0, 0.0, 1.0), "upper": (0.0, 0.0, 1.0)},
    )
    summaries = recommended.summary()["interfaces"]
    assert summaries["lower"]["interface_kinematics"] == "tie"
    assert summaries["upper"]["interface_kinematics"] == "mixed"


def test_named_interface_split_rejects_ambiguous_shared_nodes():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    specification = {
        "interface_facets": [[1, 2]],
        "positive_cells": [1],
    }
    with pytest.raises(ValueError, match="must not share source nodes"):
        interfaces.split_conforming_named_interfaces(
            coordinates,
            cells,
            {"first": specification, "second": specification},
        )
