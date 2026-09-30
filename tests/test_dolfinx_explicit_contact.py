from __future__ import annotations

import copy

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from agentfem import boundary_models, fracture, operators, problems, time
from agentfem import mesh as agent_mesh


def _cube(comm, *, cells_x: int = 1):
    return mesh.create_unit_cube(
        comm,
        cells_x,
        1,
        1,
        cell_type=mesh.CellType.tetrahedron,
    )


def _left_region(domain):
    facet_dimension = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0),
    )
    tags = mesh.meshtags(
        domain,
        facet_dimension,
        facets,
        np.full(facets.size, 27, dtype=np.int32),
    )
    return agent_mesh.tagged_boundary_region(
        domain,
        tags,
        tag=27,
        name="left_contact_slave",
    )


def _vector_space(domain):
    return fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            domain.basix_cell(),
            1,
            shape=(3,),
        ),
    )


def _triangulated_tool_plane():
    return boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.05, 0.0, 0.0),
            (0.05, 0.0, 1.0),
            (0.05, 1.0, 1.0),
            (0.05, 1.0, 0.0),
        ),
        triangles=((0, 1, 2), (0, 2, 3)),
        facet_ids=(101, 202),
        name="triangulated_tool_plane",
    )


def _bulk_residual(domain, function_space, value):
    test = ufl.TestFunction(function_space)
    source = fem.Constant(domain, np.asarray(value, dtype=float))
    return operators.OperatorForm(
        name="bulk_residual",
        expression=ufl.inner(source, test) * ufl.dx,
        kind="bulk_residual",
        role="residual",
        family="test",
    )


def _contact(
    comm,
    *,
    cells_x: int = 1,
    displacement_value: float = 0.1,
    bulk_value=(0.0, 0.0, 0.0),
    motion_schedule=None,
):
    domain = _cube(comm, cells_x=cells_x)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = displacement_value
    displacement.x.scatter_forward()
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    surface = boundary_models.rigid_plane(
        point=(0.05, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, bulk_value),
        adapter=adapter,
        displacement=displacement,
        projector=surface,
        penalty=200.0,
        maximum_stable_time_increment=1.0e-3,
        motion_schedule=motion_schedule,
        surface_reference_point=(0.0, 0.0, 0.0),
    )
    return domain, displacement, residual


def test_explicit_contact_residual_assembles_force_and_accepts_evidence():
    domain, _displacement, residual = _contact(MPI.COMM_SELF)

    vector = residual.assemble_vector()
    try:
        values = vector.array.reshape((-1, 3))
        np.testing.assert_allclose(np.sum(values, axis=0), (10.0, 0.0, 0.0))
    finally:
        vector.destroy()

    assert residual.trial_evidence is not None
    np.testing.assert_allclose(
        residual.trial_evidence.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(residual.trial_evidence.potential_energy, 0.25)
    residual.commit()
    assert residual.trial_evidence is None
    assert residual.accepted_evidence is not None
    assert residual.accepted_evaluations == 1
    assert residual.lifecycle.state.accepted is not None
    assert residual.summary()["surface_motion"] == "fixed_only"
    assert domain.comm.size == 1


def test_explicit_contact_residual_rolls_back_failed_trial():
    _domain, displacement, residual = _contact(MPI.COMM_SELF)
    vector = residual.assemble_vector()
    vector.destroy()
    residual.rollback()
    assert residual.trial_evidence is None
    assert residual.lifecycle.state.trial is None

    displacement.x.array[:] = np.nan
    with pytest.raises(ValueError, match="rejected collectively"):
        residual.assemble_vector()
    assert residual.trial_evidence is None
    assert residual.lifecycle.state.trial is None


def test_explicit_contact_residual_enforces_declared_stability_limit():
    _domain, displacement, residual = _contact(MPI.COMM_SELF)
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    integrator = time.explicit.central_difference(state=state, mass=mass)

    with pytest.raises(ValueError, match="exceeds.*stability limit"):
        problems.explicit_dynamics(
            state=state,
            integrator=integrator,
            residual=residual,
            dt=2.0e-3,
            steps=1,
            progress=False,
        )

    step = problems.explicit_dynamics(
        state=state,
        integrator=integrator,
        residual=residual,
        dt=5.0e-4,
        steps=1,
        progress=False,
    )
    assert step.dt == pytest.approx(5.0e-4)


def test_explicit_contact_residual_advances_through_central_difference():
    _domain, displacement, residual = _contact(MPI.COMM_SELF)
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    integrator = time.explicit.central_difference(state=state, mass=mass)
    step = problems.explicit_dynamics(
        state=state,
        integrator=integrator,
        residual=residual,
        dt=5.0e-4,
        steps=1,
        progress=False,
    )

    step.run()

    acceleration = state.a.value.x.array.reshape((-1, 3)).sum(axis=0)
    np.testing.assert_allclose(acceleration, (-10.0, 0.0, 0.0), atol=1.0e-13)
    assert step.completed_steps == 1
    assert residual.accepted_evaluations == 1
    assert residual.lifecycle.state.accepted is not None


def test_moving_contact_records_only_accepted_force_moment_work():
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.02, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion,
        end_time=1.0e-3,
    )
    _domain, displacement, residual = _contact(
        MPI.COMM_SELF,
        motion_schedule=schedule,
    )
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    integrator = time.explicit.central_difference(state=state, mass=mass)
    step = problems.explicit_dynamics(
        state=state,
        integrator=integrator,
        residual=residual,
        dt=5.0e-4,
        steps=1,
        progress=False,
    )

    step.run()

    assert residual.work_state is not None
    assert len(residual.work_state.accepted) == 2
    assert residual.work_state.current.factor == pytest.approx(0.5)
    assert residual.work_state.path_work == pytest.approx(-0.09)
    assert residual.work_state.latest_interval_power == pytest.approx(-180.0)
    summary = residual.summary()
    assert summary["surface_motion"] == "prescribed_proportional_rigid_motion"
    assert summary["prescribed_motion_work"]["path_work"] == pytest.approx(-0.09)


def test_moving_contact_enters_shared_dynamic_energy_balance():
    class ZeroBodyEnergy:
        def evaluate(self, *, displacement, velocity):
            return {"total_mechanical_energy": 0.0}

    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.02, 0.0, 0.0),
        ),
        end_time=1.0e-3,
    )
    _domain, displacement, residual = _contact(
        MPI.COMM_SELF,
        motion_schedule=schedule,
    )
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    integrator = time.explicit.central_difference(state=state, mass=mass)
    ledger = fracture.DynamicEnergyLedger(
        energy=ZeroBodyEnergy(),
        state=state,
        mass=mass,
        residual=residual,
    )
    step = problems.explicit_dynamics(
        state=state,
        integrator=integrator,
        residual=residual,
        dt=5.0e-4,
        steps=1,
        progress=False,
        history_monitor=ledger,
    )

    step.run()

    assert len(step.history_records) == 2
    initial, accepted = step.history_records
    assert initial["contact_potential_energy"] == pytest.approx(0.25)
    assert accepted["contact_motion_work"] == pytest.approx(-0.09)
    assert accepted["external_work"] == pytest.approx(-0.09)
    assert accepted["contact_potential_energy"] == pytest.approx(0.16)
    assert accepted["energy_balance_error"] == pytest.approx(0.0, abs=1.0e-14)


def test_moving_contact_work_snapshot_round_trip_and_identity_check():
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.02, 0.0, 0.0),
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion,
        end_time=1.0e-3,
    )
    _domain, _displacement, residual = _contact(
        MPI.COMM_SELF,
        motion_schedule=schedule,
    )
    residual.initialize_accepted_state(time=0.0)
    residual.update_time(5.0e-4)
    vector = residual.assemble_vector()
    vector.destroy()
    residual.commit()
    snapshot = residual.snapshot()

    _other_domain, _other_displacement, restored = _contact(
        MPI.COMM_SELF,
        motion_schedule=schedule,
    )
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert restored.work_state.path_work == pytest.approx(-0.09)

    pristine = restored.snapshot()
    corrupt = copy.deepcopy(snapshot)
    corrupt["accepted_evidence"]["contact_force_on_surface"][0] = 99.0
    with pytest.raises(ValueError, match="action and reaction"):
        restored.restore(corrupt)
    assert restored.snapshot() == pristine

    incompatible = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(translation=(0.03, 0.0, 0.0)),
        end_time=1.0e-3,
    )
    _third_domain, _third_displacement, rejected = _contact(
        MPI.COMM_SELF,
        motion_schedule=incompatible,
    )
    with pytest.raises(ValueError, match="identity differs"):
        rejected.restore(snapshot)


def test_moving_contact_transient_checkpoint_matches_uninterrupted_run(tmp_path):
    class ZeroBodyEnergy:
        def evaluate(self, *, displacement, velocity):
            return {"total_mechanical_energy": 0.0}

    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.02, 0.0, 0.0),
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion,
        end_time=1.0e-3,
    )

    def build_step():
        _domain, displacement, residual = _contact(
            MPI.COMM_SELF,
            motion_schedule=schedule,
        )
        state = problems.second_order_state(displacement)
        diagonal = np.ones(displacement.x.array.shape, dtype=float)
        mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
        integrator = time.explicit.central_difference(state=state, mass=mass)
        ledger = fracture.DynamicEnergyLedger(
            energy=ZeroBodyEnergy(),
            state=state,
            mass=mass,
            residual=residual,
        )
        step = problems.explicit_dynamics(
            state=state,
            integrator=integrator,
            residual=residual,
            dt=5.0e-4,
            steps=2,
            progress=False,
            history_monitor=ledger,
        )
        return step, residual

    continuous, continuous_residual = build_step()
    continuous.run()

    partial, _partial_residual = build_step()
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(tmp_path / "moving-contact")

    restarted, restarted_residual = build_step()
    restarted.load_checkpoint(checkpoint)
    restarted.run()

    np.testing.assert_allclose(
        restarted.state.u.value.x.array,
        continuous.state.u.value.x.array,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.state.v.value.x.array,
        continuous.state.v.value.x.array,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert restarted_residual.work_state.path_work == pytest.approx(
        continuous_residual.work_state.path_work
    )
    assert restarted_residual.work_state.snapshot() == (
        continuous_residual.work_state.snapshot()
    )
    assert restarted.history_records == continuous.history_records


def test_composed_contact_pairs_share_time_work_energy_and_restart():
    class ZeroBodyEnergy:
        def evaluate(self, *, displacement, velocity):
            return {"total_mechanical_energy": 0.0}

    inner_schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.02, 0.0, 0.0),
            name="inner_motion",
        ),
        end_time=1.0e-3,
        name="inner_schedule",
    )
    outer_schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.01, 0.0, 0.0),
            name="outer_motion",
        ),
        end_time=1.0e-3,
        name="outer_schedule",
    )

    def build_step():
        domain, displacement, inner = _contact(
            MPI.COMM_SELF,
            motion_schedule=inner_schedule,
        )
        outer_adapter = boundary_models.dolfinx_boundary_region_contact_trace(
            _left_region(domain),
            displacement.function_space,
        )
        outer = boundary_models.dolfinx_explicit_contact_residual(
            inner,
            adapter=outer_adapter,
            displacement=displacement,
            projector=boundary_models.rigid_plane(
                point=(0.04, 0.0, 0.0),
                normal=(-1.0, 0.0, 0.0),
                name="outer_plane",
            ),
            penalty=100.0,
            maximum_stable_time_increment=1.0e-3,
            motion_schedule=outer_schedule,
            name="outer_contact",
        )
        state = problems.second_order_state(displacement)
        diagonal = np.ones(displacement.x.array.shape, dtype=float)
        mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
        integrator = time.explicit.central_difference(state=state, mass=mass)
        ledger = fracture.DynamicEnergyLedger(
            energy=ZeroBodyEnergy(),
            state=state,
            mass=mass,
            residual=outer,
        )
        step = problems.explicit_dynamics(
            state=state,
            integrator=integrator,
            residual=outer,
            dt=5.0e-4,
            steps=1,
            progress=False,
            history_monitor=ledger,
        )
        return step, outer

    step, residual = build_step()
    step.run()

    terms = residual.contact_energy_evidence()
    assert [item["name"] for item in terms] == [
        "dolfinx_explicit_contact_residual",
        "outer_contact",
    ]
    assert sum(item["contact_motion_work"] for item in terms) == pytest.approx(
        -0.11875
    )
    accepted = step.history_records[-1]
    assert accepted["contact_motion_work"] == pytest.approx(-0.11875)
    assert accepted["contact_potential_energy"] == pytest.approx(0.31125)
    assert accepted["energy_balance_error"] == pytest.approx(0.0, abs=1.0e-14)
    result = step.solve_result()
    assert result.histories["contact_motion_work"].latest == pytest.approx(-0.11875)
    assert result.histories["contact_potential_energy"].latest == pytest.approx(
        0.31125
    )
    assert result.metadata["step"]["residual"]["name"] == "outer_contact"

    snapshot = residual.snapshot()
    _restored_step, restored = build_step()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert restored.contact_energy_evidence() == terms


def test_moving_triangle_bvh_tracks_facet_crossing_without_spurious_work():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.0, 0.5),
            name="tangential_tool_motion",
        ),
        end_time=1.0e-3,
    )
    body = boundary_models.rigid_body(
        _triangulated_tool_plane(),
        motion_schedule=schedule,
        name="sliding_tool",
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        projector=boundary_models.triangle_surface_bvh(
            body.surface
        ),
        rigid_body=body,
        penalty=200.0,
        maximum_stable_time_increment=1.0e-3,
        name="sliding_triangle_contact",
    )
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    step = problems.explicit_dynamics(
        state=state,
        integrator=time.explicit.central_difference(state=state, mass=mass),
        residual=residual,
        dt=5.0e-4,
        steps=2,
        progress=False,
    )

    trace = adapter.evaluate(displacement)
    initial = residual.lifecycle.evaluate(
        trace.query_points,
        motion=schedule.motion,
        factor=schedule.factor_at(0.0),
    ).record
    initial_entities = initial.projection.entity_ids.copy()
    initial_points = initial.point_ids.copy()
    residual.lifecycle.rollback_increment()
    step.run()
    final = residual.lifecycle.state.accepted

    np.testing.assert_array_equal(final.point_ids, initial_points)
    assert np.any(final.projection.entity_ids != initial_entities)
    assert residual.work_state.path_work == pytest.approx(0.0, abs=1.0e-14)
    assert residual.summary()["lifecycle"]["projector"]["method"] == (
        "deterministic_aabb_bvh"
    )
    assert residual.summary()["rigid_body"]["name"] == "sliding_tool"


def test_explicit_contact_rejects_projector_from_another_rigid_body():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.04, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
            name="other_tool",
        ),
        name="other_body",
    )

    with pytest.raises(ValueError, match="geometry differs"):
        boundary_models.dolfinx_explicit_contact_residual(
            _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
            adapter=adapter,
            displacement=displacement,
            projector=boundary_models.triangle_surface_bvh(
                _triangulated_tool_plane()
            ),
            rigid_body=body,
            penalty=200.0,
            maximum_stable_time_increment=1.0e-3,
        )


def test_explicit_contact_residual_preserves_global_force_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Explicit contact residual is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    _domain, _displacement, residual = _contact(
        comm,
        cells_x=2,
        bulk_value=(1.0, 1.0, 1.0),
    )

    vector = residual.assemble_vector()
    try:
        index_map = residual.adapter.function_space.dofmap.index_map
        owned = int(index_map.size_local) * 3
        local = vector.array[:owned].reshape((-1, 3)).sum(axis=0)
        total = np.empty(3, dtype=float)
        comm.Allreduce(local, total, op=MPI.SUM)
    finally:
        vector.destroy()

    # Unit-cube bulk resultant is (1, 1, 1).  The contact residual contributes
    # (10, 0, 0).  This catches accidental second reverse-scattering of the
    # already assembled bulk ghost entries.
    np.testing.assert_allclose(total, (11.0, 1.0, 1.0), atol=1.0e-13)
    np.testing.assert_allclose(
        residual.trial_evidence.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
        atol=1.0e-13,
    )
    assert residual.trial_evidence.active_point_count == 6
    residual.commit()


def test_moving_contact_work_is_rank_canonical_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Moving contact work is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.02, 0.0, 0.0),
        ),
        end_time=1.0e-3,
    )
    _domain, _displacement, residual = _contact(
        comm,
        cells_x=2,
        motion_schedule=schedule,
    )
    residual.initialize_accepted_state(time=0.0)
    residual.update_time(5.0e-4)
    vector = residual.assemble_vector()
    vector.destroy()
    residual.commit()

    snapshot = residual.snapshot()
    copies = comm.allgather(snapshot)
    assert all(item == copies[0] for item in copies)
    assert residual.work_state.path_work == pytest.approx(-0.09)


def test_moving_routed_triangle_contact_preserves_identity_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Moving distributed contact is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = _cube(comm, cells_x=2)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    surface = _triangulated_tool_plane()
    projector = boundary_models.routed_distributed_triangle_surface_bvh(
        boundary_models.partition_triangle_surface(surface, comm),
        comm,
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.0, 0.5),
            name="distributed_tangential_tool_motion",
        ),
        end_time=1.0e-3,
    )
    body = boundary_models.rigid_body(
        surface,
        motion_schedule=schedule,
        name="distributed_sliding_tool",
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        projector=projector,
        rigid_body=body,
        penalty=200.0,
        maximum_stable_time_increment=1.0e-3,
        name="distributed_sliding_triangle_contact",
    )
    trace = adapter.evaluate(displacement)
    initial = residual.lifecycle.evaluate(
        trace.query_points,
        motion=schedule.motion,
        factor=schedule.factor_at(0.0),
    ).record
    initial_entities = initial.projection.entity_ids.copy()
    initial_points = initial.point_ids.copy()
    residual.lifecycle.rollback_increment()

    state = problems.second_order_state(displacement)
    mass = operators.LumpedMassOperator.assemble(function_space, density=1.0)
    problems.explicit_dynamics(
        state=state,
        integrator=time.explicit.central_difference(state=state, mass=mass),
        residual=residual,
        dt=5.0e-4,
        steps=2,
        progress=False,
    ).run()

    final = residual.lifecycle.state.accepted
    np.testing.assert_array_equal(final.point_ids, initial_points)
    local_crossings = int(
        np.count_nonzero(final.projection.entity_ids != initial_entities)
    )
    assert comm.allreduce(local_crossings, op=MPI.SUM) > 0
    snapshots = comm.allgather(residual.snapshot())
    assert all(item == snapshots[0] for item in snapshots)
    assert residual.work_state.path_work == pytest.approx(0.0, abs=1.0e-14)
    assert projector.summary()["collective_pattern"] == (
        "two_stage_packed_alltoallv"
    )


def test_explicit_step_rolls_back_if_residual_commit_fails():
    class Residual:
        def __init__(self):
            self.rollbacks = 0

        def commit(self):
            raise RuntimeError("commit failed")

        def rollback(self):
            self.rollbacks += 1

    class Integrator:
        def step(self, *args, **kwargs):
            return None

    from agentfem._transient_problems import ExplicitDynamicsStep

    residual = Residual()
    step = ExplicitDynamicsStep(
        name="commit_failure",
        state=object(),
        integrator=Integrator(),
        residual=residual,
        dt=1.0,
        steps=1,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        step._advance_one(1.0)
    assert residual.rollbacks == 1
