from __future__ import annotations

import copy
import struct

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


def _hex_cube(comm, *, cells_x: int = 1):
    return mesh.create_unit_cube(
        comm,
        cells_x,
        1,
        1,
        cell_type=mesh.CellType.hexahedron,
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


def _write_binary_stl_tool_plane(path) -> None:
    facets = (
        (
            (0.05, 0.0, 0.0),
            (0.05, 0.0, 1.0),
            (0.05, 1.0, 1.0),
        ),
        (
            (0.05, 0.0, 0.0),
            (0.05, 1.0, 1.0),
            (0.05, 1.0, 0.0),
        ),
    )
    payload = bytearray(b"AgentFEM imported contact tool".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(facets)))
    for facet in facets:
        payload.extend(
            struct.pack(
                "<12fH",
                -1.0,
                0.0,
                0.0,
                *(coordinate for point in facet for coordinate in point),
                0,
            )
        )
    path.write_bytes(payload)


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
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
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


def test_contact_stability_estimate_is_mass_penalty_and_mpi_aware():
    domain = _cube(MPI.COMM_WORLD)
    function_space = _vector_space(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    mass = operators.LumpedMassOperator.assemble(function_space, density=1.0)

    baseline = boundary_models.estimate_dolfinx_contact_stability(
        adapter=adapter,
        lumped_mass=mass,
        normal_penalty=200.0,
        safety_factor=0.8,
    )
    stiffer = boundary_models.estimate_dolfinx_contact_stability(
        adapter=adapter,
        lumped_mass=mass,
        normal_penalty=800.0,
        safety_factor=0.8,
    )
    frictional = boundary_models.estimate_dolfinx_contact_stability(
        adapter=adapter,
        lumped_mass=mass,
        normal_penalty=200.0,
        tangential_penalty=800.0,
        friction_coefficient=0.25,
        safety_factor=0.8,
    )

    assert baseline.point_count > 0
    assert baseline.selected > 0.0
    assert baseline.mass_compatibility == "function_space_identity"
    assert stiffer.selected == pytest.approx(0.5 * baseline.selected)
    assert frictional.selected == pytest.approx(
        np.sqrt(200.0 / 1050.0) * baseline.selected
    )
    assert frictional.tangential_penalty_maximum == 800.0
    assert frictional.friction_coefficient == 0.25
    gathered = domain.comm.allgather(baseline.summary())
    assert all(item == gathered[0] for item in gathered)

    serial_domain = _cube(MPI.COMM_SELF)
    serial_space = _vector_space(serial_domain)
    serial_adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(serial_domain),
        serial_space,
    )
    serial_mass = operators.LumpedMassOperator.assemble(serial_space, density=1.0)
    serial = boundary_models.estimate_dolfinx_contact_stability(
        adapter=serial_adapter,
        lumped_mass=serial_mass,
        normal_penalty=200.0,
        safety_factor=0.8,
    )
    assert baseline.selected == pytest.approx(serial.selected)
    assert baseline.spectral_radius_upper_bound == pytest.approx(
        serial.spectral_radius_upper_bound
    )


def test_contact_stability_rejects_mass_from_another_space_instance():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    other_mass = operators.LumpedMassOperator.assemble(
        _vector_space(domain),
        density=1.0,
    )

    with pytest.raises(ValueError, match="same function-space instance"):
        boundary_models.estimate_dolfinx_contact_stability(
            adapter=adapter,
            lumped_mass=other_mass,
            normal_penalty=200.0,
        )


def test_explicit_contact_factory_can_select_automatic_stability_limit():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    mass = operators.LumpedMassOperator.assemble(function_space, density=1.0)
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        projector=boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        ),
        penalty=200.0,
        lumped_mass=mass,
        noncontact_unsafed_stability_limit=0.25,
        contact_stability_safety_factor=0.75,
    )

    summary = residual.summary()
    assert summary["stability"] == "automatic_contact_spectral_bound_available"
    assert summary["stability_controller"] == "combined_spectral_bound"
    assert summary["declared_maximum_stable_time_increment"] is None
    assert summary["contact_stability_estimate"]["safety_factor"] == 0.75
    combined = summary["combined_stability_estimate"]
    assert combined["noncontact_unsafed_limit"] == 0.25
    assert residual.maximum_stable_time_increment == pytest.approx(
        combined["selected"]
    )


def test_explicit_contact_factory_takes_stricter_declared_limit():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    mass = operators.LumpedMassOperator.assemble(function_space, density=1.0)
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        projector=boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        ),
        penalty=200.0,
        lumped_mass=mass,
        noncontact_unsafed_stability_limit=0.25,
        maximum_stable_time_increment=1.0e-8,
    )

    assert residual.maximum_stable_time_increment == pytest.approx(1.0e-8)
    assert residual.summary()["stability_controller"] == "caller_declared"
    declared = residual.summary()["declared_maximum_stable_time_increment"]
    assert declared == pytest.approx(1.0e-8)


def test_explicit_contact_factory_rejects_contact_only_automatic_limit():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    mass = operators.LumpedMassOperator.assemble(function_space, density=1.0)

    with pytest.raises(ValueError, match="noncontact_unsafed_stability_limit"):
        boundary_models.dolfinx_explicit_contact_residual(
            _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
            adapter=adapter,
            displacement=displacement,
            projector=boundary_models.rigid_plane(
                point=(0.05, 0.0, 0.0),
                normal=(-1.0, 0.0, 0.0),
            ),
            penalty=200.0,
            lumped_mass=mass,
        )


def test_explicit_contact_automatic_stability_rejects_curved_normal_geometry():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    mass = operators.LumpedMassOperator.assemble(function_space, density=1.0)

    with pytest.raises(NotImplementedError, match="piecewise-planar"):
        boundary_models.dolfinx_explicit_contact_residual(
            _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
            adapter=adapter,
            displacement=displacement,
            projector=boundary_models.rigid_sphere((0.5, 0.5, 0.5), 0.25),
            penalty=200.0,
            lumped_mass=mass,
        )


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


def test_rigid_pair_selects_bvh_without_putting_search_in_model_asset():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
        function_space,
    )
    surface = _triangulated_tool_plane()
    body = boundary_models.rigid_body(surface, name="tool_body")
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=200.0,
        name="tool_pair",
    )

    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    vector = residual.assemble_vector()
    vector.destroy()

    assert surface.summary()["kind"] == "triangulated_rigid_surface"
    assert residual.lifecycle.projector.summary()["kind"] == "triangle_surface_bvh"
    assert residual.trial_evidence.active_point_count == 6


def test_rigid_pair_selects_routed_bvh_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Automatic distributed tool search is reviewed on two ranks.")
    domain = _cube(MPI.COMM_WORLD)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
        function_space,
    )
    body = boundary_models.rigid_body(
        _triangulated_tool_plane(),
        name="distributed_tool_body",
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=200.0,
        name="distributed_tool_pair",
    )

    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    vector = residual.assemble_vector()
    vector.destroy()

    assert (
        residual.lifecycle.projector.summary()["kind"]
        == "routed_distributed_triangle_surface_bvh"
    )
    assert residual.trial_evidence.active_point_count == 6


def test_binary_stl_tool_reaches_dolfinx_contact_force_and_energy(tmp_path):
    pytest.importorskip("meshio")
    source = tmp_path / "tool.stl"
    _write_binary_stl_tool_plane(source)
    surface = boundary_models.triangulated_rigid_surface_from_mesh(
        source,
        coordinate_scale=1.0,
        name="imported_tool",
    )
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
        function_space,
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(surface, name="imported_tool_body"),
        penalty=200.0,
        name="imported_tool_pair",
    )

    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    vector = residual.assemble_vector()
    vector.destroy()

    assert residual.lifecycle.projector.summary()["kind"] == "triangle_surface_bvh"
    np.testing.assert_allclose(
        residual.trial_evidence.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
        atol=1.0e-12,
    )
    assert residual.trial_evidence.potential_energy == pytest.approx(0.25)
    assert surface.summary()["source"]["format"] == "stl"


def test_hexahedral_slave_trace_reaches_explicit_contact_residual():
    domain = _hex_cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
        function_space,
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(
            boundary_models.rigid_plane(
                point=(0.05, 0.0, 0.0),
                normal=(-1.0, 0.0, 0.0),
            ),
            name="hexahedral_tool_body",
        ),
        penalty=200.0,
        name="hexahedral_pair",
    )

    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    vector = residual.assemble_vector()
    vector.destroy()

    assert adapter.summary()["facet_topology"] == "quadrilateral"
    np.testing.assert_allclose(
        residual.trial_evidence.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
        atol=1.0e-12,
    )
    assert residual.trial_evidence.potential_energy == pytest.approx(0.25)


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

    corrupt_projection = copy.deepcopy(snapshot)
    corrupt_projection["projection_state"]["records"][0]["normal"] = None
    with pytest.raises(ValueError, match="lacks geometric values"):
        restored.restore(corrupt_projection)
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


def test_mixed_rigid_contact_pairs_compose_without_case_specific_solver():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
        function_space,
    )
    normal_schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.01, 0.0, 0.0),
            name="normal_tool_motion",
        ),
        end_time=5.0e-4,
    )
    normal_body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
            name="analytical_tool_surface",
        ),
        motion_schedule=normal_schedule,
        name="analytical_tool",
    )
    normal_pair = boundary_models.rigid_contact_pair(
        slave,
        normal_body,
        penalty=100.0,
        name="analytical_pair",
    )
    inner = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        contact_pair=normal_pair,
        maximum_stable_time_increment=1.0e-3,
        name="analytical_contact",
    )

    sliding_schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.0, 0.2),
            name="sliding_tool_motion",
        ),
        end_time=5.0e-4,
    )
    sliding_body = boundary_models.rigid_body(
        _triangulated_tool_plane(),
        motion_schedule=sliding_schedule,
        name="triangulated_tool",
    )
    sliding_pair = boundary_models.rigid_contact_pair(
        slave,
        sliding_body,
        penalty=100.0,
        name="triangulated_pair",
    )
    outer = boundary_models.dolfinx_explicit_contact_residual(
        inner,
        adapter=adapter,
        displacement=displacement,
        projector=boundary_models.triangle_surface_bvh(sliding_body.surface),
        contact_pair=sliding_pair,
        maximum_stable_time_increment=1.0e-3,
        name="triangulated_contact",
    )
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    step = problems.explicit_dynamics(
        state=state,
        integrator=time.explicit.central_difference(state=state, mass=mass),
        residual=outer,
        dt=5.0e-4,
        steps=1,
        progress=False,
    )

    step.run()

    progress = outer.contact_progress_evidence()
    assert [term["name"] for term in progress] == [
        "analytical_contact",
        "triangulated_contact",
    ]
    assert progress[0]["contact_motion_work"] != pytest.approx(0.0)
    assert progress[1]["contact_motion_work"] == pytest.approx(0.0, abs=1.0e-14)
    increment = next(
        event for event in step.execution_events if event.kind == "time_increment"
    )
    assert increment.metrics["contact_pair_count"] == 2.0
    snapshot = outer.snapshot()
    assert snapshot["contact_pair_identity"] == sliding_pair.scientific_identity
    assert snapshot["base_state"]["contact_pair_identity"] == (
        normal_pair.scientific_identity
    )


def test_moving_triangle_bvh_tracks_facet_crossing_without_spurious_work():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
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
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=200.0,
        name="sliding_pair",
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        projector=boundary_models.triangle_surface_bvh(
            body.surface
        ),
        contact_pair=pair,
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
    assert residual.summary()["contact_pair"]["name"] == "sliding_pair"
    snapshot = residual.snapshot()
    assert snapshot["contact_pair_identity"] == pair.scientific_identity
    projection_state = snapshot["projection_state"]
    assert projection_state["metadata"]["has_entity_ids"] is True
    assert projection_state["metadata"]["has_local_coordinates"] is True
    assert all(
        item["entity_id"] is not None
        and item["local_coordinates"] is not None
        for item in projection_state["records"]
    )
    increments = [
        event for event in step.execution_events if event.kind == "time_increment"
    ]
    assert increments[-1].metrics["contact_pair_count"] == 1.0
    assert increments[-1].metrics["contact_active_point_count"] > 0.0
    assert increments[-1].metrics["contact_maximum_penetration"] > 0.0
    assert increments[-1].metrics["contact_motion_work"] == pytest.approx(0.0)
    assert increments[-1].metrics["contact_motion_power"] == pytest.approx(0.0)
    assert "contact_active=" in increments[-1].message


def test_rigid_contact_pair_rejects_rank_local_pointwise_penalty():
    domain = _cube(MPI.COMM_SELF)
    slave = _left_region(domain)
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        )
    )
    law = boundary_models.frictionless_penalty_contact_law((100.0, 200.0))

    with pytest.raises(ValueError, match="scalar penalty"):
        boundary_models.RigidContactPair(slave, body, law)


def test_rigid_contact_pair_friction_enters_explicit_operator_and_commits_state():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    slave = _left_region(domain)
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        )
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=100.0,
        friction_coefficient=0.2,
        tangential_penalty=50.0,
        name="frictional_pair",
    )

    assert pair.summary()["friction"]["coefficient"] == pytest.approx(0.2)
    assert pair.scientific_identity != boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=100.0,
        name="frictionless_pair",
    ).scientific_identity
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=boundary_models.dolfinx_boundary_region_contact_trace(
            slave,
            function_space,
        ),
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    residual.initialize_accepted_state(time=0.0)
    displacement.x.array.reshape((-1, 3))[:, 2] = 0.01
    displacement.x.scatter_forward()

    vector = residual.assemble_vector()
    try:
        resultant = vector.array.reshape((-1, 3)).sum(axis=0)
    finally:
        vector.destroy()

    np.testing.assert_allclose(resultant, (5.0, 0.0, 0.5), atol=1.0e-13)
    assert residual.trial_evidence.tangential_potential_energy == pytest.approx(
        0.0025
    )
    assert residual.trial_evidence.sticking_point_count == 6
    assert residual.trial_evidence.sliding_point_count == 0
    residual.commit()
    np.testing.assert_allclose(
        residual.friction_state.accepted.elastic_slips,
        np.tile((0.0, 0.0, 0.01), (6, 1)),
    )
    snapshot = residual.snapshot()
    restored = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=boundary_models.dolfinx_boundary_region_contact_trace(
            slave,
            function_space,
        ),
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot

    displacement.x.array.reshape((-1, 3))[:, 2] = 0.1
    displacement.x.scatter_forward()
    vector = restored.assemble_vector()
    vector.destroy()
    assert restored.trial_evidence.sticking_point_count == 0
    assert restored.trial_evidence.sliding_point_count == 6
    assert restored.trial_evidence.tangential_potential_energy == pytest.approx(0.01)
    assert restored.trial_evidence.friction_dissipation == pytest.approx(0.08)
    restored.rollback()
    assert restored.friction_state.trial is None
    assert restored.friction_kinematics.trial is None


def test_rigid_contact_pair_requires_complete_friction_parameters():
    domain = _cube(MPI.COMM_SELF)
    slave = _left_region(domain)
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        )
    )

    with pytest.raises(ValueError, match="requires both"):
        boundary_models.rigid_contact_pair(
            slave,
            body,
            penalty=100.0,
            friction_coefficient=0.2,
        )


def test_explicit_friction_removes_common_prescribed_tool_translation():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.0, 0.5),
        ),
        end_time=1.0e-3,
    )
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        ),
        motion_schedule=schedule,
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=100.0,
        friction_coefficient=0.2,
        tangential_penalty=50.0,
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=boundary_models.dolfinx_boundary_region_contact_trace(
            slave,
            function_space,
        ),
        displacement=displacement,
        contact_pair=pair,
        maximum_stable_time_increment=1.0e-3,
    )
    residual.initialize_accepted_state(time=0.0)
    residual.update_time(5.0e-4)
    displacement.x.array.reshape((-1, 3))[:, 2] = 0.25
    displacement.x.scatter_forward()

    vector = residual.assemble_vector()
    try:
        resultant = vector.array.reshape((-1, 3)).sum(axis=0)
    finally:
        vector.destroy()

    np.testing.assert_allclose(resultant, (5.0, 0.0, 0.0), atol=1.0e-13)
    assert residual.trial_evidence.tangential_potential_energy == pytest.approx(0.0)
    assert residual.trial_evidence.friction_dissipation == pytest.approx(0.0)
    residual.commit()
    assert residual.work_state.path_work == pytest.approx(0.0, abs=1.0e-14)


def test_explicit_contact_rejects_projector_from_another_rigid_body():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
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
    slave = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        slave,
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
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=200.0,
        name="distributed_sliding_pair",
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
        adapter=adapter,
        displacement=displacement,
        projector=projector,
        contact_pair=pair,
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


def test_friction_state_checkpoint_is_rank_canonical_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Explicit friction checkpoint is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = _cube(comm, cells_x=2)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.scatter_forward()
    slave = _left_region(domain)
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        )
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=100.0,
        friction_coefficient=0.2,
        tangential_penalty=50.0,
        name="distributed_friction_pair",
    )

    def build_residual():
        return boundary_models.dolfinx_explicit_contact_residual(
            _bulk_residual(domain, function_space, (0.0, 0.0, 0.0)),
            adapter=boundary_models.dolfinx_boundary_region_contact_trace(
                slave,
                function_space,
            ),
            displacement=displacement,
            contact_pair=pair,
            maximum_stable_time_increment=1.0e-3,
        )

    residual = build_residual()
    residual.initialize_accepted_state(time=0.0)
    displacement.x.array.reshape((-1, 3))[:, 2] = 0.03
    displacement.x.scatter_forward()
    vector = residual.assemble_vector()
    vector.destroy()
    residual.commit()

    snapshot = residual.snapshot()
    copies = comm.allgather(snapshot)
    assert all(item == copies[0] for item in copies)
    assert residual.accepted_evidence.sliding_point_count == 6
    assert residual.accepted_evidence.friction_dissipation == pytest.approx(0.01)

    restored = build_residual()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert restored.friction_state.accepted.point_count == (
        restored.adapter.trace.point_count
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
