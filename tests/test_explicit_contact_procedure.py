from __future__ import annotations

from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import boundary_models, constitutive, fields, mesh, models, studies


def _contact_model(
    *,
    registered: bool = True,
    comm=MPI.COMM_SELF,
    cell_type=dolfinx_mesh.CellType.tetrahedron,
    cells_x: int = 1,
):
    domain = dolfinx_mesh.create_unit_cube(
        comm,
        cells_x,
        1,
        1,
        cell_type=cell_type,
    )
    model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(
            young=1.0e5,
            poisson=0.3,
            density=1000.0,
        )
    )
    facet_dimension = domain.topology.dim - 1
    facets = dolfinx_mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0),
    )
    tags = dolfinx_mesh.meshtags(
        domain,
        facet_dimension,
        facets,
        np.full(facets.size, 71, dtype=np.int32),
    )
    slave = mesh.tagged_boundary_region(
        domain,
        tags,
        tag=71,
        name="contact_slave",
    )
    body = boundary_models.rigid_body(
        boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        ),
        name="rigid_tool",
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        body,
        penalty=2.0e4,
        name="tool_contact",
    )
    if registered:
        model.add_boundary_model(pair)
    displacement.value.x.array.reshape((-1, 3))[:, 0] = 0.1
    displacement.value.x.scatter_forward()
    return model, displacement, material, pair


def test_finite_strain_explicit_discovers_registered_contact_pair():
    model, displacement, material, pair = _contact_model()

    step = model.step(
        target=displacement,
        material=material,
        steps=1,
        progress=False,
    )

    summary = step.summary()
    contributions = summary["stability"]["contributions"]
    assert [item["name"] for item in contributions] == [
        "body",
        f"contact:{pair.name}",
    ]
    assert summary["stability"]["composition"] == "additive_spectral_upper_bounds"
    assert summary["residual"]["contact_pair"]["name"] == pair.name
    assert summary["residual"]["contact_stability_estimate"]["point_count"] > 0
    assert summary["residual"]["stability_controller"] == "caller_declared"
    step.run()
    assert step.residual.accepted_evidence.active_point_count > 0
    assert step.history_records[-1]["contact_potential_energy"] >= 0.0


def test_hexahedral_finite_strain_explicit_uses_the_same_contact_procedure():
    model, displacement, material, pair = _contact_model(
        cell_type=dolfinx_mesh.CellType.hexahedron,
    )

    step = model.step(
        target=displacement,
        material=material,
        steps=1,
        progress=False,
    )
    step.run()

    assert step.residual.contact_pair is pair
    assert step.residual.adapter.summary()["facet_topology"] == "quadrilateral"
    assert step.residual.accepted_evidence.active_point_count == 4
    assert step.history_records[-1]["contact_potential_energy"] >= 0.0


def test_finite_strain_explicit_accepts_explicit_contact_pair_without_registration():
    model, displacement, material, pair = _contact_model(registered=False)

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        dt="auto",
        steps=1,
        progress=False,
    )

    assert step.residual.contact_pair is pair


def test_finite_strain_explicit_composes_multiple_contact_pairs_once():
    model, displacement, material, left_pair = _contact_model(registered=False)
    domain = displacement.space.mesh
    facet_dimension = domain.topology.dim - 1
    facets = dolfinx_mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 1.0),
    )
    tags = dolfinx_mesh.meshtags(
        domain,
        facet_dimension,
        facets,
        np.full(facets.size, 72, dtype=np.int32),
    )
    right_pair = boundary_models.rigid_contact_pair(
        mesh.tagged_boundary_region(
            domain,
            tags,
            tag=72,
            name="right_contact_slave",
        ),
        boundary_models.rigid_body(
            boundary_models.rigid_plane(
                point=(1.15, 0.0, 0.0),
                normal=(1.0, 0.0, 0.0),
            ),
            name="right_tool",
        ),
        penalty=3.0e4,
        name="right_tool_contact",
    )

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        contact_pairs=(left_pair, right_pair),
        steps=1,
        progress=False,
    )

    names = [
        item["name"] for item in step.summary()["stability"]["contributions"]
    ]
    assert names == [
        "body",
        f"contact:{left_pair.name}",
        f"contact:{right_pair.name}",
    ]
    step.run()
    terms = step.residual.contact_energy_evidence()
    assert [item["name"] for item in terms] == [left_pair.name, right_pair.name]
    assert all(item["contact_potential_energy"] > 0.0 for item in terms)


def test_finite_strain_explicit_carries_moving_friction_into_result_evidence():
    model, displacement, material, base_pair = _contact_model(registered=False)
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.02, 0.0),
            name="tool_slide",
        ),
        end_time=1.0e-5,
        name="tool_slide_schedule",
    )
    pair = boundary_models.rigid_contact_pair(
        base_pair.slave_boundary,
        boundary_models.rigid_body(
            base_pair.rigid_body.surface,
            motion_schedule=schedule,
            name="moving_rigid_tool",
        ),
        penalty=2.0e4,
        friction_coefficient=0.2,
        tangential_penalty=1.0e4,
        name="moving_frictional_contact",
    )

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        dt=5.0e-6,
        steps=2,
        progress=False,
    )
    result = step.solve_result()

    assert step.residual.accepted_evidence.sliding_point_count > 0
    assert result.histories["contact_friction_dissipation"].latest > 0.0
    assert "contact_motion_work" in result.histories
    assert result.metadata["step"]["residual"]["contact_pair"]["name"] == pair.name


def test_mass_damping_preserves_moving_contact_lifecycle_and_evidence():
    model, displacement, material, base_pair = _contact_model(registered=False)
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.02, 0.0),
            name="damped_tool_slide",
        ),
        end_time=1.0e-5,
        name="damped_tool_schedule",
    )
    pair = boundary_models.rigid_contact_pair(
        base_pair.slave_boundary,
        boundary_models.rigid_body(
            base_pair.rigid_body.surface,
            motion_schedule=schedule,
            name="damped_moving_tool",
        ),
        penalty=2.0e4,
        friction_coefficient=0.2,
        tangential_penalty=1.0e4,
        name="damped_moving_contact",
    )

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        dt=5.0e-6,
        steps=2,
        mass_damping=0.1,
        progress=False,
    )
    result = step.solve_result()

    assert result.histories["contact_friction_dissipation"].latest > 0.0
    assert "contact_motion_work" in result.histories
    assert "numerical_damping_dissipation" in result.histories
    assert step.residual.contact_progress_evidence()[0]["sliding_point_count"] > 0


def test_moving_contact_preload_transfer_initializes_work_state_before_equilibrium():
    model, displacement, material, base_pair = _contact_model(registered=False)
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.02, 0.0),
            name="preloaded_tool_slide",
        ),
        end_time=1.0e-5,
        name="preloaded_tool_schedule",
    )
    pair = boundary_models.rigid_contact_pair(
        base_pair.slave_boundary,
        boundary_models.rigid_body(
            base_pair.rigid_body.surface,
            motion_schedule=schedule,
            name="preloaded_moving_tool",
        ),
        penalty=2.0e4,
        friction_coefficient=0.2,
        tangential_penalty=1.0e4,
        name="preloaded_moving_contact",
    )
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        dt=5.0e-6,
        steps=2,
        progress=False,
    )

    report = step.initialize_from_preload(
        displacement.value,
        mode="release",
    )

    assert report.mode == "release"
    assert step.residual.accepted_evidence is not None
    assert len(step.residual.work_state.accepted) == 1
    step.run()
    assert len(step.residual.work_state.accepted) == 3


def test_failed_moving_contact_preload_transfer_is_atomic():
    model, displacement, material, base_pair = _contact_model(registered=False)
    pair = boundary_models.rigid_contact_pair(
        base_pair.slave_boundary,
        boundary_models.rigid_body(
            base_pair.rigid_body.surface,
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(
                    translation=(0.0, 0.02, 0.0),
                ),
                end_time=1.0e-5,
            ),
            name="atomic_moving_tool",
        ),
        penalty=2.0e4,
        friction_coefficient=0.2,
        tangential_penalty=1.0e4,
        name="atomic_moving_contact",
    )
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        dt=5.0e-6,
        steps=1,
        progress=False,
    )
    before = step.state.snapshot()
    incompatible = fem.Function(displacement.value.function_space)
    incompatible.x.array[:] = displacement.value.x.array
    incompatible.x.array.reshape((-1, 3))[:, 0] += 0.05
    incompatible.x.scatter_forward()

    import pytest

    with pytest.raises(RuntimeError, match="not in equilibrium"):
        step.initialize_from_preload(
            incompatible,
            mode="equilibrium",
            force_tolerance=0.0,
        )

    np.testing.assert_allclose(
        step.state.u.value.x.array,
        before["fields"]["u"],
        rtol=0.0,
        atol=0.0,
    )
    assert step.residual.accepted_evidence is None
    assert step.residual.work_state.accepted == []


def test_standard_contact_procedure_checkpoint_matches_uninterrupted_run(tmp_path):
    def build_step():
        model, displacement, material, pair = _contact_model(registered=False)
        return model.finite_strain_explicit_dynamics_step(
            target=displacement,
            material=material,
            contact_pairs=(pair,),
            dt=5.0e-6,
            steps=2,
            progress=False,
        )

    continuous = build_step()
    continuous.run()

    partial = build_step()
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(tmp_path / "standard-contact")

    restarted = build_step()
    restarted.load_checkpoint(checkpoint)
    restarted.run()

    np.testing.assert_allclose(
        restarted.state.u.value.x.array,
        continuous.state.u.value.x.array,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert restarted.history_records == continuous.history_records
    assert restarted.residual.snapshot() == continuous.residual.snapshot()


def test_standard_contact_procedure_is_rank_canonical_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        import pytest

        pytest.skip("Standard contact Procedure MPI route is reviewed on two ranks.")
    model, displacement, material, pair = _contact_model(
        registered=False,
        comm=MPI.COMM_WORLD,
    )
    step = model.step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        steps=1,
        mass_damping=0.1,
        progress=False,
    )

    result = step.solve_result()
    snapshots = MPI.COMM_WORLD.allgather(step.residual.snapshot())

    assert all(item == snapshots[0] for item in snapshots)
    assert result.histories["contact_potential_energy"].latest >= 0.0


def test_hexahedral_contact_procedure_is_rank_canonical_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        import pytest

        pytest.skip("Hexahedral contact Procedure MPI route is reviewed on two ranks.")
    model, displacement, material, pair = _contact_model(
        registered=False,
        comm=MPI.COMM_WORLD,
        cell_type=dolfinx_mesh.CellType.hexahedron,
        cells_x=2,
    )
    step = model.step(
        target=displacement,
        material=material,
        contact_pairs=(pair,),
        steps=1,
        mass_damping=0.1,
        progress=False,
    )

    result = step.solve_result()
    snapshots = MPI.COMM_WORLD.allgather(step.residual.snapshot())

    assert all(item == snapshots[0] for item in snapshots)
    assert step.residual.base.adapter.summary()["facet_topology"] == "quadrilateral"
    assert result.histories["contact_potential_energy"].latest >= 0.0


def test_finite_strain_explicit_rejects_unknown_projection_option_pair():
    model, displacement, material, pair = _contact_model(registered=False)

    try:
        model.finite_strain_explicit_dynamics_step(
            target=displacement,
            material=material,
            contact_pairs=(pair,),
            contact_projection_options={"not_a_pair": {}},
            steps=1,
            progress=False,
        )
    except ValueError as error:
        assert "unknown pair names" in str(error)
    else:
        raise AssertionError("Unknown contact projection option pair was accepted.")
