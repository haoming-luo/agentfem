# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh
from agentfem import constitutive, elements, fields, models, studies, loads
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


def problem(domain=None):
    if domain is None:
        domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.hexahedron)
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    schema = MaterialStateSchema("clock", (MaterialStateVariable("clock"),))
    def update(request):
        n = request.point_count
        return constitutive.ExplicitMaterialBatchOutput(np.zeros((n, 3, 3)),
            request.state_old + request.time_increment, schema, np.zeros(n))
    model.material(constitutive.explicit_user_material(state_schema=schema, update=update,
        identity={"law":"zero-stress-clock", "revision":1}, density=1.))
    policy = elements.uniform_strain_hex8(hourglass_modulus=1., hourglass_scale=.1,
                                          kinematics="finite_strain")
    return model, u, policy


def make_step(**options):
    model, u, policy = problem()
    config = dict(dt=1e-4, steps=4, time_increment_ceiling=1e-3,
                  stress_free_initial_state=True, progress=False)
    config.update(options)
    step = model.step(target=u, element_policy=policy, **config)
    step.state.v.value.x.array[::3] = .01
    return step


def test_public_step_result_and_restart(tmp_path):
    complete = make_step()
    result = complete.solve_result()
    assert result is not None
    np.testing.assert_allclose(complete.state.u.value.x.array[::3], 4e-6, atol=1e-15)
    assert complete.summary()["energy_scope"] == "accepted_internal_work_not_stored_energy"
    partial = make_step()
    partial.run(until_step=2)
    checkpoint = partial.save_checkpoint(tmp_path/'restart')
    resumed = make_step()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.u.value.x.array, complete.state.u.value.x.array)
    assert resumed.residual.snapshot() == complete.residual.snapshot()


def test_accepted_material_export_is_detached_and_read_only():
    step = make_step()
    step.run(until_step=2)
    record = step.accepted_material_fields()
    state = record['state'].copy()
    assert record['time'] == pytest.approx(2e-4)
    for key in ('state', 'cauchy_stress', 'deformation_gradient', 'cell_keys'):
        assert not record[key].flags.writeable
    step.run()
    np.testing.assert_array_equal(record['state'], state)
    assert not np.array_equal(step.accepted_material_fields()['state'], state)


def test_no_auto_dt_or_silent_load_drop():
    with pytest.raises(ValueError, match="numerical dt"):
        make_step(dt="auto")
    model, u, policy = problem()
    model.load(loads.body_force((1, 0, 0), domain=u.value.function_space.mesh, target=u))
    with pytest.raises(NotImplementedError, match="does not yet admit"):
        model.step(target=u, element_policy=policy, dt=1e-4, steps=4,
                   time_increment_ceiling=1e-3, stress_free_initial_state=True)
    assert not model.steps


@pytest.mark.parametrize('name,error,pattern', (
    ('output', NotImplementedError, 'does not yet lower declarative'),
    ('history', TypeError, 'Unsupported Step option')))
def test_no_silent_declarative_observer_drop(name, error, pattern):
    with pytest.raises(error, match=pattern):
        make_step(**{name: {}})


def contact_step(bond=False, cylinder=False, **options):
    from agentfem import boundary_models, mesh as af_mesh
    domain = None
    if bond:
        from test_nonmatching_global import _blocks
        domain = _blocks(1, 2, cell_type="hexahedron")
    model, u, policy = problem(domain)
    domain = u.value.function_space.mesh
    if bond:
        from agentfem import fracture, interfaces
        from test_quadrilateral_global import _trace
        a, na = _trace(u.value.function_space, False)
        b, nb = _trace(u.value.function_space, True)
        pairing = interfaces.initial_nodal_bond(a, b,
            eligible_nodes=np.ones(len(a.vertices), dtype=bool), tolerance=1e-10)
        options["cohesive_force"] = fracture.nonmatching_cohesive_force(pairing, u,
            interfaces.elastic_cohesive(normal_stiffness=100, tangential_stiffness=100),
            negative_dofs=na, positive_dofs=nb)
        options.setdefault("bond_contact_policy", "independent")
    for side in (0, 1):
        region = af_mesh.boundary(domain, lambda x, side=side: np.isclose(x[0], side),
                                  name=f"side_{side}")
        direction = 1 if side == 0 else -1
        surface = boundary_models.rigid_plane(point=(side, 0, 0), normal=(direction, 0, 0))
        if cylinder:
            surface = boundary_models.rigid_cylinder(axis_point=(side - direction * .8, 0, .5),
                axis_direction=(0, 1, 0), radius=1., axial_interval=(-1, 2))
        tool = boundary_models.rigid_body(surface,
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(translation=(direction * 1e-4, 0, 0)),
                end_time=.001), name=f"tool_{side}")
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            region, tool, penalty=100, name=f"contact_{side}"))
    return model.step(target=u, element_policy=policy, dt=1e-4, steps=4,
                      time_increment_ceiling=1e-3, stress_free_initial_state=True,
                      progress=False, **options)


@pytest.mark.parametrize("viscous", [False, True])
@pytest.mark.parametrize("factor", [None, 2.])
@pytest.mark.parametrize("bond", [False, True])
def test_multiple_public_tools_work_and_restart(tmp_path, viscous, factor, bond):
    options = {"bulk_viscosity": elements.BulkViscosityPolicy(1., 1., 1.)} if viscous else {}
    options["mass_scaling_factor"] = factor
    options["bond"] = bond
    complete = contact_step(**options)
    complete.run()
    evidence = complete.residual.contact_energy_evidence()
    assert len(evidence) == 2
    assert all(item["contact_motion_work"] != 0 for item in evidence)
    assert complete.summary()["material"]["name"] == "user_explicit_material"
    partial = contact_step(**options)
    partial.run(until_step=2)
    checkpoint = partial.save_checkpoint(tmp_path / "tools")
    resumed = contact_step(**options)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.u.value.x.array, complete.state.u.value.x.array)
    assert resumed.residual.snapshot() == complete.residual.snapshot()
    if factor is not None:
        assert complete.summary()["numerical_inertia"]["added_mass_ratio"] == pytest.approx(factor - 1)


def test_viscosity_requires_physical_density():
    with pytest.raises(ValueError, match="physical reference density"):
        make_step(bulk_viscosity=elements.BulkViscosityPolicy(2., 1., 1.))


def test_body_damping_ceiling_is_distinct_from_global_increment_cap():
    config = dict(time_increment_ceiling=1e-4,
                  bulk_viscosity=elements.BulkViscosityPolicy(1.,1.,1.))
    rejected = make_step(**config)
    with pytest.raises(ValueError, match='damping-adjusted'):
        rejected.run()
    assert rejected.completed_steps == 0
    step = make_step(**config, operator_increment_ceilings={'body':1e-3})
    step.run()
    assert step.completed_steps == 4 and step.dt == 1e-4
    assert step.summary()['operator_increment_ceilings']['body'] == 1e-3


@pytest.mark.parametrize('limits', ({'unknown':1}, {'body':True}, {'body':float('nan')},
                                   {'body':1e-9}, {'rigid_contact':1}, []))
def test_bad_operator_increment_ceilings_fail_before_execution(limits):
    with pytest.raises(ValueError, match='increment ceiling'):
        make_step(operator_increment_ceilings=limits)


def test_overlap_is_not_silent():
    with pytest.raises(ValueError, match="traces overlap"):
        contact_step(bond=True, bond_contact_policy="disjoint")
