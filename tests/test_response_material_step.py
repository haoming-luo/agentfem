# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh
from agentfem import constitutive, elements, fields, models, studies, loads
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


def problem():
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


def test_no_auto_dt_or_silent_load_drop():
    with pytest.raises(ValueError, match="numerical dt"):
        make_step(dt="auto")
    model, u, policy = problem()
    model.load(loads.body_force((1, 0, 0), domain=u.value.function_space.mesh, target=u))
    with pytest.raises(NotImplementedError, match="does not yet admit"):
        model.step(target=u, element_policy=policy, dt=1e-4, steps=4,
                   time_increment_ceiling=1e-3, stress_free_initial_state=True)
    assert not model.steps


def contact_step(**options):
    from agentfem import boundary_models, mesh as af_mesh
    model, u, policy = problem()
    domain = u.value.function_space.mesh
    for side in (0, 1):
        region = af_mesh.boundary(domain, lambda x, side=side: np.isclose(x[0], side),
                                  name=f"side_{side}")
        direction = 1 if side == 0 else -1
        tool = boundary_models.rigid_body(
            boundary_models.rigid_plane(point=(side, 0, 0), normal=(direction, 0, 0)),
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
def test_multiple_public_tools_work_and_restart(tmp_path, viscous, factor):
    options = {"bulk_viscosity": elements.BulkViscosityPolicy(1., 1., 1.)} if viscous else {}
    options["mass_scaling_factor"] = factor
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
