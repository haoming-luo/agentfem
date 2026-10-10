# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private composition gate; ordinary finite-Hex contact remains unpromoted."""

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import boundary_models, fields, fracture, models, problems, studies
from agentfem import mesh as mesh_api
from agentfem.mechanics._finite_hex_energy import FiniteHexEnergyMonitor
from test_finite_hex_material_envelope_step import BoundedNeoHookean
from test_finite_hex_step import problem


def prepare_contact(*, dt=1e-4):
    prototype, _, policy = problem(force=0)
    domain = prototype.mesh
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(BoundedNeoHookean())
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    body = model.step(target=u, element_policy=policy, dt=dt,
                      steps=round(0.02 / dt), progress=False,
                      maximum_negative_growth_per_increment=0.1)
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 41, dtype=np.int32))
    region = mesh_api.tagged_boundary_region(domain, tags, tag=41, name="tool_slave")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(region, u.value.function_space)
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(translation=(-0.01, 0, 0)), end_time=0.02,
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        body.residual, adapter=adapter, displacement=u.value,
        projector=boundary_models.rigid_plane(point=(1, 0, 0), normal=(-1, 0, 0)),
        penalty=200, motion_schedule=schedule,
        lumped_mass=body.residual.internal.mass_diagonal,
        noncontact_unsafed_stability_limit=2 / np.sqrt(body.residual.bound),
    )
    ledger = fracture.DynamicEnergyLedger(
        energy=FiniteHexEnergyMonitor(body.residual), state=body.state,
        mass=body.residual.internal.mass_diagonal, residual=residual,
        prescribed=body.prescribed,
    )
    step = problems.explicit_dynamics(
        state=body.state, integrator=body.integrator, residual=residual,
        dt=dt, steps=body.steps, prescribed=body.prescribed, constraints=body.constraints,
        update_load=body.update_load, progress=False, history_monitor=ledger,
        stability=residual.combined_stability_estimate,
    )
    return step


def test_private_finite_material_moving_contact_restart_and_energy(tmp_path):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Private composition is initially serial only.")
    full = prepare_contact()
    full.run()
    partial = prepare_contact()
    partial.run(until_step=73)
    checkpoint = partial.save_checkpoint(tmp_path / "finite-contact")
    resumed = prepare_contact()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.u.value.x.array, full.state.u.value.x.array)
    np.testing.assert_array_equal(resumed.state.v.value.x.array, full.state.v.value.x.array)
    assert resumed.history_records == pytest.approx(full.history_records)
    row = full.history_records[-1]
    assert row["contact_motion_work"] > 0
    assert row["bulk_stored_energy"] > 0
    assert row["contact_potential_energy"] > 0
    assert row["relative_energy_balance_error"] < 1e-3


def test_private_finite_contact_time_refinement_reduces_balance_error():
    errors, displacements = [], []
    for dt in (2e-4, 1e-4, 5e-5):
        step = prepare_contact(dt=dt)
        step.run()
        errors.append(step.history_records[-1]["relative_energy_balance_error"])
        displacements.append(step.state.u.value.x.array.copy())
    assert errors[-1] < 4e-7
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5
    assert np.linalg.norm(displacements[2] - displacements[1]) < (
        np.linalg.norm(displacements[1] - displacements[0]) / 3.5
    )


def test_private_finite_contact_accepted_sampling_and_commit_failure(monkeypatch):
    step = prepare_contact()
    step.run(until_step=10)
    residual = step.residual
    before = residual.snapshot()
    fields_before = step.state.snapshot()
    original_update = residual.base.material.update_array_batch

    def reject_update(request):
        raise AssertionError("Accepted force reads must not reintegrate material.")

    monkeypatch.setattr(residual.base.material, "update_array_batch", reject_update)
    vector = residual.assemble_accepted_vector()
    vector.destroy()
    assert residual.snapshot() == before
    monkeypatch.setattr(residual.base.material, "update_array_batch", original_update)
    commit = residual.commit

    def rejected_commit():
        commit()
        raise RuntimeError("injected failure after bulk and contact acceptance")

    monkeypatch.setattr(residual, "commit", rejected_commit)
    with pytest.raises(RuntimeError, match="injected failure"):
        step.run(until_step=11)
    assert step.completed_steps == 10
    assert residual.snapshot() == before
    for name, values in fields_before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)
    monkeypatch.setattr(residual, "commit", commit)
    step.run(until_step=11)
    assert step.completed_steps == 11
