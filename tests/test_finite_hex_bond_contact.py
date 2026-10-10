# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Three-operator composition through the ordinary finite Hex Step."""

import numpy as np
import pytest
from dolfinx import mesh

from agentfem import boundary_models, constitutive, elements, fields, fracture, interfaces, models, studies
from agentfem import mesh as mesh_api
from test_nonmatching_global import _blocks
from test_quadrilateral_global import _trace
from test_finite_hex_material_envelope_step import BoundedNeoHookean


def prepare(*, dt=1e-4, stiffness=100, tool_z=1.0, plastic=False, material=None):
    domain = _blocks(1, 2, cell_type="hexahedron")
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(material or (constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=0.02, hardening_modulus=5, density=2,
    ) if plastic else BoundedNeoHookean()))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(0, 1))
    model.fix(u, on=lambda x: np.isclose(x[2], -1), components=2)
    a, na = _trace(u.value.function_space, False)
    b, nb = _trace(u.value.function_space, True)
    force = fracture.nonmatching_cohesive_force(
        interfaces.pair_reference_traces(a, b, tolerance=1e-10), u,
        interfaces.elastic_cohesive(normal_stiffness=stiffness, tangential_stiffness=stiffness),
        negative_dofs=na, positive_dofs=nb,
    )
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], tool_z))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 1, dtype=np.int32))
    region = mesh_api.tagged_boundary_region(domain, tags, tag=1, name="tool")
    motion = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(translation=(0, 0, -0.01)), end_time=0.1,
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        region, boundary_models.rigid_body(
            boundary_models.rigid_plane(point=(0, 0, tool_z), normal=(0, 0, -1)),
            motion_schedule=motion,
        ), penalty=200, name="tool",
    ))
    return model.step(
        target=u, cohesive_force=force,
        element_policy=elements.uniform_strain_hex8(
            kinematics="finite_strain", hourglass_modulus=40, hourglass_scale=0.1,
        ), dt=dt, steps=round(0.1 / dt), progress=False,
        maximum_negative_growth_per_increment=0.1,
        **({"omega_squared_bound": 1e7} if plastic or material is not None else {}),
    )


def test_three_operator_work_refinement_and_stability():
    responses, errors = [], []
    for dt in (4e-4, 2e-4, 1e-4):
        step = prepare(dt=dt)
        step.run()
        row = step.history_records[-1]
        assert row["interface_stored_energy"] > 0
        assert row["bulk_stored_energy"] > 0
        assert row["contact_motion_work"] > 0
        assert row["contact_potential_energy"] > 0
        errors.append(row["relative_energy_balance_error"])
        responses.append(step.state.u.value.x.array.copy())
        contributions = step.material_residual.stability.contributions
        assert len(contributions) == 2  # bulk/hourglass and bonded interface
        assert step.residual.maximum_stable_time_increment < step.material_residual.stability.selected
    assert errors[2] < errors[1] / 3
    assert errors[1] < errors[0] / 3
    assert errors[2] < 1e-4
    assert np.linalg.norm(responses[2] - responses[1]) < np.linalg.norm(responses[1] - responses[0]) / 3


def test_three_operator_restart_and_post_commit_rollback(tmp_path, monkeypatch):
    full = prepare()
    full.run()
    partial = prepare()
    partial.run(until_step=371)
    path = partial.save_checkpoint(tmp_path / "bond-contact")
    resumed = prepare()
    resumed.load_checkpoint(path)
    state_before, residual_before = resumed.state.snapshot(), resumed.residual.snapshot()
    original = resumed.residual.commit

    def fail():
        original()
        raise RuntimeError("post three-operator commit")

    monkeypatch.setattr(resumed.residual, "commit", fail)
    with pytest.raises(RuntimeError, match="three-operator"):
        resumed.run()
    assert resumed.completed_steps == 371
    assert resumed.residual.snapshot() == residual_before
    for key, value in state_before["fields"].items():
        np.testing.assert_array_equal(resumed.state.snapshot()["fields"][key], value)
    monkeypatch.setattr(resumed.residual, "commit", original)
    resumed.run()
    for key in ("u", "v", "a"):
        np.testing.assert_array_equal(getattr(resumed.state, key).value.x.array, getattr(full.state, key).value.x.array)
    assert resumed.residual.snapshot() == full.residual.snapshot()
    assert resumed.history_records == full.history_records
    with pytest.raises(ValueError, match="identity|differ"):
        prepare(stiffness=101).load_checkpoint(path)


def test_combined_overlap_requires_explicit_future_policy():
    with pytest.raises(NotImplementedError, match="disjoint trace nodes"):
        prepare(tool_z=0)


def test_plastic_bond_contact_reports_separate_energy_and_state(tmp_path):
    full = prepare(plastic=True)
    full.run()
    row = full.history_records[-1]
    assert row["material_dissipation"] > 0
    assert row["interface_stored_energy"] > 0
    assert row["relative_energy_balance_error"] < 1e-4
    partial = prepare(plastic=True)
    partial.run(until_step=371)
    path = partial.save_checkpoint(tmp_path / "plastic-bond-contact")
    restored = prepare(plastic=True)
    restored.load_checkpoint(path)
    restored.run()
    assert restored.residual.snapshot() == full.residual.snapshot()
    np.testing.assert_array_equal(restored.state.u.value.x.array, full.state.u.value.x.array)


def reference_response(increments):
    """Independent four-mass axial system; no AgentFEM force assembly."""
    dt = 0.1 / increments
    response = np.zeros(9)  # four displacements, four velocities, tool work

    def rhs(t, y):
        q1, q2, q3, q4 = y[:4]
        stretch = np.array([1 + q1, 1 + 2 * (q3 - q2), 1 + 2 * (q4 - q3)])
        p1, p2, p3 = 30 * (stretch - 1 / stretch) + 40 * np.log(stretch) / stretch
        bond = 100 * (q2 - q1)
        contact = 200 * max(q4 + 0.1 * t, 0)
        internal = np.array([p1 - bond, bond - p2, p2 - p3, p3 + contact])
        return np.r_[y[4:8], -internal / np.array([1, 0.5, 1, 0.5]), 0.1 * contact]

    for index in range(increments):
        t = index * dt
        k1 = rhs(t, response)
        k2 = rhs(t + dt / 2, response + dt * k1 / 2)
        k3 = rhs(t + dt / 2, response + dt * k2 / 2)
        k4 = rhs(t + dt, response + dt * k3)
        response += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return response


def test_combined_response_matches_independent_lumped_reference():
    reference = reference_response(4000)
    np.testing.assert_allclose(reference, reference_response(8000), atol=1e-13, rtol=1e-10)
    errors = []
    for dt in (4e-4, 2e-4, 1e-4):
        step = prepare(dt=dt)
        step.run()
        residual = step.material_residual
        x = step.state.u.value.function_space.tabulate_dof_coordinates()
        negative = residual.cohesive.negative_dofs
        positive = residual.cohesive.positive_dofs
        layers = (negative, positive, np.flatnonzero(np.isclose(x[:, 2], 0.5)),
                  np.flatnonzero(np.isclose(x[:, 2], 1)))
        output = []
        for field in (step.state.u, step.state.v):
            z = field.value.x.array.reshape(-1, 3)[:, 2]
            output.extend(float(z[nodes].mean()) for nodes in layers)
            for nodes in layers:
                assert np.ptp(z[nodes]) < 1e-12
        output.append(step.history_records[-1]["contact_motion_work"])
        errors.append(np.linalg.norm(np.asarray(output) - reference) / np.linalg.norm(reference))
    assert errors[2] < errors[1] / 3.5
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < 1e-5
