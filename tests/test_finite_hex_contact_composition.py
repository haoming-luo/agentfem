# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private composition gate; ordinary finite-Hex contact remains unpromoted."""

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import boundary_models, constitutive, fields, fracture, models, problems, studies
from agentfem import mesh as mesh_api
from agentfem.mechanics._finite_hex_energy import FiniteHexEnergyMonitor
from test_finite_hex_material_envelope_step import BoundedNeoHookean
from test_finite_hex_step import problem


def prepare_contact(*, dt=1e-4, duration=0.02, single_cell=False, public=False,
                    rotation=0.0, automatic=False, material=None, frequency_ceiling=None,
                    comm=MPI.COMM_SELF, include_contact=True):
    prototype, _, policy = problem(comm, force=0)
    domain = prototype.mesh
    if single_cell:
        partitioner = None
        if comm.size > 1:
            from dolfinx import graph

            def partitioner(comm, parts, types, cells):
                # Exercise an empty rank without partitioning an edgeless graph.
                count = (cells.num_nodes if hasattr(cells, "num_nodes")
                         else sum(array.size // 8 for array in cells))
                result = graph.adjacencylist(np.zeros((count, 1), dtype=np.int32))
                return getattr(result, "_cpp_object", result)

        domain = mesh.create_unit_cube(comm, 1, 1, 1, cell_type=mesh.CellType.hexahedron,
                                      partitioner=partitioner)
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(material or BoundedNeoHookean())
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    options = dict(target=u, element_policy=policy, dt=dt,
                   steps=round(duration / dt), progress=False,
                   maximum_negative_growth_per_increment=0.1)
    if automatic:
        options["dt"] = "auto"
    if frequency_ceiling is not None:
        options["omega_squared_bound"] = frequency_ceiling
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 41, dtype=np.int32))
    region = mesh_api.tagged_boundary_region(domain, tags, tag=41, name="tool_slave")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(region, u.value.function_space)
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(translation=(-0.01, 0, 0), rotation=rotation), end_time=duration,
    )
    surface = boundary_models.rigid_plane(point=(1, 0, 0), normal=(-1, 0, 0))
    if public:
        pair = boundary_models.rigid_contact_pair(
            region, boundary_models.rigid_body(surface, motion_schedule=schedule),
            penalty=200, name="moving_plane",
        )
        if include_contact:
            model.add_boundary_model(pair)
        return model.step(**options)
    body = model.step(**options)
    residual = boundary_models.dolfinx_explicit_contact_residual(
        body.residual, adapter=adapter, displacement=u.value,
        projector=surface,
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


def _independent_uniform_compression(duration, intervals):
    """Scalar balance: rho*V/2*q'' = -P11(1+q)-k*positive(q-s)."""
    state = np.zeros(3)  # endpoint displacement, velocity, tool work
    h = duration / intervals

    def rhs(t, y):
        q, velocity, _ = y
        stretch = 1 + q
        piola = 30 * (stretch - 1 / stretch) + 40 * np.log(stretch) / stretch
        force = 200 * max(q + 0.01 * t / duration, 0)
        # Half of the unit-volume rho=2 mass belongs to the loaded end.
        return np.asarray((velocity, -piola - force, force * 0.01 / duration))

    for index in range(intervals):
        t = index * h
        a = rhs(t, state)
        b = rhs(t + h / 2, state + h * a / 2)
        c = rhs(t + h / 2, state + h * b / 2)
        d = rhs(t + h, state + h * c)
        state += h * (a + 2 * b + 2 * c + d) / 6
    return state


def test_private_contact_matches_independent_finite_compression_ode():
    duration = 0.1
    oracle = _independent_uniform_compression(duration, 4000)
    refined_oracle = _independent_uniform_compression(duration, 8000)
    np.testing.assert_allclose(oracle, refined_oracle, rtol=1e-10, atol=1e-13)
    errors = []
    for dt in (0.001, 0.0005, 0.00025):
        step = prepare_contact(dt=dt, duration=duration, single_cell=True)
        step.run()
        right = step.state.u.value.function_space.tabulate_dof_coordinates()[:, 0] > 0.5
        q = step.state.u.value.x.array.reshape(-1, 3)[right, 0]
        v = step.state.v.value.x.array.reshape(-1, 3)[right, 0]
        np.testing.assert_allclose(q, q[0], rtol=0, atol=1e-13)
        result = np.asarray((q[0], v[0], step.history_records[-1]["contact_motion_work"]))
        errors.append(np.linalg.norm((result - oracle) / oracle))
    assert errors[-1] < 2e-5
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5


def test_ordinary_finite_contact_matches_private_composition_and_restarts(tmp_path):
    reference = prepare_contact()
    reference.run()
    step = prepare_contact(public=True)
    result = step.solve_result(field_variables=("S", "F", "SENER"))
    np.testing.assert_allclose(step.state.u.value.x.array, reference.state.u.value.x.array,
                               rtol=3e-15, atol=1e-18)
    for actual, expected in zip(step.history_records, reference.history_records, strict=True):
        assert actual == pytest.approx(expected, rel=1e-11, abs=1e-14)
    assert "S" in result.fields
    assert step.summary()["stability"]["composition"] == "additive_spectral_upper_bounds"
    assert step.checkpoint_capabilities().rank_count_portability == "unsupported"
    partial = prepare_contact(public=True)
    partial.run(until_step=73)
    checkpoint = partial.save_checkpoint(tmp_path / "ordinary-contact")
    resumed = prepare_contact(public=True)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.u.value.x.array, step.state.u.value.x.array)
    assert resumed.history_records == pytest.approx(step.history_records)


def test_ordinary_contact_auto_increment_composes_body_and_contact_once():
    step = prepare_contact(public=True, automatic=True)
    contributions = step.stability.contributions
    assert len(contributions) == 2
    assert contributions[1].name == "contact:moving_plane"
    bound = sum(item.spectral_radius_upper_bound for item in contributions)
    assert step.dt == pytest.approx(0.8 * 2 / np.sqrt(bound))
    assert step.dt < step.material_residual.stability.selected


def test_ordinary_finite_contact_refuses_unreviewed_rotations():
    with pytest.raises(NotImplementedError, match="rotating contact"):
        prepare_contact(public=True, rotation=(0, 0, 0.1))


def test_ordinary_finite_contact_refuses_curved_tools(monkeypatch):
    monkeypatch.setattr(boundary_models, "rigid_plane", lambda **kwargs:
                        boundary_models.rigid_sphere(center=(1, 0, 0), radius=1))
    with pytest.raises(NotImplementedError, match="frictionless rigid plane"):
        prepare_contact(public=True)


def test_ordinary_finite_j2_contact_has_accepted_plastic_work_and_restart(tmp_path):
    def build():
        return prepare_contact(
            public=True, single_cell=True, duration=0.1, frequency_ceiling=1e7,
            material=constitutive.finite_strain_j2_logarithmic(
                young=100, poisson=0.3, yield_stress=0.02,
                hardening_modulus=5, density=2,
            ),
        )
    continuous = build()
    result = continuous.solve_result(field_variables=("PEEQ", "PDENER", "S"))
    assert np.max(result.fields["PEEQ"].field.x.array) > 0
    row = continuous.history_records[-1]
    assert row["material_dissipation"] > 0
    assert row["contact_motion_work"] > 0
    assert row["relative_energy_balance_error"] < 1e-3
    partial = build()
    partial.run(until_step=621)
    path = partial.save_checkpoint(tmp_path / "plastic-contact")
    restored = build()
    restored.load_checkpoint(path)
    restored.run()
    np.testing.assert_array_equal(restored.state.u.value.x.array, continuous.state.u.value.x.array)
    assert restored.material_residual.snapshot() == continuous.material_residual.snapshot()
