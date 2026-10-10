# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent rigid-translation work oracle for the assembled explicit path."""

import numpy as np
import pytest

from agentfem import (
    amplitudes,
    constitutive,
    elements,
    fields,
    loads,
    mesh,
    models,
    studies,
)


def _accelerating_block(*, prescribed=False, history_every=1):
    domain = mesh.cuboid((0, 0, 0), (1, 1, 1), (1, 1, 1), cell_type="hexahedron")
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    if prescribed:
        motion = amplitudes.Amplitude(
            "quadratic_motion",
            lambda t: 0.5 * t * t,
            metadata={"coefficient": 0.5, "power": 2},
        )
        model.fix(
            u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=0, value=motion
        )
    else:
        model.load(loads.body_force((2, 0, 0), domain=domain, target=u))
    return model.step(
        target=u,
        element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=50, hourglass_scale=0.1
        ),
        dt=1e-3,
        steps=40,
        history_every=history_every,
        progress=False,
    )


@pytest.mark.parametrize("prescribed", [False, True])
@pytest.mark.parametrize("cadence", [1, 7])
def test_rigid_acceleration_work_matches_analytic_energy(prescribed, cadence):
    step = _accelerating_block(prescribed=prescribed, history_every=cadence)
    step.run()
    row = step.history_records[-1]
    stop = step.dt * step.completed_steps
    expected = stop**2  # half mass(2) times velocity(t)^2
    np.testing.assert_allclose(
        step.state.u.value.x.array[::3], 0.5 * stop**2, atol=1e-14
    )
    assert row["kinetic_energy"] == pytest.approx(expected, rel=1e-10)
    assert row["external_work"] == pytest.approx(expected, rel=1e-10)
    key = "prescribed_motion_work" if prescribed else "natural_load_work"
    assert row[key] == pytest.approx(expected, rel=1e-10)
    assert row["relative_energy_balance_error"] < 1e-9
    assert row["hourglass_energy"] < 1e-25


@pytest.mark.parametrize("prescribed", [False, True])
def test_external_work_restart_preserves_unsaved_increments(tmp_path, prescribed):
    reference = _accelerating_block(prescribed=prescribed, history_every=7)
    reference.run()
    partial = _accelerating_block(prescribed=prescribed, history_every=7)
    partial.run(until_step=20)
    checkpoint = partial.save_checkpoint(tmp_path / "work")
    resumed = _accelerating_block(prescribed=prescribed, history_every=7)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    assert resumed.history_records[-1] == pytest.approx(reference.history_records[-1])


def test_tiny_physical_time_does_not_merge_distinct_history_frames():
    step = _accelerating_block()
    step.dt = 1e-10
    step.run()
    assert len(step.history_records) == 41
    assert step.history_records[-1]["external_work"] == pytest.approx(
        (40e-10) ** 2, rel=1e-10, abs=1e-30
    )


def test_deforming_hex_under_constant_traction_has_second_order_time_error():
    errors = []
    for dt in (0.002, 0.001):
        domain = mesh.cuboid((0, 0, 0), (1, 1, 1), (1, 1, 1), cell_type="hexahedron")
        model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
        u = model.field(fields.displacement(domain))
        model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
        model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
        model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
        right = mesh.boundary_region(domain, lambda x: np.isclose(x[0], 1))
        model.load(loads.traction((0.01, 0, 0), on=right))
        step = model.step(
            target=u,
            element_policy=elements.uniform_strain_hex8(
                hourglass_modulus=50, hourglass_scale=0.1
            ),
            dt=dt,
            steps=round(0.1 / dt),
            progress=False,
        )
        step.run()
        right_nodes = np.isclose(
            u.value.function_space.tabulate_dof_coordinates()[:, 0], 1
        )
        displacement = u.value.x.array.reshape(-1, 3)[right_nodes, 0].mean()
        # Exact solution of the independently reduced lumped system: m=1,k=100.
        exact = 0.01 / 100 * (1 - np.cos(10 * 0.1))
        errors.append(abs(displacement - exact))
        last = step.history_records[-1]
        assert last["external_work"] == pytest.approx(0.01 * displacement, rel=1e-10)
        assert last["strain_energy"] == pytest.approx(50 * displacement**2, rel=1e-10)
        assert last["relative_energy_balance_error"] < 1e-4
    assert errors[1] < errors[0] / 3.9
