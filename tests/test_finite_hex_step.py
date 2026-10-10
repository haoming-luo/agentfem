# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded ordinary-Step route for finite history materials."""

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh

from agentfem import amplitudes, constitutive, elements, fields, loads, models, studies


def problem(comm=MPI.COMM_SELF, *, density=2, force=2, finite=True):
    domain = mesh.create_unit_cube(comm, 3, 2, 2, cell_type=mesh.CellType.hexahedron)
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=100, poisson=0.3, yield_stress=1, hardening_modulus=5, density=density
        )
    )
    model.load(loads.body_force((force, 0, 0), domain=domain, target=u))
    policy = elements.uniform_strain_hex8(
        hourglass_modulus=40,
        hourglass_scale=0.1,
        kinematics="finite_strain" if finite else "small_strain",
    )
    return model, u, policy


def make_step(comm=MPI.COMM_SELF, **kwargs):
    model, u, policy = problem(comm, **kwargs)
    return model.step(
        target=u,
        element_policy=policy,
        omega_squared_bound=1e8,
        dt=1e-4,
        steps=20,
        progress=False,
    )


def test_ordinary_finite_step_body_force_result_fields_and_restart(tmp_path):
    step = make_step()
    result = step.solve_result(
        field_variables=("S", "P", "F", "SENER", "PEEQ", "PDENER", "MISES"),
        output=tmp_path / "finite.xdmf",
    )
    np.testing.assert_allclose(
        step.state.u.value.x.array[::3], 0.5 * (20e-4) ** 2, atol=1e-13
    )
    row = step.history_records[-1]
    assert row["relative_energy_balance_error"] < 1e-8
    assert row["natural_load_work"] == pytest.approx((20e-4) ** 2, rel=1e-10)
    assert result.fields["S"].processing["stress_measure"] == "cauchy"
    assert result.fields["P"].processing["stress_measure"] == "first_piola"
    assert result.fields["SENER"].processing["volume_measure"] == "reference"
    for name in ("S", "P", "F", "SENER", "PEEQ", "PDENER", "MISES"):
        assert result.fields[name].location == "cells"
        assert result.fields[name].processing["interelement_smoothing"] is False
    partial = make_step()
    partial.run(until_step=7)
    checkpoint = partial.save_checkpoint(tmp_path / "checkpoint")
    resumed = make_step()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(
        resumed.state.u.value.x.array, step.state.u.value.x.array
    )
    assert resumed.history_records == pytest.approx(step.history_records)
    wrong_load = make_step(force=3)
    with pytest.raises(ValueError, match="identity"):
        wrong_load.load_checkpoint(checkpoint)


def test_finite_step_prescribed_motion_plastic_energy_and_empty_fields():
    model, u, policy = problem(force=0)
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    # All nodal values are prescribed from a spatially constant displacement
    # only on the end plane; the interior remains an actual dynamic solve.
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(
        u,
        on=lambda x: np.isclose(x[0], 1),
        components=0,
        value=amplitudes.Amplitude(
            "ramp", lambda t: 100 * t * t, metadata={"quadratic_coefficient": 100}
        ),
    )
    step = model.step(
        target=u,
        element_policy=policy,
        omega_squared_bound=1e8,
        maximum_negative_growth_per_increment=0.1,
        dt=1e-4,
        steps=100,
        progress=False,
    )
    result = step.solve_result(field_variables=())
    assert set(result.fields) == {u.value.name}
    assert step.history_records[-1]["material_dissipation"] > 0
    assert step.history_records[-1]["prescribed_motion_work"] > 0
    assert step.history_records[-1]["relative_energy_balance_error"] < 5e-3


def test_finite_step_rejects_missing_density_bound_and_unsupported_fields():
    model, u, policy = problem(density=None)
    with pytest.raises(ValueError, match="density"):
        model.step(
            target=u, element_policy=policy, omega_squared_bound=1e8, dt=1e-4, steps=2
        )
    model, u, policy = problem()
    with pytest.raises((ValueError, TypeError), match="omega_squared_bound"):
        model.step(target=u, element_policy=policy, dt=1e-4, steps=2)
    step = make_step()
    with pytest.raises(ValueError, match="Unsupported finite"):
        step.solve_result(field_variables=("DAMAGE",))
    assert step.completed_steps == 0


def test_finite_restart_binds_strong_boundary_layout(tmp_path):
    source = make_step(force=0)
    source.run(until_step=4)
    checkpoint = source.save_checkpoint(tmp_path / "free-body")
    model, u, policy = problem(force=0)
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    target = model.step(
        target=u,
        element_policy=policy,
        omega_squared_bound=1e8,
        dt=1e-4,
        steps=20,
        progress=False,
    )
    with pytest.raises(ValueError, match="identity"):
        target.load_checkpoint(checkpoint)
    assert target.completed_steps == 0
    np.testing.assert_array_equal(target.state.u.value.x.array, 0)


def test_finite_load_must_not_silently_add_displacement_stiffness():
    model, u, policy = problem()
    model.load(loads.body_force(u.value, domain=u.value.function_space.mesh, target=u))
    with pytest.raises(NotImplementedError, match="follower loads"):
        model.step(
            target=u, element_policy=policy, omega_squared_bound=1e8, dt=1e-4, steps=20
        )


def test_density_does_not_change_material_response():
    from dataclasses import replace
    from test_material_array_batch import fixture

    law, request = fixture(count=3)
    inertial = replace(law, density=7800)
    original, updated = (
        law.update_array_batch(request),
        inertial.update_array_batch(request),
    )
    np.testing.assert_array_equal(original.cauchy_stress, updated.cauchy_stress)
    np.testing.assert_array_equal(
        original.consistent_tangent, updated.consistent_tangent
    )
    np.testing.assert_array_equal(original.state_new, updated.state_new)
    for density in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="density"):
            replace(law, density=density)
