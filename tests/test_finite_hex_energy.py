# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite energy lifecycle, not general finite dynamics admission."""

from dataclasses import replace

import numpy as np
import pytest

from agentfem import fracture
from agentfem.mechanics._finite_hex_energy import FiniteHexEnergyMonitor
from test_finite_hex_explicit import make_step


def energy_step(*, growth_resolution=None):
    step = make_step(growth_resolution=growth_resolution)
    residual = step.residual
    residual.enable_energy(
        residual.material.initial_array_response(len(residual.internal.cell_nodes))
    )
    step.history_monitor = fracture.DynamicEnergyLedger(
        energy=FiniteHexEnergyMonitor(residual),
        state=step.state,
        mass=residual.internal.mass_diagonal,
        residual=residual,
    )
    return step


def test_initial_response_never_advances_material(monkeypatch):
    step = make_step()
    law = step.residual.material

    def forbid(*args, **kwargs):
        raise AssertionError("Initial response must not call the time integrator")

    monkeypatch.setattr(type(law), "_integrate_batch", forbid)
    initial = law.initial_array_response(3)
    np.testing.assert_array_equal(initial.cauchy_stress, 0)
    np.testing.assert_array_equal(initial.strain_energy_density, 0)
    np.testing.assert_allclose(
        initial.consistent_tangent[0, 0, 0],
        law.bulk_modulus + 4 * law.shear_modulus / 3,
    )


def test_initial_response_requires_matching_virgin_state():
    step = make_step()
    r = step.residual
    initial = r.material.initial_array_response(len(r.internal.cell_nodes))
    state = initial.state_new.copy()
    state[0, -2] = 0.1
    with pytest.raises(ValueError, match="identity/state"):
        r.enable_energy(replace(initial, state_new=state))
    assert r._accepted_energy is None


def test_initial_energy_populates_response_fields_without_advancement():
    step = energy_step()
    residual = step.residual
    expected = residual.material.initial_array_response(
        len(residual.internal.cell_nodes)
    )
    np.testing.assert_array_equal(
        residual.internal.response.tangent.values.reshape(-1, 9, 9),
        expected.consistent_tangent,
    )
    assert residual.accepted_time == 0


def test_initial_tangent_matches_fixed_state_stress_difference():
    from test_material_array_batch import fixture

    law, request = fixture(count=1)
    initial = law.initial_array_response(1)
    epsilon = 1e-6
    numerical = np.zeros((9, 9))
    for component in range(9):
        forces = []
        for sign in (-1, 1):
            gradient = np.eye(3).reshape(1, 9).copy()
            gradient[0, component] += sign * epsilon
            gradient = gradient.reshape(1, 3, 3)
            result = law.update_array_batch(
                replace(request, deformation_gradient_new=gradient)
            )
            piola = np.linalg.det(gradient)[:, None, None] * (
                result.cauchy_stress @ np.linalg.inv(gradient).transpose(0, 2, 1)
            )
            forces.append(piola.ravel())
        numerical[:, component] = (forces[1] - forces[0]) / (2 * epsilon)
    np.testing.assert_allclose(
        initial.consistent_tangent[0], numerical, rtol=1e-8, atol=1e-4
    )


def test_accepted_energy_restart_and_no_repeat_material_update(tmp_path, monkeypatch):
    reference = energy_step()
    reference.run()
    partial = energy_step()
    partial.run(until_step=2)
    checkpoint = partial.save_checkpoint(tmp_path / "finite-energy")
    resumed = energy_step()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert resumed.history_records == reference.history_records
    before = resumed.residual.snapshot()

    def forbid(*args, **kwargs):
        raise AssertionError("Energy sampling must not advance material")

    monkeypatch.setattr(resumed.residual.internal.response, "update", forbid)
    resumed.history_monitor.evaluate(
        displacement=resumed.state.u, velocity=resumed.state.v
    )
    assert resumed.residual.snapshot() == before


def test_failed_commit_restores_accepted_energy(monkeypatch):
    step = energy_step()
    step.run(until_step=2)
    before = step.residual.snapshot()
    original = step.residual.commit

    def fail():
        original()
        raise RuntimeError("after commit")

    monkeypatch.setattr(step.residual, "commit", fail)
    with pytest.raises(RuntimeError, match="after commit"):
        step.run()
    assert step.residual.snapshot() == before


def test_physical_discrete_and_dissipated_energy_have_distinct_meanings():
    step = energy_step(growth_resolution=0.1)
    xyz = step.state.u.value.function_space.tabulate_dof_coordinates()
    step.state.v.value.x.array[:] = np.column_stack(
        (0.1 * xyz[:, 1] * xyz[:, 2], np.zeros(len(xyz)), np.zeros(len(xyz)))
    ).ravel()
    step.run()
    row = step.history_records[-1]
    assert row["hourglass_energy"] > 1e-15
    assert row["total_mechanical_energy"] == pytest.approx(
        row["kinetic_energy"] + row["bulk_stored_energy"] + row["interface_stored_energy"]
    )
    assert row["total_discrete_energy"] == pytest.approx(
        row["total_mechanical_energy"] + row["hourglass_energy"]
    )
    assert row["accounted_internal_kinetic_energy"] == pytest.approx(
        row["total_discrete_energy"] + row["material_dissipation"]
    )


def prescribed_bar(dt, *, yield_stress=1e9, growth_resolution=None):
    """One-cell constrained stretch, with analytically prescribed acceleration."""
    from mpi4py import MPI
    from dolfinx import mesh as dm
    from agentfem import (
        amplitudes,
        constitutive,
        fields,
        models,
        problems,
        state,
        studies,
        time,
    )
    from agentfem.constitutive.material_driver import MaterialQuadratureResponse
    from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
    from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual

    domain = dm.create_box(
        MPI.COMM_SELF,
        [[0, 0, 0], [1, 1, 1]],
        [1, 1, 1],
        cell_type=dm.CellType.hexahedron,
    )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(
        u,
        on=lambda x: np.isclose(x[0], 1),
        components=0,
        value=amplitudes.Amplitude(
            "quadratic_stretch",
            lambda t: 20 * t * t,
            metadata={"coefficient": 20, "power": 2},
        ),
    )
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=yield_stress, hardening_modulus=5
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    history = state.second_order_state(u)
    internal = FiniteUniformHexResidual(
        history.u, response, density=2, hourglass_modulus=40, hourglass_scale=0.1
    )
    residual = FiniteHexExplicitResidual(
        internal,
        law,
        omega_squared_bound=10000,
        maximum_negative_growth_per_increment=growth_resolution,
    )
    residual.enable_energy(law.initial_array_response(1))
    ledger = fracture.DynamicEnergyLedger(
        energy=FiniteHexEnergyMonitor(residual),
        state=history,
        mass=internal.mass_diagonal,
        residual=residual,
        prescribed=tuple(model.constraints),
    )
    return problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        dt=dt,
        steps=round(0.1 / dt),
        prescribed=tuple(model.constraints),
        history_monitor=ledger,
        progress=False,
    )


@pytest.mark.parametrize("yield_stress", [1e9, 1.0])
def test_finite_stretch_energy_against_independent_coaxial_solution(yield_stress):
    errors = []
    for dt in (0.002, 0.001, 0.0005):
        step = prescribed_bar(dt, yield_stress=yield_stress, growth_resolution=0.1)
        step.run()
        row = step.history_records[-1]
        # Exact proportional logarithmic J2 return in uniaxial strain, volume=1.
        mu, bulk, hardening = 100 / 2.6, 100 / 1.2, 5
        log_stretch = np.log(1.2)
        peeq = max(0, (2 * mu * log_stretch - yield_stress) / (3 * mu + hardening))
        stored = (
            bulk / 2 * log_stretch**2
            + 2 * mu / 3 * (log_stretch - 1.5 * peeq) ** 2
            + hardening / 2 * peeq**2
        )
        assert row["bulk_stored_energy"] == pytest.approx(stored, rel=1e-10)
        assert row["material_dissipation"] == pytest.approx(
            yield_stress * peeq, abs=1e-12
        )
        assert row["kinetic_energy"] == pytest.approx(8, rel=1e-10)
        assert row["hourglass_energy"] < 1e-24
        errors.append(abs(row["energy_balance_error"]))
    assert errors[-1] < errors[0] / 10
    assert errors[-1] < 2e-4


def test_plastic_stretch_requires_explicit_curvature_policy_and_rolls_back():
    step = prescribed_bar(0.002, yield_stress=1.0)
    with pytest.raises(ValueError, match="curvature policy"):
        step.run()
    assert step.completed_steps > 0
    accepted = step.residual.snapshot()
    assert accepted["time"] == pytest.approx(step.completed_steps * step.dt)
    assert step.residual._trial is None


def test_negative_curvature_resolution_rejection_is_atomic():
    step = prescribed_bar(0.002, yield_stress=1.0, growth_resolution=1e-12)
    with pytest.raises(ValueError, match="under-resolves"):
        step.run()
    assert step.residual._trial is None
    assert step.residual.accepted_time == pytest.approx(step.dt * step.completed_steps)
    step.history_monitor.evaluate(displacement=step.state.u, velocity=step.state.v)


def test_plastic_energy_restart_retains_work_dissipation_and_signed_spectrum(tmp_path):
    reference = prescribed_bar(0.001, yield_stress=1, growth_resolution=0.1)
    reference.run()
    partial = prescribed_bar(0.001, yield_stress=1, growth_resolution=0.1)
    partial.run(until_step=63)
    path = partial.save_checkpoint(tmp_path / "plastic-energy")
    resumed = prescribed_bar(0.001, yield_stress=1, growth_resolution=0.1)
    resumed.load_checkpoint(path)
    resumed.run()
    assert resumed.history_records == reference.history_records
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert resumed.residual.last_spectrum["negative_material_curvature_cells"] == 1
    assert resumed.history_records[-1]["material_dissipation"] > 0


def test_interface_energy_is_cached_from_force_trial_and_restartable(
    tmp_path, monkeypatch
):
    from test_finite_hex_interface import dynamic_step

    def build():
        step = dynamic_step()
        residual = step.residual
        residual.enable_energy(
            residual.material.initial_array_response(len(residual.internal.cell_nodes))
        )
        step.history_monitor = fracture.DynamicEnergyLedger(
            energy=FiniteHexEnergyMonitor(residual),
            state=step.state,
            mass=residual.internal.mass_diagonal,
            residual=residual,
        )
        return step

    reference = build()
    reference.run()
    partial = build()
    partial.run(until_step=2)
    path = partial.save_checkpoint(tmp_path / "interface-energy")
    resumed = build()
    resumed.load_checkpoint(path)
    resumed.run()
    assert resumed.history_records == reference.history_records
    row = resumed.history_records[-1]
    assert row["interface_stored_energy"] > 0
    assert row["relative_energy_balance_error"] < 1e-7

    def forbid(*args, **kwargs):
        raise AssertionError("Energy sampling must reuse the force trial")

    monkeypatch.setattr(resumed.residual.cohesive.assembler, "evaluate", forbid)
    resumed.history_monitor.evaluate(
        displacement=resumed.state.u, velocity=resumed.state.v
    )
