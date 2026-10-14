# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem import fracture, time
from agentfem.time._scaled_inertia import FixedScaledInertia
from agentfem.mechanics._response_hex_explicit import ResponseHexWorkMonitor
from test_deformable_contact_explicit import make_step


def scaled_step(factor=4.0, dt=1e-4):
    step = make_step(dt=dt)
    body = step.residual.base
    inertia = FixedScaledInertia(body.internal.mass_diagonal, factor=factor)
    step.integrator = time.explicit.central_difference(state=step.state, mass=inertia)
    step.history_monitor = fracture.DynamicEnergyLedger(
        energy=ResponseHexWorkMonitor(body, numerical_inertia=inertia),
        state=step.state,
        mass=inertia,
        residual=step.residual,
    )
    return step


def test_fixed_inertia_keeps_physical_density_and_positive_diagonal():
    physical = np.repeat([1.0, 2.0], 3)
    mass = FixedScaledInertia(physical, factor=[2.0, 3.0])
    physical[:] = 9
    np.testing.assert_array_equal(mass.physical_mass_diagonal, np.repeat([1.0, 2.0], 3))
    np.testing.assert_array_equal(mass.mass_diagonal, np.repeat([2.0, 6.0], 3))
    assert mass.summary()["physical_mass"] == 3
    assert mass.summary()["numerical_mass"] == 8
    assert mass.summary()["material_density_changed"] is False
    with pytest.raises(ValueError):
        mass.mass_diagonal[0] = 0


@pytest.mark.parametrize("factor", [True, 0.9, -1.0, float("nan"), [1.0, 2.0, 3.0], 1 + 1j])
def test_bad_scaling_is_rejected(factor):
    with pytest.raises(ValueError):
        FixedScaledInertia(np.ones(6), factor=factor)


def test_uniform_inertia_time_scale_and_separate_kinetic_energies():
    physical = scaled_step(1.0, dt=1e-4)
    scaled = scaled_step(4.0, dt=2e-4)
    scaled.state.v.value.x.array[:] *= 0.5
    physical.run()
    scaled.run()
    np.testing.assert_allclose(
        scaled.state.u.value.x.array,
        physical.state.u.value.x.array,
        atol=1e-15,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        scaled.state.v.value.x.array * 2,
        physical.state.v.value.x.array,
        atol=1e-15,
        rtol=1e-12,
    )
    last = scaled.history_records[-1]
    assert last["kinetic_energy"] == pytest.approx(
        4 * last["physical_mass_kinetic_energy"]
    )
    assert last["added_inertia_kinetic_energy"] == pytest.approx(
        3 * last["physical_mass_kinetic_energy"]
    )


def test_scaled_inertia_restart_and_identity_rejection(tmp_path):
    continuous = scaled_step()
    continuous.run()
    partial = scaled_step()
    partial.run(until_step=8)
    path = partial.save_checkpoint(tmp_path / "scaled")
    resumed = scaled_step()
    resumed.load_checkpoint(path)
    resumed.run()
    np.testing.assert_array_equal(
        resumed.state.u.value.x.array, continuous.state.u.value.x.array
    )
    assert resumed.history_records == continuous.history_records
    incompatible = scaled_step(9.0)
    before = incompatible.state.u.value.x.array.copy()
    with pytest.raises(ValueError, match="inertia identity"):
        incompatible.load_checkpoint(path)
    np.testing.assert_array_equal(incompatible.state.u.value.x.array, before)
    assert incompatible.completed_steps == 0
    ordinary = make_step()
    with pytest.raises(ValueError, match="inertia identity"):
        ordinary.load_checkpoint(path)


def test_old_unscaled_checkpoint_cannot_silently_enter_scaled_run(tmp_path):
    plain = make_step()
    plain.run(until_step=2)
    path = plain.save_checkpoint(tmp_path / "plain")
    scaled = scaled_step()
    with pytest.raises(ValueError, match="inertia identity"):
        scaled.load_checkpoint(path)
    with pytest.raises(NotImplementedError, match="nonportable"):
        scaled.save_checkpoint(tmp_path / "portable", portable=True)
