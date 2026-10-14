# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem import time
from agentfem.mechanics._response_hex_explicit import ResponseHexExplicitResidual
from agentfem.time._scaled_inertia import FixedScaledInertia
from test_response_hex_explicit import make_step


def make(factor):
    step = make_step(viscosity=True)
    old = step.residual
    mass = FixedScaledInertia(old.internal.mass_diagonal, factor=factor)
    step.residual = ResponseHexExplicitResidual(
        old.internal,
        old.material,
        time_increment_ceiling=old.ceiling,
        material_identity={"test": "mass-binding"},
        stress_free_initial_state=True,
        bulk_viscosity=old.bulk_viscosity,
        numerical_inertia=mass,
    )
    step.integrator = time.explicit.central_difference(state=step.state, mass=mass)
    step.history_monitor = None
    coordinates = step.state.u.value.function_space.tabulate_dof_coordinates()
    step.state.v.value.x.array[:] = (0.01 * coordinates).ravel()
    return step


def test_actual_inertia_changes_bound_not_physical_viscosity_force_or_work():
    physical, scaled = make(1), make(9)
    physical.run(until_step=1)
    scaled.run(until_step=1)
    assert scaled.residual.damping_rate_bound == pytest.approx(
        physical.residual.damping_rate_bound / 9
    )
    assert scaled.residual.viscosity_work == pytest.approx(
        physical.residual.viscosity_work
    )
    np.testing.assert_allclose(
        scaled.residual.accepted_force, physical.residual.accepted_force
    )


def test_wrong_integrator_mass_rejects_before_advancing():
    step = make(9)
    step.integrator = time.explicit.central_difference(
        state=step.state, mass=step.residual.internal
    )
    before = step.state.snapshot()
    with pytest.raises(ValueError, match="share numerical inertia"):
        step.run()
    assert step.completed_steps == 0
    for name, value in before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], value)


def test_scaled_damping_disk_restart_and_changed_mass_identity(tmp_path):
    full = make(9)
    full.run()
    partial = make(9)
    partial.run(until_step=3)
    checkpoint = partial.save_checkpoint(tmp_path / "scaled-damping")
    resumed = make(9)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(
        full.state.u.value.x.array, resumed.state.u.value.x.array
    )
    assert full.residual.snapshot() == resumed.residual.snapshot()
    with pytest.raises(ValueError):
        make(8).load_checkpoint(checkpoint)
