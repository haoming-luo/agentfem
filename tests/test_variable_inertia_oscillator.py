# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent piecewise-analytic oscillator, without material State."""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import fracture, problems, state, time
from agentfem.time._variable_inertia import VariableScaledInertia


def make(dt):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    history = state.second_order_state(u)
    history.u.value.x.array[:] = 1.0
    history.v.value.x.array[:] = 0.2
    history.a.value.x.array[:] = -4.0
    mass = VariableScaledInertia(
        np.ones(len(u.x.array)),
        policy=lambda **request: 1.0 if request["time"] < 0.4 - 1e-12 else 4.0,
        policy_identity={"test": "piecewise-oscillator", "switch": 0.4},
        maximum_added_mass_ratio=3,
    )

    class Spring:
        def assemble_vector(self):
            result = u.x.petsc_vec.duplicate()
            result.array[:] = 4 * history.u.value.x.array
            return result

    class Energy:
        def evaluate(self, *, displacement, velocity):
            x, v = displacement.value.x.array, velocity.value.x.array
            kinetic = float(0.5 * np.dot(mass.mass, v * v))
            potential = float(2 * np.dot(x, x))
            return {
                "kinetic_energy": kinetic,
                "total_mechanical_energy": kinetic + potential,
                "numerical_inertia_energy_input": mass.energy_input,
            }

    residual = Spring()
    return problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=mass),
        residual=residual,
        dt=dt,
        steps=round(1 / dt),
        history_every=1,
        progress=False,
        history_monitor=fracture.DynamicEnergyLedger(
            energy=Energy(), state=history, mass=mass, residual=residual
        ),
    )


def exact():
    first_u = np.cos(0.8) + 0.1 * np.sin(0.8)
    first_v = -2 * np.sin(0.8) + 0.2 * np.cos(0.8)
    return first_u * np.cos(0.6) + first_v * np.sin(0.6), -first_u * np.sin(
        0.6
    ) + first_v * np.cos(0.6)


def test_piecewise_mass_oscillator_second_order_and_energy_refinement():
    errors, balances = [], []
    target_u, target_v = exact()
    for dt in (0.02, 0.01, 0.005):
        step = make(dt)
        step.run()
        errors.append(
            max(
                np.max(abs(step.state.u.value.x.array - target_u)),
                np.max(abs(step.state.v.value.x.array - target_v)),
            )
        )
        balances.append(abs(step.history_records[-1]["relative_energy_balance_error"]))
        assert step.integrator.mass.update_count == 1
    assert errors[1] < 0.27 * errors[0] and errors[2] < 0.27 * errors[1]
    assert balances[1] < 0.27 * balances[0] and balances[2] < 0.27 * balances[1]


def test_inertia_only_checkpoint_without_residual_state(tmp_path):
    full = make(0.01)
    full.run()
    partial = make(0.01)
    partial.run(until_step=50)
    path = partial.save_checkpoint(tmp_path / "inertia-only")
    resumed = make(0.01)
    resumed.load_checkpoint(path)
    resumed.run()
    np.testing.assert_array_equal(
        full.state.u.value.x.array, resumed.state.u.value.x.array
    )
    assert full.history_records == resumed.history_records
    bad = dict(resumed.history_records[-1])
    bad["numerical_inertia_energy_input"] += 1
    with pytest.raises(ValueError, match="inertia State"):
        resumed.history_monitor.restore(bad)
