# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import problems, state, time
from agentfem.time._nodal_inertia import plan_nodal_inertia
from agentfem.time._variable_inertia import VariableScaledInertia

K = np.array([[30.0, 4.0, 2.0], [4.0, 20.0, -3.0], [2.0, -3.0, 10.0]])
C = np.array([[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]])


def make(budget=1000.0):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    history = state.second_order_state(u)
    history.u.value.x.array.reshape(-1, 3)[:] = [1.0, -0.3, 0.2]
    history.v.value.x.array[:] = 0.1
    history.a.value.x.array.reshape(-1, 3)[:] = (
        -(K @ [1.0, -0.3, 0.2] + C @ [0.1, 0.1, 0.1]) / 0.01
    )
    count = len(u.x.array) // 3
    physical = np.full(count, 0.01)

    def policy(**request):
        return plan_nodal_inertia(
            physical,
            stiffness_rows={"elastic": np.tile(np.abs(K).sum(axis=1), (count, 1))},
            damping_rows={"damping": np.tile(np.abs(C).sum(axis=1), (count, 1))},
            target_increment=request["dt"],
            minimum_nodal_factors=request["current_factors"],
        ).nodal_factors

    mass = VariableScaledInertia(
        np.repeat(physical, 3),
        policy=policy,
        policy_identity={"test": "combined-rows.v1", "K": K.tolist(), "C": C.tolist()},
        maximum_added_mass_ratio=budget,
    )

    class Residual:
        def assemble_vector(self):
            result = u.x.petsc_vec.duplicate()
            result.array[:] = (
                history.u.value.x.array.reshape(-1, 3) @ K.T
                + history.v_mid.value.x.array.reshape(-1, 3) @ C.T
            ).ravel()
            return result

    return problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=mass),
        residual=Residual(),
        dt=0.1,
        steps=100,
        progress=False,
    )


def test_combined_plan_drives_existing_step_and_bitwise_restart(tmp_path):
    full = make()
    full.run()
    partial = make()
    partial.run(until_step=37)
    path = partial.save_checkpoint(tmp_path / "combined-inertia")
    resumed = make()
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(full.state, name).value.x.array,
            getattr(resumed.state, name).value.x.array,
        )
    mass = full.integrator.mass.mass_diagonal[0]
    x = np.array([1.0, -0.3, 0.2])
    v = np.full(3, 0.1)
    acceleration = -(K @ x + C @ v) / mass
    for _ in range(100):
        x = x + 0.1 * v + 0.005 * acceleration
        half = v + 0.05 * acceleration
        acceleration = -(K @ x + C @ half) / mass
        v = half + 0.05 * acceleration
    np.testing.assert_allclose(
        full.state.u.value.x.array.reshape(-1, 3)[0], x, atol=1e-14
    )
    assert np.linalg.norm(x) < 1e-3
    assert full.integrator.mass.update_count == 1


def test_combined_plan_budget_failure_preserves_accepted_fields():
    step = make(budget=0.0)
    before = step.state.snapshot()
    with pytest.raises(ValueError, match="added-mass budget"):
        step.run()
    assert step.completed_steps == 0
    for name, value in before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], value)
    assert step.integrator.mass.update_count == 0
