from copy import deepcopy

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import problems, state, time
from agentfem.constitutive._explicit_material_driver import ExplicitQuadratureResponse
from agentfem.diagnostics import MechanicalEnergyMonitor
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._response_hex_explicit import ResponseHexExplicitResidual
from test_explicit_material_quadrature import ResponseOnly


def test_unaccepted_response_cannot_be_restored():
    step = make_step()
    record = step.residual.snapshot()
    record["fields"]["P"][0] = 1.0
    before = step.residual.snapshot()
    with pytest.raises(ValueError, match="unaccepted"):
        step.residual.restore(record)
    assert step.residual.snapshot() == before


def make_step(dt=1e-4):
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 2, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    history = state.second_order_state(u)
    law = ResponseOnly()
    response = ExplicitQuadratureResponse.create(domain, law.state_schema)
    internal = FiniteUniformHexResidual(
        history.u.value,
        response,
        density=1.0,
        hourglass_modulus=1.0,
        hourglass_scale=0.1,
    )
    residual = ResponseHexExplicitResidual(
        internal,
        law,
        time_increment_ceiling=1e-3,
        stress_free_initial_state=True,
        material_identity={"test": "linear-response", "modulus": 20.0},
    )
    xyz = u.function_space.tabulate_dof_coordinates()
    history.v.value.x.array[:] = (0.001 * xyz).ravel()
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        history_monitor=MechanicalEnergyMonitor(mass=internal.mass_diagonal),
        dt=dt,
        steps=10,
        progress=False,
    )
    return step


def test_response_only_advancement_and_disk_restart(tmp_path):
    reference = make_step()
    reference.run()
    partial = make_step()
    partial.run(until_step=4)
    path = partial.save_checkpoint(tmp_path / "response")
    resumed = make_step()
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(resumed.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
        )
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert reference.residual.internal_work > 0
    assert reference.residual.accepted_time == pytest.approx(0.001)


def test_material_failure_rolls_back_complete_step():
    step = make_step()
    step.residual.material.reduce = True
    before = step.state.snapshot()
    material_before = step.residual.snapshot()
    with pytest.raises(ValueError, match="smaller increment"):
        step.run()
    assert step.completed_steps == 0
    assert step.residual.snapshot() == material_before
    for k, v in before["fields"].items():
        np.testing.assert_array_equal(v, step.state.snapshot()["fields"][k])


def test_corrupt_restore_atomic_and_ceiling_guard():
    step = make_step()
    step.run(until_step=2)
    before = step.residual.snapshot()
    bad = deepcopy(before)
    bad["gradient"][0][0][0] = -1
    with pytest.raises(ValueError, match="gradient"):
        step.residual.restore(bad)
    assert step.residual.snapshot() == before
    with pytest.raises(ValueError, match="ceiling"):
        step.residual.validate_time_increment(0.1)


def test_late_commit_failure_restores_accepted_material_and_nodal_state(monkeypatch):
    step = make_step()
    step.run(until_step=2)
    nodal = step.state.snapshot()
    accepted = step.residual.snapshot()
    original = step.residual.internal.response.commit

    def fail_after_commit():
        original()
        raise RuntimeError("late commit failure")

    monkeypatch.setattr(step.residual.internal.response, "commit", fail_after_commit)
    with pytest.raises(RuntimeError, match="late commit"):
        step.run()
    assert step.residual.snapshot() == accepted
    for k, v in nodal["fields"].items():
        np.testing.assert_array_equal(v, step.state.snapshot()["fields"][k])


def test_rigid_translation_has_zero_work():
    step = make_step()
    step.state.v.value.x.array.reshape(-1, 3)[:] = [0.1, -0.2, 0.3]
    step.run()
    assert abs(step.residual.internal_work) < 1e-20
    np.testing.assert_allclose(
        step.state.u.value.x.array.reshape(-1, 3),
        np.tile([0.0001, -0.0002, 0.0003], (12, 1)),
        atol=1e-14,
    )
