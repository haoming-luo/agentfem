# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
import pytest
from dolfinx import fem

from agentfem import fracture, problems, state, time
from agentfem._deformable_q1_contact import DeformableQ1Penalty
from agentfem._interface_quadrilateral import BilinearReferenceTrace
from agentfem.constitutive._explicit_material_driver import ExplicitQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._deformable_contact_explicit import DeformableQ1ContactResidual
from agentfem.mechanics._response_hex_explicit import (
    ResponseHexExplicitResidual,
    ResponseHexWorkMonitor,
)
from test_explicit_material_quadrature import ResponseOnly
from test_nonmatching_global import _blocks
from test_quadrilateral_global import _trace


def make_step(dt=1e-4, steps=20, *, closed=False, separated=False):
    domain = _blocks(1, 2, cell_type="hexahedron")
    # A smaller lower block puts its contact nodes strictly inside the upper
    # surface. Coincident boundary edges need the separate normal-cone gate.
    cells = domain.geometry.dofmaps[0]
    lower = np.unique(cells[domain.geometry.x[cells, 2].mean(axis=1) < 0])
    domain.geometry.x[lower, :2] = 0.25 + 0.5 * domain.geometry.x[lower, :2]
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
    body = ResponseHexExplicitResidual(
        internal,
        law,
        time_increment_ceiling=1e-3,
        stress_free_initial_state=True,
        material_identity={"test": "response_only"},
    )
    a, na = _trace(u.function_space, False)
    b, nb = _trace(u.function_space, True)
    a = BilinearReferenceTrace(a.vertices, a.quadrilaterals, tolerance=1e-10)
    b = BilinearReferenceTrace(b.vertices, b.quadrilaterals, tolerance=1e-10)
    if closed:
        xyz = u.function_space.tabulate_dof_coordinates()
        cells = internal.cell_nodes[xyz[internal.cell_nodes, 2].mean(axis=1) > 0]
        faces = cells[
            :,
            [
                [0, 2, 3, 1],
                [4, 5, 7, 6],
                [0, 1, 5, 4],
                [1, 3, 7, 5],
                [3, 2, 6, 7],
                [2, 0, 4, 6],
            ],
        ].reshape(-1, 4)
        _, first, count = np.unique(
            np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
        )
        faces = faces[first[count == 1]]
        nb, inverse = np.unique(faces, return_inverse=True)
        b = BilinearReferenceTrace(xyz[nb], inverse.reshape(-1, 4), tolerance=1e-10)
    operator = DeformableQ1Penalty(
        a,
        b,
        candidate_nodes=np.ones(len(a.vertices), dtype=bool),
        penalty=100.0,
        search_radius=0.5,
        geometry_policy="reviewed_closed_solid" if closed else "oriented_open_patch",
    )
    residual = DeformableQ1ContactResidual(
        body,
        operator=operator,
        displacement=history.u.value,
        secondary_dofs=na,
        primary_dofs=nb,
        time_increment_ceiling=1e-3,
    )
    xyz = u.function_space.tabulate_dof_coordinates()
    for nodes in internal.cell_nodes:
        history.v.value.x.array.reshape(-1, 3)[nodes, 2] = -0.1 * np.sign(
            xyz[nodes, 2].mean()
        )
    if separated:
        history.v.value.x.array[:] *= -1
    ledger = fracture.DynamicEnergyLedger(
        energy=ResponseHexWorkMonitor(body),
        state=history,
        mass=internal,
        residual=residual,
    )
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        dt=dt,
        steps=steps,
        history_monitor=ledger,
        history_every=1,
        progress=False,
    )
    return step


@pytest.mark.parametrize("closed", [False, True])
def test_actual_two_body_contact_restart_and_momentum(tmp_path, closed):
    continuous = make_step(closed=closed)
    mass = continuous.residual.base.internal.mass_diagonal.reshape(-1, 3)
    initial_momentum = np.sum(
        mass * continuous.state.v.value.x.array.reshape(-1, 3), axis=0
    )
    continuous.run()
    partial = make_step(closed=closed)
    partial.run(until_step=8)
    path = partial.save_checkpoint(tmp_path / "deformable")
    resumed = make_step(closed=closed)
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(resumed.state, name).value.x.array,
            getattr(continuous.state, name).value.x.array,
        )
    assert resumed.history_records == continuous.history_records
    assert resumed.residual.snapshot() == continuous.residual.snapshot()
    assert continuous.residual.accepted[2].potential_energy > 0
    mass = continuous.residual.base.internal.mass_diagonal.reshape(-1, 3)
    momentum = np.sum(mass * continuous.state.v.value.x.array.reshape(-1, 3), axis=0)
    np.testing.assert_allclose(momentum, initial_momentum, atol=1e-14)


def test_deformable_contact_outer_commit_failure_rolls_back(monkeypatch):
    step = make_step()
    step.run(until_step=2)
    before = step.residual.snapshot()
    displacement = step.state.u.value.x.array.copy()
    original = step.residual.commit

    def fail():
        original()
        raise RuntimeError("injected after contact commit")

    monkeypatch.setattr(step.residual, "commit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        step.run(until_step=3)
    assert step.residual.snapshot() == before
    np.testing.assert_array_equal(step.state.u.value.x.array, displacement)


def test_corrupt_contact_restore_does_not_touch_body():
    step = make_step()
    step.run(until_step=2)
    before = step.residual.snapshot()
    corrupt = deepcopy(before)
    corrupt["accepted"][0][0][0] = np.nan
    corrupt["base"]["gradient"][0][0][0] = 2
    with pytest.raises(ValueError, match="finite real"):
        step.residual.restore(corrupt)
    assert step.residual.snapshot() == before


def test_two_body_energy_time_refinement():
    errors = []
    for dt, steps in ((1e-3, 20), (5e-4, 40), (2.5e-4, 80)):
        step = make_step(dt=dt, steps=steps)
        step.run()
        last = step.history_records[-1]
        errors.append(abs(last["energy_balance_error"]))
    assert errors[1] < 0.35 * errors[0]
    assert errors[2] < 0.35 * errors[1]


def test_outside_to_outside_tunneling_is_rejected_before_material_update():
    step = make_step(steps=1, closed=True)
    residual = step.residual
    values = step.state.u.value.x.array.reshape(-1, 3)
    # Contact-only swept-path test: start .1 below the primary solid and end
    # .2 above it. Both endpoint penalty forces vanish, but the path crosses.
    values[residual.secondary_dofs, 2] = -0.1
    residual.initialize_accepted_state()
    accepted = residual.accepted
    before_material = (
        residual.base.internal.response.state.committed_state_vectors().copy()
    )
    values[residual.secondary_dofs, 2] = 1.2
    with pytest.raises(ValueError, match="Potential missed contact crossing"):
        residual.assemble_vector()
    assert residual.trial is None and residual.accepted is accepted
    np.testing.assert_array_equal(
        residual.base.internal.response.state.committed_state_vectors(), before_material
    )


def test_proved_separation_skips_search_without_fabricating_projection(monkeypatch):
    step = make_step(closed=True)
    residual = step.residual
    step.state.u.value.x.array.reshape(-1, 3)[residual.secondary_dofs, 2] = -0.1
    residual.initialize_accepted_state()
    search = residual.operator.evaluate
    calls = []

    def counted(*values):
        calls.append(1)
        return search(*values)

    monkeypatch.setattr(residual.operator, "evaluate", counted)
    values = tuple(a.copy() for a in residual.accepted[:2])
    values[0][:, 2] += 0.01
    bounded = residual._evaluate_current(values)
    assert calls == [] and bounded.projection is None
    assert bounded.potential_energy == 0
    assert np.all(bounded.separation_lower_bounds > 0)
    exact = search(*values)
    assert np.all(bounded.separation_lower_bounds <= exact.projection.signed_gaps)
    np.testing.assert_array_equal(bounded.secondary_residual, exact.secondary_residual)
    # Approaching through the old separation invalidates the bound; exact
    # search discovers penetration and supplies a real projection again.
    values[0][:, 2] += 0.1
    active = residual._evaluate_current(values)
    assert calls == [1] and active.projection is not None
    assert active.potential_energy > 0


def test_bound_only_response_rejects_forces_or_false_separation():
    from agentfem._deformable_q1_contact import DeformableContactResponse

    with pytest.raises(ValueError, match="force-free"):
        DeformableContactResponse(
            np.ones((1, 3)), np.zeros((1, 3)), 0.0, None, None, None, np.array([0.1])
        )


def test_separated_contact_restart_preserves_physics(tmp_path):
    def separated():
        return make_step(steps=5, closed=True, separated=True)

    continuous = separated()
    continuous.run()
    partial = separated()
    partial.run(until_step=3)
    assert partial.residual.accepted[2].projection is None
    path = partial.save_checkpoint(tmp_path / "separated")
    resumed = separated()
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(resumed.state, name).value.x.array,
            getattr(continuous.state, name).value.x.array,
        )
    assert resumed.history_records == continuous.history_records


def test_identical_contact_geometry_reuses_search_but_not_material_update(monkeypatch):
    step = make_step(steps=3, closed=True)
    step.state.v.value.x.array[:] = 0
    calls = {"search": 0, "material": 0}
    search = step.residual.operator.evaluate
    material = step.residual.base.assemble_vector

    def counted_search(*args):
        calls["search"] += 1
        return search(*args)

    def counted_material():
        calls["material"] += 1
        return material()

    monkeypatch.setattr(step.residual.operator, "evaluate", counted_search)
    monkeypatch.setattr(step.residual.base, "assemble_vector", counted_material)
    step.run()
    assert calls == {"search": 1, "material": 3}
