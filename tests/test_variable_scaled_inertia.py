# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import numpy as np
import pytest

from agentfem import fracture, time
from agentfem.time._variable_inertia import VariableScaledInertia
from agentfem.mechanics._response_hex_explicit import ResponseHexWorkMonitor
from test_deformable_contact_explicit import make_step


def policy(**request):
    return 2.0 if request["increment"] < 9 else 4.0


def variable_step(*, selected_policy=policy, identity=None, budget=4):
    step = make_step()
    body = step.residual.base
    mass = VariableScaledInertia(
        body.internal.mass_diagonal,
        policy=selected_policy,
        policy_identity={"test": "two-stage"} if identity is None else identity,
        maximum_added_mass_ratio=budget,
    )
    step.integrator = time.explicit.central_difference(state=step.state, mass=mass)
    step.history_monitor = fracture.DynamicEnergyLedger(
        energy=ResponseHexWorkMonitor(body, numerical_inertia=mass),
        state=step.state,
        mass=mass,
        residual=step.residual,
    )
    return step


def assert_snapshot_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        if isinstance(first[name], dict):
            assert_snapshot_equal(first[name], second[name])
        else:
            np.testing.assert_array_equal(first[name], second[name])


def test_variable_mass_free_translation_accounts_artificial_energy_and_momentum():
    step = variable_step()
    velocity = np.array([0.1, 0.2, -0.05])
    step.state.v.value.x.array.reshape(-1, 3)[:] = velocity
    physical = step.integrator.mass.physical_mass_diagonal.copy()
    coordinates = step.state.u.value.function_space.tabulate_dof_coordinates()
    step.run()
    mass = step.integrator.mass
    assert mass.update_count == 2 and mass.last_increment == step.steps
    np.testing.assert_array_equal(mass.physical_mass_diagonal, physical)
    np.testing.assert_allclose(
        step.state.v.value.x.array.reshape(-1, 3),
        np.broadcast_to(velocity, (len(physical) // 3, 3)),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        mass.momentum_input, 3 * physical[::3].sum() * velocity, atol=1e-14
    )
    np.testing.assert_allclose(
        mass.angular_momentum_input,
        np.cross(coordinates, 3 * physical[::3, None] * velocity).sum(axis=0),
        atol=1e-14,
    )
    assert mass.energy_input == pytest.approx(
        1.5 * physical[::3].sum() * float(velocity @ velocity)
    )
    last = step.history_records[-1]
    assert last["external_work"] == pytest.approx(0, abs=1e-14)
    assert last["energy_balance_error"] == pytest.approx(0, abs=1e-13)
    assert last["numerical_inertia_energy_input"] == mass.energy_input


def test_mass_revision_preserves_force_and_velocity_before_prediction():
    step = variable_step()
    mass = step.integrator.mass
    step.state.a.value.x.array[:] = np.arange(len(mass.mass_diagonal))
    force = mass.mass_diagonal * step.state.a.value.x.array
    velocity = step.state.v.value.x.array.copy()
    mass.prepare_increment(step.state, increment=1, time=0, dt=step.dt)
    np.testing.assert_array_equal(
        mass.mass_diagonal * step.state.a.value.x.array, force
    )
    np.testing.assert_array_equal(step.state.v.value.x.array, velocity)


def test_unchanged_policy_reuses_immutable_mass_storage():
    step = variable_step(selected_policy=lambda **request: 1.0)
    mass = step.integrator.mass
    diagonal, inverse = mass.mass_diagonal, mass.inv_mass
    step.run()
    assert mass.mass_diagonal is diagonal and mass.inv_mass is inverse
    assert mass.update_count == 0 and mass.last_increment == step.steps


def test_boolean_policy_is_rejected_even_when_numerically_unchanged():
    step = variable_step(
        selected_policy=lambda **request: np.ones_like(
            request["current_factors"], dtype=bool
        )
    )
    with pytest.raises(ValueError, match="booleans"):
        step.run()
    assert step.completed_steps == step.integrator.mass.last_increment == 0


def test_variable_inertia_checkpoint_reproduces_all_fields_history_and_mass(tmp_path):
    whole = variable_step()
    whole.run()
    partial = variable_step()
    partial.run(until_step=8)
    path = partial.save_checkpoint(tmp_path / "variable")
    resumed = variable_step()
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(whole.state, name).value.x.array,
            getattr(resumed.state, name).value.x.array,
        )
    assert whole.history_records == resumed.history_records
    assert_snapshot_equal(
        whole.integrator.mass.snapshot(), resumed.integrator.mass.snapshot()
    )
    wrong = variable_step(identity={"test": "changed-policy"})
    with pytest.raises(ValueError, match="inertia identity"):
        wrong.load_checkpoint(path)
    assert wrong.completed_steps == wrong.integrator.mass.last_increment == 0


def test_failed_increment_rolls_back_mass_after_body_commit(monkeypatch):
    step = variable_step()
    step.run(until_step=8)
    old_mass = step.integrator.mass.snapshot()
    old_state = step.state.snapshot()
    commit = step.residual.commit

    def fail():
        commit()
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(step.residual, "commit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        step.run(until_step=9)
    assert step.completed_steps == 8
    assert_snapshot_equal(old_mass, step.integrator.mass.snapshot())
    assert_snapshot_equal(old_state, step.state.snapshot())


def test_failed_checkpoint_restore_rolls_back_mass_and_fields(tmp_path, monkeypatch):
    source = variable_step()
    source.run(until_step=10)
    path = source.save_checkpoint(tmp_path / "source")
    target = variable_step()
    target.run(until_step=2)
    old_mass, old_state = target.integrator.mass.snapshot(), target.state.snapshot()
    restore = target.residual.restore
    calls = []

    def fail_once(record):
        restore(record)
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("after inertia restore")

    monkeypatch.setattr(target.residual, "restore", fail_once)
    with pytest.raises(RuntimeError, match="after inertia"):
        target.load_checkpoint(path)
    assert target.completed_steps == 2
    assert_snapshot_equal(old_mass, target.integrator.mass.snapshot())
    assert_snapshot_equal(old_state, target.state.snapshot())


@pytest.mark.parametrize("case", ["decrease", "nonfinite", "budget"])
def test_invalid_policy_proposal_rejects_atomically(case):
    def invalid(**request):
        if request["increment"] == 1:
            return 2.0
        return {"decrease": 1.0, "nonfinite": np.nan, "budget": 100.0}[case]

    step = variable_step(selected_policy=invalid)
    step.run(until_step=1)
    old_mass, old_state = step.integrator.mass.snapshot(), step.state.snapshot()
    with pytest.raises(ValueError):
        step.run(until_step=2)
    assert_snapshot_equal(old_mass, step.integrator.mass.snapshot())
    assert_snapshot_equal(old_state, step.state.snapshot())


def test_corrupt_inertia_state_and_unsupported_constraints_fail_closed(tmp_path):
    step = variable_step()
    original = step.integrator.mass.snapshot()
    bad = deepcopy(original)
    bad["factors"][0] = np.nan
    with pytest.raises(ValueError):
        step.integrator.mass.restore(bad)
    assert_snapshot_equal(original, step.integrator.mass.snapshot())
    step.constraints = (object(),)
    with pytest.raises(NotImplementedError, match="MPC"):
        step.run()
    step.constraints = ()
    with pytest.raises(NotImplementedError, match="nonportable"):
        step.save_checkpoint(tmp_path / "portable", portable=True)
