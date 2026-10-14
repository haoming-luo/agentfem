# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""The private Procedure primitive retains ordinary state transaction rules."""
import numpy as np
import pytest

from agentfem._transient_problems import _record_transient_history
from test_deformable_contact_explicit import make_step
from test_variable_scaled_inertia import variable_step, assert_snapshot_equal


def test_subincrements_match_fine_steps_without_changing_macro_schedule():
    fine = make_step(dt=1e-4, steps=4)
    fine.run()
    coarse = make_step(dt=2e-4, steps=2)
    # Initialize through the ordinary lifecycle, then exercise the primitive
    # from a separately matched accepted station.
    coarse.run(until_step=1)
    matched = make_step(dt=1e-4, steps=4)
    matched.run(until_step=2)
    for name in ("u", "v", "a"):
        getattr(coarse.state, name).value.x.array[:] = getattr(matched.state, name).value.x.array
    coarse.residual.restore(matched.residual.snapshot())
    coarse.history_monitor.restore(matched.history_records[-1])
    for end in (3e-4, 4e-4):
        coarse._advance_one(end, increment_size=1e-4)
        _record_transient_history(coarse, end, store=False)
    assert coarse.completed_steps == 1 and coarse.dt == 2e-4
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(getattr(coarse.state, name).value.x.array,
                                   getattr(fine.state, name).value.x.array,
                                   atol=1e-14, rtol=1e-12)


def test_subincrement_requires_prepared_inertia_and_restores_failure():
    step = variable_step()
    step.run(until_step=1)
    with pytest.raises(ValueError, match="inertia prepared"):
        step._advance_one(1.5*step.dt, increment_size=step.dt/2)
    mass = step.integrator.mass
    mass.prepare_increment(step.state, increment=2, time=step.dt, dt=step.dt)
    before = mass.snapshot()
    state = step.state.snapshot()
    residual = step.residual.snapshot()
    original = step.residual.commit
    def fail():
        original()
        raise RuntimeError("subincrement commit failed")
    step.residual.commit = fail
    with pytest.raises(RuntimeError, match="subincrement commit"):
        step._advance_one(1.5*step.dt, increment_size=step.dt/2)
    assert_snapshot_equal(before, mass.snapshot())
    assert_snapshot_equal(state, step.state.snapshot())
    assert residual == step.residual.snapshot()
    assert step.residual.time == pytest.approx(step.dt)


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), 1])
def test_invalid_subincrement_is_rejected_without_mutation(size):
    step = make_step()
    before = step.state.snapshot()
    with pytest.raises(ValueError, match="subincrement"):
        step._advance_one(step.dt, increment_size=size)
    assert_snapshot_equal(before, step.state.snapshot())
