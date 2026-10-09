# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from test_transient_restart import _dynamic_step


@pytest.mark.parametrize("where", ["load", "residual", "commit"])
def test_explicit_failure_restores_accepted_motion_and_can_retry(where):
    reference = _dynamic_step(implicit=False)
    reference.run()
    step = _dynamic_step(implicit=False)
    step.run(until_step=1)
    before = step.state.snapshot()
    history = list(step.history_records)
    times = []
    original = step.residual
    original_update = step.update_load

    class FailingResidual:
        def __init__(self):
            self.accepted = 1

        def snapshot(self):
            return self.accepted

        def restore(self, saved):
            self.accepted = saved

        def assemble_vector(self):
            if where == "residual":
                raise RuntimeError("injected residual failure")
            return original.assemble_vector()

        def commit(self):
            self.accepted = 999
            if where == "commit":
                raise RuntimeError("injected commit failure")

    def update(t):
        times.append(t)
        if where == "load" and t > step.dt:
            step.state.u.value.x.array[:] = 999
            raise RuntimeError("injected load failure")
        if original_update:
            original_update(t)

    failing = FailingResidual()
    step.residual = failing
    step.update_load = update
    with pytest.raises(RuntimeError, match="injected"):
        step.run()
    assert step.completed_steps == 1
    assert step.history_records == history
    assert failing.accepted == 1
    assert times[-1] == step.dt
    for name, values in before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)
    step.residual = original
    step.update_load = original_update
    step.run()
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(step.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
        )
