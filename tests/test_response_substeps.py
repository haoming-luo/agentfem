# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem import constitutive, time
from test_response_material_step import problem


def make_step(terminal=False, material_storage="fields"):
    model, u, policy = problem()
    original = model.materials[0].item
    def update(request):
        if request.time_increment > 5.1e-5 or (terminal and request.time > 1.4e-4):
            raise constitutive.ExplicitMaterialCutbackRequest("test resolution limit")
        return original.update(request)
    model.materials.clear()
    model.material(constitutive.explicit_user_material(state_schema=original.state_schema,
        update=update, identity={"test": "bounded_cutback", "terminal": terminal}, density=1.))
    step = model.step(target=u, element_policy=policy, dt=1e-4, steps=4,
        time_increment_ceiling=1e-3, stress_free_initial_state=True, progress=False,
        substep_policy=time.ExplicitSubstepPolicy(3, True), history_every=1,
        material_storage=material_storage)
    step.state.v.value.x.array[::3] = .01
    return step


def test_typed_retry_keeps_macro_clock_and_restart(tmp_path):
    whole = make_step()
    whole.run()
    assert whole.completed_steps == 4
    assert len(whole._substeps.events) == 4
    np.testing.assert_allclose(whole.state.u.value.x.array[::3], 4e-6, atol=1e-15)
    partial = make_step()
    partial.run(until_step=2)
    path = partial.save_checkpoint(tmp_path / "retry")
    resumed = make_step()
    resumed.load_checkpoint(path)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.u.value.x.array, whole.state.u.value.x.array)
    assert resumed.residual.snapshot() == whole.residual.snapshot()


@pytest.mark.parametrize('storage', ('fields', 'owned_arrays'))
def test_terminal_retry_restores_entire_macro(storage):
    step = make_step(terminal=True, material_storage=storage)
    step.run(until_step=1)
    before = step.residual.snapshot()
    u = step.state.u.value.x.array.copy()
    history = list(step.history_records)
    with pytest.raises(constitutive.ExplicitMaterialCutbackRequest):
        step.run()
    assert step.completed_steps == 1
    np.testing.assert_array_equal(step.state.u.value.x.array, u)
    assert step.residual.snapshot() == before
    assert step.history_records == history
