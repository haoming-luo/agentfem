"""Failed restoration must preserve the live, already accepted solution."""

import numpy as np
import pytest
from pathlib import Path

from agentfem import steps
from test_small_strain_material_step import _problem


def _step():
    model, displacement, material = _problem()
    return model.step(
        target=displacement, incrementation=steps.fixed(4), progress=False
    )


@pytest.mark.parametrize(
    "failure",
    [
        "quadrature",
        "response",
        "rank_local_provider",
        "rank_local_manifest",
    ],
)
def test_failed_restore_preserves_live_state(tmp_path, monkeypatch, failure):
    source = _step()
    source.solve(until=0.75)
    comm = source.solution.function_space.mesh.comm
    directory = comm.bcast(str(tmp_path), root=0)
    checkpoint = source.save_checkpoint(directory + "/source")
    live = _step()
    live.solve(until=0.25)
    before_u = live.solution.x.array.copy()
    before_state = live.state.snapshot()
    before_stress = live.response.cauchy_stress.values.copy()
    before_tangent = live.response.tangent.values.copy()
    before_events = list(live.execution_events)
    before_info = live.last_solve_info
    before_trial = live.state.trial_state_vectors().copy()
    before_strain = live._accepted_strain.copy()
    before_constants = [
        np.array(item.value, copy=True)
        for item in [live.load_factor, *(item[0] for item in live.prescribed_values)]
    ]

    def fail(*args, **kwargs):
        if failure == "response":
            live.response.cauchy_stress.assign(before_stress + 3)
            live.response.tangent.assign(before_tangent + 7)
            live.state.assign_trial_state_vectors(before_trial + 1)
            live.response.last_diagnostics = ({"injected": True},)
        raise RuntimeError("injected restore failure")

    with monkeypatch.context() as patch:
        if failure == "quadrature":
            patch.setattr(live.state, "load", fail)
        elif failure == "rank_local_provider":
            if comm.rank == 0:
                patch.setattr(live.material, "update_batch", fail)
        elif failure == "rank_local_manifest":
            if comm.rank == 0:
                patch.setattr(Path, "read_text", fail)
        else:
            patch.setattr(live, "_update_material", fail)
        with pytest.raises(
            (RuntimeError, ValueError), match="injected restore failure"
        ):
            live.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(live.solution.x.array, before_u)
    for name, values in before_state.items():
        np.testing.assert_array_equal(live.state.snapshot()[name], values)
    np.testing.assert_array_equal(live.state.trial_state_vectors(), before_trial)
    np.testing.assert_array_equal(live.response.cauchy_stress.values, before_stress)
    np.testing.assert_array_equal(live.response.tangent.values, before_tangent)
    assert live.accepted_load_factor == 0.25
    assert live.execution_events == before_events
    assert live.last_solve_info is before_info
    np.testing.assert_array_equal(live._accepted_strain, before_strain)
    for value, constant in zip(
        before_constants,
        [live.load_factor, *(item[0] for item in live.prescribed_values)],
    ):
        np.testing.assert_array_equal(constant.value, value)
    live.load_checkpoint(checkpoint)
    live.solve()
    source.solve()
    np.testing.assert_allclose(live.solution.x.array, source.solution.x.array)
