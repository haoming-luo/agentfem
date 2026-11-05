"""Failed restoration must preserve the live, already accepted solution."""

import numpy as np
import pytest
from pathlib import Path
import json

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


@pytest.mark.parametrize("defect", ["coordinate", "accepted", "attempts", "boundary"])
def test_restore_rejects_inconsistent_accepted_path(tmp_path, defect):
    source = _step()
    source.solve(until=0.5)
    comm = source.solution.function_space.mesh.comm
    directory = comm.bcast(str(tmp_path), root=0)
    checkpoint = source.save_checkpoint(directory + "/source")
    damaged = checkpoint.with_name("inconsistent.checkpoint.json")
    if comm.rank == 0:
        payload = json.loads(checkpoint.read_text())
        if defect == "coordinate":
            payload["coordinate"] = 0.75
        elif defect == "accepted":
            payload["accepted_increments"][-1]["converged"] = False
        elif defect == "attempts":
            payload["attempted_increments"] = []
        else:
            # A self-consistent path ledger but displacement from another time.
            payload["coordinate"] = 0.75
            for records in (
                payload["accepted_increments"],
                payload["attempted_increments"],
            ):
                records[-1]["load_factor"] = 0.75
                records[-1]["increment_size"] = 0.5
        damaged.write_text(json.dumps(payload))
    comm.barrier()
    live = _step()
    live.solve(until=0.25)
    before = live.solution.x.array.copy()
    with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
        live.load_checkpoint(damaged)
    np.testing.assert_array_equal(live.solution.x.array, before)
    assert live.accepted_load_factor == 0.25


@pytest.mark.parametrize("failed_attempt", [False, True])
def test_valid_zero_or_rejected_path_checkpoint_continues(
    tmp_path, monkeypatch, failed_attempt
):
    source = _step()
    comm = source.solution.function_space.mesh.comm
    if failed_attempt:
        source.solve(until=0.25)
        original = source.material.update_batch

        def reject(request):
            if request.time > 0.25:
                raise RuntimeError("rejected next increment")
            return original(request)

        with monkeypatch.context() as patch:
            patch.setattr(source.material, "update_batch", reject)
            with pytest.raises(RuntimeError):
                source.solve(until=0.5)
        assert source.accepted_load_factor == 0.25
        assert not source.attempted_increments[-1].converged
    directory = comm.bcast(str(tmp_path), root=0)
    checkpoint = source.save_checkpoint(directory + "/valid")
    live = _step()
    live.load_checkpoint(checkpoint)
    live.solve()
    reference = _step()
    reference.solve()
    np.testing.assert_allclose(live.solution.x.array, reference.solution.x.array)
    np.testing.assert_allclose(
        live.state.committed_state_vectors(), reference.state.committed_state_vectors()
    )
