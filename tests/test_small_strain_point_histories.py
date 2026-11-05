"""One accepted material point, one physical clock, across rollback and restart."""

import json
import numpy as np
import pytest
from agentfem import results, steps
from test_small_strain_physical_time import make_step, RelaxingMaterial

POINTS = (results.material_point("a", cell=0), results.material_point("b", cell=1))


def build(**kwargs):
    return make_step(material_points=POINTS, **kwargs)


def test_raw_history_and_result_roundtrip(tmp_path):
    step = build()
    result = step.solve_result()
    for prefix, index in zip(("a", "b"), step._point_history.indices):
        np.testing.assert_array_equal(
            result.histories[prefix + ".S"].latest,
            step.response.cauchy_stress.values[index],
        )
        np.testing.assert_array_equal(
            result.histories[prefix + ".E"].latest, step._accepted_strain[index]
        )
        np.testing.assert_allclose(
            result.histories[prefix + ".AGE"].values,
            result.histories[prefix + ".AGE"].abscissa,
        )
        assert result.histories[prefix + ".S"].abscissa[0] > 0
        assert len(result.histories[prefix + ".S"].abscissa) == len(
            step.accepted_increments
        )
    manifest = result.write_manifest(
        tmp_path / "result.json", include_histories=True, history_storage="npz"
    )
    histories = {h.name: h for h in results.read_histories(manifest)}
    np.testing.assert_array_equal(
        histories["a.E"].values, result.histories["a.E"].values
    )


def test_restart_preserves_all_accepted_frames(tmp_path):
    full = build()
    full.solve()
    partial = build()
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "mid")
    resumed = build()
    resumed.load_checkpoint(path)
    resumed.solve()
    for a, b in zip(
        full.material_point_histories, resumed.material_point_histories, strict=True
    ):
        np.testing.assert_array_equal(a.abscissa, b.abscissa)
        np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("mutation", ["missing", "value", "time"])
def test_corrupt_history_restore_is_atomic(tmp_path, mutation):
    source = build()
    source.solve(until=0.5)
    path = source.save_checkpoint(tmp_path / "source")
    record = json.loads(path.read_text())
    if mutation == "missing":
        del record["material_point_history"]
    elif mutation == "value":
        record["material_point_history"]["frames"][0]["values"]["AGE"][0] = 999
    else:
        record["material_point_history"]["frames"][0]["time"] = 999
    path.write_text(json.dumps(record))
    target = build()
    target.solve(until=0.25)
    before = target._point_history.payload()
    u = target.solution.x.array.copy()
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    assert target._point_history.payload() == before
    np.testing.assert_array_equal(target.solution.x.array, u)


@pytest.mark.parametrize("stage", ["prepare", "commit", "after_commit"])
def test_failed_acceptance_leaves_previous_point_state(monkeypatch, stage):
    step = build()
    step.solve(until=0.25)
    before = step._point_history.payload()
    u = step.solution.x.array.copy()
    original_commit = step.response.commit
    state = {name: value.values.copy() for name, value in step.state.committed.items()}

    def fail(*args, **kwargs):
        if stage == "after_commit":
            original_commit()
        raise RuntimeError("injected accepted publication failure")

    monkeypatch.setattr(
        step._point_history if stage == "prepare" else step.response,
        "prepare" if stage == "prepare" else "commit",
        fail,
    )
    with pytest.raises(RuntimeError):
        step.solve()
    assert step.accepted_load_factor == 0.25
    assert step._point_history.payload() == before
    np.testing.assert_array_equal(step.solution.x.array, u)
    for name, values in state.items():
        np.testing.assert_array_equal(step.state.committed[name].values, values)


def test_selection_change_and_missing_duration_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match="physical duration"):
        from test_small_strain_material_step import LinearBatchMaterial

        build(material=LinearBatchMaterial(), duration=None)
    step = build()
    step.solve(until=0.25)
    step.material_points = ()
    with pytest.raises(ValueError, match="scientific inputs changed"):
        step.save_checkpoint(tmp_path / "invalid")


def test_rejected_trials_never_enter_history():
    step = build(
        material=RelaxingMaterial(max_dt=0.2),
        increments=steps.automatic(initial=0.25, minimum=0.01),
    )
    step.solve()
    assert len(step.attempted_increments) > len(step.accepted_increments)
    assert len(step.material_point_histories[0].abscissa) == len(
        step.accepted_increments
    )


@pytest.mark.parametrize(
    "selection", [(), (results.material_point("changed", cell=0),)]
)
def test_restart_cannot_relabel_or_remove_selection(tmp_path, selection):
    source = build()
    source.solve(until=0.5)
    path = source.save_checkpoint(tmp_path / "selected")
    target = make_step(material_points=selection)
    with pytest.raises(ValueError, match="identity differs"):
        target.load_checkpoint(path)


def test_complete_result_rejects_changed_selection():
    step = build()
    step.solve()
    step.material_points = ()
    with pytest.raises(ValueError, match="scientific inputs changed"):
        step.solve_result()
