"""Declared physical clocks are part of rate-dependent restart identity."""
import numpy as np
import pytest
import json

from test_rate_material_history_units import build, assert_same_state


def committed(step):
    snapshot = step.state.snapshot()
    # Maxwell portable restart reconstructs the response tangent for the next
    # increment; that derivative is not a committed material history variable.
    return snapshot.get("state", snapshot)


@pytest.mark.parametrize("family", ["creep", "joint", "viscoelastic"])
@pytest.mark.parametrize("portable", [False, True])
def test_checkpoint_preserves_clock_and_rejects_other_clock_atomically(tmp_path, family, portable):
    original = build(family, None, "s")
    original.solve(until=1.0 if family == "viscoelastic" else .5)
    checkpoint = original.save_checkpoint(tmp_path / "accepted", portable=portable)
    same = build(family, None, "s")
    same.load_checkpoint(checkpoint)
    assert same.accepted_time == original.accepted_time
    np.testing.assert_array_equal(same.solution.x.array, original.solution.x.array)
    assert_same_state(committed(same), committed(original))
    original.solve()
    same.solve()
    np.testing.assert_array_equal(same.solution.x.array, original.solution.x.array)
    assert_same_state(committed(same), committed(original))
    np.testing.assert_allclose(same.state.stress.values, original.state.stress.values,
                               rtol=1e-12, atol=1e-10)

    wrong = build(family, None, "h")
    before = wrong.solution.x.array.copy()
    state = wrong.state.snapshot()
    with pytest.raises(ValueError):
        wrong.load_checkpoint(checkpoint)
    assert wrong.accepted_time == 0.
    np.testing.assert_array_equal(wrong.solution.x.array, before)
    assert_same_state(wrong.state.snapshot(), state)


@pytest.mark.parametrize("family", ["creep", "joint", "viscoelastic"])
def test_undeclared_clock_does_not_gain_an_invented_unit(family):
    step = build(family, None, None)
    assert "physical_time_unit" not in step._checkpoint_identity()
    assert "physical_time_unit" not in step._portable_checkpoint_identity()


@pytest.mark.parametrize("family", ["creep", "joint", "viscoelastic"])
@pytest.mark.parametrize("portable", [False, True])
def test_bad_live_temperature_label_cannot_be_saved(tmp_path, family, portable):
    step = build(family, "s", "s")
    step.temperature.unit = "degC"
    with pytest.raises(ValueError, match="kelvin"):
        step.save_checkpoint(tmp_path / "bad", portable=portable)
    assert not list(tmp_path.iterdir())


def test_legacy_maxwell_active_sample_does_not_replace_history_identity(tmp_path):
    source = build("viscoelastic", None, None)
    source.solve(until=1.)
    saved = source.save_checkpoint(tmp_path / "original", portable=False)
    with np.load(saved, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    identity = json.loads(str(payload["step_identity"]))
    identity["temperature"] = source._temperature_summary()
    payload["step_identity"] = json.dumps(identity, sort_keys=True)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **payload)
    restored = build("viscoelastic", None, None)
    restored.load_checkpoint(legacy)
    assert restored.accepted_time == 1.
    source.solve()
    restored.solve()
    assert_same_state(committed(restored), committed(source))
