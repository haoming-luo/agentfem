# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Scalar energy observations preserve roles, atomicity and bounded snapshots."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentfem.results._accepted_energy_components import AcceptedEnergyComponents


def station(i, n=1000):
    return SimpleNamespace(index=i, load_factor=i / n)


def recorder():
    return AcceptedEnergyComponents(
        {"elastic": "stored", "plastic": "cumulative_dissipation"},
        lambda s: {"elastic": float(s.index**2), "plastic": float(s.index)},
    )


@pytest.mark.parametrize("count", [1, 63, 64, 65, 129, 1000])
def test_constant_size_snapshot_can_restore_alternative_history(count):
    history = recorder()
    history.reset(station(0))
    for i in range(1, count):
        history.accept(station(i))
    token = history.snapshot_runtime_state()
    expected = history.checkpoint_state()
    assert len(token["tail"]) < 64
    history.accept(station(count))
    future = history.snapshot_runtime_state()
    future_state = history.checkpoint_state()
    history.reset(station(0))
    history.restore_runtime_state(token)
    assert history.checkpoint_state() == expected
    history.restore_runtime_state(future)
    assert history.checkpoint_state() == future_state


def test_snapshot_and_accept_do_not_materialize_all_frames(monkeypatch):
    history = recorder()
    history.reset(station(0))
    for i in range(1, 130):
        history.accept(station(i))
    def forbidden(self):
        raise AssertionError("Scanned accepted energy history during normal advancement")
    monkeypatch.setattr(AcceptedEnergyComponents, "frames", property(forbidden))
    token = history.snapshot_runtime_state()
    history.accept(station(130))
    history.restore_runtime_state(token)
    assert history.summary()["frame_count"] == 130


def test_durable_roundtrip_and_missing_history_stays_incomplete():
    source = recorder()
    source.reset(station(0))
    for i in range(1, 4):
        source.accept(station(i))
    restored = recorder()
    restored.restore_checkpoint_state(source.checkpoint_state(), current_snapshot=station(3))
    restored.prepare_resume(station(3))
    assert restored.checkpoint_state() == source.checkpoint_state()
    legacy = recorder()
    legacy.restore_missing_checkpoint_state(current_snapshot=station(3))
    legacy.accept(station(4))
    resumed = recorder()
    resumed.restore_checkpoint_state(legacy.checkpoint_state(), current_snapshot=station(4))
    assert not resumed.complete and len(resumed.frames) == 2


@pytest.mark.parametrize("fault", ["roles", "nan", "negative", "decrease", "skip", "start", "end", "value", "complete"])
def test_corrupt_checkpoint_rejected_atomically(fault):
    source = recorder()
    source.reset(station(0))
    source.accept(station(1))
    source.accept(station(2))
    payload = deepcopy(source.checkpoint_state())
    if fault == "roles":
        payload["roles"]["plastic"] = "stored"
    elif fault == "nan":
        payload["frames"][1]["values"]["elastic"] = float("nan")
    elif fault == "negative":
        payload["frames"][1]["values"]["plastic"] = -1.
    elif fault == "decrease":
        payload["frames"][1]["values"]["plastic"] = 3.
    elif fault == "skip":
        del payload["frames"][1]
    elif fault == "start":
        payload["frames"][0]["factor"] = .0001
    elif fault == "end":
        payload["frames"][-1]["factor"] = .003
    elif fault == "value":
        payload["frames"][-1]["values"]["elastic"] = 5.
    else:
        payload["complete"] = "yes"
    target = recorder()
    target.reset(station(0))
    before = target.checkpoint_state()
    with pytest.raises(ValueError):
        target.restore_checkpoint_state(payload, current_snapshot=station(2))
    assert target.checkpoint_state() == before


def test_loss_monotonicity_does_not_prevent_elastic_unloading():
    history = AcceptedEnergyComponents({"energy": "stored"}, lambda s: {"energy": 2. - s.index})
    history.reset(station(0))
    history.accept(station(1))
    assert history.frames[-1].values == (1.,)
    with pytest.raises(ValueError, match="consecutive"):
        history.accept(station(3))
