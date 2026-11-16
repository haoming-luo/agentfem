"""Nominal loading must preserve history and report incomplete paths honestly."""

from types import SimpleNamespace
import numpy as np
import pytest

from tools import p92_nominal_cycles as module
from tools.p92_published_recovery_reference import published_parameters


def test_continuous_state_and_published_waveform(monkeypatch):
    calls = []
    def integrate(old, duration, rate, parameters, **kwargs):
        calls.append((old.copy(), duration, rate))
        new = old.copy()
        new[4] += .01
        return SimpleNamespace(y=np.column_stack((old, new)))
    monkeypatch.setattr(module, "integrate_segment", integrate)
    initial = np.zeros(5)
    report = module.nominal_cycles(published_parameters(600), initial, cycles=2)
    assert report["completed_cycles"] == 2
    assert report["failure"] is None
    assert calls[5][0][4] == pytest.approx(.05)
    assert report["last_completed_state"][4] == pytest.approx(.1)
    assert sum(duration for _, duration, _ in calls[:5]) == 368.
    assert sum(duration*rate for _, duration, rate in calls[:5]) == 0.
    np.testing.assert_array_equal(initial, np.zeros(5))


def test_failed_segment_does_not_publish_partial_cycle(monkeypatch):
    counter = 0
    def integrate(old, *args, **kwargs):
        nonlocal counter
        counter += 1
        if counter == 8:
            raise RuntimeError("domain rejected")
        new = old.copy()
        new[4] += .01
        return SimpleNamespace(y=np.column_stack((old, new)))
    monkeypatch.setattr(module, "integrate_segment", integrate)
    report = module.nominal_cycles(published_parameters(600), np.zeros(5), cycles=3)
    assert report["completed_cycles"] == 1
    assert report["failure"] == {"cycle": 2, "segment": 2, "reason": "domain rejected"}
    assert report["last_completed_state"][4] == pytest.approx(.05)
    assert not report["engineering_qualified"]


@pytest.mark.parametrize("cycles", [True, 0, -1, 1.5])
def test_invalid_cycle_requests(cycles):
    with pytest.raises(ValueError):
        module.nominal_cycles(published_parameters(600), np.zeros(5), cycles=cycles)
