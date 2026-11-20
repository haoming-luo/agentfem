from pathlib import Path

import numpy as np
import pytest

from tests.test_p92_two_rate_probe import seed
from tools.p92_two_rate_probe import bridge_seed
from tools.p92_two_rate_joint import decode


def test_joint_coordinates_do_not_introduce_hidden_history_parameters():
    original = bridge_seed(seed())
    changed = decode(original, [.1, .2, 1., -.1, .2, -.2])
    a, b = changed.branches
    assert a.strength == original.branches[0].strength+5.
    assert a.hardening == original.branches[0].hardening
    assert a.saturation_ratio == b.saturation_ratio == 1.
    assert a.softening_rate == b.softening_rate == 0.
    assert b.strength == original.branches[1].strength
    assert b.reference_flow == original.branches[1].reference_flow
    assert changed.modulus == original.modulus
    with pytest.raises(ValueError):
        decode(original, np.zeros(5))


def test_rank_deficient_observations_stop_before_optimization(monkeypatch):
    from tools import p92_two_rate_joint as module
    inputs = seed()
    inputs["source_sha256"] = {p: p for p in ("RF", "SLR", "SLC")}
    monkeypatch.setattr(module, "read_source", lambda path, protocol: dict(sha256=protocol))
    def early(source, *, cycles):
        assert cycles == 3
        return source
    monkeypatch.setattr(module, "measured_path", early)
    monkeypatch.setattr(module, "attach_waveforms", lambda path, source: path)
    monkeypatch.setattr(module, "replay", lambda path, card: (np.ones(10), {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not optimize unobservable directions")
    monkeypatch.setattr(module, "least_squares", forbidden)
    result = module.identify(inputs, Path("unused"))
    assert not result["conditional_fit"]["success"]
    assert result["initial_singular_values"] == [0.]*6
