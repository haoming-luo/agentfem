from copy import deepcopy
from hashlib import sha256
import json

import pytest
import numpy as np

from tools.p92_two_rate_probe import bridge_seed
from tools.p92_two_rate_sensitivity import perturb


def seed():
    p = dict(E=150000., k=100., Z=600., n=3., a1=20., a2=50.,
             C1=2000., C2=500., gamma1=1e-8, gamma2=2e-8, m=3.)
    cards = [dict(parameters=p, enabled=True, exponent=3.) for _ in range(2)]
    return dict(parameters=cards, parameter_sha256=sha256(
        json.dumps(cards, sort_keys=True).encode()).hexdigest())


def test_bridge_is_explicit_initial_scale_mapping_not_old_state_transfer():
    card = bridge_seed(seed())
    branch = card.branches[0]
    assert branch.reference_flow == pytest.approx((120/600)**3)
    assert branch.hardening == (40000., 25000.)
    assert branch.static_recovery == pytest.approx((1.2e-5, 7.5e-5))
    assert branch.strength == 100.
    for component in range(4):
        changed = perturb(card, component, 1.)
        for j in range(4):
            old = card.branches[j//2].static_recovery[j % 2]
            new = changed.branches[j//2].static_recovery[j % 2]
            assert new == pytest.approx(old*(10 if component == j else 1))


def test_seed_identity_and_recovery_direction_checks():
    original = seed()
    broken = deepcopy(original)
    broken["parameters"][0]["parameters"]["k"] = 1.
    with pytest.raises(ValueError, match="identity"):
        bridge_seed(broken)
    with pytest.raises(ValueError):
        perturb(bridge_seed(original), 4, 1.)


def test_history_grid_uses_only_three_early_cycles_and_preserves_seed(monkeypatch):
    from pathlib import Path
    from tools import p92_two_rate_history_probe as module
    original = seed()
    original["source_sha256"] = {p: p for p in ("RF", "SLR", "SLC")}
    before = deepcopy(original)
    seen = []
    monkeypatch.setattr(module, "read_source", lambda path, protocol:
                        dict(protocol=protocol, sha256=protocol))

    def controls(source, *, cycles):
        assert cycles == 3
        return source

    monkeypatch.setattr(module, "measured_path", controls)
    monkeypatch.setattr(module, "attach_waveforms", lambda path, source: path)

    def evaluate(path, card):
        seen.append((path["protocol"], card.branches[0].saturation_ratio,
                     card.branches[0].softening_rate))
        return np.array([1.]), {"synthetic": True}

    monkeypatch.setattr(module, "replay", evaluate)
    result = module.probe(original, Path("unused"))
    assert len(seen) == 15
    assert len(result["candidates"]) == 5
    assert result["training_cycles"] == [1, 2, 3]
    assert all(c["cost"] == 1.5 for c in result["candidates"])
    assert original == before
