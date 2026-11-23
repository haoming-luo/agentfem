from dataclasses import asdict

import numpy as np
import pytest

from tools.p92_published_recovery_reference import published_parameters
from tools.p92_ratcheting_screen import corrected_rates
from tools.p92_two_mechanism import Mechanism
from tools.p92_recovery_discriminator import terms, variants, screen


@pytest.mark.parametrize("sign", [-1., 1.])
def test_signed_decomposition_reconstructs_existing_rate_equations(sign):
    p = published_parameters(600)
    m = Mechanism(p)
    state = np.array([sign*300., sign*15., sign*40., -30., .02])
    before = state.copy()
    actual = corrected_rates(state, 0., p, exponent=m.exponent)
    ledger = terms(state, m)
    assert -actual[0]/p.E == pytest.approx(ledger["signed_flow_per_s"])
    for i in range(2):
        assert actual[i+1] == pytest.approx(sum(ledger["backstress_MPa_per_s"][i].values()))
    assert actual[3] == pytest.approx(sum(ledger["isotropic_MPa_per_s"].values()))
    np.testing.assert_array_equal(state, before)


def test_disabled_branch_has_no_kinetic_contributions():
    ledger = terms([300., 15., 40., -30., .02], Mechanism(published_parameters(600), enabled=False))
    assert ledger["signed_flow_per_s"] == 0
    assert all(sum(v.values()) == 0 for v in ledger["backstress_MPa_per_s"])
    assert sum(ledger["isotropic_MPa_per_s"].values()) == 0


def test_ablation_preserves_baseline_and_other_parameters():
    original = (Mechanism(published_parameters(600)),)*2
    before = [asdict(m) for m in original]
    candidates = variants(original)
    assert len(candidates) == 6
    assert candidates["frozen_baseline"] is original
    slow = candidates["slow_no_time_isotropic"]
    assert slow[0] is original[0]
    assert slow[1].parameters.gamma_r1 == slow[1].parameters.gamma_r2 == 0
    assert slow[1].parameters.gamma1 == original[1].parameters.gamma1
    assert [asdict(m) for m in original] == before


def test_changed_frozen_parameters_are_rejected():
    with pytest.raises(ValueError, match="Changed frozen"):
        screen(dict(parameters=[], parameter_sha256="bad"), [])


def test_archived_ablation_is_not_promoted_for_merely_finishing_ten_cycles():
    import json
    from pathlib import Path
    r = json.loads((Path(__file__).resolve().parents[1]/
        "evidence/creep/2026-11-17-recovery-discriminator.json").read_text())
    assert len(r["records"]) == 18
    assert not r["engineering_qualified"] and not r["life_qualified"]
    index = {(q["variant"], q["protocol"]): q for q in r["records"]}
    assert index["frozen_baseline", "SLC"]["failure"][0]["cycle"] == 5
    assert index["slow_no_time_isotropic", "SLC"]["completed_cycles"] == 10
    for q in r["records"]:
        delta = np.array(q["integrator_max_state_difference"])
        assert max(delta[[0, 1, 2, 3, 5, 6, 7]]) < 1e-4
        assert max(delta[[4, 8, 9]]) < 1e-8
