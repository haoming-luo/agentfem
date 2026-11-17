from dataclasses import asdict

import numpy as np
import pytest

from tools.p92_published_recovery_reference import published_parameters
from tools.p92_two_mechanism import Mechanism
from tools.p92_transient_history import decode, entry_probes, identify


def test_transient_scales_change_only_C1_and_preserve_seed():
    base = (Mechanism(published_parameters(600)),)*2
    assert decode([0., 0.], base) == base
    changed = decode([1., -1.], base)
    for old, new, factor in zip(base, changed, (10., .1), strict=True):
        expected = asdict(old)
        expected["parameters"]["C1"] *= factor
        assert asdict(new) == expected
    with pytest.raises(ValueError, match="two finite"):
        decode([np.nan, 0.], base)


def test_entry_probes_do_not_modify_state_or_confuse_softening_with_backstress():
    base = (Mechanism(published_parameters(600)),)*2
    state = np.array([170., 20., 5., -10., .001, -20., -5., -10., .001, .002])
    old = state.copy()
    rows = entry_probes(state, base)
    np.testing.assert_array_equal(state, old)
    for r in rows:
        assert r["zero_backstress_probe"]["effective_stress_MPa"] == 170.
        assert r["zero_isotropic_probe"]["effective_stress_MPa"] == r["actual"]["effective_stress_MPa"]
        assert r["zero_backstress_probe"]["strength_MPa"] == r["actual"]["strength_MPa"]
    assert rows[0]["actual"]["signed_flow_per_s"] < rows[1]["actual"]["signed_flow_per_s"]


def test_bad_frozen_identity_rejected_before_fit():
    with pytest.raises(ValueError, match="intact"):
        identify(dict(parameters=[], training_cycles=[1, 2, 3], fit={"success": True},
                      parameter_sha256="wrong"), [])
    with pytest.raises(ValueError, match="budget"):
        identify({}, [], max_nfev=0)


def test_time_shape_separates_initial_rate_from_target_without_state_reset():
    from tools.p92_recovery_discriminator import terms
    from tools.p92_published_recovery_reference import isotropic_strength_lower_bound
    base = (Mechanism(published_parameters(600)),)*2
    assert decode([0., 0.], base, direction="fast_time_shape") == base
    changed = decode([.5, -.7], base, direction="fast_time_shape")
    assert changed[1] == base[1]
    assert changed[0].parameters.C1 == base[0].parameters.C1
    assert changed[0].parameters.Qr == pytest.approx(base[0].parameters.Qr*10**(-.7))
    for accumulated in (0., .001, .1):
        state = np.array([170., 2., 3., 0., accumulated])
        before = terms(state, base[0])["isotropic_MPa_per_s"]["time_recovery"]
        after = terms(state, changed[0])["isotropic_MPa_per_s"]["time_recovery"]
        assert after == pytest.approx(before*10**.5)
        state[3] = changed[0].parameters.Qr
        assert terms(state, changed[0])["isotropic_MPa_per_s"]["time_recovery"] == 0.
    assert isotropic_strength_lower_bound(changed[0].parameters) > 0
    with pytest.raises(ValueError, match="Unknown"):
        decode([0., 0.], base, direction="unknown")


def test_archived_conditional_fit_keeps_unresolved_physics_and_frozen_identity():
    import json
    from hashlib import sha256
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]/"evidence/creep"
    report = json.loads((root/"2026-11-17-transient-history.json").read_text())
    seed = json.loads((root/"2026-11-17-both-time-identification.json").read_text())
    assert report["training_cycles"] == [1, 2, 3]
    assert report["seed_parameter_sha256"] == seed["parameter_sha256"]
    assert report["parameter_sha256"] == sha256(json.dumps(report["parameters"], sort_keys=True).encode()).hexdigest()
    assert not report["engineering_qualified"] and not report["life_qualified"]
    assert report["fit"]["active_bounds"][0] == -1
    for old, new in zip(seed["parameters"], report["parameters"], strict=True):
        before, after = old["parameters"].copy(), new["parameters"].copy()
        before.pop("C1")
        after.pop("C1")
        assert before == after
    for prediction in report["frozen_predictions"]:
        assert prediction["numerical_refinement_passed"]
        assert prediction["parameter_sha256"] == report["parameter_sha256"]
        assert len(prediction["cycles"]) == 10
        assert not prediction["reversed_post_hold_ramps"]
    slc = next(p for p in report["frozen_predictions"] if p["protocol"] == "SLC")
    first = next(h for h in slc["hold_diagnostics"] if h["stress_control"])["fields"]["strain"]
    assert first["early_secant_predicted_per_s"] < first["early_secant_observed_per_s"]
    assert first["last_predicted"] < first["last_observed"]


def test_time_shape_evidence_keeps_history_gap_despite_peak_improvement():
    import json
    from hashlib import sha256
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]/"evidence/creep"
    report = json.loads((root/"2026-11-17-time-shape.json").read_text())
    seed = json.loads((root/"2026-11-17-transient-history.json").read_text())
    assert report["optimizer_initial"] == [0., -.05]
    assert report["training_cycles"] == [1, 2, 3]
    assert report["seed_parameter_sha256"] == seed["parameter_sha256"]
    assert report["parameter_sha256"] == sha256(json.dumps(report["parameters"], sort_keys=True).encode()).hexdigest()
    assert report["parameters"][1] == seed["parameters"][1]
    for key, value in seed["parameters"][0]["parameters"].items():
        if key not in {"Qr", "gamma_r1", "gamma_r2"}:
            assert report["parameters"][0]["parameters"][key] == value
    assert not report["engineering_qualified"] and not report["life_qualified"]
    for prediction in report["frozen_predictions"]:
        assert prediction["numerical_refinement_passed"]
        assert prediction["parameter_sha256"] == report["parameter_sha256"]
        assert len(prediction["cycles"]) == 10
    slc = next(p for p in report["frozen_predictions"] if p["protocol"] == "SLC")
    last = [h for h in slc["hold_diagnostics"] if h["stress_control"]][-1]
    assert len(last["late_secant_source_indices"]) == 2
    assert last["fields"]["strain"]["early_secant_predicted_per_s"] > last["fields"]["strain"]["early_secant_observed_per_s"]
