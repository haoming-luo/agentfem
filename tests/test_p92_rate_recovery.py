from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from tools import p92_rate_recovery_reference as rate
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_frozen import decode_material
from tools.p92_spectrum_recovery_joint import identify
from tools.p92_two_rate_reference import Branch, Material


def material(chi=3.):
    branch = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02), (chi, chi))
    return rate.Material(spectrum.Material(Material(10000., (branch, branch)),
                                           (20000., 40000.), (.5, 10.)))


def test_no_parameters_or_states_added_and_roundtrip():
    mat = material()
    assert list(asdict(mat)) == ["spectrum"]
    assert len(rate.STATE_LAYOUT) == 12
    module, restored = decode_material(asdict(mat), rate.MODEL_ID)
    assert module is rate and restored == mat


def test_zero_exponent_exact_control_and_recoverability():
    mat = material(0.)
    y = np.array([100., 2., -3., .01, 1., 4., .02, .04, 3., .1, .001, -.002])
    for stress_control in (False, True):
        np.testing.assert_array_equal(rate.rates(y, .001, mat, stress_control=stress_control),
            spectrum.rates(y, .001, mat.spectrum, stress_control=stress_control))
    assert np.any(rate.rates(y, 0., mat)[10:] != 0.)


def test_normalized_branch_rate_is_not_imposed_loading_rate():
    mat = material()
    y = np.zeros(12)
    y[0] = 140.
    changed = rate.Material(replace(mat.spectrum, backbone=replace(mat.spectrum.backbone,
        branches=tuple(replace(b, reference_flow=b.reference_flow*10.)
                       for b in mat.spectrum.backbone.branches))))
    original = rate.current_material(y, mat)
    scaled = rate.current_material(y, changed)
    assert original.backbone.branches[0].ratcheting == scaled.backbone.branches[0].ratcheting
    assert original.backbone.branches[0].ratcheting == (4.5, 4.5)


@pytest.mark.parametrize("stress", [-300., -50., 0., 50., 300.])
def test_power_identity_and_inward_invariant(stress):
    mat = material()
    y = np.zeros(12)
    y[0], y[1:3], y[4:6], y[10:] = stress, [10., -20.], [-10., 20.], [.001, -.002]
    dy = rate.rates(y, .003, mat)
    energy_rate = y[0]*dy[0]/mat.spectrum.backbone.modulus
    for offset, branch in zip((1, 4), mat.spectrum.backbone.branches, strict=True):
        energy_rate += np.sum(y[offset:offset+2]*dy[offset:offset+2]/branch.hardening)
        assert np.all(y[offset:offset+2]*dy[offset:offset+2] <= 1e-12)
    energy_rate += np.sum(np.asarray(mat.spectrum.moduli)*y[10:]*dy[10:])
    assert dy[9] >= 0.
    assert energy_rate+dy[9] == pytest.approx(dy[8], abs=1e-12)


def test_sign_symmetry_rate_limits_and_history_continuation():
    mat = material()
    zero = np.zeros(12)
    assert rate.current_material(zero, mat).backbone.branches[0].ratcheting == (6., 6.)
    fast = zero.copy()
    fast[0] = 1e6
    np.testing.assert_allclose(rate.current_material(fast, mat).backbone.branches[0].ratcheting, (3., 3.))
    first = rate.integrate(zero, 2., .01, mat).y[:, -1]
    flipped = first.copy()
    flipped[[0, 1, 2, 4, 5, 7, 10, 11]] *= -1
    expected = rate.rates(first, .01, mat)
    expected[[0, 1, 2, 4, 5, 7, 10, 11]] *= -1
    np.testing.assert_allclose(rate.rates(flipped, -.01, mat), expected)
    whole = rate.integrate(first, 2., -.01, mat).y[:, -1]
    bdf = rate.integrate(first, 2., -.01, mat, method="BDF", rtol=1e-10).y[:, -1]
    half = rate.integrate(first, 1., -.01, mat).y[:, -1]
    split = rate.integrate(half, 1., -.01, mat).y[:, -1]
    np.testing.assert_allclose(whole, bdf, atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(whole, split, atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(zero, np.zeros(12))


@pytest.mark.parametrize("other", ["shared_history", "directional", "primary", "recoverable_kinetics",
    "transient_strength", "replacement_space", "recoverable_mobility", "critical_recovery", "recovery_shape"])
def test_isolated_mechanism(other):
    with pytest.raises(ValueError, match="isolate"):
        identify({"model_id": spectrum.MODEL_ID}, Path("unused"), (0., 0.),
                 rate_recovery=True, **{other: True})


def test_reject_combined_material():
    with pytest.raises(ValueError):
        rate.Material(replace(material().spectrum, directional_recovery=True))


def test_fixed_card_evidence_does_not_promote_numerical_success():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    raw = (root/"2026-11-22-recovery-shape-control-frozen.json").read_bytes()
    control = json.loads(raw)
    result = json.loads((root/"2026-11-23-rate-recovery-fixed.json").read_text())
    assert result["seed_sha256"] == sha256(raw).hexdigest()
    assert result["parameters"] == {"spectrum": control["parameters"]}
    assert result["source_sha256"] == control["source_sha256"]
    assert not any(result[k] for k in ("fitted", "engineering_qualified", "life_qualified", "independent_validation"))
    assert set(result["protocols"]) == {"RF", "SLR", "SLC"}
    for name, evidence in result["protocols"].items():
        assert evidence["numerical"]["passed"]
        assert evidence["response_screen"]["status"] == "failed"
        assert evidence["response_screen"]["policy"] == control["protocols"][name]["response_screen"]["policy"]
        assert [c["cycle"] for c in evidence["prediction"]["cycles"]] == list(range(1, 11))


def test_matched_identification_retains_failed_frozen_responses():
    from tools.p92_matched_response_report import check_pair

    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    def read(suffix):
        return json.loads((root/f"2026-11-23-rate-recovery{suffix}.json").read_text())
    control, candidate = read("-control-matched"), read("-matched")
    check_pair(control, candidate)
    paired = read("-paired")
    assert -.004 < paired["relative_cost_change"] < -.003
    for stem, fit in (("-control", control), ("", candidate)):
        frozen = read(stem+"-frozen")
        assert frozen["parameters"] == fit["fit"]["parameters"]
        assert frozen["fit_file_sha256"] == sha256(
            (root/f"2026-11-23-rate-recovery{stem}-matched.json").read_bytes()).hexdigest()
        for result in frozen["protocols"].values():
            assert result["numerical"]["passed"]
            assert result["response_screen"]["status"] == "failed"
    a, b = (read(stem+"-frozen")["protocols"]["SLC"]["prediction"]["hold_diagnostics"][2]
            ["fields"]["stress_MPa"] for stem in ("-control", ""))
    assert b["first_residual"] > a["first_residual"] > 1.
    assert b["last_predicted"] < a["last_predicted"] < a["last_observed"]
    assert not any(paired[k] for k in ("engineering_qualified", "life_qualified", "independent_validation"))
