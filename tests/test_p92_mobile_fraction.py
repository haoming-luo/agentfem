from dataclasses import replace

import numpy as np
import pytest

from tools import p92_mobile_fraction_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_mobile_fraction_probe import candidates
from tools.p92_two_rate_reference import Branch, Material


def base():
    b = Branch(20., 1e-5, 3., (100., 200.), (100., 200.), (.01, .02))
    return spectrum.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("fraction", [0., .2, .8])
def test_literal_mobility_scaling_matches_and_retains_recoverable_modes(control, fraction):
    s = base()
    y = np.array([200., 10., -5., .01, 5., -2., .02, .03, 10., 1., .001, -.001, fraction])
    before = y.copy()
    reference = replace(s, backbone=replace(s.backbone, branches=tuple(
        replace(b, reference_flow=b.reference_flow*(1-fraction)) for b in s.backbone.branches)))
    result = kernel.rates(y, .001, kernel.Material(s, .01, 250.), stress_control=control)
    np.testing.assert_allclose(result[:12], spectrum.rates(y[:12], .001, reference, stress_control=control), atol=1e-13)
    np.testing.assert_array_equal(y, before)


def test_zero_flow_analytic_aging_and_fully_blocked_recovery():
    m = kernel.Material(base(), .01, 250.)
    y = np.zeros(13)
    sol = kernel.integrate(y, 100., 0., m, stress_control=True)
    assert sol.y[12, -1] == pytest.approx(1.-np.exp(-1.), abs=1e-9)
    y[[1, 10, 11, 12]] = [10., .001, .002, 1.]
    dy = kernel.rates(y, 0., m, stress_control=True)
    assert dy[3] == dy[6] == dy[12] == 0.
    assert dy[10] < 0 and dy[11] < 0 and dy[1] < 0


@pytest.mark.parametrize("control", [False, True])
def test_power_identity_symmetry_and_nonnegative_dissipation(control):
    m, y = kernel.Material(base(), .01, 250.), np.zeros(13)
    y[[0, 1, 2, 4, 5, 10, 11, 12]] = [200., 10., -5., 3., 2., .001, -.002, .4]
    dy = kernel.rates(y, .002, m, stress_control=control)
    grad = np.zeros(13)
    grad[0] = y[0]/m.spectrum.backbone.modulus
    for offset, b in zip((1, 4), m.spectrum.backbone.branches, strict=True):
        grad[offset:offset+2] = y[offset:offset+2]/b.hardening
    grad[10:12] = np.array(m.spectrum.moduli)*y[10:12]
    assert dy[8]-grad@dy == pytest.approx(dy[9], abs=1e-12)
    assert dy[9] >= 0
    signed = [0, 1, 2, 4, 5, 7, 10, 11]
    reflected = y.copy()
    reflected[signed] *= -1
    expected = dy.copy()
    expected[signed] *= -1
    np.testing.assert_allclose(kernel.rates(reflected, -.002, m, stress_control=control), expected)


def test_continuation_independent_integration_and_zero_limit():
    m, y = kernel.Material(base(), .001, 250.), np.zeros(13)
    a = kernel.integrate(y, 10., .001, m)
    b = kernel.integrate(y, 10., .001, m, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], rtol=1e-6, atol=1e-8)
    first = kernel.integrate(y, 5., .001, m).y[:, -1]
    end = kernel.integrate(first, 5., .001, m).y[:, -1]
    np.testing.assert_allclose(end, a.y[:, -1], rtol=1e-7, atol=1e-8)
    zero = kernel.integrate(y, 10., .001, kernel.Material(base(), 0., 250.)).y[:, -1]
    original = spectrum.integrate(y[:12], 10., .001, base()).y[:, -1]
    np.testing.assert_allclose(zero[:12], original, rtol=1e-7, atol=1e-9)
    assert zero[12] == 0 and not y.any()


def test_replacement_budget_and_invalid_input():
    original, deleted, proposed = candidates(base())
    assert original[1].spectrum == base()
    assert proposed[1].spectrum == deleted[1].spectrum
    assert proposed[1].spectrum.backbone.branches[0].static_recovery == (0., 0.)
    for z in (-.01, 1.01, float("nan")):
        y = np.zeros(13)
        y[12] = z
        with pytest.raises(ValueError):
            kernel.integrate(y, 1., 0., proposed[1])


def test_population_fixed_backbone_equilibria_expose_kinetic_cutoff():
    m, y = kernel.Material(base(), .001, 250.), np.zeros(13)
    y[0] = 200.
    unblocked = kernel.rates(y, 0., m, stress_control=True)
    speed = unblocked[3]+unblocked[6]
    y[12] = m.immobilization/(m.remobilization*speed)
    dy = kernel.rates(y, 0., m, stress_control=True)
    assert 0 < y[12] < 1 and dy[12] == pytest.approx(0., abs=1e-15)
    assert dy[3]+dy[6] == pytest.approx(speed-m.immobilization/m.remobilization)
    y[12] = 1.
    assert kernel.rates(y, 0., m, stress_control=True)[12] == 0.


def test_decoder_replaces_exactly_two_coordinates_and_rejects_mixing():
    from dataclasses import asdict
    from tools.p92_spectrum_recovery_joint import decode_mobile, identify
    card = asdict(base())
    a = decode_mobile(card, np.zeros(22))
    z = np.zeros(22)
    z[7:9] = [.1, -.1]
    b = decode_mobile(card, z)
    assert a.spectrum == b.spectrum
    assert b.immobilization/a.immobilization == pytest.approx(10**.1)
    assert b.remobilization/a.remobilization == pytest.approx(10**-.1)
    with pytest.raises(ValueError, match="isolate"):
        identify(dict(model_id=spectrum.MODEL_ID), None, (0., 0.),
                 mobile_fraction=True, aging_resistance=True)


def test_frozen_adapter_rejects_unidentified_card_before_source_access():
    from tools.p92_mobile_fraction_frozen import evaluate
    with pytest.raises(ValueError, match="identifiable"):
        evaluate(dict(model_id=kernel.MODEL_ID, training_cycles=[1, 2, 3],
                      fit=dict(success=True), history_audit=dict(passed=False)), None)


def test_free_recovery_drains_both_modes_without_permanent_flow():
    m, y = kernel.Material(base(), .001, 250.), np.zeros(13)
    y[10:12] = [.001, -.002]
    original = y.copy()
    end = kernel.integrate(y, 100., 0., m, stress_control=True).y[:, -1]
    assert not end[[3, 6]].any()
    assert abs(end[10]) < 1e-9 and abs(end[11]) < 1e-6
    assert end[9] > 0 and kernel.stored_energy(end, m) < kernel.stored_energy(y, m)
    np.testing.assert_array_equal(y, original)


def test_screen_preserves_conflicting_response_and_deletion_control():
    import json
    from hashlib import sha256
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/"evidence/creep/2026-11-23-mobile-fraction-probe.json").read_text())
    original, deleted, proposed = report["records"]
    assert all(r["completed"] for r in report["records"])
    assert abs(deleted["cost"]-original["cost"]) < 1e-6
    assert proposed["cost"] > original["cost"]
    def hold(row):
        return next(h["fields"]["stress_MPa"] for h in row["protocols"]["SLC"]["hold_diagnostics"]
                    if h["cycle"] == 1 and h["segment"] == 5)
    a, b = hold(original), hold(proposed)
    assert abs(b["last_residual"]) < abs(a["last_residual"])
    assert abs(b["first_residual"]) > abs(a["first_residual"])
    assert proposed["protocols"]["SLR"]["cycles"][2]["sampled_peak_MPa"] > original["protocols"]["SLR"]["cycles"][2]["sampled_peak_MPa"]
    assert all(p["response_screen"]["status"] == "failed"
               for row in report["records"] for p in row["protocols"].values())
    assert not report["fitted"] and report["added_independent_coefficients"] == 0
    for name, digest in report["driver_sha256"].items():
        assert sha256((root/"tools"/name).read_bytes()).hexdigest() == digest


def test_matched_identification_preserves_budget_and_parameter_replacement():
    import json
    from hashlib import sha256
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/"evidence/creep"
    result = json.loads((directory/"2026-11-23-mobile-fraction-matched.json").read_text())
    assert result["seed_sha256"] == sha256((directory/"2026-11-23-rate-recovery-control-frozen.json").read_bytes()).hexdigest()
    assert result["training_cycles"] == [1, 2, 3]
    assert len(result["coordinates"]) == 22
    assert result["coordinates"][7:9] == ["mobile.immobilization", "mobile.remobilization"]
    assert result["maximum_log_coordinate_norm"] == .3
    assert result["history_audit"]["passed"] and result["fit"]["success"]
    assert 1 <= result["fit"]["nfev"] <= 40
    assert len(result["fit"]["coordinates"]) == 10
    assert np.linalg.norm(result["fit"]["log_multipliers"]) <= .3+1e-12
    s = result["fit"]["parameters"]["spectrum"]
    assert s["backbone"]["branches"][0]["static_recovery"] == [0., 0.]
    assert len(s["times"]) == len(s["moduli"]) == 2
    assert not result["engineering_qualified"] and not result["life_qualified"]
    for name, digest in result["driver_sha256"].items():
        assert sha256((root/"tools"/name).read_bytes()).hexdigest() == digest


def test_paired_frozen_evidence_retains_failed_cross_protocol_gate():
    import json
    from hashlib import sha256
    from pathlib import Path
    from tools.p92_matched_response_report import compare
    root = Path(__file__).resolve().parents[1]/"evidence/creep"
    names = ("control-matched", "matched", "control-frozen", "frozen")
    raw = [(root/f"2026-11-23-mobile-fraction-{name}.json").read_bytes() for name in names]
    pair = compare(*(json.loads(r) for r in raw), *(sha256(r).hexdigest() for r in raw[:2]))
    saved = json.loads((root/"2026-11-23-mobile-fraction-paired.json").read_text())
    assert all(saved[k] == v for k, v in pair.items())
    assert saved["input_sha256"] == [sha256(r).hexdigest() for r in raw]
    assert pair["relative_cost_change"] > 0
    for side in ("control", "candidate"):
        assert all(p["numerical_passed"] and p["status"] == "failed"
                   for p in pair[side]["protocols"].values())
    old = pair["control"]["protocols"]["SLR"]["peaks"][-1]
    new = pair["candidate"]["protocols"]["SLR"]["peaks"][-1]
    assert abs(new["peak_relative_error"]) > abs(old["peak_relative_error"])
    assert not pair["engineering_qualified"] and not pair["life_qualified"]
