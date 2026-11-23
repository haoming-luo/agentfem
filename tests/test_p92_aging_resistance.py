from dataclasses import asdict, replace
from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tools import p92_aging_resistance_reference as aging
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_aging_resistance_probe import candidates
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_recoverable_frozen import decode_material
from tools.p92_spectrum_recovery_joint import decode_aging, identify


def test_substitution_decoder_and_frozen_identity():
    original = material().spectrum
    decoded = decode_aging(asdict(original), np.zeros(22))
    assert decoded == dict(candidates(original))["rate-target-memory"]
    module, restored = decode_material(asdict(decoded), aging.MODEL_ID)
    assert module is aging and restored == decoded
    coordinates = np.zeros(22)
    coordinates[7:9] = [.1, -.1]
    moved = decode_aging(asdict(original), coordinates)
    assert moved.spectrum == decoded.spectrum
    assert moved.sensitivity == pytest.approx(.1*10**.1)
    assert moved.build == pytest.approx(250.*10**(-.1))


@pytest.mark.parametrize("other", ["rate_recovery", "shared_history", "directional", "primary",
    "recoverable_kinetics", "transient_strength", "replacement_space", "recoverable_mobility",
    "critical_recovery", "recovery_shape"])
def test_identification_isolates_replacement(other):
    with pytest.raises(ValueError, match="isolate"):
        identify({"model_id": spectrum.MODEL_ID}, Path("unused"), (0., 0.),
                 aging_resistance=True, **{other: True})


def material(sensitivity=.1):
    b = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return aging.Material(spectrum.Material(Material(10000., (b, b)),
                                           (20000., 40000.), (.5, 10.)), sensitivity, 250.)


def test_no_extra_free_coefficients_in_replacement():
    arms = dict(candidates(material().spectrum))
    assert arms["original"].sensitivity == arms["deletion-only"].sensitivity == 0
    for name in ("deletion-only", "rate-target-memory"):
        m = arms[name]
        assert m.spectrum.backbone.branches[0].static_recovery == (0., 0.)
        assert m.spectrum.moduli == material().spectrum.moduli
        assert m.spectrum.times == material().spectrum.times
    assert len(aging.STATE_LAYOUT) == 13


def test_zero_sensitivity_exact_reduction_and_zero_flow_limit():
    m = material(0.)
    y = np.zeros(13)
    y[:3] = [100., 2., -3.]
    for stress_control in (False, True):
        result = aging.rates(y, .001, m, stress_control=stress_control)
        np.testing.assert_array_equal(result[:12], spectrum.rates(y[:12], .001, m.spectrum,
                                                                  stress_control=stress_control))
        assert result[12] == 0.
    assert aging.memory_rate(0., .2, 1e-5, .1, 250.) == 0.
    assert abs(aging.memory_rate(1e-300, .2, 1e-5, .1, 250.)) < 1e-290


def test_same_accumulated_strain_different_rate_history():
    # Exact constant-rate solution in accumulated strain, not a P92 fit.
    dp, build, slope, ref = .002, 250., .1, 1e-5
    slow_target = slope*np.log10(ref/1e-7)
    fast_target = slope*np.log10(ref/1e-3)
    slow = slow_target*(1-np.exp(-build*dp))
    fast = fast_target*(1-np.exp(-build*dp))
    assert slow > 0 > fast
    assert aging.memory_rate(1e-7, slow_target, ref, slope, build) == pytest.approx(0., abs=1e-18)
    assert aging.memory_rate(1e-3, fast_target, ref, slope, build) == pytest.approx(0., abs=1e-18)
    assert aging.memory_rate(1., -1., ref, slope, build) >= 0
    for speed, exact in ((1e-7, slow), (1e-3, fast)):
        solution = solve_ivp(lambda t, y, speed=speed: [aging.memory_rate(speed, y[0], ref, slope, build)],
                             (0., dp/speed), [0.], rtol=1e-10, atol=1e-12)
        assert solution.success
        assert solution.y[0, -1] == pytest.approx(exact, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("increment", [-1., -.1, 0., .3])
def test_mechanical_power_symmetry_and_nonnegative_dissipation(increment):
    m = material()
    y = np.array([100., 2., -3., .01, 1., 4., .02, .04, 3., .1, .001, -.002, increment])
    r = aging.rates(y, .001, m)
    power = y[0]*r[0]/m.spectrum.backbone.modulus
    for off, branch in zip((1, 4), m.spectrum.backbone.branches, strict=True):
        power += np.sum(y[off:off+2]*r[off:off+2]/branch.hardening)
    power += np.sum(np.asarray(m.spectrum.moduli)*y[10:12]*r[10:12])
    assert power+r[9] == pytest.approx(r[8], abs=1e-12)
    assert r[9] >= 0
    signed = [0, 1, 2, 4, 5, 7, 10, 11]
    mirrored = y.copy()
    mirrored[signed] *= -1
    r[signed] *= -1
    np.testing.assert_allclose(aging.rates(mirrored, -.001, m), r)


def test_independent_integration_continuation_and_input_retained():
    m, old = material(), np.zeros(13)
    first = aging.integrate(old, 2., .01, m).y[:, -1]
    full = aging.integrate(first, 2., -.01, m).y[:, -1]
    other = aging.integrate(first, 2., -.01, m, method="BDF", rtol=1e-10).y[:, -1]
    half = aging.integrate(first, 1., -.01, m).y[:, -1]
    split = aging.integrate(half, 1., -.01, m).y[:, -1]
    np.testing.assert_allclose(full, other, atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(full, split, atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(old, np.zeros(13))
    assert full[12] > -1


def test_free_recovery_preserves_memory_without_creating_permanent_strain():
    m = material()
    old = np.zeros(13)
    old[10:12] = [.001, -.0003]
    old[7], old[12] = sum(old[10:12]), .04
    end = aging.integrate(old, 3., 0., m, stress_control=True).y[:, -1]
    expected = old[10:12]*np.exp(-3./np.asarray(m.spectrum.times))
    np.testing.assert_allclose(end[10:12], expected, rtol=1e-8, atol=1e-12)
    assert end[7] == pytest.approx(sum(expected), abs=1e-12)
    np.testing.assert_array_equal(end[[0, 3, 6, 8]], 0.)
    assert end[12] == old[12]
    assert end[9] == pytest.approx(aging.stored_energy(old, m)-aging.stored_energy(end, m), abs=1e-10)


def test_invalid_memory_and_combined_hypotheses_rejected():
    with pytest.raises(ValueError):
        aging.Material(replace(material().spectrum, directional_recovery=True), .1, 250.)
    old = np.zeros(13)
    old[12] = -1.1
    with pytest.raises(ValueError):
        aging.integrate(old, 1., .01, material())


def test_early_evidence_keeps_conflicting_observations():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    result = json.loads((root/"2026-11-23-aging-resistance-probe.json").read_text())
    raw = (root/"2026-11-23-rate-recovery-control-frozen.json").read_bytes()
    assert result["seed_sha256"] == sha256(raw).hexdigest()
    assert result["source_sha256"] == json.loads(raw)["source_sha256"]
    assert result["training_cycles"] == [1, 2, 3]
    assert result["added_independent_coefficients"] == 0
    assert result["substituted_coordinates"] == ["A.static1", "A.static2"]
    assert not result["life_qualified"]
    original, deletion, candidate = result["records"]
    assert abs(deletion["cost"]/original["cost"]-1) < .001
    assert .89 < candidate["cost"]/original["cost"] < .90
    for arm in (original, deletion, candidate):
        assert arm["completed"]
        for protocol in arm["protocols"].values():
            assert protocol["response_screen"]["status"] == "failed"
            assert [r["cycle"] for r in protocol["comparison"]["records"]] == [1, 2, 3]
    a, b = (arm["protocols"]["SLC"]["hold_diagnostics"][2]["fields"]["stress_MPa"]
            for arm in (original, candidate))
    assert a["last_predicted"] < b["last_predicted"] < a["last_observed"]
    assert b["first_residual"] > a["first_residual"] > 0


def test_matched_fit_keeps_budget_and_replacement_accounting():
    from tools.p92_matched_response_report import check_pair

    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    control = json.loads((root/"2026-11-23-aging-resistance-control-matched.json").read_text())
    candidate = json.loads((root/"2026-11-23-aging-resistance-matched.json").read_text())
    check_pair(control, candidate)
    assert candidate["history_audit"]["passed"]
    assert candidate["model_id"] == aging.MODEL_ID
    assert candidate["coordinates"][7:9] == ["aging.sensitivity", "aging.build"]
    parameters = candidate["fit"]["parameters"]
    assert parameters["spectrum"]["backbone"]["branches"][0]["static_recovery"] == [0., 0.]
    assert len(parameters["spectrum"]["moduli"]) == len(parameters["spectrum"]["times"]) == 2
    assert .88 < candidate["fit"]["cost"]/control["fit"]["cost"] < .89
    assert not candidate["engineering_qualified"] and not candidate["life_qualified"]


def test_frozen_evidence_preserves_unresolved_cross_protocol_conflict():
    from tools.p92_matched_response_report import compare

    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    stems = ["-control-matched", "-matched", "-control-frozen", "-frozen"]
    raw = [(root/f"2026-11-23-aging-resistance{s}.json").read_bytes() for s in stems]
    result = compare(*(json.loads(r) for r in raw), *(sha256(r).hexdigest() for r in raw[:2]))
    recorded = json.loads((root/"2026-11-23-aging-resistance-paired.json").read_text())
    assert result == {k: v for k, v in recorded.items() if k != "input_sha256"}
    for arm in (result["control"], result["candidate"]):
        assert all(p["numerical_passed"] for p in arm["protocols"].values())
        assert all(p["status"] == "failed" for p in arm["protocols"].values())
    a, b = result["control"]["protocols"], result["candidate"]["protocols"]
    assert b["SLR"]["peaks"][-1]["peak_relative_error"] > a["SLR"]["peaks"][-1]["peak_relative_error"] > .05
    ha, hb = (p["SLC"]["holds"][2]["fields"]["stress_MPa"] for p in (a, b))
    assert ha["last_predicted"] < hb["last_predicted"] < hb["last_observed"]
    assert hb["first_residual"] > ha["first_residual"] > 0
    assert not any(result[k] for k in ("engineering_qualified", "life_qualified", "independent_validation"))
