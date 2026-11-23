from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as spectrum
from tools import p92_aging_resistance_reference as aging
from tools.p92_rate_fingerprint import components, inverse, fingerprint, evaluate, RATES
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(20., 1e-5, 3., (100., 200.), (100., 200.), (.01, .02))
    return spectrum.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


def test_analytic_equal_branch_fingerprint_and_unchanged_input():
    m, y = material(), np.zeros(12)
    y[[1, 4]] = 30.
    old = y.copy()
    result = fingerprint(y, spectrum, m)
    assert result["inactive_interval_MPa"] == [10., 50.]
    for row in result["records"]:
        expected = 20.+120.*(row["rate_per_s"]/(2e-5))**(1/3)
        assert row["irreversible"]["center_MPa"] == pytest.approx(30., abs=1e-8)
        assert row["irreversible"]["half_width_MPa"] == pytest.approx(expected, abs=1e-8)
    np.testing.assert_array_equal(y, old)


def test_recoverable_history_masks_rate_not_irreversible_strength():
    m, y = material(), np.zeros(12)
    original = fingerprint(y, spectrum, m)
    y[10:12] = [.001, -.0002]
    changed = fingerprint(y, spectrum, m)
    for a, b in zip(original["records"], changed["records"], strict=True):
        assert a["irreversible"] == b["irreversible"]
        assert b["total"]["center_MPa"] > a["total"]["center_MPa"]


def test_opposing_branch_cancellation_is_not_an_inactive_interval():
    m, y = material(), np.zeros(12)
    y[1], y[4] = 50., -50.
    result = fingerprint(y, spectrum, m)
    assert result["inactive_interval_MPa"] is None
    rates = components(y, 0., spectrum, m)
    assert rates["irreversible"] == 0.
    assert rates["branch_rates"][0] < 0 < rates["branch_rates"][1]


def test_reflection_and_aging_zero_limit():
    m, y = material(), np.zeros(12)
    y[[1, 4, 10, 11]] = [10., -5., .001, -.0002]
    mirrored = y.copy()
    mirrored[[0, 1, 2, 4, 5, 7, 10, 11]] *= -1
    a, b = fingerprint(y, spectrum, m), fingerprint(mirrored, spectrum, m)
    for x, z in zip(a["records"], b["records"], strict=True):
        for channel in ("irreversible", "total"):
            assert z[channel]["center_MPa"] == pytest.approx(-x[channel]["center_MPa"], abs=1e-8)
            assert z[channel]["half_width_MPa"] == pytest.approx(x[channel]["half_width_MPa"], abs=1e-8)
    assert fingerprint(np.r_[y, 0.], aging, aging.Material(m, 0., 250.)) == a


def test_bracket_and_zero_rate_are_not_silently_interpreted():
    for target in (0., float("nan"), 1e20):
        with pytest.raises(ValueError):
            inverse(np.zeros(12), spectrum, material(), target, "irreversible")
    with pytest.raises(ValueError):
        components(np.zeros(11), 0., spectrum, material())


def test_frozen_diagnostic_never_uses_later_cycles_or_changes_card():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    frozen = json.loads((root/"2026-11-23-aging-resistance-control-frozen.json").read_text())
    before = deepcopy(frozen)
    a = evaluate(frozen)
    assert frozen == before
    for protocol in frozen["protocols"].values():
        prediction = protocol["prediction"]
        for s in prediction["segments"]:
            if s["cycle"] > 3:
                s["end_state"] = [float("nan")]*12
    assert evaluate(frozen) == a
    assert len(a["records"]) == 24 and a["cycles"] == [1, 2, 3]
    intermediate = [r for r in a["records"] if r["protocol"] == "SLR" and r["hold_duration_s"] > 1000.]
    assert len(intermediate) == 6 and all(abs(r["hold_strain"]) < 1e-12 for r in intermediate)
    assert not any(a[k] for k in ("fitted", "experimental_validation", "engineering_qualified", "life_qualified"))
    assert all([r["rate_per_s"] for r in row["records"]] == list(RATES) for row in a["records"])


@pytest.mark.parametrize("label,source", [("control", "control-frozen"), ("aging", "frozen")])
def test_archived_fingerprint_identities_and_independent_roots(label, source):
    root = Path(__file__).resolve().parents[1]
    directory = root/"evidence"/"creep"
    report = json.loads((directory/f"2026-11-23-rate-fingerprint-{label}.json").read_text())
    raw = (directory/f"2026-11-23-aging-resistance-{source}.json").read_bytes()
    assert report["frozen_sha256"] == sha256(raw).hexdigest()
    for name, digest in report["driver_sha256"].items():
        assert sha256((root/"tools"/name).read_bytes()).hexdigest() == digest
    assert len(report["records"]) == 24
    for station in report["records"]:
        for row in station["records"]:
            for channel in ("irreversible", "total"):
                for sign in ("negative", "positive"):
                    check = row[channel][sign]
                    assert check["independent_difference_MPa"] <= 1e-6
                    assert check["residual_per_s"] <= 1e-12+1e-7*row["rate_per_s"]
    assert not report["experimental_validation"]
