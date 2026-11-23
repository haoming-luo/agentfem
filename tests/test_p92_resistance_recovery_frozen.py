from dataclasses import asdict

import pytest

from tools import p92_resistance_recovery_frozen as driver
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def frozen():
    b = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    m = spectrum.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))
    return dict(model_id=spectrum.MODEL_ID, parameters=asdict(m),
                source_sha256={p: p for p in ("RF", "SLR", "SLC")})


def prepare(monkeypatch):
    monkeypatch.setattr(driver, "read_source", lambda path, protocol: dict(protocol=protocol, sha256=protocol))
    monkeypatch.setattr(driver, "measured_path", lambda source, cycles: dict(protocol=source["protocol"], cycles=cycles))
    monkeypatch.setattr(driver, "attach_waveforms", lambda path, source: path)
    monkeypatch.setattr(driver, "refine_knots", lambda path: {**path, "refined": True})
    monkeypatch.setattr(driver, "compare_candidate", lambda source, prediction:
                        dict(groups={}, records=[dict(cycle=1), dict(cycle=10)]))
    monkeypatch.setattr(driver, "screen", lambda prediction, comparison, requested_cycles: dict(status="failed"))


@pytest.mark.parametrize("kind,tau", [("strength", 180.), ("mobility", 1800.)])
def test_fixed_parameters_full_state_partition_and_no_promotion(monkeypatch, tmp_path, kind, tau):
    prepare(monkeypatch)
    calls, comparisons = [], []
    def replay(path, material, **kwargs):
        calls.append((path, material, kwargs))
        return [], dict(hold_diagnostics=[], mechanical_balance_MPa=0.)
    def difference(a, b, **kwargs):
        comparisons.append(kwargs)
        return dict(passed=True)
    monkeypatch.setattr(driver, "replay", replay)
    monkeypatch.setattr(driver, "numerical_difference", difference)
    result = driver.evaluate(frozen(), tmp_path, kind)
    assert len(calls) == 9 and len(comparisons) == 6
    assert all(m.kind == kind and m.recovery == 1/tau and m.minimum == .9 for _, m, _ in calls)
    assert all(p["cycles"] == 10 for p, _, _ in calls)
    assert all(c == dict(extra_strain_indices=(10, 11), dimensionless_indices=(12, 13)) for c in comparisons)
    assert calls[1][2]["method"] == "BDF" and calls[2][0]["refined"]
    assert not result["fitted"] and not result["engineering_qualified"]
    for p in result["protocols"].values():
        assert p["numerical"]["passed"] and p["response_screen"]["status"] == "failed"
        assert p["comparison"]["records"][-1]["role"] == "inspected_development"


def test_rejected_path_retained_not_silently_dropped(monkeypatch, tmp_path):
    prepare(monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError("incompatible continuation")
    monkeypatch.setattr(driver, "replay", fail)
    result = driver.evaluate(frozen(), tmp_path, "strength")
    assert len(result["protocols"]) == 3
    assert all(p["failure"]["stage"] == "prediction" and not p["numerical"]["passed"]
               for p in result["protocols"].values())


def test_source_mismatch_not_relabelled_model_failure(monkeypatch, tmp_path):
    prepare(monkeypatch)
    source = frozen(); source["source_sha256"]["RF"] = "changed"
    with pytest.raises(ValueError, match="Source identity"):
        driver.evaluate(source, tmp_path, "mobility")
    with pytest.raises(ValueError, match="declared mechanism"):
        driver.evaluate(frozen(), tmp_path, "unknown")
