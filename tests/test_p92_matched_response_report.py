from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tools.p92_matched_response_report import check_pair, response_summary, fit_identity


def card():
    return dict(source_sha256={"RF": "source"}, seed_sha256="seed", training_cycles=[1, 2, 3],
        maximum_log_coordinate_norm=.3, driver_sha256={"driver": "hash"}, coordinates=list(range(22)),
        model_id="research", fit=dict(success=True, coordinates=[0.]*10, nfev=4,
        parameters={}, cost=.1, residual_evaluations=169))


def test_same_budget_does_not_require_same_number_of_calls():
    a, b = card(), card()
    b["fit"].update(nfev=8, residual_evaluations=253)
    check_pair(a, b)


@pytest.mark.parametrize("key,value", [("seed_sha256", "other"), ("training_cycles", [1, 2, 3, 4]),
    ("source_sha256", {}), ("maximum_log_coordinate_norm", .6), ("driver_sha256", {})])
def test_unmatched_experiments_rejected(key, value):
    a, b = card(), card()
    b[key] = value
    with pytest.raises(ValueError, match="Unmatched"):
        check_pair(a, b)


def test_nonconverged_or_overspent_fit_cannot_be_compared():
    for change in (dict(success=False), dict(nfev=41), dict(coordinates=[0.]*11),
                   dict(cost=0.), dict(cost=float("nan")), dict(nfev=0),
                   dict(coordinates=[.1]*10), dict(coordinates=[float("nan")]*10),
                   dict(residual_evaluations=1)):
        a, b = card(), card()
        b["fit"].update(change)
        with pytest.raises(ValueError, match="converged"):
            check_pair(a, b)


def test_failed_protocols_are_retained_not_dropped():
    fit = card()
    frozen = dict(fit_file_sha256="digest", parameters={}, model_id="research",
                  source_sha256=fit["source_sha256"], protocols={p: dict(failure={"stage": "prediction"})
                  for p in ("RF", "SLR", "SLC")})
    result = response_summary(fit, frozen, "digest")
    assert len(result["protocols"]) == 3
    assert all(v["status"] == "failed" for v in result["protocols"].values())
    wrong = deepcopy(frozen)
    wrong["parameters"] = {"modified": 1}
    with pytest.raises(ValueError, match="does not belong"):
        response_summary(fit, wrong, "digest")


def test_real_frozen_checks_are_recomputed_not_trusted_as_labels():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    fit = json.loads((root/"2026-11-22-recovery-shape-matched.json").read_text())
    frozen = json.loads((root/"2026-11-22-recovery-shape-frozen.json").read_text())
    summary = response_summary(fit, frozen, fit["archived_source_sha256"])
    assert set(summary["protocols"]) == {"RF", "SLR", "SLC"}
    assert all(p["status"] == "failed" and p["numerical_passed"]
               for p in summary["protocols"].values())
    slc = summary["protocols"]["SLC"]
    assert not slc["observation_resolution_ready"]
    assert len(slc["unresolved_observations"]) == 10
    assert all(c["name"] == "late_strain_secant" for c in slc["unresolved_observations"])
    frozen["protocols"]["SLC"]["response_screen"]["status"] = "passed"
    with pytest.raises(ValueError, match="do not reproduce"):
        response_summary(fit, frozen, fit["archived_source_sha256"])


def test_lossless_archived_fit_is_reproducible_without_original_temp_file():
    raw = (json.dumps(card(), indent=2)+"\n").encode()
    fit, digest = fit_identity(raw)
    assert digest == sha256(raw).hexdigest()
    archived = dict(fit, archived_source_sha256=digest)
    assert fit_identity(json.dumps(archived).encode())[1] == digest
    archived["fit"]["cost"] = .01
    with pytest.raises(ValueError, match="lossless"):
        fit_identity(json.dumps(archived).encode())


def test_real_archived_fit_has_verified_original_identity():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    for stem in ("recovery-shape", "recovery-shape-control"):
        fit, digest = fit_identity((root/f"2026-11-22-{stem}-matched.json").read_bytes())
        frozen = json.loads((root/f"2026-11-22-{stem}-frozen.json").read_text())
        assert response_summary(fit, frozen, digest)["cost"] == fit["fit"]["cost"]
        frozen["protocols"]["SLC"]["prediction"]["source_sha256"] = "another source"
        with pytest.raises(ValueError, match="identity"):
            response_summary(fit, frozen, digest)
