"""Preserve the distinction between numerical checks and material admission."""

import json
from pathlib import Path

import pytest

from tools.p92_response_screen import POLICY

ROOT = Path(__file__).resolve().parents[1]/"evidence"/"creep"


def read(name):
    return json.loads((ROOT/f"2026-11-22-{name}.json").read_text())


def test_history_kinetics_split_retains_signed_cancellation_not_admission():
    evidence = read("history-kinetics-split")
    assert not any(evidence[k] for k in ("fitted", "engineering_qualified", "life_qualified"))
    assert set(evidence["records"]) == {"RF", "SLR", "SLC"}
    for protocol in evidence["records"].values():
        assert len(protocol["rows"]) == 6
        for row in protocol["rows"]:
            assert row["numerical_passed"]
            assert row["two_scale_relative_error"] < .002
            assert all(d["chain_rule_relative_error"] < .002 for d in row["derivatives"])
    row = next(r for r in evidence["records"]["SLC"]["rows"]
               if r["coordinate"] == "recoverable.slow_modulus")
    fine = row["derivatives"][-1]
    assert fine["history"][2] < fine["full"][2] < 0 < fine["kinetics"][2]


def test_inviscid_mapping_cannot_be_selected_on_one_improved_secant():
    evidence = read("inviscid-branch-probe")
    assert evidence["added_coefficients"] == evidence["added_state"] == []
    assert len(evidence["inactive_original_coefficients"]) == 2
    assert not evidence["fitted"] and not evidence["engineering_qualified"]
    control, candidate = evidence["records"]
    assert all(r["completed"] for r in (control, candidate))
    assert candidate["cost"] > 8*control["cost"]
    old = control["protocols"]["SLC"]["hold_diagnostics"][2]["fields"]["stress_MPa"]
    new = candidate["protocols"]["SLC"]["hold_diagnostics"][2]["fields"]["stress_MPa"]
    assert abs(new["early_secant_predicted_per_s"]-new["early_secant_observed_per_s"]) < .1
    assert abs(new["first_residual"]) > 15.
    assert abs(old["first_residual"]) < 1.
    for protocol in candidate["protocols"].values():
        assert protocol["response_screen"]["status"] == "failed"


def test_budget_sensitivity_is_finite_diagnostic_not_frozen_validation():
    evidence = read("recovery-budget-sensitivity")
    assert len(evidence["records"]) == 211
    assert evidence["maximum_log10_coordinate_norm"] == .3
    assert not any(evidence[k] for k in ("fitted", "continuous_prediction",
                                       "engineering_qualified", "life_qualified"))
    assert "not a certified" in evidence["scope"]
    assert all(r["contrast_interval_per_s"][0] > 0 for r in evidence["records"])
    assert evidence["minimum_sampled_lower_contrast"] > 0


def test_cross_resistance_replacement_stops_after_failed_screen():
    evidence = read("cross-resistance-probe")
    assert not any(evidence[k] for k in ("fitted", "engineering_qualified", "life_qualified"))
    assert evidence["added_coefficients"] == evidence["added_state"] == []
    assert evidence["training_cycles"] == [1, 2, 3]
    control, candidate = evidence["records"]
    assert control["cost"] == pytest.approx(.0023607800272608744, abs=1e-15)
    assert candidate["cost"] > 1.3*control["cost"]
    assert control["model_id"] != candidate["model_id"]
    for record in (control, candidate):
        assert record["completed"]
        for protocol in record["protocols"].values():
            assert protocol["response_screen"]["status"] == "failed"
            assert protocol["response_screen"]["policy"] == POLICY


def test_critical_fit_does_not_promote_failed_frozen_responses():
    fit, frozen = read("critical-recovery-matched"), read("critical-recovery-frozen")
    assert fit["critical_recovery"] and fit["fit"]["success"]
    assert fit["training_cycles"] == [1, 2, 3]
    assert len(fit["coordinates"]) == 22 and len(fit["fit"]["coordinates"]) == 10
    assert fit["maximum_log_coordinate_norm"] == .3
    assert fit["fit"]["nfev"] <= 40
    assert fit["fit"]["cost"] > .0023607800272608744
    assert frozen["fit_file_sha256"] == fit["archived_source_sha256"]
    assert frozen["parameters"] == fit["fit"]["parameters"]
    assert frozen["source_sha256"] == fit["source_sha256"]
    assert not frozen["independent_validation"]
    assert not frozen["engineering_qualified"] and not frozen["life_qualified"]
    for protocol in frozen["protocols"].values():
        assert protocol["numerical"]["passed"]
        assert protocol["response_screen"]["status"] == "failed"
        assert protocol["response_screen"]["policy"] == POLICY
        assert len(protocol["comparison"]["records"]) == 10
        assert all(row["role"] == ("early_calibration" if row["cycle"] <= 3 else "inspected_development")
                   for row in protocol["comparison"]["records"])


@pytest.mark.parametrize("name,count", [("static-modes-probe", 3), ("recoverable-modes-probe", 3),
                                        ("static-exchange-probe", 2),
                                        ("kelvin-time-exchange-probe", 2)])
def test_discrete_mode_experiments_keep_all_failed_arms(name, count):
    evidence = read(name)
    assert not evidence["fitted"] and not evidence["engineering_qualified"]
    assert evidence["added_coefficients"] == evidence["added_state"] == []
    assert evidence["recovery_eigenvalues_unchanged"]
    assert evidence["training_cycles"] == [1, 2, 3]
    assert len(evidence["records"]) == count
    baseline = evidence["records"][0]["cost"]
    assert baseline == pytest.approx(.0023607800272608744, abs=1e-15)
    for arm in evidence["records"]:
        assert arm["completed"] and arm["cost"] >= baseline
        assert set(arm["protocols"]) == {"RF", "SLR", "SLC"}
        for protocol in arm["protocols"].values():
            assert protocol["response_screen"]["policy"] == POLICY


def test_conditional_state_is_never_a_prediction_or_fit_of_material():
    evidence = read("hold-state-discriminator")
    assert not evidence["material_parameters_changed"]
    assert not evidence["state_reset_for_prediction"]
    assert not evidence["engineering_qualified"] and not evidence["life_qualified"]
    for record in evidence["records"].values():
        assert record["fit"]["success"] and record["fit"]["nfev"] <= 20
        assert record["numerical"]["passed"]
        assert record["state_reconstruction"] and not record["continuous_prediction"]
        assert not record["candidate"]["all_hold_gates_passed"]
        assert abs(record["candidate"]["normalized_errors"]["late"]) > 1
        old, new = record["original_state"], record["reconstructed_state"]
        for i in (0, 3, 6, 7, 8, 9):
            assert new[i] == old[i]
        assert new[10]+new[11] == pytest.approx(old[10]+old[11], abs=1e-15)


def test_absolute_difference_reaudit_preserves_failed_conditional_conclusion():
    original = read("hold-state-discriminator")
    checked = read("hold-state-absolute-jacobian")
    assert checked["source_sha256"] == original["source_sha256"]
    assert checked["parameters"] == original["parameters"]
    for protocol, row in checked["records"].items():
        assert row["numerical"]["passed"] and row["fit"]["success"]
        assert row["optimizer_difference"] == dict(kind="bounded_absolute_coordinate", step=.0005)
        assert not row["candidate"]["all_hold_gates_passed"]
        assert row["fit"]["cost"] == pytest.approx(original["records"][protocol]["fit"]["cost"], abs=1e-8)
        assert not row["continuous_prediction"] and not row["engineering_qualified"]


def test_recorded_stress_diagnostic_cannot_claim_input_agreement_as_validation():
    evidence = read("stress-ripple-audit")
    assert evidence["numerical"]["passed"]
    assert evidence["prescribed_stress_is_not_validation"]
    assert not evidence["fitted"] and not evidence["engineering_qualified"]
    assert not evidence["acceptance_thresholds_changed"]
    assert set(evidence["results"]) == {"nominal", "relative_ripple", "independent"}
    for screen in evidence["response_screens"].values():
        assert screen["policy"] == POLICY and screen["status"] == "failed"
    nominal = evidence["results"]["nominal"]["hold_diagnostics"][2]["fields"]["stress_MPa"]
    ripple = evidence["results"]["relative_ripple"]["hold_diagnostics"][2]["fields"]["stress_MPa"]
    assert abs(ripple["last_predicted"]-nominal["last_predicted"]) < .1
    assert ripple["last_observed"]-ripple["last_predicted"] > 14


def test_combined_input_replay_preserves_failed_admission_and_numerical_checks():
    evidence = read("mixed-control-audit")
    assert evidence["numerical"]["passed"]
    assert evidence["numerical"]["independent"]["passed"]
    assert evidence["numerical"]["refined"]["passed"]
    assert evidence["prescribed_stress_is_not_validation"]
    assert not any(evidence[k] for k in ("fitted", "engineering_qualified", "life_qualified",
                                       "acceptance_thresholds_changed"))
    assert set(evidence["results"]) == {
        "nominal", "strain_only", "stress_only", "combined", "independent", "refined"}
    for screen in evidence["response_screens"].values():
        assert screen["policy"] == POLICY and screen["status"] == "failed"
    hold = evidence["results"]["combined"]["hold_diagnostics"][2]["fields"]["stress_MPa"]
    assert hold["last_observed"]-hold["last_predicted"] > 14


@pytest.mark.parametrize("name", ["common-memory-probe", "slow-common-memory-probe"])
def test_common_memory_rejected_without_fitting_or_parameter_growth(name):
    evidence = read(name)
    assert not evidence["fitted"] and not evidence["engineering_qualified"]
    assert evidence["added_coefficients"] == evidence["added_state"] == []
    control, candidate = evidence["records"]
    assert control["shared"] is False
    assert control["cost"] == pytest.approx(.0023607800272608744, abs=1e-15)
    assert candidate["cost"] > 10*control["cost"]
    assert all(r["completed"] for r in evidence["records"])
    assert evidence["training_cycles"] == [1, 2, 3]


def test_observed_budget_is_conditional_not_a_plastic_measurement():
    evidence = read("observed-recovery-budget")
    assert not any(evidence[k] for k in ("fitted", "continuous_prediction",
                                       "engineering_qualified", "life_qualified"))
    bands = {}
    for protocol in ("SLR", "SLC"):
        candidates = [r for r in evidence["records"][protocol]["equal_stress_bands"]
                      if r["cycle"] == 1 and r["band_center_MPa"] == 210.]
        assert len(candidates) == 1
        bands[protocol] = candidates[0]
        assert candidates[0]["status"] == "resolved"
        assert "not measured plastic" in candidates[0]["scope"]
    # Conditional deterministic intervals, not independent uncertainty coverage.
    assert (bands["SLR"]["remainder_interval"][0]/bands["SLR"]["duration_s"]
            > bands["SLC"]["remainder_interval"][1]/bands["SLC"]["duration_s"])


@pytest.mark.parametrize("name,modulus", [("modulus-probe", 162000.),
                                        ("source-modulus-probe", 151000.)])
def test_modulus_sensitivity_does_not_claim_measured_card_or_fit(name, modulus):
    evidence = read(name)
    assert not evidence["fitted"] and not evidence["engineering_qualified"]
    assert evidence["modulus_is_not_specimen_measurement"]
    control, candidate = evidence["records"]
    assert candidate["modulus_MPa"] == modulus
    assert candidate["completed"] and candidate["cost"] > control["cost"]
    assert all(p["response_screen"]["status"] == "failed"
               for p in candidate["protocols"].values())
