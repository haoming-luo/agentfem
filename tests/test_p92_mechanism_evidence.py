"""Scientific status must survive compact evidence archival unchanged."""
import json
from pathlib import Path

import pytest

from tools.p92_response_screen import POLICY

ROOT = Path(__file__).resolve().parents[1]/"evidence"/"creep"


def read(name):
    return json.loads((ROOT/f"2026-11-22-{name}.json").read_text())


@pytest.mark.parametrize("name", ["recoverable-kinetics", "transient-strength"])
def test_fit_and_frozen_status_are_not_conflated(name):
    fit = read(f"{name}-matched")
    frozen = read(f"{name}-frozen")
    residual = read(f"{name}-residual")
    assert fit["fit"]["success"]
    assert fit["history_audit"]["passed"]
    assert fit["training_cycles"] == [1, 2, 3]
    assert len(fit["coordinates"]) == 22
    assert len(fit["fit"]["coordinates"]) == 10
    assert fit["maximum_log_coordinate_norm"] == .3
    assert fit["fit"]["nfev"] <= 40
    assert fit["fit"]["residual_evaluations"] >= fit["fit"]["nfev"]
    assert fit["fit"]["parameters"] == frozen["parameters"]
    assert frozen["fit_file_sha256"] == fit["archived_source_sha256"]
    assert fit["source_sha256"] == frozen["source_sha256"] == residual["source_sha256"]
    assert fit["fit"]["cost"] == pytest.approx(residual["candidate_cost"], abs=1e-14)
    assert residual["candidate_cost"] > residual["baseline_cost"]
    assert not frozen["independent_validation"]
    assert not frozen["engineering_qualified"] and not frozen["life_qualified"]
    for item in frozen["protocols"].values():
        assert item["numerical"]["passed"]
        assert item["response_screen"]["status"] == "failed"
        assert item["response_screen"]["policy"] == POLICY
        assert all(r["role"] == ("early_calibration" if r["cycle"] <= 3 else "inspected_development")
                   for r in item["comparison"]["records"])


@pytest.mark.parametrize("name", ["recoverable-kinetics", "static-strength", "transient-strength", "subthreshold", "recovery-bandwidth"])
def test_fixed_screens_preserve_common_control_and_all_declared_arms(name):
    report = read(f"{name}-probe")
    assert not report["fitted"] and not report["engineering_qualified"]
    assert report["training_cycles"] == [1, 2, 3]
    assert len(report["records"]) == 4
    assert report["records"][0]["cost"] == pytest.approx(.002576080096967705, abs=1e-10)
    for arm in report["records"]:
        assert set(arm["protocols"]) == {"RF", "SLR", "SLC"}
        if not arm["completed"]:
            assert arm["cost"] is None
            assert any(not p["completed"] and p.get("failure") for p in arm["protocols"].values())
        for item in arm["protocols"].values():
            if item["completed"]:
                assert item["response_screen"]["policy"] == POLICY


@pytest.mark.parametrize("name,fit_name", [
    ("static-memory", "2026-11-21-static-memory-matched-shared"),
    ("recoverable-kinetics", "2026-11-22-recoverable-kinetics-matched"),
    ("transient-strength", "2026-11-22-transient-strength-matched"),
])
def test_fit_space_audit_reproduces_the_archived_fit_geometry(name, fit_name):
    from tools.p92_fit_subspace_audit import audit
    fit = json.loads((ROOT/f"{fit_name}.json").read_text())
    saved = read(f"{name}-fit-space")
    assert saved["fit_sha256"] == fit["archived_source_sha256"]
    actual = audit(fit)
    assert actual["coordinates"] == saved["coordinates"]
    assert actual["replacement"]["projection_gram_eigenvalues"] == pytest.approx(
        saved["replacement"]["projection_gram_eigenvalues"], abs=1e-14)
    assert actual["replacement"]["weakest_combination_reachable_half_width"] == pytest.approx(
        saved["replacement"]["weakest_combination_reachable_half_width"], abs=1e-14)
    assert not saved["engineering_qualified"]


def test_replacement_mapping_keeps_failed_continuation_instead_of_fitting_it():
    report = read("recovery-replacement-probe")
    assert not report["fitted"] and not report["engineering_qualified"]
    assert len(report["records"]) == 1
    record = report["records"][0]
    assert record["cost"] is None and not record["completed"]
    assert record["protocols"]["RF"]["completed"]
    assert record["protocols"]["SLR"]["completed"]
    assert not record["protocols"]["SLC"]["completed"]
    assert "ramp direction" in record["protocols"]["SLC"]["failure"]["message"]
    assert "backbone" not in report["mapped_parameters"]


@pytest.mark.parametrize("name,frozen_name", [
    ("shared", "2026-11-21-static-memory-matched-shared-frozen"),
    ("baseline", "2026-11-21-static-memory-matched-baseline-frozen"),
])
def test_memory_bounds_reproduce_verified_trajectory(name, frozen_name):
    from tools.p92_history_memory_audit import attenuation_bounds
    from tools.p92_recoverable_frozen import decode_material
    frozen = json.loads((ROOT/f"{frozen_name}.json").read_text())
    saved = read(f"{name}-history-memory-bounds")
    assert saved["parameters"] == frozen["parameters"]
    _, material = decode_material(saved["parameters"], saved["model_id"])
    for item in saved["protocols"].values():
        inputs = item["audit_input"]
        actual = attenuation_bounds(material, inputs["start_state"], inputs["end_state"], inputs["duration_s"])
        for computed, saved_row in zip(actual, item["long_hold"], strict=True):
            assert computed.keys() == saved_row.keys()
            for key, value in computed.items():
                if isinstance(value, str):
                    assert value == saved_row[key]
                else:
                    assert value == pytest.approx(saved_row[key], rel=1e-12, abs=1e-100)
    assert frozen["source_sha256"] == saved["source_sha256"]
    assert not saved["fitted"] and not saved["engineering_qualified"]


def test_recoverable_interaction_does_not_claim_deletion_benefit():
    report = read("recoverable-backstress-probe")
    assert not report["fitted"] and not report["engineering_qualified"]
    rows = report["records"]
    assert [r["variant"] for r in rows] == ["original", "deletion-only", "0.1", "0.25", "-0.1"]
    assert all(r["completed"] for r in rows)
    assert all(r["cost"] > rows[0]["cost"] for r in rows[2:])
    assert rows[1]["cost"] < rows[0]["cost"]


@pytest.mark.parametrize("name", ["shared", "baseline"])
def test_signed_regeneration_budgets_keep_cancellation_and_close(name):
    import numpy as np
    report = read(f"{name}-memory-contributions")
    assert not report["fitted"] and not report["engineering_qualified"]
    for protocol in report["protocols"].values():
        for budget in protocol["contribution_budgets"].values():
            terms = np.asarray(budget["signed_increments_MPa"])
            np.testing.assert_allclose(terms.sum(axis=1), budget["net_change_MPa"], atol=1e-8)
            assert budget["quadrature_error_estimate_MPa"] < 1e-8
    b = report["protocols"]["SLC"]["contribution_budgets"]["5"]
    assert b["signed_increments_MPa"][2][0] > 300.
    assert b["signed_increments_MPa"][2][2] < -200.
    assert abs(b["net_change_MPa"][2]) < 20.


def test_compact_evidence_is_not_misrepresented_as_full_trajectory():
    from tools.p92_history_memory_audit import audit
    frozen = json.loads((ROOT/"2026-11-21-static-memory-matched-shared-frozen.json").read_text())
    with pytest.raises(ValueError, match="full frozen trajectories"):
        audit(frozen)


def test_mobility_screen_preserves_adverse_hold_slope():
    report = read("recoverable-mobility-probe")
    baseline, flow, recovery = report["records"]
    assert [r["feedback"] for r in report["records"]] == ["none", "flow", "static_recovery"]
    assert recovery["cost"] < baseline["cost"] < flow["cost"]
    assert report["added_coefficients"] == report["added_state"] == []
    def hold(row):
        return next(h["fields"]["stress_MPa"] for h in row["protocols"]["SLC"]["hold_diagnostics"]
                    if h["segment"] == 5)
    old, new = hold(baseline), hold(recovery)
    assert new["rms"] < old["rms"]
    assert abs(new["late_secant_predicted_per_s"]-new["late_secant_observed_per_s"]) > abs(
        old["late_secant_predicted_per_s"]-old["late_secant_observed_per_s"])
    assert not report["engineering_qualified"] and not report["fitted"]


def test_replacement_geometry_is_not_admitted_by_optimization_success():
    report = read("static-memory-replacement-matched")
    frozen = read("static-memory-replacement-frozen")
    audit = read("static-memory-replacement-fit-space")
    assert report["fit"]["success"] and report["replacement_space"]
    assert len(report["coordinates"]) == 22 and len(report["fit"]["coordinates"]) == 10
    assert report["maximum_log_coordinate_norm"] == .3 and report["fit"]["nfev"] <= 40
    assert audit["replacement"]["projection_gram_eigenvalues"] == [1., 1.]
    assert frozen["fit_file_sha256"] == report["archived_source_sha256"]
    assert frozen["parameters"] == report["fit"]["parameters"]
    assert all(p["numerical"]["passed"] and p["response_screen"]["status"] == "failed"
               for p in frozen["protocols"].values())


def test_recovery_capacity_does_not_turn_quantization_into_validation():
    from tools.p92_recovery_capacity_audit import capacity
    report = read("recovery-capacity")
    for row in report["records"]:
        computed = capacity(row["conditional_bound"]["amplitudes"], *row["tail_times_s"])
        for key, value in computed.items():
            if isinstance(value, (str, bool)):
                assert value == row["conditional_bound"][key]
            else:
                assert value == pytest.approx(row["conditional_bound"][key], rel=1e-12, abs=1e-100)
        assert not row["resolved_at_five_quanta"]
        assert row["original_recoverable_rate_per_s"] < row["conditional_bound"]["maximum_average_rate_per_s"]
    assert not report["fitted"] and not report["acceptance_thresholds_changed"]


def test_activated_flow_keeps_failed_continuation_without_joint_cost():
    report = read("activated-flow-probe")
    baseline, replacement = report["records"]
    assert baseline["completed"] and not replacement["completed"]
    assert replacement["cost"] is None
    assert replacement["protocols"]["RF"]["completed"]
    assert replacement["protocols"]["SLR"]["completed"]
    assert "ramp direction" in replacement["protocols"]["SLC"]["failure"]["message"]
    assert report["added_coefficients"] == report["added_state"] == []
    assert not report["engineering_qualified"]


def test_current_code_replays_original_control_without_changed_response():
    report = read("control-replay-audit")
    baseline = json.loads((ROOT/"2026-11-21-static-memory-matched-baseline-frozen.json").read_text())
    assert report["source_sha256"] == baseline["source_sha256"]
    assert report["passed"] and not report["engineering_qualified"]
    for check in report["checks"].values():
        assert check["passed"]
        assert check["stress_state_difference_MPa"] == 0.
        assert check["strain_state_difference"] == 0.
        assert check["work_dissipation_difference_MPa"] == 0.


def test_mobility_fit_improves_cost_but_does_not_pass_frozen_response():
    fit = read("recoverable-mobility-matched")
    frozen = read("recoverable-mobility-frozen")
    residual = read("recoverable-mobility-residual")
    partition = read("recoverable-mobility-hold-partition")
    assert fit["fit"]["success"] and fit["recoverable_mobility"]
    assert fit["training_cycles"] == [1, 2, 3]
    assert len(fit["coordinates"]) == 22 and len(fit["fit"]["coordinates"]) == 10
    assert fit["maximum_log_coordinate_norm"] == .3 and fit["fit"]["nfev"] <= 40
    assert fit["fit"]["parameters"] == frozen["parameters"]
    assert frozen["fit_file_sha256"] == fit["archived_source_sha256"]
    assert partition["frozen_sha256"] == frozen["archived_source_sha256"]
    assert fit["source_sha256"] == frozen["source_sha256"] == residual["source_sha256"]
    assert fit["fit"]["cost"] == pytest.approx(residual["candidate_cost"], abs=1e-14)
    assert residual["candidate_cost"] < residual["baseline_cost"]
    assert not frozen["independent_validation"] and not frozen["engineering_qualified"]
    for item in frozen["protocols"].values():
        assert item["numerical"]["passed"]
        assert item["response_screen"]["status"] == "failed"
        assert item["response_screen"]["policy"] == POLICY
    late = next(r for r in residual["records"] if r["protocol"] == "SLC"
                and r["segment"] == 5 and r["window"] == "late")
    assert abs(late["candidate_bias"]) < abs(late["baseline_bias"])
    assert abs(late["candidate_anchor_residual"]) > abs(late["baseline_anchor_residual"])
    slr = frozen["protocols"]["SLR"]["comparison"]["records"][-1]
    assert slr["peak_relative_error"] > .09


def test_resistance_screen_rejects_local_improvement_with_worse_joint_cost():
    report = read("recoverable-resistance-probe")
    baseline, candidate = report["records"]
    assert [r["feedback"] for r in report["records"]] == ["none", "strength"]
    assert all(r["completed"] for r in report["records"])
    assert candidate["cost"] > 3*baseline["cost"]
    assert report["added_coefficients"] == report["added_state"] == []
    assert not report["fitted"] and not report["engineering_qualified"]
    def hold(row):
        return next(h["fields"]["stress_MPa"] for h in row["protocols"]["SLC"]["hold_diagnostics"]
                    if h["segment"] == 5)
    old, new = hold(baseline), hold(candidate)
    for window, improves in (("early", True), ("late", False)):
        key = f"{window}_secant_predicted_per_s"
        observed = new[f"{window}_secant_observed_per_s"]
        assert (abs(new[key]-observed) < abs(old[key]-observed)) == improves


@pytest.mark.parametrize("name", ["baseline", "mobility"])
def test_switch_perturbations_keep_common_observations_and_late_mismatch(name):
    report = read(f"{name}-switch-audit")
    assert report["common_observations"]
    assert not report["fitted"] and not report["acceptance_thresholds_changed"]
    for arms in report["protocols"].values():
        assert all(a["completed"] for a in arms.values())
        for low, high in zip(arms["lower"]["segments"], arms["upper"]["segments"], strict=True):
            assert low["source_indices"] == high["source_indices"]
    for arm in report["protocols"]["SLC"].values():
        hold = next(h["fields"]["stress_MPa"] for h in arm["holds"] if h["segment"] == 5)
        assert hold["last_observed"]-hold["last_predicted"] > 10.
        assert abs(hold["late_secant_predicted_per_s"] / hold["late_secant_observed_per_s"]-1) > .2


def test_local_feasibility_stops_on_tradeoff_or_unreliable_derivative():
    original = read("baseline-local-feasibility")
    mobility = read("mobility-local-feasibility")
    assert original["derivative_relative_discrepancy"] <= .002
    assert original["response_evaluations"] == 44
    assert len(original["candidates"]) == 3
    assert all(c["completed"] and not c["merits_frozen_continuation"]
               for c in original["candidates"])
    assert mobility["derivative_relative_discrepancy"] > .002
    assert mobility["response_evaluations"] == 41 and not mobility["candidates"]
    for report in (original, mobility):
        assert not report["fitted"] and not report["engineering_qualified"]
        assert not report["acceptance_thresholds_changed"]


@pytest.mark.parametrize("name", ["seed", "baseline", "mobility"])
def test_recorded_hold_sensitivity_preserves_numerical_and_scientific_distinction(name):
    report = read(f"{name}-recorded-hold")
    assert not report["fitted"] and not report["acceptance_thresholds_changed"]
    assert not report["engineering_qualified"]
    for arms in report["protocols"].values():
        assert all(arm["completed"] for arm in arms.values())
        candidate = arms["recorded_strain_sensitivity"]
        assert candidate["independent"]["passed"] and candidate["refinement"]["passed"]
        for arm in arms.values():
            assert arm["response_screen"]["policy"] == POLICY
            assert arm["response_screen"]["status"] == "failed"
    slc = report["protocols"]["SLC"]["recorded_strain_sensitivity"]
    hold = next(h for h in slc["prediction"]["hold_diagnostics"] if h["segment"] == 5)
    assert hold["fields"]["stress_MPa"]["last_observed"]-hold["fields"]["stress_MPa"]["last_predicted"] > 10.


def test_absolute_metric_derivative_issue_is_localized_not_hidden():
    import numpy as np
    report = read("mobility-derivative-detail")
    a, b = np.asarray(report["derivative_matrices"])
    delta = np.linalg.norm(a-b, axis=1)
    largest = np.argsort(delta)[-4:]
    assert np.sum(delta[largest]**2)/np.sum(delta**2) > .999
    assert all(report["initial_values"][i] < .02 for i in largest)
    assert np.linalg.norm(a-b)/np.linalg.norm(b) == pytest.approx(report["derivative_relative_discrepancy"])
    assert len(report["perturbations"]) == 20 and not report["candidates"]


@pytest.mark.parametrize("name", ["baseline", "mobility"])
def test_signed_constraints_resolve_derivatives_without_promoting_tradeoffs(name):
    import numpy as np
    report = read(f"{name}-signed-feasibility")
    original = read(f"{name}-local-feasibility")
    assert report["signed_constraint_linearization"]
    assert report["derivative_relative_discrepancy"] <= .002
    assert report["response_evaluations"] == 44
    np.testing.assert_allclose(abs(np.asarray(report["initial_signed_values"])),
                               original["initial_values"], rtol=0., atol=1e-12)
    assert report["keys"] == original["keys"] and report["unresolved"] == original["unresolved"]
    assert report["source_sha256"] == original["source_sha256"]
    assert report["fit_sha256"] == original["fit_sha256"]
    assert len(report["candidates"]) == 3
    assert all(c["completed"] and not c["merits_frozen_continuation"]
               for c in report["candidates"])
    assert not report["acceptance_thresholds_changed"] and not report["engineering_qualified"]
