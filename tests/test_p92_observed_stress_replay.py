from types import SimpleNamespace
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_hold_flow_budget import budget
from tools.p92_observed_stress_replay import scalar_or_vector, evaluate, mechanical_balances
from tools.p92_stress_ripple_audit import stress_path
from tools.p92_two_rate_reference import Branch, Material


@pytest.mark.parametrize("sign", [-1., 1.])
def test_recorded_stress_dense_budget_supports_scalar_and_vector_queries(sign):
    branch = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    material = kernel.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))
    initial = np.zeros(12)
    initial[0] = sign*110.
    times = np.array([0., 1., 2.])
    result = stress_path(initial, times, sign*np.array([0., 2., -1.]), material, kernel)
    dense = scalar_or_vector(result.sol)
    assert dense(.5).shape == (12,)
    assert dense([]).shape == (12, 0)
    np.testing.assert_array_equal(dense(.5), dense([.5])[:, 0])
    row = budget(SimpleNamespace(sol=dense, t=times), material, .1, 1.9)
    assert row["accounting_passed"]
    assert all(sign*x > 0 for x in row["signed_branch_strain_changes"])
    with pytest.raises(ValueError, match="outside"):
        dense(-.1)


def test_adapter_does_not_flatten_invalid_queries():
    def vector_only(query):
        if query.ndim != 1:
            raise ValueError("vector only")
        return np.zeros((12, len(query)))
    with pytest.raises(ValueError, match="vector only"):
        scalar_or_vector(vector_only)([[0.]])


@pytest.mark.parametrize("work_error", [1e-6, float("nan"), float("inf")])
def test_independent_arm_energy_failure_is_not_hidden(work_error):
    branch = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    material = kernel.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))
    initial = np.zeros(12)
    good = initial[:, None].copy()
    assert mechanical_balances([good, good, good], initial, material)["passed"]
    broken = good.copy()
    broken[8, -1] = work_error
    result = mechanical_balances([good, broken, good], initial, material)
    assert result["values_MPa"][0] == 0.
    assert not result["passed"]


@pytest.mark.parametrize("cycles", [0, -1, True, 1.5])
def test_invalid_cycle_count_before_source_access(cycles):
    with pytest.raises(ValueError, match="cycle count"):
        evaluate({}, Path("missing"), cycles=cycles)


def test_archived_conditional_budget_preserves_failure_and_signed_accounting():
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root/"evidence/creep/2026-11-22-observed-stress-replay.json").read_text())
    assert not record["fitted"] and not record["forward_validation"]
    assert not record["engineering_qualified"] and not record["life_qualified"]
    for protocol in record["records"].values():
        assert protocol["numerical_passed"]
        assert abs(protocol["mechanical_balance_MPa"]) < 1e-9
        assert protocol["mechanical_balance_checks"]["passed"]
        assert len(protocol["mechanical_balance_checks"]["values_MPa"]) == 3
        for row in protocol["records"]:
            b = row["signed_flow_budget"]
            assert b["accounting_passed"] and b["independent_signed_flow_passed"]
            reconstructed = (b["elastic_strain_change"]+sum(b["recoverable_strain_changes"])
                             +sum(b["signed_branch_strain_changes"]))
            assert reconstructed == pytest.approx(row["predicted_strain_change"], abs=1e-9)
    slc = {(r["segment"], r["window"]): r for r in record["records"]["SLC"]["records"]}
    assert slc[5, "early"]["strain_change_residual"] > 2e-4
    assert abs(slc[5, "late"]["strain_change_residual"]) < 1e-5


def test_continuous_prefix_is_not_reset_when_more_cycles_are_requested():
    root = Path(__file__).resolve().parents[1]/"evidence/creep"
    first = json.loads((root/"2026-11-22-observed-stress-replay.json").read_text())
    three = json.loads((root/"2026-11-22-observed-stress-three-cycles.json").read_text())
    frozen = (root/"2026-11-22-recovery-shape-control-frozen.json").read_bytes()
    assert first["frozen_sha256"] == three["frozen_sha256"] == sha256(frozen).hexdigest()
    assert first["parameters"] == three["parameters"]
    assert first["source_sha256"] == three["source_sha256"]
    for name, count in (("RF", 4), ("SLR", 6), ("SLC", 6)):
        rows = three["records"][name]["records"]
        assert three["records"][name]["numerical_passed"]
        assert len(rows) == count*3
        assert {r["cycle"] for r in rows} == {1, 2, 3}
        for old, new in zip(first["records"][name]["records"], rows[:count], strict=True):
            for key in ("observed_strain_change", "predicted_strain_change", "strain_change_residual"):
                assert new[key] == old[key]


def test_ten_cycle_prefix_and_independent_energy_remain_continuous():
    root = Path(__file__).resolve().parents[1]/"evidence/creep"
    three = json.loads((root/"2026-11-22-observed-stress-three-cycles.json").read_text())
    ten = json.loads((root/"2026-11-22-observed-stress-ten-cycles.json").read_text())
    for key in ("frozen_sha256", "driver_sha256", "source_sha256", "parameters"):
        assert three[key] == ten[key]
    assert ten["cycles"] == 10
    assert not ten["forward_validation"] and not ten["fitted"]
    for name, count in (("RF", 4), ("SLR", 6), ("SLC", 6)):
        record = ten["records"][name]
        assert record["numerical_passed"]
        assert len(record["records"]) == count*10
        check = record["mechanical_balance_checks"]
        assert check["passed"] and len(check["values_MPa"]) == 3
        assert np.max(np.abs(check["values_MPa"])) <= check["absolute_limit_MPa"]
        assert record["records"][:count*3] == three["records"][name]["records"]


@pytest.mark.parametrize("cycles", ["three", "ten"])
def test_retained_independent_energy_failure_is_not_a_response_pass(cycles):
    root = Path(__file__).resolve().parents[1]/"evidence/creep"
    failed = json.loads((root/f"2026-11-22-observed-stress-{cycles}-cycles-energy-failure.json").read_text())
    tightened = json.loads((root/f"2026-11-22-observed-stress-{cycles}-cycles.json").read_text())
    assert not failed["forward_validation"] and not failed["engineering_qualified"]
    for field in ("parameters", "source_sha256", "frozen_sha256", "model_id"):
        assert failed[field] == tightened[field]
    for name, record in failed["records"].items():
        assert not record["numerical_passed"]
        check = record["mechanical_balance_checks"]
        assert check["absolute_limit_MPa"] == 1e-8
        assert abs(check["values_MPa"][0]) < 1e-8
        assert abs(check["values_MPa"][1]) > 1e-8
        assert not check["passed"]
        updated = tightened["records"][name]
        assert updated["mechanical_balance_checks"]["absolute_limit_MPa"] == check["absolute_limit_MPa"]
        for before, after in zip(record["records"], updated["records"], strict=True):
            assert before["source_indices"] == after["source_indices"]
            assert before["observed_strain_change"] == after["observed_strain_change"]
            assert before["predicted_strain_change"] == after["predicted_strain_change"]


def test_frozen_path_positive_rate_scaling_cannot_repair_both_early_holds():
    """A cone/accounting test, not a nonlinear model-family impossibility claim."""
    path = Path(__file__).resolve().parents[1]/"evidence/creep/2026-11-22-observed-stress-ten-cycles.json"
    record = json.loads(path.read_text())
    for cycle in range(1, 11):
        rows = [next(r for r in record["records"][name]["records"]
                     if r["cycle"] == cycle and r["segment"] == 5+7*(cycle-1)
                     and r["window"] == "early") for name in ("SLR", "SLC")]
        flows = np.array([r["signed_flow_budget"]["signed_branch_strain_changes"] for r in rows])
        required = np.array([r["observed_strain_change"]
                             - r["signed_flow_budget"]["elastic_strain_change"]
                             - sum(r["signed_flow_budget"]["recoverable_strain_changes"]) for r in rows])
        assert np.all(flows > 0) and np.all(required > 0)
        # Any nonnegative common pair of branch multipliers produces a ratio
        # between these column ratios. No material parameters are fitted here.
        cone_max = max(flows[0]/flows[1])
        assert required[0]/required[1] > cone_max
        # A deliberately labelled accounting-band sensitivity, not an
        # experimental confidence interval or a new response acceptance gate.
        assert (required[0]-2e-5)/(required[1]+2e-5) > cone_max
