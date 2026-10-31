# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from types import SimpleNamespace

import pytest
from agentfem import cli, results
from agentfem.results import compare_records


def record(value=2., unit="m"):
    result = results.SimulationResult("example")
    result.add_quantity("displacement", value, unit=unit)
    result.add_history("reaction", [0., 1.], [0., 10.], unit="N")
    return result.manifest(include_histories=True)


def test_compare_nonmutating_with_absolute_zero_reference():
    a, b = record(0.), record(1e-9)
    original = copy.deepcopy((a, b))
    comparison = compare_records(a, b, quantities=["displacement"], atol=1e-8)
    assert comparison["status"] == "within_tolerance"
    assert comparison["checks"][0]["relative_l2_error"] is None
    assert (a, b) == original
    assert compare_records(a, b, quantities=["displacement"])["status"] == "outside_tolerance"


@pytest.mark.parametrize("unit", ["mm", None, ""])
def test_units_fail_closed(unit):
    assert compare_records(record(), record(unit=unit), quantities=["displacement"])["status"] == "incompatible"


def test_history_no_implicit_resampling():
    a, b = record(), record()
    assert compare_records(a, b, histories=["reaction"])["status"] == "within_tolerance"
    b["history_records"][0]["abscissa"] = [0., 1.000001]
    assert compare_records(a, b, histories=["reaction"])["status"] == "incompatible"
    del b["history_records"][0]["values"]
    assert compare_records(a, b, histories=["reaction"])["status"] == "incompatible"


@pytest.mark.parametrize("change", ["nan", "shape", "missing", "duplicate", "failed"])
def test_bad_inputs(change):
    a, b = record(), record()
    if change == "nan":
        b["quantity_records"][0]["value"] = float("nan")
    elif change == "shape":
        b["quantity_records"][0]["value"] = [2., 2.]
    elif change == "missing":
        b["quantity_records"] = []
    elif change == "duplicate":
        b["quantity_records"] *= 2
    else:
        b["status"] = "failed"
    if change in ("duplicate", "failed"):
        with pytest.raises(ValueError, match="AFM-COMPARE-RECORD"):
            compare_records(a, b, quantities=["displacement"])
    else:
        assert compare_records(a, b, quantities=["displacement"])["status"] == "incompatible"


@pytest.mark.parametrize("kwargs", [{}, {"quantities": ["displacement", "displacement"]},
    {"quantities": ["displacement"], "rtol": -1},
    {"quantities": ["displacement"], "atol": float("inf")}])
def test_invalid_selection_and_tolerances(kwargs):
    with pytest.raises(ValueError, match="AFM-COMPARE"):
        compare_records(record(), record(), **kwargs)


def test_cli_compare_and_missing_files(tmp_path, capsys):
    left, right = tmp_path / "a.json", tmp_path / "b.json"
    a, b = record(), record(3.)
    left.write_text(json.dumps(a))
    right.write_text(json.dumps(b))
    args = ["compare", str(left), str(right), "--quantity", "displacement", "--atol", "0", "--json"]
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "outside_tolerance"
    assert cli.main(args[:-1]) == 2
    assert "max abs: 1 m" in capsys.readouterr().out
    a["artifacts"] = {"fields": "missing.xdmf"}
    left.write_text(json.dumps(a))
    assert cli.main(["show", str(left), "--files", "--json"]) == 2
    shown = json.loads(capsys.readouterr().out)
    assert shown["files"][0]["path"] == str(tmp_path / "missing.xdmf")
    assert not shown["files"][0]["exists"]


def test_preflight_does_not_execute_case(tmp_path, monkeypatch, capsys):
    (tmp_path / "case.py").write_text("raise RuntimeError('must not execute')")
    (tmp_path / "agentfem.toml").write_text('[project]\nname="safe"\nentrypoint="case.py"\n')
    monkeypatch.setattr(cli.platforms, "runtime_report", lambda: SimpleNamespace(
        solver_ready=False, summary=lambda: {"solver_ready": False}))
    assert cli.main(["check", "--project", str(tmp_path), "--runtime", "--json"]) == 2
    check = json.loads(capsys.readouterr().out)
    assert check["model_executed"] is False
    assert any("AFM-PREFLIGHT-RUNTIME" in e for e in check["errors"])


def test_summary_bounds_file_list_without_changing_manifest(tmp_path):
    from agentfem import presentation
    data = record()
    data["artifacts"] = {str(i): f"checkpoint-{i}.npz" for i in range(10)}
    data["artifacts"].update(fields="fields.xdmf", duplicate="fields.xdmf")
    original = copy.deepcopy(data)
    text = presentation.format_result(data, tmp_path / "result.json")
    assert "files: fields.xdmf" in text
    assert "+ 5 more files" in text
    assert data == original


@pytest.mark.parametrize("case", ["unassessed", "failed_claim", "required_quality"])
def test_empty_evidence_is_not_presented_as_a_failed_verification(tmp_path, case):
    from agentfem import presentation
    data = record()
    data["status"] = "partial"
    data["verification"] = {"acceptable": False, "claims": []}
    if case == "failed_claim":
        data["verification"]["claims"] = [{"name": "equilibrium", "status": "failed"}]
    elif case == "required_quality":
        data["verification"]["minimum_trust_level"] = "verified"
    original = copy.deepcopy(data)
    text = presentation.format_result(data, tmp_path / "result.json")
    assert ("verification: not assessed" in text) == (case == "unassessed")
    assert ("verification: not accepted" in text) == (case != "unassessed")
    assert data == original  # Human wording never changes the actual decision.


def test_malformed_artifacts_and_string_selection(tmp_path, capsys):
    with pytest.raises(ValueError, match="AFM-COMPARE-SELECTION"):
        compare_records(record(), record(), quantities="displacement")
    path = tmp_path / "result.json"
    data = record()
    data["artifacts"] = []
    path.write_text(json.dumps(data))
    assert cli.main(["show", str(path), "--files", "--json"]) == 2
    assert "AFM-RECORD-ARTIFACTS" in capsys.readouterr().out
