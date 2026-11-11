import json

import pytest

from agentfem.cli import _rank_error
from agentfem.step_providers import StepOptionContract, StepOptionError


def test_diagnostics_are_identical_to_preflight_and_typeerror_compatible():
    contract = StepOptionContract(accepted=("dt", "steps"), required=("dt", "steps"))
    options = {"dt": .1, "step": "private-input-value"}
    with pytest.raises(TypeError) as caught:
        contract.validate(options, provider="example")
    error = caught.value
    assert isinstance(error, StepOptionError)
    assert error.details()["issues"] == contract.issues(options)
    assert error.details()["provider"] == "example"
    assert "private-input-value" not in json.dumps(error.details())
    altered = error.details()
    altered["issues"][0]["code"] = "changed"
    assert error.details()["issues"][0]["code"] == "AFM-STEP-OPTION-001"


def test_execution_failure_preserves_repair_data():
    contract = StepOptionContract(accepted=("frequency", "omega"),
                                  exactly_one_of=(("frequency", "omega"),))
    try:
        contract.validate({"frequency": 1., "omega": 1.}, provider="harmonic-example")
    except StepOptionError as error:
        record = _rank_error(error, rank=0)
    assert record["stage"] == "step_options"
    assert record["code"] == "AFM-STEP-OPTION-003"
    assert record["details"]["issues"][0]["selected"] == ("frequency", "omega")
    json.dumps(record, allow_nan=False)


def test_ordinary_exceptions_keep_existing_semantics():
    assert _rank_error(None, rank=0) is None
    record = _rank_error(TypeError("ordinary"), rank=2)
    assert record["stage"] == "case_execution"
    assert record["rank"] == 2
    assert "details" not in record


def test_project_run_persists_actionable_error_without_running_solver(tmp_path, capsys):
    from agentfem import cli

    (tmp_path / "agentfem.toml").write_text("[project]\nname='option-error'\nentrypoint='case.py'\n")
    (tmp_path / "case.py").write_text(
        "from agentfem.step_providers import StepOptionContract\n"
        "StepOptionContract(accepted=('dt', 'steps'), required=('dt',)).validate(\n"
        "    {'dtt': .1}, provider='test-provider')\n"
        "raise AssertionError('Execution must stop before this point')\n"
    )
    assert cli.main(["run", "--project", str(tmp_path), "--run-id", "invalid", "--json"]) == 1
    capsys.readouterr()
    paths = list((tmp_path / "outputs").rglob("execution.json"))
    assert len(paths) == 1
    execution = json.loads(paths[0].read_text())
    assert execution["status"] == "failed"
    assert execution["stage"] == "step_options"
    assert execution["error"]["details"]["provider"] == "test-provider"
    assert execution["error"]["details"]["issues"][0]["option"] == "dtt"
    assert execution["error"]["details"]["issues"][0]["suggestions"] == ["dt"]
