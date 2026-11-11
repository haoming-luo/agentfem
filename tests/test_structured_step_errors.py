import json

import pytest

from agentfem.cli import _rank_error
from agentfem.step_providers import StepOptionContract, StepOptionError


@pytest.mark.parametrize("options, expected", [({}, "ready"), ({"dt": .1}, "requires_dt")])
def test_capability_prefers_the_same_ready_provider_as_execution(monkeypatch, options, expected):
    from types import SimpleNamespace
    from agentfem import step_providers as providers

    registry = providers.StepProviderRegistry()
    for name, priority, required in (("requires_dt", 10, ("dt",)), ("ready", 0, ())):
        registry.register(providers.StepProvider(
            name=name, analyses=("custom",), priority=priority,
            accepts=lambda model, request: True,
            lower=lambda model, request: None,
            option_contract=StepOptionContract(accepted=("dt",), required=required),
        ))
    monkeypatch.setattr(providers, "_ensure_builtin_providers", lambda: None)
    monkeypatch.setattr(providers, "_DEFAULT_REGISTRY", registry)
    model = SimpleNamespace(study=SimpleNamespace(analysis="custom"), fields=())
    request = providers.StepRequest(analysis="custom", target=None, options=options)
    capability = providers.step_capability(model, options=options)
    assert capability["provider"]["name"] == registry.resolve(model, request).name
    assert capability["provider"]["name"] == expected
    assert capability["ready"] is True


def test_partial_capability_does_not_claim_readiness(monkeypatch):
    from types import SimpleNamespace
    from agentfem import step_providers as providers

    registry = providers.StepProviderRegistry()
    registry.register(providers.StepProvider(
        name="transient-example", analyses=("custom",),
        accepts=lambda model, request: True,
        lower=lambda model, request: pytest.fail("Missing inputs must not execute"),
        option_contract=StepOptionContract(accepted=("dt",), required=("dt",)),
    ))
    monkeypatch.setattr(providers, "_ensure_builtin_providers", lambda: None)
    monkeypatch.setattr(providers, "_DEFAULT_REGISTRY", registry)
    model = SimpleNamespace(study=SimpleNamespace(analysis="custom"), fields=())
    capability = providers.step_capability(model)
    assert capability["supported"] is True
    assert capability["ready"] is False
    assert capability["option_issues"] == ()
    with pytest.raises(StepOptionError) as caught:
        registry.resolve(model, providers.StepRequest(analysis="custom", target=None, options={}))
    assert capability["readiness_issues"] == caught.value.details()["issues"]


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


def test_unsupported_capability_retains_the_same_reasons_as_inspection(monkeypatch):
    from types import SimpleNamespace
    from agentfem import step_providers as providers

    registry = providers.StepProviderRegistry()
    registry.register(providers.StepProvider(
        name="bounded", analyses=("custom",),
        accepts=lambda model, request: False,
        rejection_reason=lambda model, request: "This route requires a scalar field.",
        lower=lambda model, request: pytest.fail("Rejected provider must not execute"),
    ))
    monkeypatch.setattr(providers, "_ensure_builtin_providers", lambda: None)
    monkeypatch.setattr(providers, "_DEFAULT_REGISTRY", registry)
    model = SimpleNamespace(study=SimpleNamespace(analysis="custom"), fields=())
    capability = providers.step_capability(model)
    assert not capability["supported"]
    with pytest.raises(NotImplementedError) as caught:
        registry.resolve(model, providers.StepRequest(analysis="custom", target=None, options={}))
    error = caught.value
    assert isinstance(error, providers.StepCapabilityError)
    assert error.details()["rejection_reasons"] == capability["rejection_reasons"]
    assert error.details()["candidate_providers"] == capability["candidate_providers"]
    record = _rank_error(error, rank=0)
    assert record["stage"] == "step_capability"
    assert record["code"] == "AFM-STEP-CAPABILITY-001"
    record["details"]["rejection_reasons"][0]["message"] = "modified"
    assert error.details()["rejection_reasons"][0]["message"].startswith("This route")
    json.dumps(record, allow_nan=False)


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
    from agentfem import feedback
    diagnosis = feedback.diagnose(paths[0])
    assert "error.details" in diagnosis["suggestions"][0]
    assert not any("agentfem check" in item for item in diagnosis["suggestions"])


def test_project_run_persists_unsupported_capability(tmp_path, capsys):
    from agentfem import cli

    (tmp_path / "agentfem.toml").write_text("[project]\nname='unsupported'\nentrypoint='case.py'\n")
    (tmp_path / "case.py").write_text(
        "from types import SimpleNamespace\n"
        "from agentfem.step_providers import StepProviderRegistry, StepRequest\n"
        "StepProviderRegistry().resolve(SimpleNamespace(materials=()),\n"
        "    StepRequest(analysis='unavailable', target=None, options={}))\n"
        "raise AssertionError('Unsupported request must not continue')\n"
    )
    assert cli.main(["run", "--project", str(tmp_path), "--run-id", "unsupported", "--json"]) == 1
    capsys.readouterr()
    execution_path, = (tmp_path / "outputs").rglob("execution.json")
    execution = json.loads(execution_path.read_text())
    assert execution["status"] == "failed"
    assert execution["stage"] == "step_capability"
    assert execution["error"]["details"]["analysis"] == "unavailable"
    assert execution["error"]["details"]["candidate_providers"] == []
    from agentfem import feedback
    diagnosis = feedback.diagnose(execution_path)
    assert "rejection reasons" in diagnosis["suggestions"][0]
    assert not any("agentfem check" in item for item in diagnosis["suggestions"])
