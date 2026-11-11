"""Installed callable introspection, not guessed or separately maintained APIs."""

import inspect
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agentfem import cli, fields, models
from agentfem._api_discovery import callable_record


@pytest.mark.parametrize("name, function", [
    ("fields.displacement", fields.displacement),
    ("agentfem.fields.temperature", fields.temperature),
    ("models.create", models.create),
])
def test_function_parameters_match_installed_signature(name, function):
    record = callable_record(name)
    assert [item["name"] for item in record["parameters"]] == list(inspect.signature(function).parameters)
    assert record["case_executed"] is False
    assert record["requires_model_instance"] is False
    json.dumps(record, allow_nan=False)


def test_field_parameters_do_not_invent_a_name_keyword():
    record = callable_record("fields.displacement")
    parameters = {item["name"]: item for item in record["parameters"]}
    assert "name" not in parameters
    assert parameters["domain"]["required"] is True
    assert parameters["degree"]["default"] == 1
    assert parameters["degree"]["kind"] == "keyword_only"
    assert record["returns"] == "UnknownField"


def test_model_method_excludes_self_and_keeps_lifecycle_and_provider_boundary():
    record = callable_record("models.Model.step")
    assert record["requires_model_instance"] is True
    assert record["lifecycle"]["tier"] == "core"
    expected = list(inspect.signature(models.Model.step).parameters)
    assert [item["name"] for item in record["parameters"]] == [name for name in expected if name != "self"]
    assert record["provider_options"].startswith("models.step_capability")


def test_compatibility_method_remains_identified_as_compatibility():
    record = callable_record("models.Model.add_field")
    assert record["lifecycle"]["tier"] == "compatibility"
    assert record["lifecycle"]["replacement"] == "model.field(...)"


@pytest.mark.parametrize("name", [
    "os.system", "fields.__dict__", "models.Model.__init__", "models.Model.name",
    "fields.not_a_function", "fields.displacement()", "models.Model.step.extra",
])
def test_private_arbitrary_and_unknown_queries_rejected(name):
    with pytest.raises(ValueError):
        callable_record(name)


def test_cli_api_query_and_mutually_exclusive_topic(capsys):
    assert cli._dispatch(["capabilities", "--api", "fields.displacement", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "agentfem.api-callable"
    assert record["callable"] == "agentfem.fields.displacement"
    assert cli.main(["capabilities", "materials", "--api", "fields.displacement", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "argument_parsing"


def test_introspection_does_not_call_function_evaluate_annotations_or_repr_defaults(monkeypatch):
    class RuntimeDefault:
        def __repr__(self):
            pytest.fail("Do not evaluate runtime defaults while describing an API")

    def sample(domain, *, value=RuntimeDefault()):
        pytest.fail("API discovery must not execute the function")

    sample.__module__ = "agentfem.fields"
    sample.__annotations__ = {"domain": "must_not_be_evaluated()"}
    monkeypatch.setattr(fields, "displacement", sample)
    record = callable_record("fields.displacement")
    assert record["parameters"][0]["annotation"] == "must_not_be_evaluated()"
    assert record["parameters"][1]["default"]["kind"] == "runtime_default"
    assert callable_record("fields.displacement") == record


def test_core_call_paths_and_declared_model_methods_have_json_safe_descriptions():
    from agentfem import public_api

    names = [f"models.Model.{name}" for name in models.model_api("all")]
    for module_name in public_api("core"):
        module = importlib.import_module(f"agentfem.{module_name}")
        names.extend(
            f"{module_name}.{name}" for name, function in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_") and function.__module__.startswith("agentfem.")
        )
        names.extend(
            f"{module_name}.{name}" for name in getattr(module, "__all__", ())
            if inspect.isclass(function := getattr(module, name, None))
            and function.__module__.startswith("agentfem.")
            and not issubclass(function, BaseException)
        )
    assert names
    for name in names:
        record = callable_record(name)
        assert record["callable"] == f"agentfem.{name}"
        json.dumps(record, allow_nan=False)


def test_exported_material_constructor_reports_inputs_without_constructing(monkeypatch):
    from agentfem import constitutive

    # The real constructor compiles source. Signature inspection must not.
    expected = inspect.signature(constitutive.StrainHistoryRuntime)
    record = callable_record("constitutive.StrainHistoryRuntime")
    assert record["callable_kind"] == "constructor"
    assert record["returns"] == "agentfem.constitutive.StrainHistoryRuntime"
    assert [item["name"] for item in record["parameters"]] == list(expected.parameters)
    assert record["requires_model_instance"] is False
    assert record["case_executed"] is False
    parameters = {item["name"]: item for item in record["parameters"]}
    assert parameters["manifest"]["required"] is True
    assert parameters["trusted"]["default"] is False

    class Sample:
        def __init__(self, source, *, trusted=False):
            pytest.fail("Discovery must not construct or compile a material")

    Sample.__module__ = "agentfem.constitutive"
    monkeypatch.setattr(constitutive, "StrainHistoryRuntime", Sample)
    assert callable_record("constitutive.StrainHistoryRuntime")["parameters"][0]["name"] == "source"
    monkeypatch.setattr(constitutive, "UndeclaredClass", Sample, raising=False)
    with pytest.raises(ValueError, match="exported class"):
        callable_record("constitutive.UndeclaredClass")


def test_uninspectable_signature_reports_the_requested_public_name(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("missing implementation metadata")

    monkeypatch.setattr(inspect, "signature", unavailable)
    with pytest.raises(ValueError, match="fields.displacement.*inspectable signature"):
        callable_record("fields.displacement")


@pytest.mark.parametrize("arguments,schema,code", [
    (["workflow"], "agentfem.capabilities", 0),
    (["--api", "fields.displacement"], "agentfem.api-callable", 0),
    (["--api", "constitutive.StrainHistoryRuntime"], "agentfem.api-callable", 0),
    (["--api", "fields.not_an_api"], "agentfem.cli-error", 2),
])
def test_actual_cli_discovery_never_executes_the_adjacent_case(tmp_path, arguments, schema, code):
    (tmp_path / "agentfem.toml").write_text("[project]\nname='do-not-run'\nentrypoint='case.py'\n")
    (tmp_path / "case.py").write_text(
        "from pathlib import Path\nPath('case-executed').touch()\n"
        "raise RuntimeError('Discovery executed a case')\n"
    )
    original = {path.name for path in tmp_path.iterdir()}
    completed = subprocess.run(
        [sys.executable, "-m", "agentfem._cli_entry", "capabilities", *arguments, "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).resolve().parents[1]),
             "AGENTFEM_STATE_HOME": str(tmp_path / "state"), "AGENTFEM_TELEMETRY": "off"},
    )
    assert completed.returncode == code, completed.stdout + completed.stderr
    record = json.loads(completed.stdout)
    assert record["schema"] == schema
    if code == 0 and "--api" in arguments:
        assert record["case_executed"] is False
    assert {path.name for path in tmp_path.iterdir()} == original
