"""Installed callable introspection, not guessed or separately maintained APIs."""

import inspect
import importlib
import json

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
    assert names
    for name in names:
        record = callable_record(name)
        assert record["callable"] == f"agentfem.{name}"
        json.dumps(record, allow_nan=False)
