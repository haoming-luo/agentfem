# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspect installed public call signatures without evaluating a case."""

from importlib import import_module
import inspect
import math

from . import __version__, public_api
from ._api_contract import model_method_contract, model_methods


def _default(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return {"kind": "runtime_default", "python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _annotation(value):
    return value if isinstance(value, str) else inspect.formatannotation(value)


def callable_record(name: str) -> dict:
    """Describe a public module function or a declared Model method.

    Imports the selected installed module only. No expression evaluation,
    arbitrary attribute chain, user case execution or dependency installation.
    """

    selected = str(name).removeprefix("agentfem.")
    parts = selected.split(".")
    if len(parts) not in (2, 3) or any(not part.isidentifier() or part.startswith("_") for part in parts):
        raise ValueError("Use a public MODULE.FUNCTION or models.Model.METHOD name.")
    module_name = parts[0]
    if module_name not in public_api():
        raise ValueError(f"{module_name!r} is not a public AgentFEM module.")
    is_method = len(parts) == 3
    if is_method and (parts[:2] != ["models", "Model"] or parts[2] not in model_methods("all")):
        raise ValueError("Only declared models.Model methods support nested API discovery.")
    try:
        module = import_module(f"agentfem.{module_name}")
    except ImportError as exc:
        raise RuntimeError(f"Cannot inspect installed module agentfem.{module_name}: {exc}") from exc
    owner = module.Model if is_method else module
    try:
        function = inspect.getattr_static(owner, parts[-1])
    except AttributeError as exc:
        raise ValueError(f"No public callable {selected!r} in this installed version.") from exc
    if not inspect.isfunction(function) or not function.__module__.startswith("agentfem."):
        raise ValueError(f"{selected!r} is not an AgentFEM public function or Model method.")
    signature = inspect.signature(function, eval_str=False)
    parameters = []
    for parameter in signature.parameters.values():
        if is_method and parameter.name == "self":
            continue
        item = {
            "name": parameter.name,
            "kind": parameter.kind.name.lower(),
            "required": parameter.default is inspect.Parameter.empty
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
        }
        if parameter.default is not inspect.Parameter.empty:
            item["default"] = _default(parameter.default)
        if parameter.annotation is not inspect.Parameter.empty:
            item["annotation"] = _annotation(parameter.annotation)
        parameters.append(item)
    record = {
        "schema": "agentfem.api-callable", "schema_version": "0.1.0",
        "agentfem_version": __version__, "callable": f"agentfem.{selected}",
        "requires_model_instance": is_method, "parameters": parameters,
        "summary": (inspect.getdoc(function) or "").split("\n\n")[0],
        "case_executed": False,
    }
    if signature.return_annotation is not inspect.Signature.empty:
        record["returns"] = _annotation(signature.return_annotation)
    if is_method:
        record["lifecycle"] = next(item for item in model_method_contract() if item["name"] == parts[-1])
    if selected == "models.Model.step":
        record["provider_options"] = "models.step_capability(model, target=..., options=...)"
    return record


def format_callable(record):
    lines = [str(record["callable"]), str(record["summary"])]
    for item in record["parameters"]:
        requirement = "required" if item["required"] else "optional"
        default = ""
        if "default" in item:
            shown = "<runtime default>" if isinstance(item["default"], dict) else repr(item["default"])
            default = f", default={shown}"
        lines.append(f"  {item['name']}: {item['kind']}, {requirement}{default}")
    if "returns" in record:
        lines.append(f"Returns: {record['returns']}")
    if "provider_options" in record:
        lines.append(f"Provider options: {record['provider_options']}")
    return "\n".join(lines)
