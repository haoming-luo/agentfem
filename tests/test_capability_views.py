"""Scoped discovery retains the full selected scientific evidence."""

from copy import deepcopy
import json

import pytest

from agentfem import cli, presentation


@pytest.fixture
def record():
    return {
        "schema": "agentfem.capabilities", "schema_version": "test", "agentfem_version": "test",
        "constitutive": ({"name": "bounded_material", "limitations": ["serial only"]},),
        "constitutive_evidence": ({"capability": "bounded_material", "gaps": ["external"],
                                   "meets_declared_maturity": False},),
        "mesh_cells": [{"solver_ready": False}],
        "step_providers": [{"name": "provider", "options": {"required": ["dt"]}}],
        "commands": ["run"], "templates": ["static-solid"],
        "public_api": {"core": ["models"]}, "model_api": {"core": ["step"]},
        "model_api_contract": {"compatibility": "retained"}, "ownership_contract": {},
        "runtime": {"solver_ready": False}, "extensions": {"installed": []},
    }


@pytest.mark.parametrize("topic, keys", [
    ("materials", {"constitutive", "constitutive_evidence"}),
    ("meshes", {"mesh_cells"}),
    ("procedures", {"step_providers"}),
    ("workflow", {"commands", "templates", "public_api", "model_api", "model_api_contract", "ownership_contract"}),
    ("runtime", {"runtime"}),
    ("extensions", {"extensions"}),
])
def test_topic_selects_exact_existing_sections(record, topic, keys):
    before = deepcopy(record)
    selected = presentation.capability_record(record, topic)
    assert selected["scope"] == {"topic": topic, "complete": False}
    assert set(selected) == keys | {"schema", "schema_version", "agentfem_version", "scope"}
    for key in keys:
        assert selected[key] == record[key]
    assert record == before


def test_full_view_is_backward_compatible_and_material_view_keeps_gaps(record):
    assert presentation.capability_record(record) == record
    selected = presentation.capability_record(record, "Bounded-Material")
    assert selected["constitutive"] == record["constitutive"]
    assert selected["constitutive_evidence"] == record["constitutive_evidence"]
    assert selected["constitutive_evidence"][0]["meets_declared_maturity"] is False


def test_unknown_topic_rejected_in_both_views(record):
    for view in (presentation.capability_record, presentation.format_capabilities):
        with pytest.raises(ValueError, match="Unknown capability topic"):
            view(record, "typo")


def test_cli_material_query_is_scoped_and_preserves_exact_evidence(capsys):
    assert cli._dispatch(["capabilities", "--json"]) == 0
    full_output = capsys.readouterr().out
    full = json.loads(full_output)
    name = full["constitutive"][0]["name"]
    assert cli._dispatch(["capabilities", name, "--json"]) == 0
    selected_output = capsys.readouterr().out
    selected = json.loads(selected_output)
    assert selected["constitutive"] == [item for item in full["constitutive"] if item["name"] == name]
    assert selected["constitutive_evidence"] == [item for item in full["constitutive_evidence"] if item["capability"] == name]
    assert "runtime" not in selected
    assert selected["scope"]["complete"] is False
    assert len(selected_output) < len(full_output)


def test_workflow_discovery_does_not_probe_unrelated_runtime(monkeypatch, capsys):
    monkeypatch.setattr(cli.platforms, "runtime_report", lambda: pytest.fail("Unrelated runtime probe"))
    assert cli._dispatch(["capabilities", "workflow", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert "step" in record["model_api"]["core"]
    assert "runtime" not in record


def test_standalone_workflow_discovery_does_not_import_solver(tmp_path):
    import subprocess
    import sys

    completed = subprocess.run([
        sys.executable, "-c",
        "import sys; from agentfem._capability_discovery import build_capability_record; "
        "r=build_capability_record('workflow', templates=lambda: ()); "
        "assert 'step' in r['model_api']['core']; "
        "assert 'dolfinx' not in sys.modules; assert 'mpi4py.MPI' not in sys.modules",
    ], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
