from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / "src" / "agentfem" / "manifests.py"


def _portable_module():
    spec = importlib.util.spec_from_file_location("agentfem_portable_manifests", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_result_manifest_reader_has_no_solver_import_and_checks_duplicates(tmp_path):
    manifests = _portable_module()
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema": "agentfem.simulation-result",
                "schema_version": "0.1.0",
                "field_records": [{"name": "S"}, {"name": "E"}],
                "quantity_records": [{"name": "tip"}],
            }
        ),
        encoding="utf-8",
    )

    assert manifests.result(path)["field_records"][0]["name"] == "S"

    path.write_text(
        json.dumps(
            {
                "schema": "agentfem.simulation-result",
                "schema_version": "0.1.0",
                "field_records": [{"name": "S"}, {"name": "S"}],
                "quantity_records": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        manifests.result(path)


def test_reader_rejects_unknown_schema_versions(tmp_path):
    manifests = _portable_module()
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {
                "schema": "agentfem.simulation-result",
                "schema_version": "99.0.0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported"):
        manifests.read(path)
