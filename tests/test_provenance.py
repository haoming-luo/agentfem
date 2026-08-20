from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from agentfem import cli, provenance, results


def _sealed_result(tmp_path):
    artifact = tmp_path / "field.bin"
    artifact.write_bytes(b"finite-element-field\n")
    result = results.SimulationResult("sealed-case")
    result.add_quantity("response", 1.25)
    result.add_artifact("field", artifact)
    manifest = result.write_manifest(tmp_path / "result.json")
    return manifest, artifact


def test_result_manifest_seal_verifies_registered_artifacts(tmp_path):
    manifest, artifact = _sealed_result(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    report = provenance.verify_manifest(manifest)

    assert artifact.is_file()
    assert record["provenance_seal"]["producer"] == "AgentFEM"
    assert record["provenance_seal"]["origin"]["initiated_by"] == "Haoming Luo"
    assert record["provenance_seal"]["origin"]["repository"].endswith(
        "/haoming-luo/agentfem"
    )
    assert record["provenance_seal"]["completeness"] == "complete"
    assert record["runtime"]["schema"] == "agentfem.runtime-lock"
    assert record["runtime"]["identity"]["mpi"]["rank_count"] >= 1
    assert report.status == "verified"
    assert report.verified is True


def test_provenance_detects_artifact_and_manifest_changes(tmp_path):
    manifest, artifact = _sealed_result(tmp_path)
    artifact.write_bytes(b"changed\n")
    artifact_report = provenance.verify_manifest(manifest)
    assert artifact_report.status == "modified"
    assert "AFM-SEAL-004" in {item["code"] for item in artifact_report.issues}

    manifest, _ = _sealed_result(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["quantity_records"][0]["value"] = 999.0
    manifest.write_text(json.dumps(record), encoding="utf-8")
    manifest_report = provenance.verify_manifest(manifest)
    assert manifest_report.status == "modified"
    assert "AFM-SEAL-003" in {item["code"] for item in manifest_report.issues}


def test_missing_artifact_produces_truthful_incomplete_seal(tmp_path):
    result = results.SimulationResult("incomplete")
    result.add_artifact("future-field", tmp_path / "missing.xdmf")
    manifest = result.write_manifest(tmp_path / "result.json")
    report = provenance.verify_manifest(manifest)

    assert report.status == "incomplete"
    assert report.verified is False
    assert {item["code"] for item in report.issues} == {"AFM-SEAL-005"}


def test_cli_verify_is_machine_readable(tmp_path, capsys):
    manifest, _ = _sealed_result(tmp_path)

    assert cli.main(["verify", str(manifest), "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "agentfem.provenance-verification"
    assert record["status"] == "verified"


def test_cli_verify_follows_project_latest_pointer(tmp_path, capsys):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "case.py").write_text("pass\n", encoding="utf-8")
    (project_root / "agentfem.toml").write_text(
        "[project]\nname='sealed-project'\nentrypoint='case.py'\n"
        "[run]\noutput_directory='outputs'\n",
        encoding="utf-8",
    )
    run_directory = project_root / "outputs" / "sealed-project" / "run-1"
    run_directory.mkdir(parents=True)
    result = results.SimulationResult("latest-sealed")
    manifest = result.write_manifest(run_directory / "result.json")
    latest = run_directory.parent / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "schema": "agentfem.latest-run",
                "result_manifest": str(manifest),
            }
        ),
        encoding="utf-8",
    )

    assert cli.main(["verify", "--project", str(project_root), "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "verified"


def test_unsealed_legacy_manifest_remains_readable_but_not_verified(tmp_path):
    manifest = tmp_path / "legacy.json"
    manifest.write_text(
        json.dumps({"schema": "agentfem.simulation-result", "artifacts": {}}),
        encoding="utf-8",
    )

    report = provenance.verify_manifest(manifest)
    assert report.status == "unsealed"
    assert report.issues[0]["code"] == "AFM-SEAL-001"


def test_runtime_lock_round_trip_and_addressable_mismatch(tmp_path):
    lock = provenance.freeze_runtime(tmp_path / "runtime.lock.json")
    stored = json.loads(lock.read_text(encoding="utf-8"))
    matching = provenance.compare_runtime(stored, actual=deepcopy(stored))

    assert matching.compatible is True
    assert matching.mismatches == ()

    changed = deepcopy(stored)
    changed["identity"]["mpi"]["rank_count"] += 1
    changed["fingerprint"] = provenance.content_fingerprint(changed["identity"])
    comparison = provenance.compare_runtime(stored, actual=changed)
    assert comparison.compatible is False
    assert comparison.mismatches[0]["path"] == "mpi.rank_count"

    tampered = deepcopy(stored)
    tampered["identity"]["machine"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        provenance.compare_runtime(tampered, actual=stored)


def test_runtime_requirement_can_warn_or_refuse(monkeypatch):
    frozen = provenance.runtime_manifest()
    changed = deepcopy(frozen)
    changed["identity"]["packages"]["agentfem"] = "different"
    changed["fingerprint"] = provenance.content_fingerprint(changed["identity"])
    monkeypatch.setattr(provenance, "runtime_manifest", lambda: changed)

    with pytest.warns(RuntimeWarning, match="packages.agentfem"):
        report = provenance.require_runtime(frozen, policy="warn")
    assert report.compatible is False
    with pytest.raises(RuntimeError, match="packages.agentfem"):
        provenance.require_runtime(frozen, policy="error")


def test_scientific_input_manifest_hashes_files_arrays_and_declared_objects(tmp_path):
    class Declared:
        def summary(self):
            return {"kind": "material", "young_modulus": 210.0e9}

    mesh = tmp_path / "mesh.inp"
    mesh.write_text("*NODE\n1, 0, 0, 0\n", encoding="utf-8")
    inputs = {
        "mesh": mesh,
        "material": Declared(),
        "observer_points": np.asarray(((0.0, 0.0), (1.0, 0.0))),
    }

    first = provenance.scientific_input_manifest(inputs)
    mesh.write_text("*NODE\n1, 0, 0, 0\n2, 1, 0, 0\n", encoding="utf-8")
    second = provenance.scientific_input_manifest(inputs)

    assert first["complete"] is True
    assert first["record"]["mesh"]["status"] == "hashed"
    assert first["record"]["observer_points"]["kind"] == "array"
    assert first["fingerprint"] != second["fingerprint"]


def test_scientific_input_manifest_exposes_opaque_coverage_gap():
    class Opaque:
        pass

    manifest = provenance.scientific_input_manifest({"solver_object": Opaque()})

    assert manifest["complete"] is False
    assert manifest["missing"] == (
        {
            "path": "scientific_inputs.solver_object",
            "reason": "no_scientific_identity_contract",
        },
    )
    assert manifest["record"]["solver_object"]["fingerprinted"] is False


def test_result_manifest_carries_content_addressed_scientific_inputs(tmp_path):
    mesh = tmp_path / "mesh.inp"
    mesh.write_text("*NODE\n1, 0, 0, 0\n", encoding="utf-8")
    result = results.SimulationResult("identified")

    attached = result.add_scientific_inputs(
        mesh=mesh,
        material={"model": "elastic", "young": np.float64(210.0e9)},
    )
    saved = json.loads(
        result.write_manifest(tmp_path / "identified.json").read_text(
            encoding="utf-8"
        )
    )

    assert attached["complete"] is True
    assert saved["scientific_inputs"]["fingerprint"] == attached["fingerprint"]
    assert saved["scientific_inputs"]["record"]["mesh"]["status"] == "hashed"


def test_scientific_input_manifest_reports_recursive_containers():
    recursive = []
    recursive.append(recursive)

    manifest = provenance.scientific_input_manifest({"recursive": recursive})

    assert manifest["complete"] is False
    assert manifest["missing"] == (
        {
            "path": "scientific_inputs.recursive[0]",
            "reason": "cyclic_object_graph",
        },
    )


def test_callable_fingerprint_includes_bound_defaults_and_source_file():
    def evaluator_factory(scale):
        def evaluator(value, factor=scale):
            return factor * value

        return evaluator

    first = provenance.scientific_input_manifest(evaluator_factory(1.0))
    second = provenance.scientific_input_manifest(evaluator_factory(2.0))

    assert first["complete"] is True
    assert first["record"]["source_file_sha256"]
    assert first["record"]["defaults"] == [1.0]
    assert first["fingerprint"] != second["fingerprint"]


def test_empty_result_does_not_claim_complete_input_coverage():
    manifest = results.SimulationResult("undeclared").scientific_input_manifest()

    assert manifest["complete"] is False
    assert manifest["missing"] == (
        {
            "path": "result:undeclared",
            "reason": "no_scientific_inputs_declared",
        },
    )
