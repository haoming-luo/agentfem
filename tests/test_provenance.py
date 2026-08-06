from __future__ import annotations

import json

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
