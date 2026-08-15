from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentfem import __version__, cli, project, results, upgrades


def test_cohesive_checkpoint_migration_is_explicit_and_nonmutating():
    source = {
        "schema": "agentfem.dof-mapped-cohesive-force.v3",
        "interface_identity": {"sha256": "physical-interface"},
        "law": {"mode": "normal", "initial_stiffness": 1000.0},
        "maximum_opening_by_key": {"facet:0": [0.1, 0.2]},
    }
    encoded = json.dumps(source, sort_keys=True)
    migrated = upgrades.migrate_cohesive_checkpoint(
        source,
        tangential="free",
    )
    assert json.dumps(source, sort_keys=True) == encoded
    assert migrated["schema"] == upgrades.COHESIVE_CHECKPOINT_SCHEMA
    assert migrated["interface_kinematics"] == "free"
    assert migrated["tangential_stiffness"] == 0.0
    assert migrated["state_by_field_and_key"] == {
        "maximum_opening": {"facet:0": [0.1, 0.2]}
    }
    with pytest.raises(ValueError, match="mixed-mode initiation"):
        upgrades.migrate_cohesive_checkpoint(
            source,
            tangential="mixed",
            acknowledge_physics_change=True,
        )


def test_project_config_and_run_context_are_portable(tmp_path):
    (tmp_path / "case.py").write_text("print('case')\n", encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        """[project]
name = "portable-case"
entrypoint = "case.py"

[run]
output_directory = "outputs"
""",
        encoding="utf-8",
    )
    config = project.ProjectConfig.load(tmp_path)
    assert config.check() == ()
    assert project.discover(tmp_path / "case.py") == config

    run = project.RunContext.create(config, run_id="test-run").prepare()
    assert run.artifact("fields/result.xdmf") == (
        tmp_path / "outputs" / "portable-case" / "test-run" / "fields" / "result.xdmf"
    )
    simulation = results.SimulationResult("portable")
    simulation.add_quantity("response", 2.0, unit="m")
    run.publish(simulation)

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    execution = json.loads(run.execution_path.read_text(encoding="utf-8"))
    latest = json.loads(
        (tmp_path / "outputs" / "portable-case" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "agentfem.simulation-result"
    assert execution["status"] == "completed"
    assert execution["structured_result"] is True
    assert latest["run_id"] == "test-run"


def test_artifacts_cannot_escape_run_directory(tmp_path):
    config = project.ProjectConfig(
        root=tmp_path,
        name="safe",
        entrypoint=tmp_path / "case.py",
        output_directory=tmp_path / "outputs",
    )
    run = project.RunContext.create(config, run_id="safe-run")
    try:
        run.artifact("../outside.txt")
    except ValueError as exc:
        assert "inside the run directory" in str(exc)
    else:
        raise AssertionError("Escaping artifact path should fail.")


def test_cli_init_and_check_use_installed_template(tmp_path):
    target = tmp_path / "case"
    assert cli.main(["init", str(target), "--name", "my-case"]) == 0
    assert (target / "case.py").is_file()
    assert (target / "agentfem.toml").is_file()
    assert cli.main(["check", "--project", str(target)]) == 0
    assert cli.main(["templates", "--json"]) == 0


def test_template_copy_ignores_runtime_cache_directories(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "templates" / "static-solid"
    cache = source / "__pycache__"
    cache.mkdir(exist_ok=True)
    target = tmp_path / "cached-template"
    try:
        assert cli.main(["init", str(target), "--template", "static-solid"]) == 0
        assert not (target / "__pycache__").exists()
    finally:
        try:
            cache.rmdir()
        except OSError:
            pass


def test_cli_check_reports_syntax_failure(tmp_path):
    (tmp_path / "case.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        "[project]\nname='broken'\nentrypoint='case.py'\n",
        encoding="utf-8",
    )
    assert cli.main(["check", "--project", str(tmp_path), "--json"]) == 2


def test_capability_command_is_json_serializable(capsys):
    assert cli.main(["capabilities", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "agentfem.capabilities"
    assert "upgrade" in record["commands"]
    assert "verify" in record["commands"]
    assert "extensions" in record["commands"]
    assert "project" in record["public_modules"]
    assert "extensions" in record["public_modules"]
    assert "provenance" in record["public_modules"]
    assert "models" in record["public_api"]["core"]
    assert "surrogates" in record["public_api"]["advanced"]
    assert "backends" in record["public_api"]["expert"]
    assert set(record["public_modules"]) == set().union(
        *map(set, record["public_api"].values())
    )
    assert any(item["name"] == "linear_elasticity" for item in record["constitutive"])
    assert record["extensions"]["schema"] == "agentfem.extensions"


def test_upgrade_report_is_location_aware_and_does_not_rewrite_case(tmp_path):
    case = tmp_path / "case.py"
    source = """from agentfem import io\nwith io.XDMFTimeSeries('old.xdmf', domain) as writer:\n    writer.write_fields(0.0, u)\n"""
    case.write_text(source, encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        "[project]\nname='legacy'\nentrypoint='case.py'\n",
        encoding="utf-8",
    )

    report = upgrades.inspect_project(project.ProjectConfig.load(tmp_path))

    assert report.status == "review_recommended"
    assert {item.code for item in report.findings} == {
        "AFM-UPG-004",
        "AFM-UPG-101",
    }
    xdmf = next(item for item in report.findings if item.code == "AFM-UPG-101")
    assert xdmf.line == 2
    assert xdmf.semantic_review is True
    assert case.read_text(encoding="utf-8") == source


def test_upgrade_can_apply_only_safe_metadata_and_write_plan(tmp_path):
    (tmp_path / "case.py").write_text("print('safe')\n", encoding="utf-8")
    config = tmp_path / "agentfem.toml"
    original = "[project]\nname='safe'\nentrypoint='case.py'\n"
    config.write_text(original, encoding="utf-8")

    assert cli.main(
        [
            "upgrade",
            "--project",
            str(tmp_path),
            "--apply-safe",
            "--write-plan",
            "upgrade.json",
            "--json",
        ]
    ) == 0

    assert 'schema_version = "0.2.0"' in config.read_text(encoding="utf-8")
    assert config.with_suffix(".toml.bak").read_text(encoding="utf-8") == original
    saved = json.loads((tmp_path / "upgrade.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "agentfem.upgrade-report"
    assert saved["status"] == "current"


def test_check_blocks_project_schema_newer_than_runtime(tmp_path):
    (tmp_path / "case.py").write_text("print('future')\n", encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        "[project]\nname='future'\nentrypoint='case.py'\nschema_version='99.0.0'\n",
        encoding="utf-8",
    )

    assert cli.main(["check", "--project", str(tmp_path), "--json"]) == 2


def test_safe_upgrade_replaces_an_older_operational_schema(tmp_path):
    (tmp_path / "case.py").write_text("print('old')\n", encoding="utf-8")
    config = tmp_path / "agentfem.toml"
    config.write_text(
        "[project]\nname='old'\nentrypoint='case.py'\nschema_version='0.0.1'\n",
        encoding="utf-8",
    )

    changed = upgrades.apply_safe_metadata(project.ProjectConfig.load(tmp_path))

    assert changed
    assert 'schema_version="0.2.0"' in config.read_text(encoding="utf-8")


def test_upgrade_scans_project_modules_but_ignores_generated_outputs(tmp_path):
    (tmp_path / "case.py").write_text("from helpers import area\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text(
        "area = results.integral(1.0, measure=surface.measure)\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "generated.py").write_text(
        "with io.XDMFTimeSeries('ignored.xdmf', domain):\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "agentfem.toml").write_text(
        "[project]\nname='modules'\nentrypoint='case.py'\nschema_version='0.2.0'\n",
        encoding="utf-8",
    )

    report = upgrades.inspect_project(project.ProjectConfig.load(tmp_path))

    assert [(item.code, item.path.name) for item in report.findings] == [
        ("AFM-UPG-103", "helpers.py")
    ]


def test_all_installed_templates_are_current_and_version_stamped(tmp_path):
    for template in cli._templates():
        target = tmp_path / template
        assert cli.main(
            ["init", str(target), "--template", template, "--name", template]
        ) == 0
        config = project.ProjectConfig.load(target)
        report = upgrades.inspect_project(config)
        assert config.created_with == __version__
        assert report.status == "current"
        source = config.entrypoint.read_text(encoding="utf-8")
        assert "io.XDMFTimeSeries" not in source
