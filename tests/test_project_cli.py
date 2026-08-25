from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentfem import __version__, cli, materials, project, results, upgrades


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


def test_failed_run_records_stage_traceback_and_machine_error_fields(tmp_path):
    (tmp_path / "case.py").write_text(
        "raise RuntimeError('deliberate failure')\n",
        encoding="utf-8",
    )
    (tmp_path / "agentfem.toml").write_text(
        "[project]\nname='failed-case'\nentrypoint='case.py'\n",
        encoding="utf-8",
    )

    assert cli.main(
        [
            "run",
            "--project",
            str(tmp_path),
            "--run-id",
            "failed-run",
            "--json",
        ]
    ) == 1

    execution = json.loads(
        (
            tmp_path
            / "outputs"
            / "failed-case"
            / "failed-run"
            / "execution.json"
        ).read_text(encoding="utf-8")
    )
    assert execution["status"] == "failed"
    assert execution["stage"] == "case_execution"
    assert execution["error"]["stage"] == "case_execution"
    assert execution["error"]["code"] is None
    assert "RuntimeError: deliberate failure" in execution["error"]["traceback"]


def test_cli_init_and_check_use_installed_template(tmp_path):
    target = tmp_path / "case"
    assert cli.main(["init", str(target), "--name", "my-case"]) == 0
    assert (target / "case.py").is_file()
    assert (target / "agentfem.toml").is_file()
    assert cli.main(["check", "--project", str(target)]) == 0
    assert cli.main(["templates", "--json"]) == 0


def test_template_copy_ignores_runtime_cache_directories(tmp_path, monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentfem"
        / "templates"
        / "static-solid"
    )
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
    assert "procedures" in record["public_api"]["advanced"]
    assert "solvers" in record["public_api"]["advanced"]
    assert "io" in record["public_api"]["expert"]
    assert "problems" in record["public_api"]["expert"]
    assert "time" in record["public_api"]["expert"]
    assert "io" not in record["public_api"]["core"]
    assert "step" in record["model_api"]["core"]
    assert "stiffness" in record["model_api"]["advanced"]
    assert "linear_static_step" in record["model_api"]["compatibility"]
    lifecycle = {
        item["name"]: item for item in record["model_api_contract"]
    }
    assert lifecycle["step"]["lifecycle"] == "recommended"
    assert lifecycle["linear_static_step"]["replacement"] == "model.step(...)"
    assert set(record["public_modules"]) == set().union(
        *map(set, record["public_api"].values())
    )
    assert any(item["name"] == "linear_elasticity" for item in record["constitutive"])
    evidence = {
        item["capability"]: item
        for item in record["constitutive_evidence"]
    }
    assert evidence["linear_elasticity"]["meets_declared_maturity"] is True
    assert evidence["mixed_mode_cohesive_interface"]["maturity"].startswith(
        "experimental_"
    )
    linear = next(
        item
        for item in record["step_providers"]
        if item["name"] == "linear_static_operators"
    )
    assert "solver_options" in linear["options"]["accepted"]
    assert record["extensions"]["schema"] == "agentfem.extensions"


def test_cli_inspects_abaqus_deck_without_converting_or_solving(tmp_path, capsys):
    source = tmp_path / "one_hex.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1,0,0,0", "2,1,0,0", "3,1,1,0", "4,0,1,0",
                "5,0,0,1", "6,1,0,1", "7,1,1,1", "8,0,1,1",
                "*Element, type=C3D8R, elset=SOLID",
                "1,1,2,3,4,5,6,7,8",
                "*Material, name=STEEL",
                "*Elastic",
                "2e11,0.3",
            )
        ),
        encoding="utf-8",
    )
    report = tmp_path / "migration.json"

    assert cli.main(
        ["inspect-abaqus", str(source), "--write", str(report), "--json"]
    ) == 0

    emitted = json.loads(capsys.readouterr().out)
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert emitted["schema"] == "agentfem.abaqus-migration-report"
    assert emitted["topology_only_elements"] == ["C3D8R"]
    assert saved["source_sha256"] == emitted["source_sha256"]
    assert emitted["source_graph"]["complete"] is True
    assert emitted["source_graph"]["files"][0]["logical_path"] == "one_hex.inp"


def test_cli_creates_fail_closed_abaqus_migration_project(tmp_path, capsys):
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    source = source_root / "model.inp"
    included = source_root / "mesh.inc"
    source.write_text(
        "\n".join(
            (
                "*Include, input=mesh.inc",
                "*Solid Section, elset=SOLID, material=STEEL",
                "*Material, name=STEEL",
                "*Elastic",
                "210000.,0.3",
                "*Density",
                "7.85e-9",
            )
        ),
        encoding="utf-8",
    )
    included.write_text(
        "*Node\n1,0,0,0\n*Element, type=C3D4, elset=SOLID\n1,1,1,1,1\n",
        encoding="utf-8",
    )
    target = tmp_path / "migrated"

    assert (
        cli.main(
            [
                "migrate-abaqus",
                str(source),
                str(target),
                "--name",
                "legacy-bracket",
                "--json",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    migration = json.loads((target / "migration.json").read_text(encoding="utf-8"))

    assert record["schema"] == "agentfem.abaqus-migration-project"
    assert migration["schema"] == "agentfem.abaqus-migration-plan"
    assert migration["ready_to_solve"] is False
    assert Path(record["migration_report"]).is_file()
    assert "Abaqus migration review" in (target / "migration.md").read_text()
    assert "Native lowering gate" in (target / "migration.md").read_text()
    assert (target / "source" / "model.inp").is_file()
    assert (target / "source" / "mesh.inc").is_file()
    candidates = (target / "materials" / "candidates.py").read_text()
    assert "young=210000.0" in candidates
    assert '"migration_status": "candidate_not_activated"' in candidates
    assert "MATERIAL_CANDIDATES" in candidates
    loaded = materials.load(
        target / "materials" / "candidates.py",
        symbol="material_1",
    )
    assert loaded.name == "STEEL"
    assert loaded.behavior("mechanical").density == 7.85e-9
    assert loaded.metadata["migration_status"] == "candidate_not_activated"
    assert "fail-closed migration scaffold" in (target / "case.py").read_text()
    assert cli.main(["check", "--project", str(target), "--json"]) == 0


def test_cli_refuses_incomplete_abaqus_source_graph_atomically(tmp_path, capsys):
    source = tmp_path / "model.inp"
    source.write_text("*Include, input=missing.inc\n", encoding="utf-8")
    target = tmp_path / "migrated"

    assert cli.main(["migrate-abaqus", str(source), str(target), "--json"]) == 2

    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed"
    assert "incomplete Abaqus source graph" in failure["error"]["message"]
    assert not target.exists()


def test_cli_lowers_and_activates_reviewed_abaqus_static_project(tmp_path, capsys):
    source = tmp_path / "static.inp"
    source.write_text(
        "\n".join(
            (
                "*Node",
                "1,0,0,0",
                "2,1,0,0",
                "3,0,1,0",
                "4,0,0,1",
                "*Nset, nset=FIXED",
                "1,2,3",
                "*Nset, nset=MOVED",
                "4",
                "*Element, type=C3D4, elset=SOLID",
                "1,1,2,3,4",
                "*Material, name=MAT",
                "*Elastic",
                "1000.,0.3",
                "*Density",
                "1.0",
                "*Solid Section, elset=SOLID, material=MAT",
                "*Step, name=LOAD",
                "*Static",
                "*Boundary",
                "FIXED,1,3,0.",
                "MOVED,1,2,0.",
                "MOVED,3,3,0.01",
                "*End Step",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "migrated"
    assert cli.main(["migrate-abaqus", str(source), str(target), "--json"]) == 0
    capsys.readouterr()

    assert (
        cli.main(
            [
                "lower-abaqus",
                str(target),
                "--reviewed-by",
                "Test Engineer",
                "--unit-system",
                "SI",
                "--activate",
                "--json",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "activated"
    assert Path(record["entrypoint"]).is_file()
    assert cli.main(["check", "--project", str(target), "--json"]) == 0


def test_cli_reports_structured_abaqus_lowering_findings(tmp_path, capsys):
    source = tmp_path / "unsupported.inp"
    source.write_text(
        "*Node\n1,0,0,0\n*Element, type=C3D8R, elset=SOLID\n"
        "1,1,1,1,1,1,1,1,1\n",
        encoding="utf-8",
    )
    target = tmp_path / "migrated"
    assert cli.main(["migrate-abaqus", str(source), str(target), "--json"]) == 0
    capsys.readouterr()

    assert (
        cli.main(
            [
                "lower-abaqus",
                str(target),
                "--reviewed-by",
                "Test Engineer",
                "--unit-system",
                "SI",
                "--json",
            ]
        )
        == 2
    )
    failure = json.loads(capsys.readouterr().out)
    details = failure["error"]["details"]
    assert details["status"] == "blocked"
    assert "AFM-ABAQUS-LOWER-ELEMENT-002" in {
        item["code"] for item in details["findings"]
    }


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


def test_upgrade_reports_material_specific_step_spelling_without_rewriting(tmp_path):
    case = tmp_path / "case.py"
    source = "step = model.hyperelastic_step(target=u, material=material)\n"
    case.write_text(source, encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        "[project]\nname='legacy-step'\nentrypoint='case.py'\n",
        encoding="utf-8",
    )

    report = upgrades.inspect_project(project.ProjectConfig.load(tmp_path))
    finding = next(item for item in report.findings if item.code == "AFM-UPG-104")

    assert finding.replacement == "model.step(...)"
    assert finding.semantic_review is True
    assert finding.automatic is False
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
