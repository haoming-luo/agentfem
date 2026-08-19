from pathlib import Path
import tomllib

import release_gate
from agentfem import __version__, cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_gate_rejects_build_machine_bytecode_and_cache_members():
    members = {
        "agentfem/models.py",
        "agentfem/__pycache__/models.cpython-311.pyc",
        "agentfem/old.pyo",
    }

    assert release_gate._forbidden_distribution_members(members) == [
        "agentfem/__pycache__/models.cpython-311.pyc",
        "agentfem/old.pyo",
    ]


def test_release_contract_is_complete_and_references_real_workflows():
    contract = release_gate.check_release_contract(source_root=PROJECT_ROOT)

    assert contract["target_version"] == __version__
    assert release_gate.release_contract_path().name == f"{__version__}.json"
    assert {item["maturity"] for item in contract["workflows"]} <= {
        "release",
        "engineering",
        "experimental",
    }
    assert len(contract["required_gates"]) >= 10


def test_release_gate_exercises_every_installed_project_template():
    assert release_gate.INSTALLED_PROJECT_TEMPLATES == cli._templates()


def test_release_facing_examples_never_insert_the_checkout_into_sys_path():
    for relative, _ in release_gate.SMOKE_COMMANDS:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, relative
        assert "SOURCE_PARENT" not in source, relative


def test_publish_workflow_verifies_the_same_artifacts_it_builds_once():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "publish-pypi.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("python -m build") == 1
    assert "python release_gate.py --dist dist --tag \"${GITHUB_REF_NAME}\" --smoke" in workflow
    assert "needs: [verify, ml-verify]" in workflow
    assert "needs: attest" in workflow


def test_test_workflow_runs_the_versioned_critical_static_analysis_gate():
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "test.yml"
    ).read_text(encoding="utf-8")

    assert configuration["tool"]["ruff"]["target-version"] == "py311"
    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert configuration["tool"]["ruff"]["lint"]["select"] == [
        "E9",
        "F63",
        "F7",
        "F82",
        "B023",
        "RUF009",
    ]
    assert "ruff check . --no-cache" in workflow


def test_source_and_installed_distribution_evidence_are_separate():
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert "_check_installed_identity(wheel=wheel)" in (
        PROJECT_ROOT / "release_gate.py"
    ).read_text()
