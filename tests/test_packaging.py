from pathlib import Path

from agentfem import cli, release_gate


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

    assert contract["target_version"] == "0.2.0a3"
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
