from pathlib import Path
import subprocess
import tomllib

import pytest

import release_gate
from agentfem import __version__, cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unreleased_checkout_does_not_reuse_a_published_tag_version():
    """Keep a moving checkout distinguishable from its latest release."""

    if not (PROJECT_ROOT / ".git").exists():
        pytest.skip("release identity requires a Git checkout")

    exact = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "--match", "v[0-9]*"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if exact.returncode == 0:
        assert exact.stdout.strip() == f"v{__version__}"
        return

    latest = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if latest.returncode != 0:
        pytest.skip("checkout has no release tag history")
    assert latest.stdout.strip() != f"v{__version__}", (
        "HEAD has moved beyond the latest release but still reports its published "
        "version; advance to the next development or release-candidate version"
    )


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


def test_release_gate_exercises_agent_machine_entrypoints():
    source = (PROJECT_ROOT / "release_gate.py").read_text(encoding="utf-8")
    assert "run_agent_entrypoint_smoke" in source
    for command in (
        "doctor",
        "workspace",
        "capabilities",
        "init",
        "check",
        "run",
        "inspect",
        "verify",
    ):
        assert f'"{command}"' in source


def test_platform_acceptance_distinguishes_native_and_wsl_routes(tmp_path, monkeypatch):
    wheel = tmp_path / "agentfem.whl"
    wheel.write_bytes(b"immutable candidate")
    monkeypatch.setattr(release_gate, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(release_gate, "_source_identity", lambda: ("b" * 40, False))
    acceptance = {
        "agentfem_version": __version__,
        "runtime_fingerprint": {
            "platform": {"route": "Windows via WSL2/Linux"},
            "operating_system": {
                "system": "Linux",
                "release": "6.6.87.2-microsoft-standard-WSL2",
                "version": "x",
            },
            "python": "3.11.0",
            "machine": "x86_64",
            "packages": {"agentfem": __version__},
            "mpi": {"rank_count": 1},
        },
        "templates": {
            "static-solid": {"provenance": "verified"},
        },
        "mpi_smoke": {"status": "passed", "rank_count": 2},
    }

    report = release_gate.platform_acceptance(acceptance, wheel=wheel)

    assert report["schema"] == "agentfem.platform-acceptance"
    assert report["platform_id"] == "wsl2"
    assert report["status"] == "passed"
    assert report["installed_wheel"] is True
    assert report["source_commit"]
    assert "generated_at" in report

    weak = release_gate.platform_acceptance(
        {**acceptance, "mpi_smoke": None},
        wheel=wheel,
    )
    assert weak["status"] == "failed"


def test_wsl2_platform_requirement_rejects_wsl1_and_requires_mpi():
    accepted = {
        "status": "passed",
        "platform_id": "wsl2",
        "route": "Windows via WSL2/Linux",
        "operating_system": {
            "system": "Linux",
            "release": "6.6.87.2-microsoft-standard-WSL2",
        },
        "wsl": {"kernel_mentions_wsl2": True},
        "mpi_smoke": {"status": "passed", "rank_count": 2},
    }

    release_gate.require_platform_acceptance(accepted, expected="wsl2")

    with pytest.raises(RuntimeError, match="two-rank"):
        release_gate.require_platform_acceptance(
            {**accepted, "mpi_smoke": None},
            expected="wsl2",
        )
    with pytest.raises(RuntimeError, match="real"):
        release_gate.require_platform_acceptance(
            {
                **accepted,
                "route": "Windows via WSL1/Linux",
                "wsl": {"kernel_mentions_wsl2": False},
            },
            expected="wsl2",
        )


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
    assert "workflow_dispatch:" in workflow
    assert "run-id: ${{ inputs.candidate_run_id }}" in workflow
    assert 'actions/runs/${CANDIDATE_RUN_ID}' in workflow
    assert "--jq .head_sha" in workflow
    assert "--site-dir /tmp/agentfem-release-site" in workflow
    assert "python release_gate.py --dist dist --tag \"${RELEASE_TAG}\" --smoke" in workflow
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
