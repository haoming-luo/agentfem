import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile

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


def test_release_gate_ignores_repaired_wheel_directory_records(tmp_path):
    wheel = tmp_path / "agentfem-0.4.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("agentfem/", "")
        archive.writestr("agentfem/backends/", "")
        archive.writestr("agentfem/__init__.py", "__version__ = '0.4.3'\n")
        archive.writestr("agentfem/backends/base.py", "class Backend: pass\n")

    assert release_gate._archive_members(wheel) == {
        "agentfem/__init__.py",
        "agentfem/backends/base.py",
    }


def test_release_gate_ignores_source_archive_directory_records(tmp_path):
    source = tmp_path / "agentfem-0.4.3.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        directory = tarfile.TarInfo("agentfem-0.4.3/src/agentfem")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        payload = b"__version__ = '0.4.3'\n"
        member = tarfile.TarInfo("agentfem-0.4.3/src/agentfem/__init__.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    assert release_gate._archive_members(source) == {
        "src/agentfem/__init__.py",
    }


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

    previous = json.loads(
        (PROJECT_ROOT / "src/agentfem/release/0.3.2.json").read_text(
            encoding="utf-8"
        )
    )
    workflow_ids = {item["id"] for item in contract["workflows"]}
    workflow_maturity = {
        item["id"]: item["maturity"] for item in contract["workflows"]
    }
    previous_ids = {item["id"] for item in previous["workflows"]}
    assert previous_ids <= workflow_ids
    assert set(previous["required_gates"]) <= set(contract["required_gates"])
    assert {
        "structural-modal",
        "structural-direct-harmonic",
        "generalized-maxwell-transient",
        "generalized-maxwell-harmonic",
    } <= workflow_ids
    assert workflow_maturity["structural-direct-harmonic"] == "engineering"
    assert workflow_maturity["generalized-maxwell-harmonic"] == "experimental"
    if "bounded-serial-periodic-j2-contact" in workflow_maturity:
        assert workflow_maturity["bounded-serial-periodic-j2-contact"] == "experimental"
        assert "mesh-independent nonuniform periodic plastic-contact response without spatial and path convergence" in contract["not_release_claims"]
    assert {
        "partitioned-finite-strain-explicit",
        "multi-material-thermoelastic-static",
        "composite-material-and-fabric-foundation",
        "mesh-cell-compatibility-and-quality",
    } <= workflow_ids
    assert (
        workflow_maturity["partitioned-finite-strain-explicit"] == "experimental"
    )
    assert (
        workflow_maturity["multi-material-thermoelastic-static"] == "engineering"
    )
    assert (
        workflow_maturity["composite-material-and-fabric-foundation"]
        == "experimental"
    )
    assert {
        "nafems-r0016-test5h-external-comparison",
        "harmonic-three-level-observable-stability",
        "harmonic-two-rank-complete-sweep-equivalence",
        "harmonic-cross-rank-scalar-checkpoint-portability",
        "harmonic-prepared-configuration-fail-closed",
        "harmonic-nonfinite-evidence-rejection",
    } <= set(contract["required_gates"])
    assert {
        "NAFEMS-authored Test 5H acceptance tolerances",
        (
            "external generalized-Maxwell harmonic validation from the elastic "
            "Rayleigh-damped Test 5H benchmark"
        ),
        "full-field harmonic checkpoint recovery",
    } <= set(contract["not_release_claims"])
    assert {
        "thermoelastic-free-expansion-and-region-partition",
        "finite-strain-explicit-material-load-partition",
        "fibrous-shell-hybrid-serial-and-two-rank",
        "rotation-free-edge-ownership",
        "mesh-cell-compatibility-contract",
        "quad-hex-real-import-and-quality",
        "unknown-cell-fail-closed",
    } <= set(contract["required_gates"])
    if contract["target_version"].startswith(("0.3.8", "0.4.")):
        foundation_gate_ids = {
            item["id"] for item in contract.get("foundation_gates", ())
        }
        assert contract.get("promotion_target") == "0.4-foundation"
        assert foundation_gate_ids == {
            "mesh-element-function-space-compatibility",
            "mesh-quality-and-unsupported-topology-fail-closed",
            "typed-time-input-effects",
            "constraint-force-work-energy-closure",
            "mpi-result-state-checkpoint-lifecycle",
            "external-provider-without-core-modification",
        }
        assert foundation_gate_ids <= set(contract["required_gates"])
    assert {
        "forming-capable composite shell, tool contact, friction or inter-ply slip",
        "executable rotation-free shell edge constraints or boundary moments",
        "universal multi-material support across every analysis provider",
    } <= set(contract["not_release_claims"])


@pytest.mark.parametrize("tag", [None, f"v{__version__}"])
def test_release_gate_rejects_stale_contract_content_without_relying_on_filename(
    tmp_path, monkeypatch, tag,
):
    record = json.loads(release_gate.release_contract_path().read_text())
    record["target_version"] = "0.0.0"
    path = tmp_path / f"{__version__}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(release_gate, "release_contract_path", lambda: path)

    with pytest.raises(RuntimeError, match="not package version"):
        release_gate.check_release_contract(tag=tag)


@pytest.mark.parametrize("payload, message", [
    ([], "must be a JSON object"),
    (None, "must be a JSON object"),
    ({"workflows": [None]}, "workflow must be a JSON object"),
])
def test_release_gate_reports_invalid_contract_structure(
    tmp_path, monkeypatch, payload, message,
):
    if isinstance(payload, dict):
        record = json.loads(release_gate.release_contract_path().read_text())
        record.update(payload)
        payload = record
    path = tmp_path / f"{__version__}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(release_gate, "release_contract_path", lambda: path)

    with pytest.raises(RuntimeError, match=message):
        release_gate.check_release_contract()


@pytest.mark.parametrize("arguments, scope", [
    ([], "source"),
    (["--dist", "candidate"], "source and distribution"),
    (["--smoke"], "source and installed-use smoke"),
    (["--dist", "candidate", "--smoke"], "source, distribution and installed-use smoke"),
])
def test_release_gate_reports_only_the_checks_actually_requested(
    monkeypatch, capsys, arguments, scope,
):
    monkeypatch.setattr(sys, "argv", ["release_gate.py", *arguments])
    monkeypatch.setattr(release_gate, "check_versions", lambda **_: __version__)
    monkeypatch.setattr(release_gate, "check_dependency_boundaries", lambda: None)
    monkeypatch.setattr(release_gate, "check_release_contract", lambda **_: {})
    monkeypatch.setattr(release_gate, "check_distributions", lambda _: Path("candidate.whl"))
    monkeypatch.setattr(release_gate, "run_smoke", lambda **_: {})

    release_gate.main()

    assert capsys.readouterr().out.strip() == (
        f"AgentFEM {__version__} release gate passed ({scope} checks only)."
    )


def test_release_wheel_requires_the_direct_harmonic_evidence_chain():
    assert {
        "agentfem/backends/_harmonic.py",
        "agentfem/benchmarks/forced_vibration.py",
        "agentfem/checkpointing.py",
        "agentfem/mechanics/harmonic.py",
        "agentfem/operators/harmonic.py",
        "agentfem/operators/identity.py",
        "agentfem/results/_harmonic.py",
        "agentfem/results/harmonic.py",
        (
            "agentfem/knowledge/benchmarks/"
            "nafems_r0016_test5h_forced_vibration.json"
        ),
        (
            "agentfem/knowledge/benchmarks/"
            "nafems_r0016_test5h_spatial_refinement.json"
        ),
    } <= set(release_gate.REQUIRED_WHEEL_MEMBERS)


def test_release_gate_exercises_every_installed_project_template():
    assert release_gate.INSTALLED_PROJECT_TEMPLATES == cli._templates()


@pytest.mark.parametrize("workflow", ["test.yml", "publish-pypi.yml"])
def test_ci_and_publication_keep_distributed_recovery_and_rejection_gates(workflow):
    source = (PROJECT_ROOT / ".github" / "workflows" / workflow).read_text()
    for gate in (
        "tests/test_quadrature_physical_recovery.py",
        "tests/test_finite_strain_j2_contact_mpi_boundary.py",
        "tests/test_affine_adaptive_preconditioner.py::test_distributed_reuse_rejected_before_mutation",
    ):
        assert gate in source, f"{workflow} omits the distributed gate {gate}"


def test_release_smoke_includes_both_global_viscoelastic_routes():
    examples = {relative for relative, _arguments in release_gate.SMOKE_COMMANDS}

    assert "examples/viscoelastic_relaxation_3d.py" in examples
    assert "examples/viscoelastic_harmonic_3d.py" in examples


def test_release_smoke_includes_bounded_installed_contact_routes():
    examples = {relative for relative, _arguments in release_gate.SMOKE_COMMANDS}

    assert {
        "examples/finite_j2_opposing_rigid_tools.py",
        "examples/finite_j2_closed_convex_contact.py",
        "examples/finite_j2_moving_plane_friction.py",
        "examples/finite_j2_hertz_contact.py",
    } <= examples
    hertz_arguments = dict(release_gate.SMOKE_COMMANDS)[
        "examples/finite_j2_hertz_contact.py"
    ]
    assert "--displacement-degree" in hertz_arguments
    assert "--linear-policy" in hertz_arguments


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


def test_source_distribution_retains_the_external_provider_acceptance_assets():
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "release_gate.py").read_text(encoding="utf-8")

    assert "include extension_gate.py" in manifest
    assert "include foundation_gate.py" in manifest
    assert "recursive-include examples/extensions *.toml" in manifest
    assert '"extension_gate.py"' in source
    assert '"foundation_gate.py"' in source
    assert '"examples/extensions/reference_material/pyproject.toml"' in source


def test_release_workflow_fans_in_linux_and_macos_foundation_evidence():
    workflow = (PROJECT_ROOT / ".github/workflows/test.yml").read_text(
        encoding="utf-8"
    )

    assert "macos-foundation-platform:" in workflow
    assert "needs:\n      - change-scope\n      - fenicsx" in workflow
    assert "name: tested-python-package-distributions" in workflow
    assert 'name: agent-acceptance-evidence' in workflow
    assert "release_gate.py --dist dist --smoke --mpi-ranks 2" in workflow
    assert (
        '--evidence "${LINUX}/agentfem-platform-acceptance.json"'
        in workflow
    )
    assert (
        "--evidence /tmp/agentfem-macos-platform-acceptance.json"
        in workflow
    )
    assert 'report.get("status") != "passed"' in workflow


def test_pypi_publication_requires_exact_wheel_foundation_audit():
    workflow = (PROJECT_ROOT / ".github/workflows/publish-pypi.yml").read_text(
        encoding="utf-8"
    )
    assert "needs: [verify, foundation-verify, ml-verify, chaboche-release-verify]" in workflow
    assert "foundation_gate.py" in workflow
    assert "extension_gate.py" in workflow
    assert "--target 0.4-foundation --require-complete" in workflow
    assert "--evidence-directory /tmp/agentfem-foundation-linux" in workflow
    macos_job = workflow.split("  foundation-verify:", 1)[1].split("  ml-verify:", 1)[0]
    assert "needs: verify" in macos_job
    assert "name: python-package-distributions" in macos_job
    assert "--require-platform macos" in macos_job
    assert "python -m build" not in macos_job
    assert "build_platform_wheel.py --source-dir dist" in macos_job
    assert "--dist /tmp/agentfem-release-macos-dist" in macos_job
    assert "import agentfem._hex_native, agentfem._body_pipeline" in macos_job


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


def test_history_integration_checks_installed_project_and_release_tag_history():
    workflow = (PROJECT_ROOT / ".github/workflows/test.yml").read_text()
    numerical = workflow.split("  fenicsx:\n", 1)[1].split("  macos", 1)[0]
    assert "fetch-depth: 0" in numerical
    assert "tools/check_installed_history_project.py" in numerical
    checker = (PROJECT_ROOT / "tools/check_installed_history_project.py").read_text()
    assert 'environment.pop("PYTHONPATH", None)' in checker
    assert '"-m", "agentfem._cli_entry"' in checker
    assert "sys.path.insert" not in checker
    record = release_gate.check_release_contract(source_root=PROJECT_ROOT)
    workflows = {item["id"]: item for item in record["workflows"]}
    for name in (
        "bounded-finite-strain-history-material",
        "bounded-thermal-cyclic-creep",
        "bounded-strain-life-creep-assessment",
    ):
        assert workflows[name]["maturity"] == "experimental"


def test_publish_workflow_verifies_the_same_artifacts_it_builds_once():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "publish-pypi.yml"
    ).read_text(encoding="utf-8")

    build_job = workflow.split("  build:", 1)[1].split("  verify:", 1)[0]
    assert build_job.count("python -m build") == 1
    # The other build is the independent provider, never a replacement core wheel.
    assert workflow.count("python -m build") == 2
    assert "--source-dir source-dist --output dist --repair linux" in build_job
    assert "examples/extensions/reference_material" in workflow
    assert "workflow_dispatch:" in workflow
    assert "run-id: ${{ inputs.candidate_run_id }}" in workflow
    assert 'actions/runs/${CANDIDATE_RUN_ID}' in workflow
    assert "--jq .head_sha" in workflow
    assert "--site-dir /tmp/agentfem-release-site" in workflow
    assert "python release_gate.py --dist dist --tag \"${RELEASE_TAG}\" --smoke" in workflow
    assert "needs: [verify, foundation-verify, ml-verify, chaboche-release-verify]" in workflow
    assert "Verify every published ratcheting cycle" in workflow
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


def test_test_workflow_proves_the_external_provider_from_two_wheels():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "test.yml"
    ).read_text(encoding="utf-8")

    assert "Prove an independently installed external provider" in workflow
    assert "python extension_gate.py" in workflow
    assert "--extension agentfem-reference-material" in workflow
    assert (
        '--evidence "${LINUX}/agentfem-extension-acceptance.json"'
        in workflow
    )
    assert "python foundation_gate.py" in workflow
    assert "--target 0.4-foundation" in workflow
    assert (
        '--evidence "${LINUX}/agentfem-foundation-acceptance.json"'
        in workflow
    )


def test_source_and_installed_distribution_evidence_are_separate():
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert "_check_installed_identity(wheel=wheel)" in (
        PROJECT_ROOT / "release_gate.py"
    ).read_text()
