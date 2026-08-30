from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from agentfem import cli, portability, project, results
from agentfem.backends import runtime as backend_runtime
from agentfem.project_bundle import inspect_bundle, pack_project, unpack_bundle


def _write_project(root: Path) -> project.ProjectConfig:
    root.mkdir()
    (root / "case.py").write_text(
        """from agentfem import project, results
run = project.current_run()
result = results.SimulationResult("portable-bundle")
result.add_quantity("answer", 42.0, unit="1")
run.publish(result)
""",
        encoding="utf-8",
    )
    (root / "mesh.dat").write_text("scientific asset\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=must-not-travel\n", encoding="utf-8")
    (root / "outputs").mkdir()
    (root / "outputs" / "old.txt").write_text("old output\n", encoding="utf-8")
    (root / "agentfem.toml").write_text(
        """[project]
name = "portable"
entrypoint = "case.py"
schema_version = "0.2.0"

[run]
output_directory = "outputs"
default_profile = "local"

[execution.local]
runtime = "auto"
ranks = 1

[execution.cluster]
runtime = "fenicsx-petsc"
ranks = 8
required_capabilities = ["mpi_distributed_mesh"]
""",
        encoding="utf-8",
    )
    return project.ProjectConfig.load(root)


def test_execution_profiles_are_operational_not_scientific(tmp_path):
    config = _write_project(tmp_path / "project")
    assert config.check() == ()
    assert config.execution_profile().name == "local"
    cluster = config.execution_profile("cluster")
    assert cluster.runtime == "fenicsx-petsc"
    assert cluster.ranks == 8
    assert cluster.capabilities == ("mpi_distributed_mesh",)
    summary = config.summary()
    assert summary["default_profile"] == "local"
    assert len(summary["execution_profiles"]) == 2


def test_cluster_profile_fails_closed_on_native_only_runtime(tmp_path, monkeypatch):
    config = _write_project(tmp_path / "project")
    available = {"dolfinx", "scipy"}
    monkeypatch.setattr(
        backend_runtime,
        "_available",
        lambda module: module in available,
    )
    record = cli._check_project(
        config,
        profile_name="cluster",
        check_runtime=True,
    )
    assert record["status"] == "failed"
    compatibility = record["runtime_compatibility"]
    assert compatibility["status"] == "incompatible"
    assert compatibility["error"]["code"] == "AFM-BACKEND-RUNTIME-001"


def test_execution_profile_rejects_scientific_or_solver_keys(tmp_path):
    root = tmp_path / "invalid"
    root.mkdir()
    (root / "case.py").write_text("pass\n", encoding="utf-8")
    (root / "agentfem.toml").write_text(
        """[project]
name = "invalid"
entrypoint = "case.py"

[execution.local]
runtime = "auto"
young = 210e9
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot redefine scientific model"):
        project.ProjectConfig.load(root)


def test_bundle_rejects_obvious_machine_absolute_inputs(tmp_path):
    config = _write_project(tmp_path / "project")
    config.entrypoint.write_text(
        "from pathlib import Path\nmesh = Path('/Users/example/mesh.xdmf')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="AFM-PROJECT-PORTABILITY-001"):
        pack_project(config, tmp_path / "project.afm")


def test_bundle_is_deterministic_integrity_checked_and_secret_free(tmp_path):
    config = _write_project(tmp_path / "project")
    first = pack_project(config, tmp_path / "first.afm")
    second = pack_project(config, tmp_path / "second.afm")
    assert first.bundle_sha256 == second.bundle_sha256
    names = {record["path"] for record in first.files}
    assert names == {"agentfem.toml", "case.py", "mesh.dat"}
    assert ".env" in first.excluded
    assert "outputs/old.txt" in first.excluded

    destination = tmp_path / "unpacked"
    unpack_bundle(first.path, destination)
    assert (destination / "mesh.dat").read_text(encoding="utf-8") == (
        "scientific asset\n"
    )
    assert not (destination / ".env").exists()


def test_bundle_rejects_tampered_registered_file(tmp_path):
    config = _write_project(tmp_path / "project")
    report = pack_project(config, tmp_path / "project.afm")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(report.path, "a") as archive:
            archive.writestr("mesh.dat", b"tampered")
    with pytest.raises(ValueError, match="duplicate members"):
        inspect_bundle(report.path)


def test_bundle_rejects_path_traversal_before_materialization(tmp_path):
    malicious = tmp_path / "malicious.afm"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../outside.txt", b"do not extract")
        archive.writestr("agentfem.bundle.json", b"{}")
    with pytest.raises(ValueError, match="Unsafe AgentFEM bundle member"):
        inspect_bundle(malicious)
    assert not (tmp_path.parent / "outside.txt").exists()

    windows_escape = tmp_path / "windows-escape.afm"
    with zipfile.ZipFile(windows_escape, "w") as archive:
        archive.writestr("..\\outside.txt", b"do not extract")
        archive.writestr("agentfem.bundle.json", b"{}")
    with pytest.raises(ValueError, match="Unsafe AgentFEM bundle member"):
        inspect_bundle(windows_escape)


def test_cli_runs_verified_bundle_outside_its_source_tree(tmp_path):
    config = _write_project(tmp_path / "source")
    transport = tmp_path / "transport"
    transport.mkdir()
    bundle = pack_project(config, transport / "portable.afm")

    assert cli.main(["inspect", str(bundle.path), "--json"]) == 0
    assert cli.main(
        [
            "run",
            "--project",
            str(bundle.path),
            "--run-id",
            "bundle-run",
            "--json",
        ]
    ) == 0
    manifest_path = (
        transport
        / "outputs"
        / "portable"
        / "bundle-run"
        / "result.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest["metadata"]["run"]
    assert run["execution_profile"]["name"] == "local"
    assert run["project_source"]["kind"] == "agentfem_project_bundle"
    assert run["project_source"]["sha256"] == bundle.bundle_sha256
    assert manifest["quantities"] == ["answer"]


def test_cli_pack_and_unpack_round_trip(tmp_path):
    _write_project(tmp_path / "source")
    bundle = tmp_path / "transport.afm"
    destination = tmp_path / "destination"
    assert cli.main(
        [
            "pack",
            "--project",
            str(tmp_path / "source"),
            "--output",
            str(bundle),
        ]
    ) == 0
    assert cli.main(["unpack", str(bundle), str(destination)]) == 0
    assert project.ProjectConfig.load(destination).check() == ()


def test_runtime_equivalence_compares_declared_quantities_with_tolerance(tmp_path):
    paths = []
    for name, value in (("serial", 1.0), ("mpi", 1.0 + 2.0e-10)):
        result = results.SimulationResult(name)
        result.add_quantity("tip_displacement", value, unit="m")
        path = tmp_path / f"{name}.json"
        result.write_manifest(path)
        paths.append(path)
    comparison = portability.compare_results(
        paths,
        relative_tolerance=1.0e-8,
        absolute_tolerance=1.0e-12,
    )
    assert comparison.accepted
    assert comparison.quantities[0]["name"] == "tip_displacement"

    assert cli.main(
        [
            "compare-runs",
            *(str(path) for path in paths),
            "--quantity",
            "tip_displacement",
            "--json",
        ]
    ) == 0


def test_runtime_equivalence_rejects_scientific_difference(tmp_path):
    paths = []
    for name, value in (("serial", 1.0), ("mpi", 1.1)):
        result = results.SimulationResult(name)
        result.add_quantity("reaction", value, unit="N")
        path = tmp_path / f"{name}.json"
        result.write_manifest(path)
        paths.append(path)
    comparison = portability.compare_results(paths, relative_tolerance=1.0e-4)
    assert not comparison.accepted
    assert comparison.quantities[0]["status"] == "rejected"


def test_runtime_equivalence_rejects_tampered_manifest(tmp_path):
    paths = []
    for name in ("serial", "mpi"):
        result = results.SimulationResult(name)
        result.add_quantity("response", 1.0, unit="m")
        path = tmp_path / f"{name}.json"
        result.write_manifest(path)
        paths.append(path)
    tampered = json.loads(paths[1].read_text(encoding="utf-8"))
    tampered["quantity_records"][0]["value"] = 1.0
    tampered["metadata"]["tampered"] = True
    paths[1].write_text(json.dumps(tampered), encoding="utf-8")
    comparison = portability.compare_results(paths, quantities=["response"])
    assert not comparison.accepted
    assert any("provenance integrity" in item for item in comparison.issues)
