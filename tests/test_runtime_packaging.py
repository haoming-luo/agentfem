from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "packaging" / "runtime"


def test_runtime_specs_keep_heavy_optional_packs_out_of_core():
    specs = (RUNTIME / "runtime-specs.txt").read_text(encoding="utf-8").splitlines()
    names = {line.split("=")[0].strip() for line in specs if line.strip()}
    assert {"fenics-dolfinx", "dolfinx_mpc", "mpich", "petsc4py", "c-compiler"} <= names
    assert not ({"gmsh", "python-gmsh", "pytorch", "pyvista"} & names)


def test_complete_runtime_adds_gmsh_without_learning_or_visualization():
    specs = (RUNTIME / "runtime-specs-complete.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    names = {line.split("=")[0].strip() for line in specs if line.strip()}
    assert {"gmsh", "python-gmsh"} <= names
    assert not ({"pytorch", "pyvista"} & names)


def test_wsl_distribution_has_oobe_and_stable_default_name():
    configuration = (RUNTIME / "wsl" / "wsl-distribution.conf").read_text(
        encoding="utf-8"
    )
    assert "command = /usr/lib/agentfem/oobe.sh" in configuration
    assert "defaultUid = 1000" in configuration
    assert "defaultName = AgentFEM" in configuration
    assert "windowsterminal" in configuration


def test_windows_installer_fails_closed_on_existing_distribution():
    installer = (RUNTIME / "wsl" / "Install-AgentFEM.ps1").read_text(
        encoding="utf-8"
    )
    assert "will not overwrite it" in installer
    assert "Get-FileHash -Algorithm SHA256" in installer
    assert "--install --from-file" in installer
    assert "@IMAGE_FILENAME@" in installer
    assert "@IMAGE_SHA256@" in installer


def test_macos_public_artifact_requires_signing_identity():
    constructor = (RUNTIME / "macos" / "construct.yaml.in").read_text(
        encoding="utf-8"
    )
    assert 'environ.get("AGENTFEM_INSTALLER_IDENTITY")' in constructor
    assert "signing_identity_name" in constructor
    assert "notarization_identity_name" in constructor
    assert "environment_file: ../runtime-lock.txt" in constructor
    post_install = (RUNTIME / "macos" / "post_install.sh").read_text(
        encoding="utf-8"
    )
    assert "agentfem-*.whl" in post_install
    assert 'user_home=$(dirname "$runtime_parent")' in post_install


def test_complete_profile_pins_redistributed_gmsh_source_contract():
    builder = (RUNTIME / "build_runtime.py").read_text(encoding="utf-8")
    constructor = (RUNTIME / "macos" / "construct.yaml.in").read_text(
        encoding="utf-8"
    )
    assert '"4.15.2"' in builder
    assert "Gmsh-{version}-corresponding-source.tar.gz" in builder
    assert "feedstock_commit" in builder
    assert "runtime-sbom.cdx.json" in builder
    assert "initialize_conda: false" in constructor


def test_linux_cross_solver_declares_supported_glibc_abi():
    builder = (RUNTIME / "build_runtime.py").read_text(encoding="utf-8")
    assert 'solver_env["CONDA_OVERRIDE_GLIBC"] = "2.17"' in builder
    assert "run_with_retries(" in builder


def test_complete_wsl_bundle_carries_gmsh_compliance_materials():
    builder = (RUNTIME / "build_runtime.py").read_text(encoding="utf-8")
    dockerfile = (RUNTIME / "wsl" / "Dockerfile").read_text(encoding="utf-8")
    assert 'BUILD / "GMSH-LICENSE.txt"' in builder
    assert 'output / f"Gmsh-{version}-corresponding-source.tar.gz"' in builder
    assert 'path.suffix in {".wsl", ".gz"}' in builder
    assert "runtime-release.json" in dockerfile


def test_runtime_acceptance_requires_embedded_release_identity():
    verifier = (RUNTIME / "verify_installed_runtime.py").read_text(encoding="utf-8")
    assert '["runtime-record", str(runtime_record)]' in verifier
    assert '"passed": runtime_record.is_file()' in verifier


def test_manifest_command_is_deterministic_for_empty_output(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME / "build_runtime.py"),
            "manifest",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((tmp_path / "runtime-artifacts.json").read_text())
    assert manifest["schema"] == "agentfem.runtime-artifacts"
    assert manifest["agentfem_version"]
    assert manifest["artifacts"] == []
    assert (tmp_path / "SHA256SUMS").read_text() == ""
